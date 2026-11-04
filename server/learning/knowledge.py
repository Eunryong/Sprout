"""Knowledge Synthesis & Progressive Intelligence Scaling Engine (1B -> 7B Trajectory).

Manages the child's internal knowledge graph across turns, extracting concepts,
synthesizing cross-episode principles, computing cognitive scaling metrics (1B to 7B),
and converting insights into synthetic facts for TinyLM training.
"""
from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field

from server.learning.codec import TOK, Example, Fact
from server.world.situation import Observation, OptionView


# ---------------------------------------------------------------------------
# 1. Models
# ---------------------------------------------------------------------------

class EpisodeNode(BaseModel):
    turn: int
    scene_text: str
    action_text: str
    outcome_text: str
    takeaway: str
    outcome: str = "solved"
    parent_mode: str = "leave"


class ConceptNode(BaseModel):
    name: str
    observations: list[str] = Field(default_factory=list)
    valence_sum: float = 0.0
    count: int = 0


class PrincipleNode(BaseModel):
    id: str
    principle: str
    source_turns: list[int] = Field(default_factory=list)
    hop_depth: int = 2
    outcome_tendency: str = "solved"
    confidence: float = 1.0


class PersonalityNode(BaseModel):
    label: str = ""
    traits: list[str] = Field(default_factory=list)
    tone_guide: str = ""
    core_self: str = ""
    history: list[str] = Field(default_factory=list)


class KnowledgeGraph(BaseModel):
    episodes: list[EpisodeNode] = Field(default_factory=list)
    concepts: dict[str, ConceptNode] = Field(default_factory=dict)
    principles: list[PrincipleNode] = Field(default_factory=list)
    personality: PersonalityNode = Field(default_factory=PersonalityNode)


class ConceptObservation(BaseModel):
    name: str
    observation: str = ""


class SynthesisResponse(BaseModel):
    concepts: list[ConceptObservation] = Field(default_factory=list)
    synthesized_principle: str | None = None
    source_turns: list[int] = Field(default_factory=list)
    hop_depth: int = 2
    personality_label: str = Field(default="", description="Short open-ended description of the child's current evolving personality shaped by experiences")
    personality_traits: list[str] = Field(default_factory=list, description="2-4 dynamic personality traits evolved from experiences")
    personality_tone: str = Field(default="", description="How the child now speaks, reacts, and feels toward the parent and world")
    personality_shift: str = Field(default="", description="1-sentence first-person self-reflection on how this experience changed or reinforced my attitude/personality")


# ---------------------------------------------------------------------------
# 2. Cognitive Scaling Calculator
# ---------------------------------------------------------------------------

VALENCE_MAP: dict[str, float] = {
    "solved": 1.0,
    "partial": 0.5,
    "nothing": 0.0,
    "worse": -1.0,
}

def _compute_empirical_scores(graph: KnowledgeGraph) -> dict[str, Any]:
    guided_eps = [ep for ep in graph.episodes if ep.parent_mode in ("show", "say")]
    solo_eps = [ep for ep in graph.episodes if ep.parent_mode not in ("show", "say")]

    guided_count = len(guided_eps)
    solo_count = len(solo_eps)
    solved_count = sum(1 for ep in graph.episodes if ep.outcome == "solved")
    worse_count = sum(1 for ep in graph.episodes if ep.outcome == "worse")

    att_raw = sum(VALENCE_MAP.get(ep.outcome, 0.0) * 0.35 for ep in guided_eps) - sum(
        0.30 for ep in solo_eps if ep.outcome == "worse"
    )
    attachment = round(max(-1.0, min(1.0, att_raw)), 2)

    conf_raw = sum(VALENCE_MAP.get(ep.outcome, 0.0) * 0.30 for ep in graph.episodes) + sum(
        0.15 for ep in solo_eps if ep.outcome in ("solved", "partial")
    )
    confidence = round(max(-1.0, min(1.0, conf_raw)), 2)

    expressiveness = round(max(-1.0, min(1.0, (attachment + confidence) / 2.0)), 2)

    return {
        "attachment": attachment,
        "confidence": confidence,
        "expressiveness": expressiveness,
        "guided_count": guided_count,
        "solo_count": solo_count,
        "solved_count": solved_count,
        "worse_count": worse_count,
    }


def _synthesize_empirical_personality(
    graph: KnowledgeGraph, scores: dict[str, Any], is_ko: bool
) -> PersonalityNode:
    if not graph.episodes:
        return PersonalityNode()

    terms = extract_key_terms(
        " ".join(f"{ep.takeaway} {ep.outcome_text} {ep.action_text}" for ep in graph.episodes[-4:])
    )[:4]
    latest = graph.episodes[-1]
    core_self = (latest.takeaway or latest.outcome_text or latest.action_text).strip()
    traits = list(dict.fromkeys([*(graph.personality.traits or []), *terms]))[-4:]
    trait_summary = ", ".join(traits[:3]) if traits else core_self[:20]

    if is_ko:
        label = f"{trait_summary} 경험이 스며든 아이 (애착 {scores['attachment']:+.2f} · 자신감 {scores['confidence']:+.2f})"
        tone_guide = (
            f"내면화된 태도('{core_self}')와 경험 키워드({trait_summary}), 현재 정서(애착 {scores['attachment']:+.2f}, "
            f"자신감 {scores['confidence']:+.2f}, 표현력 {scores['expressiveness']:+.2f})가 말투에 자연스럽게 드러나도록 유아 눈높이에서 대답하세요."
        )
    else:
        label = f"Child shaped by {trait_summary} (att {scores['attachment']:+.2f}, conf {scores['confidence']:+.2f})"
        tone_guide = (
            f"Respond in a child voice naturally reflecting internalized attitude ('{core_self}'), "
            f"traits ({trait_summary}), and emotional state (attachment {scores['attachment']:+.2f}, confidence {scores['confidence']:+.2f})."
        )

    history = list(graph.personality.history)
    if core_self and (not history or history[-1] != core_self):
        history.append(core_self)
        history = history[-6:]

    return PersonalityNode(
        label=label,
        traits=traits,
        tone_guide=tone_guide,
        core_self=core_self,
        history=history,
    )


def compute_personality(graph: KnowledgeGraph, lang: str = "ko") -> dict[str, Any]:
    """Computes empirical continuous personality metrics and merges with graph.personality."""
    scores = _compute_empirical_scores(graph)
    is_ko = lang.startswith("ko")

    if graph.personality.label:
        label = graph.personality.label
        traits = list(graph.personality.traits)
        tone_guide = graph.personality.tone_guide
        core_self = graph.personality.core_self
        history = list(graph.personality.history)
    elif graph.episodes:
        p_node = _synthesize_empirical_personality(graph, scores, is_ko)
        label = p_node.label
        traits = list(p_node.traits)
        tone_guide = p_node.tone_guide
        core_self = p_node.core_self
        history = list(p_node.history)
    else:
        label = ""
        traits = []
        tone_guide = ""
        core_self = ""
        history = []

    return {
        "label": label,
        "traits": traits,
        "tone_guide": tone_guide,
        "core_self": core_self,
        "history": history,
        **scores,
    }


def cognitive_status(
    graph: KnowledgeGraph,
    lang: str = "ko",
    base_scale_b: float = 3.0,
) -> dict[str, Any]:
    """Computes the child's current effective intelligence scale from 1.0 (1B) to 7.0 (7B).

    Because 1B = 1세 기준 and 1 year = 12 months (12 scenes = 24 episodes in 2-step mode),
    intelligence starts at base_scale_b (default 3.0B at age 3) and grows by ~1.0B per 12-month year
    (~0.083B per scene / month):
    ep_bonus = min(2.4, n_ep * 0.025)
    concept_bonus = min(0.8, n_concepts * 0.004)
    prin_bonus = min(0.8, n_prin * 0.012)
    effective_depth = min(max_depth, 2 + n_ep // 12)
    depth_bonus = min(0.2, max(0, effective_depth - 2) * 0.04)
    raw_scale = base_scale_b + ep_bonus + concept_bonus + prin_bonus + depth_bonus
    scale_b = min(7.0, max(base_scale_b, raw_scale))
    """
    n_ep = len(graph.episodes)
    n_concepts = len(graph.concepts)
    n_prin = len(graph.principles)
    max_depth = max((p.hop_depth for p in graph.principles), default=1)

    ep_bonus = min(2.4, n_ep * 0.025)
    concept_bonus = min(0.8, n_concepts * 0.004)
    prin_bonus = min(0.8, n_prin * 0.012)
    effective_depth = min(max_depth, 2 + n_ep // 12)
    depth_bonus = min(0.2, max(0, effective_depth - 2) * 0.04)

    raw_scale = base_scale_b + ep_bonus + concept_bonus + prin_bonus + depth_bonus
    scale_b = min(7.0, max(base_scale_b, raw_scale))

    if scale_b < 2.0:
        hops = 1
        tier_name = (
            "1B (1세 수준 · 감각/단편 직관)"
            if lang.startswith("ko")
            else "1B (Age 1 · Sensory Intuition)"
        )
    elif scale_b < 3.0:
        hops = 1
        tier_name = (
            "2B (2세 수준 · 단어·행동 연상)"
            if lang.startswith("ko")
            else "2B (Age 2 · Action Association)"
        )
    elif scale_b < 4.0:
        hops = 2
        tier_name = (
            "3B (3세 수준 · 기초 개념 형성)"
            if lang.startswith("ko")
            else "3B (Age 3 · Basic Concepts)"
        )
    elif scale_b < 5.0:
        hops = 3
        tier_name = (
            "4B (4세 수준 · 인과 연결)"
            if lang.startswith("ko")
            else "4B (Age 4 · Causal Linking)"
        )
    elif scale_b < 6.0:
        hops = 4
        tier_name = (
            "5B (5세 수준 · 경험 비교·응용)"
            if lang.startswith("ko")
            else "5B (Age 5 · Experience Comparison)"
        )
    elif scale_b < 6.6:
        hops = 4
        tier_name = (
            "6B (6세 수준 · 복합 경험 결합)"
            if lang.startswith("ko")
            else "6B (Age 6 · Experience Synthesis)"
        )
    else:
        hops = 5
        tier_name = (
            "7B (7세 수준 · 원리 일반화)"
            if lang.startswith("ko")
            else "7B (Age 7 · Principle Generalization)"
        )

    return {
        "scale_b": round(scale_b, 1),
        "tier_name": tier_name,
        "reasoning_hops": hops,
        "n_episodes": n_ep,
        "n_concepts": n_concepts,
        "n_principles": n_prin,
        "recent_principles": [p.principle for p in graph.principles[-5:]],
        "personality": compute_personality(graph, lang=lang),
    }


# ---------------------------------------------------------------------------
# Dynamic Term Extraction (Fallback without hardcoded ontology)
# ---------------------------------------------------------------------------

GENERIC_STOPWORDS: set[str] = {
    # English common function words
    "the", "a", "an", "is", "are", "was", "were", "and", "or", "to", "in", "on",
    "at", "by", "for", "with", "it", "this", "that", "of", "from", "as", "be",
    "do", "did", "so", "not", "my", "your", "then", "have", "had", "all",
    # Korean common particles / functional auxiliaries
    "을", "를", "이", "가", "은", "는", "에", "의", "와", "과", "로", "으로",
    "에서", "하고", "하고는", "하면", "한다", "했다", "이다", "있다", "없다",
    "그", "저", "이것", "저것", "너무", "매우", "다", "더",
}

KO_SUFFIXES = (
    "으로", "에서", "에게", "한테", "처럼", "보다", "부터", "까지",
    "은", "는", "이", "가", "을", "를", "에", "의", "로", "도",
)


def extract_key_terms(text: str) -> list[str]:
    """Dynamically extracts meaningful content tokens without hardcoded ontology dictionaries."""
    if not text:
        return []

    tokens = re.findall(r"[A-Za-z0-9가-힣]+", text)
    terms: list[str] = []
    seen: set[str] = set()

    for tok in tokens:
        raw = tok.strip()
        cand = raw
        if len(raw) > 2:
            for sfx in KO_SUFFIXES:
                if raw.endswith(sfx) and len(raw) - len(sfx) >= 2:
                    cand = raw[:-len(sfx)]
                    break

        cand_lower = cand.lower()
        if len(cand) >= 2 and cand_lower not in GENERIC_STOPWORDS and cand_lower not in seen:
            seen.add(cand_lower)
            terms.append(cand)

    return terms


# ---------------------------------------------------------------------------
# Synthetic Fact & Example Conversion
# ---------------------------------------------------------------------------

def principle_to_fact(principle_node: PrincipleNode) -> Fact:
    """Converts a PrincipleNode into a synthetic Fact so TinyLM can train on text tokens."""
    return Fact(
        option=OptionView(verb="WAIT", option_text=principle_node.principle),
        result=principle_node.outcome_tendency or "solved",
        weight=1.0,
        action_text=principle_node.principle,
        outcome_text=principle_node.outcome_tendency or "solved",
        takeaway=principle_node.principle,
    )


def principle_to_example(principle_node: PrincipleNode, day: int = 1) -> Example:
    """Wraps a synthesized principle Fact in a valid Example for TinyLM training."""
    fact = principle_to_fact(principle_node)
    tag = "TIRED" if "TAG_TIRED" in TOK else "REQUEST"
    obs = Observation(
        tags=[tag] if f"TAG_{tag}" in TOK else [],
        scene_text=principle_node.principle,
        options=[fact.option],
    )
    return Example(
        id=f"synth_{principle_node.id}",
        day=day,
        source="principle",
        obs=obs,
        chosen=0,
        weight=fact.weight,
        facts=[fact],
    )


def personality_to_fact(personality: PersonalityNode, outcome: str = "solved") -> Fact:
    """Converts the child's evolving PersonalityNode into a Fact so TinyLM trains on personality text tokens."""
    text = f"{personality.label} | {personality.core_self} | {personality.tone_guide}".strip(" |")
    return Fact(
        option=OptionView(verb="WAIT", option_text=text),
        result=outcome if outcome in VALENCE_MAP else "solved",
        weight=1.0,
        action_text=personality.core_self or text,
        outcome_text=personality.tone_guide or text,
        takeaway=text,
    )


def personality_to_example(personality: PersonalityNode, day: int = 1, outcome: str = "solved") -> Example:
    """Wraps a dynamic personality Fact into an Example for TinyLM gradient updates."""
    fact = personality_to_fact(personality, outcome=outcome)
    tag = "TIRED" if "TAG_TIRED" in TOK else "REQUEST"
    obs = Observation(
        tags=[tag] if f"TAG_{tag}" in TOK else [],
        scene_text=personality.core_self or personality.label,
        options=[fact.option],
    )
    return Example(
        id=f"pers_day_{day}",
        day=day,
        source="personality",
        obs=obs,
        chosen=0,
        weight=fact.weight,
        facts=[fact],
    )


# ---------------------------------------------------------------------------
# 3. API Knowledge Expansion & Synthesis
# ---------------------------------------------------------------------------

SYNTHESIS_SYSTEM = """You are the cognitive synthesizer of a growing child agent.
Given the child's world premise, current episode, past episodes, existing principles, and current personality, extract at most 1 to 2 core child-level concepts (short concrete words/phrases, not adult psychological jargon) per turn, synthesize higher-order principles that link past and present experiences (only return a non-null synthesized_principle when there is a clear pattern across 2+ episodes, otherwise null), and synthesize the child's evolving personality: personality_label, personality_traits (2-4 traits), personality_tone, and personality_shift (a 1-sentence first-person self-reflection on how this experience shaped the child's attitude/personality, with zero static archetype names).

Respond strictly in valid JSON matching this schema:
{
  "concepts": [
    {"name": "<concept_name>", "observation": "<what was observed about this concept>"}
  ],
  "synthesized_principle": "<synthesized higher-order insight linking turns, or null if nothing to synthesize>",
  "source_turns": [<turn_numbers_linked>],
  "hop_depth": <integer >= 2, how many reasoning hops connect this insight>,
  "personality_label": "<short open-ended description of the child's current evolving personality shaped by experiences, with zero static archetype names>",
  "personality_traits": ["<trait1>", "<trait2>"],
  "personality_tone": "<how the child now speaks, reacts, and feels toward the parent and world>",
  "personality_shift": "<1-sentence first-person self-reflection on how this experience shaped or reinforced my attitude/personality>"
}
"""


def _apply_llm_synthesis(
    graph: KnowledgeGraph,
    episode: EpisodeNode,
    client: Any,
    world_premise: str,
    lang: str,
) -> SynthesisResponse:
    payload = {
        "world_premise": world_premise,
        "current_personality": graph.personality.model_dump(),
        "current_episode": {
            "turn": episode.turn,
            "scene_text": episode.scene_text,
            "action_text": episode.action_text,
            "outcome_text": episode.outcome_text,
            "takeaway": episode.takeaway,
            "outcome": episode.outcome,
            "parent_mode": episode.parent_mode,
        },
        "recent_episodes": [
            {
                "turn": ep.turn,
                "scene_text": ep.scene_text,
                "action_text": ep.action_text,
                "outcome_text": ep.outcome_text,
                "takeaway": ep.takeaway,
                "outcome": ep.outcome,
            }
            for ep in graph.episodes[:-1][-6:]
        ],
        "existing_principles": [
            {
                "id": p.id,
                "principle": p.principle,
                "source_turns": p.source_turns,
                "hop_depth": p.hop_depth,
            }
            for p in graph.principles[-4:]
        ],
        "language": "Korean" if lang.startswith("ko") else "English",
    }

    raw = client.complete_json(
        agent="narrator",
        system=SYNTHESIS_SYSTEM,
        user=json.dumps(payload, ensure_ascii=False),
        schema=SynthesisResponse,
    )

    if isinstance(raw, SynthesisResponse):
        return raw
    if isinstance(raw, dict):
        return SynthesisResponse.model_validate(raw)

    # Try model_validate as fallback
    return SynthesisResponse.model_validate(raw)


def _apply_dynamic_fallback(
    graph: KnowledgeGraph,
    episode: EpisodeNode,
    lang: str,
) -> tuple[list[ConceptObservation], str | None, list[int], int]:
    curr_terms = extract_key_terms(f"{episode.action_text} {episode.takeaway}")[:3]
    concepts_out: list[ConceptObservation] = [
        ConceptObservation(name=term, observation=episode.takeaway or episode.action_text)
        for term in curr_terms
    ]

    prior_episodes = graph.episodes[:-1]
    if not prior_episodes:
        return concepts_out, None, [], 2

    # Search for prior episode sharing key terms
    curr_terms_set = set(curr_terms)
    best_match: EpisodeNode | None = None
    matched_term: str | None = None

    for prev in reversed(prior_episodes):
        prev_terms = set(extract_key_terms(f"{prev.action_text} {prev.takeaway}"))
        overlap = curr_terms_set & prev_terms
        if overlap:
            best_match = prev
            matched_term = next(iter(overlap))
            break

    # If no shared term, check if they share outcome pattern
    if best_match is None:
        for prev in reversed(prior_episodes):
            if prev.outcome == episode.outcome:
                best_match = prev
                break

    existing_source_turns = {t for p in graph.principles for t in p.source_turns}
    can_synthesize = False
    if matched_term is not None:
        can_synthesize = True
    elif (
        best_match is not None
        and len(prior_episodes) >= 2
        and best_match.turn not in existing_source_turns
    ):
        can_synthesize = True

    if not can_synthesize or best_match is None:
        return concepts_out, None, [], 2

    source_turns = [best_match.turn, episode.turn]
    prev_depths = [p.hop_depth for p in graph.principles if best_match.turn in p.source_turns]
    hop_depth = max(prev_depths, default=1) + 1

    is_ko = lang.startswith("ko")
    if matched_term:
        if is_ko:
            synth = (
                f"[{matched_term}] 턴 {best_match.turn}과 턴 {episode.turn}의 경험: "
                f"'{best_match.takeaway}' & '{episode.takeaway}' -> {episode.outcome} 일관성"
            )
        else:
            synth = (
                f"[{matched_term}] Turns {best_match.turn} & {episode.turn}: "
                f"'{best_match.takeaway}' & '{episode.takeaway}' -> consistent {episode.outcome}"
            )
    else:
        if is_ko:
            synth = (
                f"턴 {best_match.turn}과 턴 {episode.turn}의 공통 결과({episode.outcome}): "
                f"'{best_match.action_text}'와 '{episode.action_text}' 행동 양식의 인과적 연결"
            )
        else:
            synth = (
                f"Turns {best_match.turn} & {episode.turn} shared outcome ({episode.outcome}): "
                f"causal link between '{best_match.action_text}' and '{episode.action_text}'"
            )

    return concepts_out, synth, source_turns, hop_depth


def expand_knowledge(
    graph: KnowledgeGraph,
    episode: EpisodeNode,
    client: Any = None,
    world_premise: str = "",
    lang: str = "ko",
) -> dict[str, Any]:
    """Expands the knowledge graph with a new episode, synthesizing concepts and principles."""
    # 1. Append episode to graph
    graph.episodes.append(episode)

    concepts_to_apply: list[ConceptObservation] = []
    synthesized_principle: str | None = None
    source_turns: list[int] = []
    hop_depth: int = 2

    # 2. Try LLM synthesis if client is provided
    used_llm = False
    synth_resp: SynthesisResponse | None = None
    if client is not None:
        try:
            synth_resp = _apply_llm_synthesis(
                graph, episode, client, world_premise, lang
            )
            concepts_to_apply = synth_resp.concepts[:2]
            synthesized_principle = synth_resp.synthesized_principle
            source_turns = synth_resp.source_turns
            hop_depth = synth_resp.hop_depth
            used_llm = True
        except Exception:
            used_llm = False

    # 3. Dynamic fallback if client is None or failed
    if not used_llm:
        c_list, s_principle, s_turns, depth = _apply_dynamic_fallback(graph, episode, lang)
        concepts_to_apply = c_list
        synthesized_principle = s_principle
        source_turns = s_turns
        hop_depth = depth

    # 4. Update concepts
    valence = VALENCE_MAP.get(episode.outcome, 0.0)
    updated_concepts: list[str] = []
    for co in concepts_to_apply:
        name = co.name.strip()
        if not name:
            continue
        if name not in graph.concepts:
            graph.concepts[name] = ConceptNode(name=name, observations=[], valence_sum=0.0, count=0)
        c_node = graph.concepts[name]
        c_node.count += 1
        c_node.valence_sum += valence
        if co.observation and co.observation not in c_node.observations:
            c_node.observations.append(co.observation)
        updated_concepts.append(name)

    # 5. Create PrincipleNode if synthesized
    new_principle_node: PrincipleNode | None = None
    fact: Fact | None = None
    example: Example | None = None

    if synthesized_principle and synthesized_principle.strip():
        prin_id = f"prin_{len(graph.principles) + 1:03d}"
        new_principle_node = PrincipleNode(
            id=prin_id,
            principle=synthesized_principle.strip(),
            source_turns=source_turns or [episode.turn],
            hop_depth=max(2, hop_depth),
            outcome_tendency=episode.outcome,
        )
        graph.principles.append(new_principle_node)
        fact = principle_to_fact(new_principle_node)
        example = principle_to_example(new_principle_node, day=max(1, episode.turn))

    # 6. Update personality
    if used_llm and synth_resp is not None:
        if synth_resp.personality_label:
            graph.personality.label = synth_resp.personality_label.strip()
        if synth_resp.personality_traits:
            graph.personality.traits = [t.strip() for t in synth_resp.personality_traits if t.strip()][:5]
        if synth_resp.personality_tone:
            graph.personality.tone_guide = synth_resp.personality_tone.strip()
        if synth_resp.personality_shift:
            shift = synth_resp.personality_shift.strip()
            graph.personality.core_self = shift
            graph.personality.history.append(shift)
            graph.personality.history = graph.personality.history[-6:]

    if not used_llm or not graph.personality.label or not graph.personality.core_self:
        empirical = _synthesize_empirical_personality(
            graph, _compute_empirical_scores(graph), lang.startswith("ko")
        )
        if not used_llm:
            graph.personality = empirical
        else:
            if not graph.personality.label:
                graph.personality.label = empirical.label
            if not graph.personality.traits:
                graph.personality.traits = empirical.traits
            if not graph.personality.tone_guide:
                graph.personality.tone_guide = empirical.tone_guide
            if not graph.personality.core_self:
                graph.personality.core_self = empirical.core_self
            if not graph.personality.history:
                graph.personality.history = empirical.history

    personality_example: Example | None = None
    if graph.personality.core_self or graph.personality.label:
        personality_example = personality_to_example(
            graph.personality, day=max(1, episode.turn), outcome=episode.outcome
        )

    status = cognitive_status(graph, lang=lang)

    return {
        "episode": episode,
        "new_concepts": updated_concepts,
        "new_principle": new_principle_node,
        "new_principles": [new_principle_node] if new_principle_node else [],
        "synthesized_principle": synthesized_principle,
        "fact": fact,
        "example": example,
        "personality_example": personality_example,
        "cognitive_status": status,
    }


# ---------------------------------------------------------------------------
# Persistence Helpers
# ---------------------------------------------------------------------------

def save_knowledge(graph: KnowledgeGraph, path: Path | str) -> None:
    """Saves the knowledge graph as JSON."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "w", encoding="utf-8") as f:
        f.write(graph.model_dump_json(indent=2))


def load_knowledge(path: Path | str) -> KnowledgeGraph:
    """Loads a knowledge graph from JSON. Returns empty graph if file doesn't exist."""
    p = Path(path)
    if not p.exists():
        return KnowledgeGraph()
    with open(p, "r", encoding="utf-8") as f:
        data = json.loads(f.read())
    return KnowledgeGraph.model_validate(data)
