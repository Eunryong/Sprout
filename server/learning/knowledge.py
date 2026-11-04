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


class KnowledgeGraph(BaseModel):
    episodes: list[EpisodeNode] = Field(default_factory=list)
    concepts: dict[str, ConceptNode] = Field(default_factory=dict)
    principles: list[PrincipleNode] = Field(default_factory=list)


class ConceptObservation(BaseModel):
    name: str
    observation: str = ""


class SynthesisResponse(BaseModel):
    concepts: list[ConceptObservation] = Field(default_factory=list)
    synthesized_principle: str | None = None
    source_turns: list[int] = Field(default_factory=list)
    hop_depth: int = 2


# ---------------------------------------------------------------------------
# 2. Cognitive Scaling Calculator
# ---------------------------------------------------------------------------

VALENCE_MAP: dict[str, float] = {
    "solved": 1.0,
    "partial": 0.5,
    "nothing": 0.0,
    "worse": -1.0,
}

WARM_CUES = (
    "안아", "토닥", "공감", "다정", "함께", "도와", "무릎을 꿇고",
    "알려주", "위로", "칭찬", "쓰다듬", "살펴",
    "hug", "comfort", "gentle", "together", "help", "warm",
)

SCOLD_CUES = (
    "혼내", "소리", "화내", "야단", "안 된다며", "다그치", "무섭", "혼나",
    "scold", "yell", "angry", "shout",
)

NEGLECT_CUES = (
    "스마트폰", "혼자 두", "무시", "기다리라고만", "외면", "방치",
    "혼자 남", "안 들어주",
    "phone", "ignore", "alone", "neglect",
)


def compute_personality(graph: KnowledgeGraph, lang: str = "ko") -> dict[str, Any]:
    """Computes learned personality scores and archetype based on the child's past experiences."""
    warm_count = 0
    scold_count = 0
    ignore_count = 0
    solved_count = 0
    partial_count = 0
    worse_count = 0
    solo_count = 0

    for ep in graph.episodes:
        text = f"{ep.action_text} {ep.outcome_text} {ep.takeaway}"

        has_warm_cue = any(c in text for c in WARM_CUES)
        has_scold_cue = any(c in text for c in SCOLD_CUES)
        has_neglect_cue = any(c in text for c in NEGLECT_CUES)

        # warm_count: episodes with warm cues or (parent_mode in ("show", "say") and outcome in ("solved", "partial") without scolding/neglect cues)
        if has_warm_cue or (
            ep.parent_mode in ("show", "say")
            and ep.outcome in ("solved", "partial")
            and not (has_scold_cue or has_neglect_cue)
        ):
            warm_count += 1

        # scold_count: episodes with scolding/harsh cues
        if has_scold_cue:
            scold_count += 1

        # ignore_count: episodes with neglect/ignoring cues or (parent_mode == "leave" and outcome == "worse")
        if has_neglect_cue or (ep.parent_mode == "leave" and ep.outcome == "worse"):
            ignore_count += 1

        if ep.outcome == "solved":
            solved_count += 1
        elif ep.outcome == "partial":
            partial_count += 1
        elif ep.outcome == "worse":
            worse_count += 1

        if ep.parent_mode in ("leave", "solo", "try"):
            solo_count += 1

    attachment = round(max(-1.0, min(1.0, (warm_count * 0.28 - scold_count * 0.35 - ignore_count * 0.30))), 2)
    confidence = round(max(-1.0, min(1.0, (solved_count * 0.25 + partial_count * 0.10 - worse_count * 0.22 - scold_count * 0.20))), 2)
    expressiveness = round(max(-1.0, min(1.0, (warm_count * 0.20 + solved_count * 0.15 - scold_count * 0.30 - ignore_count * 0.15))), 2)

    is_ko = lang.startswith("ko")

    if len(graph.episodes) == 0:
        archetype = "innocent_toddler"
        label = "순수하고 호기심 어린 꼬마" if is_ko else "Innocent & Curious Toddler"
        traits = ["순수함", "호기심"] if is_ko else ["innocent", "curious"]
        tone_guide = (
            "아직 많은 일을 겪지 않은 해맑고 순수한 아이입니다. 짧고 천진난만한 말투로 대답하세요."
            if is_ko
            else "A cheerful and innocent toddler who has not experienced much yet. Respond in a short, pure, childlike tone."
        )
    elif (scold_count + ignore_count) >= 1 and warm_count >= 1 and -0.5 <= attachment <= 0.35:
        archetype = "wary_sensitive"
        label = "눈치 보지만 다정함에 기대는 아이" if is_ko else "Sensitive & Cautiously Trusting"
        traits = ["섬세함", "살짝 눈치 봄", "애정 갈구"] if is_ko else ["sensitive", "cautious", "seeks warmth"]
        tone_guide = (
            "혼나거나 혼자 남겨져 서운했던 기억 때문에 처음엔 살짝 눈치를 보며 조심스럽게 말하지만, 부모가 다정하게 안아주던 따뜻함도 기억하고 있어 부모에게 기대고 싶어하는 말투로 대답하세요."
            if is_ko
            else "Cautious and hesitant at first due to memories of scolding or being left alone, but also remembering the warmth of parental hugs, wanting to lean on parents."
        )
    elif scold_count >= 1 and (attachment < 0 or confidence < 0) and scold_count >= ignore_count:
        archetype = "timid_anxious"
        label = "눈치 보고 주눅 든 소심이" if is_ko else "Timid & Cautious"
        traits = ["소심함", "눈치 봄", "주눅 듦"] if is_ko else ["timid", "anxious", "hesitant"]
        tone_guide = (
            "혼나거나 실패한 경험 때문에 자신감이 없고 부모의 눈치를 많이 봅니다. 말끝을 흐리거나('...', '그게...'), 혹시 자기가 또 잘못했는지 조심스럽게 묻는 소심한 말투로 대답하세요."
            if is_ko
            else "Lacks confidence and watches parents' reactions carefully due to scolding or failures. Hesitant tone, trailing off ('...', 'well...'), asking if they did something wrong."
        )
    elif ignore_count >= 1 and attachment < 0:
        archetype = "lonely_sulky"
        label = "외롭고 토라진 투정이" if is_ko else "Lonely & Sulky"
        traits = ["서운함", "투정", "관심 갈구"] if is_ko else ["lonely", "sulky", "wants attention"]
        tone_guide = (
            "부모가 관심을 안 주거나 혼자 둔 경험 때문에 서운함과 외로움이 쌓여 있습니다. '흥', '엄마는 맨날 바쁘잖아', '나 심심하단 말이야'처럼 뾰로통하게 투정 부리거나 관심을 달라고 조르는 말투로 대답하세요."
            if is_ko
            else "Harbors resentment and loneliness from being neglected or left alone. Sulky or whining tone ('Hmph', 'Mom is always busy', 'I'm bored!'), begging for attention."
        )
    elif confidence >= 0.25 and solo_count >= 1 and solved_count >= worse_count:
        archetype = "bold_explorer"
        label = "호기심 많고 당당한 탐험가" if is_ko else "Bold & Curious Explorer"
        traits = ["당당함", "자기주도적", "호기심"] if is_ko else ["bold", "independent", "curious"]
        tone_guide = (
            "스스로 해내며 성공한 경험이 많아 자신감이 넘치고 씩씩합니다. '응! 내가 해볼래!', '이거 재밌어!'처럼 당당하고 에너지 넘치는 말투로 대답하세요."
            if is_ko
            else "Full of confidence and high energy from solving things independently. Confident and energetic tone ('Yeah! I'll try!', 'This is fun!')."
        )
    elif attachment >= 0.15 or warm_count > (scold_count + ignore_count):
        archetype = "warm_affectionate"
        label = "다정하고 애교 많은 사랑둥이" if is_ko else "Warm & Affectionate"
        traits = ["다정함", "애교 많음", "안정된 애착"] if is_ko else ["warm", "affectionate", "secure"]
        tone_guide = (
            "부모의 따뜻한 공감과 보살핌을 받아 정서적으로 안정되어 있고 애교가 많습니다. '헤헤', '엄마랑 있는 게 제일 좋아!'처럼 밝고 사랑스러운 말투로 대답하세요."
            if is_ko
            else "Emotionally secure and affectionate from warm empathy and care. Bright and loving tone ('Hehe', 'I love being with mommy the most!')."
        )
    else:
        archetype = "innocent_toddler"
        label = "순수하고 호기심 어린 꼬마" if is_ko else "Innocent & Curious Toddler"
        traits = ["순수함", "호기심"] if is_ko else ["innocent", "curious"]
        tone_guide = (
            "세상을 조금씩 배워가는 순수한 아이입니다. 꾸밈없고 짧은 유아 말투로 대답하세요."
            if is_ko
            else "A pure child learning about the world step by step. Respond in an unadorned, short toddler tone."
        )

    return {
        "archetype": archetype,
        "label": label,
        "traits": traits,
        "tone_guide": tone_guide,
        "attachment": attachment,
        "confidence": confidence,
        "expressiveness": expressiveness,
        "warm_count": warm_count,
        "scold_count": scold_count,
        "ignore_count": ignore_count,
        "solved_count": solved_count,
        "worse_count": worse_count,
    }


def cognitive_status(
    graph: KnowledgeGraph,
    lang: str = "ko",
    base_scale_b: float = 3.0,
) -> dict[str, Any]:
    """Computes the child's current effective intelligence scale from 1.0 (1B) to 7.0 (7B).

    Because 1B = 1세 기준 and the game starts at 3세 (3년차), the child's starting
    baseline intelligence at Turn 0 (empty graph) is base_scale_b (default 3.0B).
    Grows smoothly via episodic accumulation + non-linear synthesis bonus:
    ep_bonus = min(1.6, n_ep * 0.06)
    concept_bonus = min(1.0, n_concepts * 0.03)
    prin_bonus = min(1.0, n_prin * 0.12)
    depth_bonus = min(0.6, max(0, max_depth - 2) * 0.15)
    raw_scale = base_scale_b + ep_bonus + concept_bonus + prin_bonus + depth_bonus
    scale_b = min(7.0, max(base_scale_b, raw_scale))
    """
    n_ep = len(graph.episodes)
    n_concepts = len(graph.concepts)
    n_prin = len(graph.principles)
    max_depth = max((p.hop_depth for p in graph.principles), default=1)

    ep_bonus = min(1.6, n_ep * 0.06)
    concept_bonus = min(1.0, n_concepts * 0.03)
    prin_bonus = min(1.0, n_prin * 0.12)
    depth_bonus = min(0.6, max(0, max_depth - 2) * 0.15)

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


# ---------------------------------------------------------------------------
# 3. API Knowledge Expansion & Synthesis
# ---------------------------------------------------------------------------

SYNTHESIS_SYSTEM = """You are the cognitive synthesizer of a growing child agent.
Given the child's world premise, current episode, past episodes, and existing principles, extract at most 1 to 2 core child-level concepts (short concrete words/phrases, not adult psychological jargon) per turn, and synthesize higher-order principles that link past and present experiences. Only return a non-null synthesized_principle when there is a clear pattern across 2+ episodes (otherwise null).

Respond strictly in valid JSON matching this schema:
{
  "concepts": [
    {"name": "<concept_name>", "observation": "<what was observed about this concept>"}
  ],
  "synthesized_principle": "<synthesized higher-order insight linking turns, or null if nothing to synthesize>",
  "source_turns": [<turn_numbers_linked>],
  "hop_depth": <integer >= 2, how many reasoning hops connect this insight>
}
"""


def _apply_llm_synthesis(
    graph: KnowledgeGraph,
    episode: EpisodeNode,
    client: Any,
    world_premise: str,
    lang: str,
) -> tuple[list[ConceptObservation], str | None, list[int], int]:
    payload = {
        "world_premise": world_premise,
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
        return raw.concepts, raw.synthesized_principle, raw.source_turns, raw.hop_depth
    if isinstance(raw, dict):
        parsed = SynthesisResponse.model_validate(raw)
        return parsed.concepts, parsed.synthesized_principle, parsed.source_turns, parsed.hop_depth

    # Try model_validate as fallback
    parsed = SynthesisResponse.model_validate(raw)
    return parsed.concepts, parsed.synthesized_principle, parsed.source_turns, parsed.hop_depth


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
    if client is not None:
        try:
            c_list, s_principle, s_turns, depth = _apply_llm_synthesis(
                graph, episode, client, world_premise, lang
            )
            concepts_to_apply = c_list
            synthesized_principle = s_principle
            source_turns = s_turns
            hop_depth = depth
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
    else:
        concepts_to_apply = concepts_to_apply[:2]

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

    status = cognitive_status(graph, lang=lang)

    return {
        "episode": episode,
        "new_concepts": updated_concepts,
        "new_principle": new_principle_node,
        "new_principles": [new_principle_node] if new_principle_node else [],
        "synthesized_principle": synthesized_principle,
        "fact": fact,
        "example": example,
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
