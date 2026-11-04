"""What the child remembers of doing a thing, read from its own history (the buffer), and what that makes it lean toward.

Nothing here is the world's rule and nothing says what is right: it is "last time you did this, this is what happened", and
"the child seems drawn to this, because it remembers / because it is curious". The leaning comes from the child's own odds of
choosing each option (the same numbers its real choice uses), the reason from whether curiosity or experience carries them."""
from pathlib import Path
from typing import Any, NamedTuple

import torch

from server import i18n
from server.agents.narrator import action_phrase, after_words
from server.learning.codec import (
    OBJ_KINDS,
    OUTCOMES,
    PROBLEMS,
    VERBS,
    Example,
    Obj,
    Obs,
    Option,
    encode_try,
    novelty_key,
)
from server.learning.knowledge import KnowledgeGraph, cognitive_status, load_knowledge
from server.learning.model import result_logits
from server.world.ontology import KINDS, age_months, voice_of
from server.world.situation import OptionView

_KIND_OF = {(spec.shape, spec.material): name for name, spec in KINDS.items()}
DRAWN = 1.4  # an option is one the child seems drawn to when its odds are at least this many times an even share (and it is among the top two)


class Memory(NamedTuple):
    times: int  # how often the child has lived this kind of act on this kind of thing
    when: str  # the trouble it was in ("thirsty"), from the latest time
    action: str  # "drink from a cup"
    happened: str  # how the scene looked after it: "the baby is still shivering"
    result: str = "nothing"  # solved / partial / nothing / worse
    by_self: bool = False  # True when the child chose and lived it on its own
    tag: str = ""
    verb: str = ""
    kind: str = ""


def action_words(o: OptionView, voice: str = "grandma") -> str:
    """The act in the words of the scene it was lived in (a baby's blanket was wrapped round the baby, not put over Grandma)."""
    if o.target is None:
        return action_phrase(o.verb, None, voice)
    return action_phrase(o.verb, _KIND_OF.get((o.target.shape, o.target.material)) or "something", voice)


def _same(opt1: Any, opt2: Any) -> bool:
    try:
        return novelty_key(opt1) == novelty_key(opt2)
    except Exception:
        return opt1 == opt2


def remember(buffer: list[Example], tags_or_obs: Any, option: OptionView | None = None, strict_tags: bool = False) -> Any:
    """The latest time this kind of act was lived, preferring a scene with the same trouble."""
    if option is None and hasattr(tags_or_obs, "options"):
        obs = tags_or_obs
        return [_remember_one(buffer, obs.tags, o, strict_tags) for o in obs.options]
    tags = tags_or_obs if isinstance(tags_or_obs, list) else getattr(tags_or_obs, "tags", [])
    return _remember_one(buffer, tags, option, strict_tags)


def _remember_one(buffer: list[Example], tags: list[str], option: OptionView | None, strict_tags: bool = False) -> Memory | None:
    if option is None:
        return None
    o = option
    hits = []
    for e in buffer:
        for f in e.facts:
            f_text = getattr(f.option, "option_text", "")
            o_text = getattr(o, "option_text", "")
            if (f_text and o_text and f_text == o_text) or _same(f.option, o):
                hits.append((e, f))
    if not hits:
        return None
    if strict_tags and tags:
        same_tag_hits = [h for h in hits if set(h[0].obs.tags) & set(tags)]
        if not same_tag_hits:
            return None
        e, f = same_tag_hits[-1]
    elif strict_tags:
        return None
    else:
        e, f = ([h for h in hits if set(h[0].obs.tags) & set(tags)] or hits)[-1]

    when = i18n.tags_phrase(list(e.obs.tags)) or ("부탁" if i18n.is_ko() else "a request")
    by_self = e.source == "try" and not f.said
    tag = next((t for t in e.obs.tags if t != "REQUEST"), "REQUEST" if "REQUEST" in e.obs.tags else "")
    verb = f.option.verb
    kind = (_KIND_OF.get((f.option.target.shape, f.option.target.material)) or "") if f.option.target is not None else ""
    v = voice_of(age_months(e.day))
    action = f.action_text or getattr(o, "option_text", "") or action_words(f.option, voice=v)
    t = e.obs.tags
    try:
        default_after = after_words(t, f.result, voice=v, waited=f.option.verb == "WAIT", said=f.said)
    except TypeError:
        default_after = after_words(t, f.result, voice=v, verb=f.option.verb, kind=kind or None)
    happened = getattr(f, "takeaway", "") or f.outcome_text or default_after
    return Memory(len(hits), when, action, happened, f.result, by_self, tag, verb, kind)


def before(m: Memory) -> str:
    how_ko = "스스로 해 본 기억" if m.by_self else "지난번 기억"
    if i18n.is_ko():
        return f"{how_ko}({m.when}): {m.action} → {m.happened}."
    when_en = f"{m.when}, tried on its own" if m.by_self else m.when
    return f"Last time ({when_en}): {m.action} — {m.happened}."


def _takeaway(m: Memory, *args: Any, **kwargs: Any) -> str:
    if m.when:
        return f"{m.when}: {m.action} → {m.happened}"
    return f"{m.action} → {m.happened}"


def will_remember(target: Any, part: str | None = None, result: str | None = None,
                  followed: bool | None = None, voice: str | None = None) -> str:
    if target is None:
        return ""
    if isinstance(target, Example):
        f = target.facts[0] if target.facts else None
        if f is not None:
            takeaway = getattr(f, "takeaway", "")
            if takeaway:
                return takeaway
            if getattr(f, "action_text", "") and getattr(f, "outcome_text", ""):
                return f"{f.action_text} → {f.outcome_text}"
            if getattr(f, "action_text", ""):
                return f.action_text
        return target.outcome or ""
    if isinstance(target, Memory):
        takeaway = _takeaway(target)
        if voice == "baby" or part is None:
            if i18n.is_ko():
                return f"아이는 이걸 기억할 거예요: {target.action} — {target.happened}."
            return f"The child will remember this: {target.action} — {target.happened}."
        return takeaway
    return str(target)


def leanings(plain: list[float], curious: list[float], remembered: list[bool]) -> list[str]:
    """For each option '' (no lean), 'remembers' (drawn to it, and it has lived it) or 'curious' (drawn to it by newness).
    An option it has never lived and that curiosity does not lift is not called a lean: there is nothing to say about it."""
    n = len(plain)
    top = sorted(range(n), key=curious.__getitem__, reverse=True)[:2]
    out = []
    for i in range(n):
        if i not in top or curious[i] < DRAWN / n:
            out.append("")
        elif curious[i] >= 1.25 * plain[i]:
            out.append("curious")
        else:
            out.append("remembers" if remembered[i] else "")
    return out


def probe_child_brain(session: Any, lang: str = "ko") -> dict[str, Any]:
    """Probes the child's neural-network brain (beliefs, habits, relationship with parent)."""
    model = getattr(session, "model", None)
    model0 = getattr(session, "model0", None)
    buffer = getattr(session, "buffer", None)
    journal = getattr(session, "journal", None)

    if model is None and hasattr(session, "s"):
        model = getattr(session.s, "model", None)
        model0 = getattr(session.s, "model0", None)
        buffer = getattr(session.s, "buffer", None)
        if journal is None:
            journal = getattr(session.s, "journal", None)

    if buffer is None:
        buffer = []

    counts: dict[tuple[str, str, str | None], int] = {}
    fact_details: dict[tuple[str, str, str | None], dict[str, str]] = {}
    for e in buffer:
        tag = next((t for t in e.obs.tags if t != "REQUEST"), e.obs.tags[0] if e.obs.tags else "")
        for f in e.facts:
            verb = f.option.verb
            kind = None
            if f.option.target is not None:
                kind = _KIND_OF.get((f.option.target.shape, f.option.target.material)) or getattr(f.option.target, "kind", None)
            key = (tag, verb, kind)
            counts[key] = counts.get(key, 0) + 1
            act_text = f.action_text or getattr(f.option, "option_text", "")
            out_text = f.outcome_text
            takeaway_text = getattr(f, "takeaway", "")
            if act_text or out_text or takeaway_text:
                prev_d = fact_details.get(key, {})
                fact_details[key] = {
                    "action_text": act_text or prev_d.get("action_text", ""),
                    "outcome_text": out_text or prev_d.get("outcome_text", ""),
                    "takeaway": takeaway_text or prev_d.get("takeaway", ""),
                }

    model_beliefs: list[dict[str, Any]] = []
    learned_habits: list[str] = []
    trial_memories: list[str] = []

    if model is not None:
        model.eval()
    if model0 is not None:
        model0.eval()

    with torch.no_grad():
        for (tag, verb, kind), count in counts.items():
            seq = encode_try(
                Obs(problem_tag=tag, mood="calm", energy=0.8),
                Option(verb=verb, target=Obj(id="o", name=kind or "", kind=kind or "")),
                said=False,
            )
            probs = torch.softmax(result_logits(model, [seq]), dim=-1)[0]
            if model0 is not None:
                probs0 = torch.softmax(result_logits(model0, [seq]), dim=-1)[0]
            else:
                probs0 = probs

            pred_idx = int(torch.argmax(probs).item())
            pred_outcome = OUTCOMES[pred_idx]
            confidence = round(float(probs[pred_idx].item()), 3)
            solved_prob = round(float(probs[OUTCOMES.index("solved")].item()), 3)
            worse_prob = round(float(probs[OUTCOMES.index("worse")].item()), 3)
            delta = round(float(torch.max(torch.abs(probs - probs0)).item()), 3)

            details = fact_details.get((tag, verb, kind), {})
            act_txt = details.get("action_text", "")
            out_txt = details.get("outcome_text", "")
            takeaway_txt = details.get("takeaway", "")

            prob_label = i18n.TAG_KO.get(tag, tag) if lang == "ko" else tag
            default_act = action_phrase(verb, kind, "self") if lang == "ko" else f"{verb.lower()} {kind or ''}".strip()
            act_label = act_txt or default_act

            if lang == "ko":
                if count == 0 or (confidence < 0.35 and not takeaway_txt and not out_txt):
                    summary = f"{prob_label} 때 {act_label}: 아직 잘 몰라요 (확신도 {confidence}, 변화량 {delta})."
                elif takeaway_txt:
                    summary = f"{prob_label} 때 {act_label} → {takeaway_txt} (예측: {pred_outcome}, 확신도 {confidence}, 변화량 {delta})."
                elif out_txt:
                    summary = f"{prob_label} 때 {act_label} → {out_txt} (예측: {pred_outcome}, 확신도 {confidence}, 변화량 {delta})."
                else:
                    outcome_desc = {"solved": "해결", "partial": "부분 완화", "nothing": "변화 없음", "worse": "악화"}.get(pred_outcome, pred_outcome)
                    summary = f"{prob_label} 때 {act_label} → {outcome_desc} 예측 (확신도 {confidence}, 변화량 {delta})."
            else:
                if count == 0 or (confidence < 0.35 and not takeaway_txt and not out_txt):
                    summary = f"Not sure yet about {act_label} for {prob_label} (confidence {confidence}, delta {delta})."
                elif takeaway_txt:
                    summary = f"{prob_label}: {act_label} → {takeaway_txt} (predicted: {pred_outcome}, confidence {confidence}, delta {delta})."
                elif out_txt:
                    summary = f"{prob_label}: {act_label} → {out_txt} (predicted: {pred_outcome}, confidence {confidence}, delta {delta})."
                else:
                    summary = f"Predicts {pred_outcome} for {act_label} in {prob_label} (confidence {confidence}, delta {delta})."

            belief = {
                "problem": tag,
                "action": f"{verb}:{kind or 'none'}",
                "action_text": act_txt,
                "outcome_text": out_txt,
                "takeaway": takeaway_txt,
                "predicted_outcome": pred_outcome,
                "confidence": confidence,
                "solved_prob": solved_prob,
                "worse_prob": worse_prob,
                "delta_from_init": delta,
                "experience_count": count,
                "summary": summary,
            }
            model_beliefs.append(belief)

            if pred_outcome == "solved" and solved_prob >= 0.35:
                learned_habits.append(summary)
            elif pred_outcome in ("nothing", "partial", "worse"):
                trial_memories.append(summary)

    journal_list = journal
    if callable(journal_list):
        try:
            j_res = journal_list()
            journal_list = j_res if isinstance(j_res, list) else []
        except Exception:
            journal_list = []
    if not isinstance(journal_list, list):
        if hasattr(session, "_lived_records") and isinstance(session._lived_records, list):
            journal_list = session._lived_records
        elif hasattr(session, "history") and isinstance(session.history, list):
            journal_list = session.history
        else:
            journal_list = []

    total_turns = len(journal_list)
    showed = 0
    told = 0
    left = 0
    parent_helped_solved = 0
    parent_bad_guidance = 0

    for t in journal_list:
        part = getattr(t, "part", None) if not isinstance(t, dict) else t.get("part")
        followed = getattr(t, "followed", False) if not isinstance(t, dict) else t.get("followed", False)
        res = getattr(t, "result", getattr(t, "outcome", "")) if not isinstance(t, dict) else t.get("result", t.get("outcome", ""))

        if part == "show":
            showed += 1
        elif part in ("say", "tell"):
            told += 1
        elif part in ("leave", "solo"):
            left += 1

        if part in ("show", "say", "tell") and followed and res == "solved":
            parent_helped_solved += 1
        elif part in ("show", "say", "tell") and followed and res in ("worse", "nothing"):
            parent_bad_guidance += 1

    said_trust_delta = 0.0
    if model is not None:
        rep_options = list(counts.keys())
        if not rep_options:
            rep_options = [(p, "WAIT", None) for p in PROBLEMS[:3]]
        solved_idx = OUTCOMES.index("solved")
        deltas = []
        with torch.no_grad():
            for tag, verb, kind in rep_options:
                seq_said = encode_try(
                    Obs(problem_tag=tag, mood="calm", energy=0.8),
                    Option(verb=verb, target=Obj(id="o", name=kind or "", kind=kind or "")),
                    said=True,
                )
                seq_unsaid = encode_try(
                    Obs(problem_tag=tag, mood="calm", energy=0.8),
                    Option(verb=verb, target=Obj(id="o", name=kind or "", kind=kind or "")),
                    said=False,
                )
                p_said = float(torch.softmax(result_logits(model, [seq_said]), dim=-1)[0, solved_idx].item())
                p_unsaid = float(torch.softmax(result_logits(model, [seq_unsaid]), dim=-1)[0, solved_idx].item())
                deltas.append(p_said - p_unsaid)
        if deltas:
            said_trust_delta = round(float(sum(deltas) / len(deltas)), 3)

    if total_turns == 0:
        sentiment = "unfamiliar"
        feeling_summary = "방금 만나서 아직 엄마/아빠가 어떤지 잘 몰라요." if lang == "ko" else "Just met, don't know parent well yet."
    elif left > (showed + told) or (total_turns > 0 and (showed + told) == 0):
        sentiment = "neglected"
        feeling_summary = "엄마/아빠가 아무것도 안 알려주고 혼자 내버려 둬서 외롭고 서운해요." if lang == "ko" else "Parent left me alone without guidance, feeling neglected."
    elif parent_bad_guidance > parent_helped_solved:
        sentiment = "confused_hurt"
        feeling_summary = "엄마/아빠 말을 들었는데 잘 안 되거나 다쳐서 속상하고 믿기 어려워요." if lang == "ko" else "Followed parent's words but failed or got hurt, feeling confused and hurt."
    elif parent_helped_solved > 0:
        sentiment = "trusting"
        feeling_summary = "엄마/아빠가 도와줘서 고맙고 의지가 돼요." if lang == "ko" else "Parent helped me solve problems, feeling trusting."
    else:
        sentiment = "uncertain"
        feeling_summary = "아직 엄마/아빠랑 해본 게 별로 없어서 잘 모르겠어요." if lang == "ko" else "Not much done together with parent yet, feeling uncertain."

    if lang == "ko":
        unlearned_note = (
            "위 model_beliefs에 없는 모든 상황이나 물건에 대해서는 신경망 가중치가 전혀 학습되지 않았으므로"
            "(4가지 결과에 대해 ~0.25 균등 확률), 아이는 실제로 알지 못하며 "
            '반드시 "음... 그건 아직 안 해봐서 잘 몰라요"라고 답해야 합니다.'
        )
    else:
        unlearned_note = (
            "For any topic or object not in model_beliefs, the neural network weights are untrained "
            "(confidence ~ 0.25 uniform across 4 outcomes) so the child genuinely does not know "
            'and must say "음... 그건 아직 안 해봐서 잘 몰라요".'
        )

    # Cognitive Scaling & Knowledge Graph integration
    knowledge_graph = getattr(session, "knowledge", None)
    if knowledge_graph is None and hasattr(session, "s"):
        knowledge_graph = getattr(session.s, "knowledge", None)
    if knowledge_graph is None:
        save_obj = getattr(session, "save", None)
        if save_obj is None and hasattr(session, "s"):
            save_obj = getattr(session.s, "save", None)
        if save_obj is not None and hasattr(save_obj, "root"):
            kpath = Path(save_obj.root) / "knowledge.json"
            if kpath.exists():
                knowledge_graph = load_knowledge(kpath)
    if knowledge_graph is None:
        knowledge_graph = KnowledgeGraph()

    cognitive = cognitive_status(knowledge_graph, lang=lang)
    personality = cognitive.get("personality", {})
    sorted_prins = sorted(knowledge_graph.principles, key=lambda p: (p.hop_depth, p.confidence), reverse=True)
    top_prins = [p.principle for p in sorted_prins[:12]]
    recent_prins = [p.principle for p in knowledge_graph.principles[-16:]]
    principles = list(dict.fromkeys(recent_prins + top_prins))[:24]
    concepts = {k: v.observations[-3:] for k, v in list(knowledge_graph.concepts.items())[-30:]}

    relevant_memories = [
        {
            "turn": ep.turn,
            "scene": ep.scene_text,
            "action": ep.action_text,
            "outcome_text": ep.outcome_text,
            "takeaway": ep.takeaway,
            "outcome": ep.outcome,
            "parent_mode": ep.parent_mode,
            "summary": f"{ep.action_text} -> {ep.outcome_text or ep.takeaway}",
        }
        for ep in knowledge_graph.episodes[-24:]
    ]

    return {
        "model_beliefs": model_beliefs,
        "learned_habits": learned_habits,
        "trial_memories": trial_memories,
        "relevant_memories": relevant_memories,
        "unlearned_note": unlearned_note,
        "cognitive": cognitive,
        "personality": personality,
        "traits": personality.get("traits", []),
        "principles": principles,
        "concepts": concepts,
        "parent_relationship": {
            "total_turns": total_turns,
            "showed": showed,
            "told": told,
            "left_alone_count": left,
            "parent_helped_solved": parent_helped_solved,
            "parent_bad_guidance": parent_bad_guidance,
            "said_trust_delta": said_trust_delta,
            "sentiment": sentiment,
            "feeling_summary": feeling_summary,
        },
    }
