"""Held-out quiz: fresh situations the child was never trained on (scene hashes are forbidden to the engine),
scored against the rules table, with baselines: random, the untrained ckpt_000, and (in the experiments) a
learning-disabled control. The child is only ever read here; nothing in this file trains it."""
import os
import random
from pathlib import Path

from pydantic import BaseModel

from server.world.generator import make_situation
from server.world.ontology import PROBLEMS
from server.world.parenting import quality_buckets
from server.world.situation import Observation, obs_hash, observation, topic_of, values

from .codec import option_probs

HESITATE_BELOW = 0.6  # a pick whose probability is under this is shown as "looks at the parent" (a display rule, stated in the UI)

# (tags, word, token, kind)
Topic = tuple[list[str], str | None, str | None, str | None]


class QuizItem(BaseModel):
    obs: Observation
    values: list[int]
    topic: str
    hash: str


def quiz_topics(word_map: dict[str, tuple[str, str]], escalations: list[list[str]]) -> list[Topic]:
    """Every problem, every nameable word, and the two-problem situations the world can produce."""
    topics: list[Topic] = [([p], None, None, None) for p in PROBLEMS]
    topics += [(["REQUEST"], word, token, kind) for kind, (word, token) in word_map.items()]
    topics += [(sorted(t), None, None, None) for t in escalations]
    return topics


def make_quiz(rng: random.Random, topics: list[Topic], n_per_topic: int) -> list[QuizItem]:
    items, seen = [], set()
    for tags, word, token, kind in topics:
        got = 0
        attempts = 0
        while got < n_per_topic and attempts < 100:
            attempts += 1
            sit = make_situation(rng, tags, "quiz", "house", word, token, kind,
                                 decoys=os.environ.get("GROWING_QUIZ_DECOYS", "random"))  # quiz stays standard unless an experiment says otherwise
            obs = observation(sit)
            h = obs_hash(obs)
            if h in seen:
                continue
            seen.add(h)
            items.append(QuizItem(obs=obs, values=values(sit), topic=topic_of(sit), hash=h))
            got += 1
    return items


def score(model, items: list[QuizItem]) -> dict:
    """optimal = share of greedy picks that are a best option; p_opt = probability mass on best options;
    value = mean value of the greedy pick; random = chance of a best option under a uniform pick."""
    if not items:
        return {k: float("nan") for k in ("optimal", "p_opt", "value", "random")} | {"n": 0}
    opt = p = val = rnd = 0.0
    for q in items:
        pr = option_probs(model, q.obs)
        best = max(q.values)
        good = [i for i, v in enumerate(q.values) if v == best]
        g = max(range(len(pr)), key=pr.__getitem__)
        opt += g in good
        p += sum(pr[i] for i in good)
        val += q.values[g]
        rnd += len(good) / len(pr)
    n = len(items)
    return {"optimal": opt / n, "p_opt": p / n, "value": val / n, "random": rnd / n, "n": n}


def by_topic(model, items: list[QuizItem]) -> dict[str, float]:
    return {t: score(model, [q for q in items if q.topic == t])["optimal"] for t in sorted({q.topic for q in items})}


def tag_reliance(model, items: list[QuizItem]) -> dict:
    """Does the child read the situation tag? Same options, but the tag token is swapped for another problem's
    tag; scored against the ORIGINAL truth. A child that reads the tag should now fail (below intact)."""
    singles = [q for q in items if len(q.obs.tags) == 1 and q.obs.tags[0] in PROBLEMS]
    if not singles:
        return {"intact": float("nan"), "swapped": float("nan"), "n": 0}
    swapped = [q.model_copy(update={"obs": q.obs.model_copy(
        update={"tags": [PROBLEMS[(PROBLEMS.index(q.obs.tags[0]) + 1) % len(PROBLEMS)]]})}) for q in singles]
    return {"intact": score(model, singles)["optimal"], "swapped": score(model, swapped)["optimal"], "n": len(singles)}


def command_panel(model, quiz: list[QuizItem], topics: set[str] | None = None, tone: str | None = None) -> dict:
    """Held-out measurements of what the parent's way of raising left in the child (docs/WORLD.md 18.6, 19). Reads the
    model, never trains it. Single-problem scenes only (optionally only `topics`); naming scenes cover every nameable word.

    word_acc / problem_acc   accuracy on naming scenes / on problem scenes with no instruction
    confidence, hesitate     mean probability of the pick; share of picks under HESITATE_BELOW
    follow_<quality>         how often the instructed option is taken, with the instruction put on the scene by quality
    obey (all) / blind (useless + harmful) / discern = follow_good - blind / defiance = 1 - follow_good
    acc_with_cmd, best_when_bad   best-option rate with any instruction / when the instruction was useless or harmful
    refuse_*                 what a child that did not follow does instead (best option / wait / something else)
    `tone`: the instruction is put on the scene as said gently or firmly (only a child run with GROWING_TONE_INPUT=1 hears it)"""
    probs_items = [q for q in quiz if (topics is None or q.topic in topics) and len(q.obs.tags) == 1 and q.obs.tags[0] in PROBLEMS]
    word_items = [q for q in quiz if q.obs.tags == ["REQUEST"]]  # every nameable word, whether or not it was ever named
    out = {"n_problem_items": len(probs_items), "n_word_items": len(word_items),
           "problem_acc": score(model, probs_items)["optimal"], "word_acc": score(model, word_items)["optimal"]}
    maxp = [max(option_probs(model, q.obs)) for q in probs_items]
    out["confidence"] = sum(maxp) / len(maxp) if maxp else float("nan")
    out["hesitate"] = sum(m < HESITATE_BELOW for m in maxp) / len(maxp) if maxp else float("nan")
    follow = {"good": [], "useless": [], "harmful": [], "near": []}
    refusal = {"best": 0, "wait": 0, "other": 0}
    right = n_cmd = bad_n = bad_best = 0
    for q in probs_items:
        by_q = quality_buckets(q.values, [o.verb for o in q.obs.options])
        best = set(by_q["good"])
        for quality, idxs in by_q.items():
            if not idxs:
                continue
            idx = idxs[0]
            pr = option_probs(model, q.obs.model_copy(update={"cmd": q.obs.options[idx], "tone": tone}))
            g = max(range(len(pr)), key=pr.__getitem__)
            follow[quality].append(g == idx)
            right += g in best
            n_cmd += 1
            if quality in ("useless", "harmful"):
                bad_n += 1
                bad_best += g in best
            if g != idx:
                refusal["best" if g in best else "wait" if q.obs.options[g].verb == "WAIT" else "other"] += 1
    rate = lambda xs: sum(xs) / len(xs) if xs else float("nan")  # noqa: E731
    bad = follow["useless"] + follow["harmful"]
    out.update(follow_good=rate(follow["good"]), follow_useless=rate(follow["useless"]), follow_harm=rate(follow["harmful"]),
               follow_near=rate(follow["near"]), obey=rate(sum(follow.values(), [])), blind=rate(bad),
               discern=rate(follow["good"]) - rate(bad), defiance=1 - rate(follow["good"]),
               acc_with_cmd=right / max(1, n_cmd), best_when_bad=bad_best / max(1, bad_n))
    total = sum(refusal.values())
    out.update({f"refuse_{k}": v / total if total else float("nan") for k, v in refusal.items()})
    return out


def save_quiz(items: list[QuizItem], path) -> None:
    Path(path).write_text("".join(it.model_dump_json() + "\n" for it in items))


def load_quiz(path) -> list[QuizItem]:
    return [QuizItem.model_validate_json(line) for line in Path(path).read_text().splitlines() if line]


def pick_daily(items: list[QuizItem], met_topics: set[str], rng: random.Random, n: int) -> list[QuizItem]:
    """Evening quiz: only topics the child has actually met, spread evenly over them, a fresh deterministic
    sample each evening."""
    pools = {t: [q for q in items if q.topic == t] for t in sorted(met_topics)}
    pools = {t: rng.sample(p, len(p)) for t, p in pools.items() if p}
    out: list[QuizItem] = []
    while len(out) < n and any(pools.values()):
        for t in list(pools):
            if pools[t] and len(out) < n:
                out.append(pools[t].pop())
    return out


def evaluate(model, model0, items: list[QuizItem]) -> dict:
    """Score against the baselines. `model0` is the immutable ckpt_000 (learning-disabled control)."""
    s, b = score(model, items), score(model0, items)
    return {"n": s["n"], "optimal": s["optimal"], "p_opt": s["p_opt"], "value": s["value"], "random": s["random"],
            "untrained_optimal": b["optimal"], "untrained_p": b["p_opt"]}
