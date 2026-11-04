"""Does saying a lesson firmly (repeating it) help or hurt? (docs/WORLD.md 22.8)

The same parent does the same things on the 8-day age schedule (docs/WORLD.md 22.7); only the TONE of the demonstrations and
corrections differs: gentle (the lesson is learned once) or firm (it is repeated `firm_repeat` times at that moment).
The learner, the model, the quiz and the parent's random choices are the same. Run on the 0-3 years world:

  uv run python scripts/exp_tone.py --styles balanced chooser commander --jobs 12 --out scripts/out/tone.json

arms: gentle | firm2 | firm3 (every demonstration and correction) | firm3c (corrections only: "no, do this!") |
firm3w (the lesson is not repeated but stays 3x as heavy whenever it is replayed).
Metrics at the last day, on held-out scenes: problem_acc (no instruction), word_acc, blind obedience, discernment, plus
  recall   share of the right lessons it was taught that it repeats on the very same scene
  gap      recall - problem_acc (how much better it knows what it was shown than what it was not shown)
  forget   mean over problems of (best accuracy so far - accuracy now)
  speed    accuracy on problems that opened on day 3 and 5, on the day after they opened
"""
import argparse
import json
import os
import random
import shutil
import statistics as st
import sys
import tempfile
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
os.environ.setdefault("GROWING_LLM", "mock")
os.environ.setdefault("GROWING_ONTOLOGY", str(Path(__file__).resolve().parents[1] / "config" / "experiments" / "onto_age03.yaml"))

import exp_ea  # noqa: E402

ARMS = {"gentle": (1, ("gentle", "gentle"), "repeat"), "firm2": (2, ("firm", "firm"), "repeat"),
        "firm3": (3, ("firm", "firm"), "repeat"), "firm3c": (3, ("gentle", "firm"), "repeat"),
        "firm3w": (3, ("firm", "firm"), "weight")}  # w: the lesson stays 3x as heavy in the replay buffer
COLS = ["problem_acc", "word_acc", "recall", "gap", "forget", "speed", "blind", "discern", "confidence", "latency"]


def run(seed: int, style: str, arm: str, days: int, steps: int = 0) -> dict:
    import torch
    torch.set_num_threads(1)
    from dataclasses import replace

    from server.config import load_configs
    from server.game.parents import play_day
    from server.game.session import Session
    from server.learning.codec import option_probs
    from server.world.ontology import PROBLEMS, opened_on

    repeat, tones, mode = ARMS[arm]
    g, _, o = load_configs()
    tmp = Path(tempfile.mkdtemp(prefix=f"tone{seed}-"))
    try:
        s = Session.new(tmp / "s", seed=seed, game=replace(g, firm_repeat=repeat, firm_mode=mode), ocfg=replace(o, steps=steps or o.steps))
        rng = random.Random(f"{seed}:{style}")  # the parent's own randomness: identical across arms
        peak: dict[str, float] = {}
        by_day = {}
        for day in range(1, days + 1):
            play_day(s, style, 1, rng, 0.0, tones=tones)
            res = s.end_day()
            singles = {t: v for t, v in res["by_topic"].items() if t in PROBLEMS}
            by_day[day] = singles
            for t, v in singles.items():
                peak[t] = max(peak.get(t, 0.0), v)
        panel = s.panel()
        taught = [e for e in s.buffer if e.source in ("show", "correct") and e.was_right]  # lessons that were right
        hits = []
        for e in taught:
            pr = option_probs(s.model, e.obs)
            hits.append(max(range(len(pr)), key=pr.__getitem__) == e.chosen)
        recall = sum(hits) / len(hits)
        last = by_day[days]
        speed = [by_day[opened_on(t) + 1][t] for t in PROBLEMS if opened_on(t) in (3, 5) and t in by_day.get(opened_on(t) + 1, {})]
        lat = [e.payload["report"]["seconds"] for e in s.engine.log if e.type.value == "learned"]
        return {"seed": seed, "style": style, "arm": arm, "problem_acc": panel["problem_acc"], "word_acc": panel["word_acc"],
                "recall": recall, "gap": recall - panel["problem_acc"],
                "forget": st.mean(peak[t] - last[t] for t in last), "speed": st.mean(speed) if speed else float("nan"),
                "blind": panel["blind"], "discern": panel["discern"], "confidence": panel["confidence"],
                "latency": st.mean(lat), "firm_moves": sum(d["firm"] for d in s.journal().values()),
                "taught": len(taught)}
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def job(a):
    return run(*a)


def report(results, styles, arms) -> str:
    out = []
    for style in styles:
        by = {a: sorted((r for r in results if r["style"] == style and r["arm"] == a), key=lambda r: r["seed"]) for a in arms}
        out.append(f"\n== parent: {style}, mean over {len(by[arms[0]])} seeds (last day) ==")
        out.append(f"{'arm':8s}" + "".join(f"{c:>12s}" for c in COLS) + f"{'firm moves':>12s}")
        for a in arms:
            out.append(f"{a:8s}" + "".join(f"{st.mean(r[c] for r in by[a]):12.3f}" for c in COLS)
                       + f"{st.mean(r['firm_moves'] for r in by[a]):12.1f}")
        out.append("paired difference to gentle, mean (sd of the difference):")
        for a in arms[1:]:
            cells = []
            for c in ("problem_acc", "word_acc", "recall", "gap", "forget", "speed", "blind", "discern"):
                d = [x[c] - y[c] for x, y in zip(by[a], by[arms[0]]) if x[c] == x[c] and y[c] == y[c]]
                cells.append(f"{c} {st.mean(d):+.3f}({st.pstdev(d):.3f})")
            out.append(f"  {a:8s}" + "  ".join(cells))
    return "\n".join(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", default="200-223")
    ap.add_argument("--days", type=int, default=8)
    ap.add_argument("--styles", nargs="+", default=["balanced", "chooser", "commander"])
    ap.add_argument("--arms", nargs="+", default=list(ARMS))
    ap.add_argument("--steps", type=int, default=0, help="gradient steps of one learning call (0 = the configured 60)")
    ap.add_argument("--jobs", type=int, default=12)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    tasks = [(s, st_, arm, a.days, a.steps) for st_ in a.styles for arm in a.arms for s in exp_ea.seeds_of(a.seeds)]
    t0 = time.time()
    with ProcessPoolExecutor(a.jobs) as ex:
        results = list(ex.map(job, tasks))
    from server.world.ontology import ONTOLOGY_PATH
    text = f"ontology: {ONTOLOGY_PATH.name}  seeds: {a.seeds}  days: {a.days}  steps per learning call: {a.steps or 60}" + report(results, a.styles, a.arms)
    print(text)
    print(f"\nwall={time.time() - t0:.0f}s")
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps({"results": results}))
    Path(a.out).with_suffix(".txt").write_text(text + "\n")


if __name__ == "__main__":
    main()
