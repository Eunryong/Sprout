"""Learn a wrong choice by what it did (docs/WORLD.md 22.11)

When the child chooses by itself and chooses wrong, the parent puts it right. Here the strength of that lesson depends on what
the child's own choice DID: it helped a little (partial), nothing happened (nothing), or it made things worse (worse). Run on the
0-3 years world with -1 rules (config/experiments/onto_age03_harm.yaml), 8 days, engine path, the same parent in every arm.

arms (learning rounds of: praise | partial | nothing | worse):  base 1|1|1|1   flat3 3|3|3|3   severity 1|1|3|5   worse-only 1|1|1|5
Measured on the last day, on held-out scenes with no instruction and on the lessons themselves:
  harm      share of scenes where the child's pick makes things worse (harm_avg: averaged over the 8 days)     waste   share where it does nothing useful (value 0)
  repeat    mean probability of choosing again the very option it had chosen wrong (lower = learned from it), by what it did
  rounds    learning rounds per run (the compute each arm spends)

  uv run python scripts/exp_outcome.py --out scripts/out/outcome.json
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
os.environ.setdefault("GROWING_ONTOLOGY", str(Path(__file__).resolve().parents[1] / "config" / "experiments" / "onto_age03_harm.yaml"))

import exp_ea  # noqa: E402

# name: (own_rounds = praise, partial, nothing, worse)
ARMS = {"base": (1, 0, 0, 0), "flat3": (3, 0, 0, 0), "severity": (1, 1, 3, 5), "worse-only": (1, 1, 1, 5)}
OUTCOMES = ("partial", "nothing", "worse")
COLS = ["problem_acc", "harm", "harm_avg", "waste", "follow_harm", "blind", "word_acc", "speed", "forget", "rounds", "ms"]


def run(seed: int, style: str, arm: str, days: int) -> dict:
    import torch
    torch.set_num_threads(1)
    from dataclasses import replace

    from server.config import load_configs
    from server.game.parents import play_day
    from server.game.session import Session
    from server.learning.codec import option_probs
    from server.world.ontology import PROBLEMS, opened_on

    own, partial, nothing, worse = ARMS[arm]
    g, _, o = load_configs()
    game = replace(g, own_rounds=own, partial_rounds=partial, nothing_rounds=nothing, worse_rounds=worse, firm_repeat=1)
    tmp = Path(tempfile.mkdtemp(prefix=f"oc{seed}-"))
    try:
        s = Session.new(tmp / "s", seed=seed, game=game)
        rng = random.Random(f"{seed}:{style}")
        peak: dict[str, float] = {}
        by_day = {}
        curve = []  # share of scenes, day by day, where the child's pick (no instruction) makes things worse
        for day in range(1, days + 1):
            play_day(s, style, 1, rng, 0.0)
            res = s.end_day()
            qs = [q for q in s.quiz if q.topic in (s.met_topics() & set(PROBLEMS)) and len(q.obs.tags) == 1]
            curve.append(st.mean(q.values[max(range(len(q.values)), key=option_probs(s.model, q.obs).__getitem__)] < 0 for q in qs))
            by_day[day] = {t: v for t, v in res["by_topic"].items() if t in PROBLEMS}
            for t, v in by_day[day].items():
                peak[t] = max(peak.get(t, 0.0), v)
        panel = s.panel()
        last = by_day[days]
        speed = [by_day[opened_on(t) + 1][t] for t in PROBLEMS if opened_on(t) in (3, 5) and t in by_day.get(opened_on(t) + 1, {})]
        met = s.met_topics() & set(PROBLEMS)
        harm = waste = n = 0
        for q in s.quiz:
            if q.topic in met and len(q.obs.tags) == 1:
                pr = option_probs(s.model, q.obs)
                v = q.values[max(range(len(pr)), key=pr.__getitem__)]
                harm += v < 0
                waste += 0 <= v < max(q.values) and v == 0
                n += 1
        repeat = {}
        for oc in OUTCOMES:
            xs = [option_probs(s.model, e.obs)[e.rejected] for e in s.buffer if e.outcome == oc and e.rejected is not None]
            repeat[oc] = (st.mean(xs) if xs else float("nan"), len(xs))
        lat = [e.payload["report"]["seconds"] for e in s.engine.log if e.type.value == "learned"]
        return {"seed": seed, "style": style, "arm": arm, "problem_acc": panel["problem_acc"], "word_acc": panel["word_acc"],
                "blind": panel["blind"], "follow_harm": panel["follow_harm"], "harm": harm / n, "waste": waste / n, "harm_curve": curve, "harm_avg": st.mean(curve),
                "forget": st.mean(peak[t] - last[t] for t in last), "speed": st.mean(speed) if speed else float("nan"),
                "rounds": sum(e.strength for e in s.buffer), "ms": 1000 * st.mean(lat), "repeat": repeat,
                "n_wrong": {oc: repeat[oc][1] for oc in OUTCOMES}}
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def job(a):
    return run(*a)


def m(xs):
    xs = [x for x in xs if x == x]
    return st.mean(xs) if xs else float("nan")


def report(results, styles, arms) -> str:
    out = []
    for style in styles:
        by = {a: sorted((r for r in results if r["style"] == style and r["arm"] == a), key=lambda r: r["seed"]) for a in arms}
        out.append(f"\n== parent: {style}, mean over {len(by[arms[0]])} seeds (last day) ==")
        out.append(f"{'arm':12s}" + "".join(f"{c:>12s}" for c in COLS) + " | chosen again after it had chosen wrong (partial / nothing / worse)")
        for a in arms:
            rs = by[a]
            rep = " / ".join(f"{m(r['repeat'][oc][0] for r in rs):.2f}" for oc in OUTCOMES)
            out.append(f"{a:12s}" + "".join(f"{m(r[c] for r in rs):12.3f}" if c not in ("rounds", "ms") else f"{m(r[c] for r in rs):12.0f}" for c in COLS)
                       + f" | {rep}")
        out.append(f"paired difference to {arms[0]}, mean (sd):")
        for a in arms[1:]:
            cells = []
            for c in ("problem_acc", "harm", "harm_avg", "waste", "follow_harm", "blind", "word_acc", "speed"):
                d = [x[c] - y[c] for x, y in zip(by[a], by[arms[0]]) if x[c] == x[c] and y[c] == y[c]]
                cells.append(f"{c} {m(d):+.3f}({st.pstdev(d):.3f})")
            out.append(f"  {a:12s}" + "  ".join(cells))
        out.append("harm by day (share of scenes where its own pick makes things worse), mean:")
        for a in arms:
            out.append(f"  {a:12s}" + " ".join(f"{m(r['harm_curve'][d] for r in by[a]):.3f}" for d in range(len(by[a][0]["harm_curve"]))))
        if "flat3" in by and "severity" in by:
            out.append("severity minus flat3 (the same kind of extra learning, spent by outcome instead of everywhere):")
            cells = []
            for c in ("problem_acc", "harm", "harm_avg", "waste", "follow_harm", "blind", "word_acc", "speed", "rounds"):
                d = [x[c] - y[c] for x, y in zip(by["severity"], by["flat3"])]
                cells.append(f"{c} {m(d):+.3f}({st.pstdev(d):.3f})" if c != "rounds" else f"rounds {m(d):+.0f}")
            out.append("  " + "  ".join(cells))
    return "\n".join(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", default="200-223")
    ap.add_argument("--days", type=int, default=8)
    ap.add_argument("--styles", nargs="+", default=["balanced", "chooser"])
    ap.add_argument("--arms", nargs="+", default=list(ARMS))
    ap.add_argument("--jobs", type=int, default=12)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    tasks = [(s, st_, arm, a.days) for st_ in a.styles for arm in a.arms for s in exp_ea.seeds_of(a.seeds)]
    t0 = time.time()
    with ProcessPoolExecutor(a.jobs) as ex:
        results = list(ex.map(job, tasks))
    from server.world.ontology import ONTOLOGY_PATH
    text = f"ontology: {ONTOLOGY_PATH.name}  seeds: {a.seeds}  days: {a.days}" + report(results, a.styles, a.arms)
    print(text)
    print(f"\nwall={time.time() - t0:.0f}s")
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps({"results": results}))
    Path(a.out).with_suffix(".txt").write_text(text + "\n")


if __name__ == "__main__":
    main()
