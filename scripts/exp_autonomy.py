"""Learn the child's own experience more, and remember like a person (docs/WORLD.md 22.10)

Two things are varied, nothing else: how many learning rounds a teaching move gets, and how old lessons are rehearsed.
  rounds  own_rounds: the child chose first and the parent praised or put it right (self-made experience)
          firm_rounds: an instruction or demonstration said firmly (the parent's strong signal)
  replay  uniform (as before) | by strength (a lesson that got more rounds is rehearsed more) | and fading (replay_decay per day)
The parent's choices are the same in every arm. 0-3 years world, 8 days, engine path. At the last day, on the quiz and on the
lessons themselves (probability of the taught option on the very scene it was taught on, right lessons only):
  alone / words / blind / discern   the usual held-out panel
  own_wrong, own_right, command, show   how well each kind of lesson is still known (all days | the first three days only)
  forget   mean over problems of (best accuracy so far - accuracy now);  speed   accuracy the day after a problem opened
  ms       learning time per taught example

  uv run python scripts/exp_autonomy.py --out scripts/out/autonomy.json
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

# name: (own_rounds, firm_rounds, replay_by_strength, replay_decay)
ARMS = {"base": (1, 1, False, 1.0), "own3": (3, 1, False, 1.0), "own3+firm2": (3, 2, False, 1.0),
        "strong replay": (3, 2, True, 1.0), "strong+fading": (3, 2, True, 0.85),
        "fading 0.5": (3, 2, True, 0.5), "fading 0.3": (3, 2, True, 0.3)}  # replay_decay per day
MODES = {"own_wrong": ("own_wrong",), "own_right": ("own_right",), "command": ("command_firm", "command_gentle"),
         "show": ("show_firm", "show_gentle")}
COLS = ["problem_acc", "word_acc", "blind", "discern", "forget", "speed", "ms"]


def run(seed: int, style: str, arm: str, days: int) -> dict:
    import torch
    torch.set_num_threads(1)
    from dataclasses import replace

    from server.config import load_configs
    from server.game.parents import play_day
    from server.game.session import Session
    from server.learning.codec import option_probs
    from server.world.ontology import PROBLEMS, opened_on

    own, firm, by_strength, decay = ARMS[arm]
    g, _, o = load_configs()
    game = replace(g, own_rounds=own, firm_rounds=firm, firm_repeat=1)
    ocfg = replace(o, replay_by_strength=by_strength, replay_decay=decay)
    tmp = Path(tempfile.mkdtemp(prefix=f"au{seed}-"))
    try:
        s = Session.new(tmp / "s", seed=seed, game=game, ocfg=ocfg)
        rng = random.Random(f"{seed}:{style}")  # the parent's choices are the same in every arm
        peak: dict[str, float] = {}
        by_day = {}
        for day in range(1, days + 1):
            play_day(s, style, 1, rng, 0.0, tones=("half", "gentle"))
            res = s.end_day()
            by_day[day] = {t: v for t, v in res["by_topic"].items() if t in PROBLEMS}
            for t, v in by_day[day].items():
                peak[t] = max(peak.get(t, 0.0), v)
        panel = s.panel()
        last = by_day[days]
        speed = [by_day[opened_on(t) + 1][t] for t in PROBLEMS if opened_on(t) in (3, 5) and t in by_day.get(opened_on(t) + 1, {})]
        known = {}
        for name, modes in MODES.items():
            for early in (False, True):
                ps = [option_probs(s.model, e.obs)[e.chosen] for e in s.buffer
                      if e.mode in modes and e.was_right and (not early or e.day <= 3)]
                known[f"{name}{'_early' if early else ''}"] = (st.mean(ps) if ps else float("nan"), len(ps))
        lat = [e.payload["report"]["seconds"] for e in s.engine.log if e.type.value == "learned"]
        return {"seed": seed, "style": style, "arm": arm, "problem_acc": panel["problem_acc"], "word_acc": panel["word_acc"],
                "blind": panel["blind"], "discern": panel["discern"],
                "forget": st.mean(peak[t] - last[t] for t in last), "speed": st.mean(speed) if speed else float("nan"),
                "ms": 1000 * st.mean(lat), "known": known, "examples": len(s.buffer)}
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
        n = len(by[arms[0]])
        out.append(f"\n== parent: {style}, mean over {n} seeds (last day) ==")
        out.append(f"{'arm':15s}" + "".join(f"{c:>9s}" for c in COLS) + " | lesson known (all days / first 3 days): "
                   + "".join(f"{k:>16s}" for k in MODES))
        for a in arms:
            rs = by[a]
            known = "".join(f"{m(r['known'][k][0] for r in rs):9.2f}/{m(r['known'][k + '_early'][0] for r in rs):.2f}" for k in MODES)
            out.append(f"{a:15s}" + "".join(f"{m(r[c] for r in rs):9.3f}" if c != "ms" else f"{m(r[c] for r in rs):9.0f}" for c in COLS)
                       + " |                                      " + known)
        out.append(f"paired difference to {arms[0]}, mean (sd):")
        for a in arms[1:]:
            cells = []
            for c in ("problem_acc", "word_acc", "forget", "speed", "blind", "discern"):
                d = [x[c] - y[c] for x, y in zip(by[a], by[arms[0]]) if x[c] == x[c] and y[c] == y[c]]
                cells.append(f"{c} {m(d):+.3f}({st.pstdev(d):.3f})")
            out.append(f"  {a:15s}" + "  ".join(cells))
    return "\n".join(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", default="200-223")
    ap.add_argument("--days", type=int, default=8)
    ap.add_argument("--styles", nargs="+", default=["balanced", "chooser", "commander"])
    ap.add_argument("--arms", nargs="+", default=list(ARMS)[:5])
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
