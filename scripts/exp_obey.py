"""When does blind obedience appear? (docs/WORLD.md section 21)

E-A (sections 19-20) found that raising the child only by instructions leaves it unable to choose alone, but not a child
that follows ANY instruction (blind obedience 0.34-0.37 against the pre-set 0.60). Here the commanding parent is varied
along what the engine lets a parent do: how often they are wrong (p_wrong), what the mistake looks like (wrong_kind), how
long stage 1 lasts, and which world the child lives in (GROWING_ONTOLOGY=config/experiments/onto_harm.yaml adds -1 rules).
One run per (seed, cell) goes `--stage1` days with that parent, then 3 days with the same discerning parent for everybody
(stage 2), and measures the held-out panel on the days in `--panel`. Nothing about the learner changes between cells.

A cell is `style[:p_wrong[:wrong_kind]]`, e.g. `commander:0.5:useless`; `chooser` is the reference.

  uv run python scripts/exp_obey.py --cells chooser commander:0 commander:0.25 commander:0.5 commander:0.75 --out scripts/out/obey_default.json
  GROWING_ONTOLOGY=config/experiments/onto_harm.yaml uv run python scripts/exp_obey.py --cells chooser commander:0.5:harmful ...
"""
import argparse
import json
import os
import statistics as st
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
os.environ.setdefault("GROWING_LLM", "mock")

import exp_ea  # noqa: E402
import exp_ea_engine  # noqa: E402

BLIND_GOAL = 0.60  # the pre-set criterion of docs/WORLD.md 18.10
COLS = [("blind", "blind"), ("follow_good", "good"), ("follow_useless", "useless"), ("follow_harm", "harm"),
        ("discern", "discern"), ("best_when_bad", "best|bad"), ("problem_acc", "alone"), ("confidence", "conf")]


def parse(cell: str) -> tuple[str, float, str]:
    style, *rest = cell.split(":")
    return style, float(rest[0]) if rest else 0.25, rest[1] if len(rest) > 1 else "any"


def job(a):
    path, seed, cell, days, stage1, panel = a
    style, p, kind = parse(cell)
    mod = exp_ea_engine if path == "engine" else exp_ea
    r = mod.run(seed, style, days, stage1, 0.0, "", p, kind, tuple(panel))
    r["cell"] = cell
    return r


def mean(xs):
    xs = [x for x in xs if x == x]
    return st.mean(xs) if xs else float("nan")


def report(results, cells, panel, stage1, days) -> str:
    by = {c: sorted((r for r in results if r["cell"] == c), key=lambda r: r["seed"]) for c in cells}
    ref = next((c for c in cells if c.startswith("chooser")), None)
    out = []
    for day in panel:
        stage = "stage 1" if day <= stage1 else "after 3 days of the same discerning parent"
        out.append(f"\n== day {day} ({stage}), mean over {len(next(iter(by.values())))} seeds ==")
        out.append(f"{'cell':26s}" + "".join(f"{n:>9s}" for _, n in COLS) + f"{'blind>=.6':>11s}{'blind-ref':>11s}")
        for c in cells:
            rs = by[c]
            vals = [mean([r["rows"][day][k] for r in rs]) for k, _ in COLS]
            share = mean([float(r["rows"][day]["blind"] >= BLIND_GOAL) for r in rs if r["rows"][day]["blind"] == r["rows"][day]["blind"]])
            diff = ""
            if ref and c != ref:
                d = [a["rows"][day]["blind"] - b["rows"][day]["blind"] for a, b in zip(rs, by[ref])
                     if a["rows"][day]["blind"] == a["rows"][day]["blind"] and b["rows"][day]["blind"] == b["rows"][day]["blind"]]
                diff = f"{mean(d):+.2f}({st.pstdev(d):.2f})" if d else ""
            out.append(f"{c:26s}" + "".join(f"{v:9.2f}" for v in vals) + f"{share:11.2f}{diff:>11s}")
    first = next(iter(by.values()))
    if all(0 in r["rows"] for r in first):
        out.append(f"\n(untrained child on the same quiz, day {stage1} topics: " + ", ".join(
            f"{n} {mean([r['rows'][0][k] for r in first]):.2f}" for k, n in COLS) + ")")
    out.append("\ncolumns: blind = follows useless+harmful instructions; good/useless/harm = follows instructions of that quality; "
               "discern = good - blind; best|bad = takes the best option although told a wrong one; alone = accuracy with no instruction; "
               "blind>=.6 = share of seeds past the pre-set goal; blind-ref = paired difference to the chooser (sd).")
    return "\n".join(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cells", nargs="+", required=True)
    ap.add_argument("--seeds", default="200-239")
    ap.add_argument("--stage1", type=int, default=10)
    ap.add_argument("--stage2", type=int, default=3)
    ap.add_argument("--panel", default="3,6,10", help="days to measure; the last day of stage 2 is always added")
    ap.add_argument("--path", choices=["harness", "engine"], default="harness")
    ap.add_argument("--jobs", type=int, default=16)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    days = a.stage1 + a.stage2
    panel = sorted({int(x) for x in a.panel.split(",") if int(x) <= days} | {a.stage1, days})
    tasks = [(a.path, s, c, days, a.stage1, panel) for c in a.cells for s in exp_ea.seeds_of(a.seeds)]
    t0 = time.time()
    with ProcessPoolExecutor(a.jobs) as ex:
        results = list(ex.map(job, tasks))
    from server.world.ontology import ONTOLOGY_PATH
    text = f"ontology: {ONTOLOGY_PATH.name}  path: {a.path}  seeds: {a.seeds}  stage 1 = {a.stage1} days, stage 2 = {a.stage2} days" \
           + report(results, a.cells, panel, a.stage1, days)
    print(text)
    print(f"\nwall={time.time() - t0:.0f}s")
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps({"ontology": ONTOLOGY_PATH.name, "path": a.path, "cells": a.cells, "panel": panel, "results": results}))
    Path(a.out).with_suffix(".txt").write_text(text + "\n")


if __name__ == "__main__":
    main()
