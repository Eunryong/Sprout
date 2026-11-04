"""Being burned by the parent's word: does the child learn that the parent is not always right? (docs/WORLD.md 22.12)

The parent's choices are the same in every arm. When the child does what the parent said and the result is bad, the arms differ in
what is learned from it (GameCfg.burn_*_rounds):
  base      as before: the parent's choice is copied (shown) or the parent's alternative is learned (corrected)
  worse1    a result that made things worse is learned as "not this, here" (unlikelihood), one round, instead
  worse3    the same in three rounds
  both      worse in three rounds, and "nothing came of it" in one
  own       what the child chose itself and went badly (no word of the parent behind it) is learned the same way, instead of the parent's correction
  all       both + own          all+regret   all, and ignoring a good word of the parent and being hurt makes the child learn the word (2 rounds)
0-3 years world with harmful options (onto_age03_harm.yaml), 8 days. Parents: a commander who is wrong 25% / 50% of the time,
a balanced one, and a commander for three days who then tells the child things and puts it right (disc). Every evening the
held-out panel measures what the child makes of an instruction:
  follow_good  instructions to the best option followed (trust in a good word)        blind  bad ones followed (useless + harmful)
  discern = follow_good - blind                follow_harm  a harmful instruction followed
  alone        accuracy with no instruction    words        accuracy on naming scenes

  uv run python scripts/exp_burn.py --out scripts/out/burn.json
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

# (burn_worse_rounds, burn_nothing_rounds, own_burn_worse_rounds, own_burn_nothing_rounds, regret_rounds)
ARMS = {"base": (0, 0, 0, 0, 0), "worse1": (1, 0, 0, 0, 0), "worse3": (3, 0, 0, 0, 0), "both": (3, 1, 0, 0, 0),
        "own": (0, 0, 3, 1, 0), "all": (3, 1, 3, 1, 0), "all+regret": (3, 1, 3, 1, 2)}
DEFAULT_ARMS = ["base", "worse1", "worse3", "both"]
KEYS = ["follow_good", "blind", "follow_harm", "discern", "problem_acc", "word_acc"]
DISC_FROM = 4  # the 'disc' parent changes from commanding to telling-and-putting-right on this day


def parse(cell: str) -> tuple[str, float, bool]:
    head, *rest = cell.split(">")
    style, *p = head.split(":")
    return style, float(p[0]) if p else 0.25, bool(rest)


def run(seed: int, cell: str, arm: str, days: int) -> dict:
    import torch
    torch.set_num_threads(1)
    from dataclasses import replace

    from server.config import load_configs
    from server.game.parents import play_day
    from server.game.session import Session

    w, n, ow, on, rg = ARMS[arm]
    g, _, o = load_configs()
    game = replace(g, burn_worse_rounds=w, burn_nothing_rounds=n, own_burn_worse_rounds=ow, own_burn_nothing_rounds=on, regret_rounds=rg)
    style, p_wrong, disc = parse(cell)
    tmp = Path(tempfile.mkdtemp(prefix=f"bu{seed}-"))
    try:
        s = Session.new(tmp / "s", seed=seed, game=game, ocfg=o)
        rng = random.Random(f"{seed}:{cell}")  # the parent's choices are the same in every arm
        per_day = []
        for day in range(1, days + 1):
            play_day(s, style, 2 if disc and day >= DISC_FROM else 1, rng, 0.0, p_wrong)
            res = s.end_day()
            per_day.append({k: res["panel"][k] for k in KEYS})
        burned = [e for e in s.buffer if e.mode == "burned"]
        told = [e.payload for e in s.engine.log if e.type.value == "commanded"]
        return {"seed": seed, "cell": cell, "arm": arm, "per_day": per_day, "burns": len(burned),
                "burn_days": [e.day for e in burned], "commands": len(told),
                "wrong_commands": sum(p["quality"] != "good" for p in told),
                "rounds": sum(e.strength for e in s.buffer)}
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def job(a):
    return run(*a)


def m(xs):
    xs = [x for x in xs if x == x]
    return st.mean(xs) if xs else float("nan")


def report(results, cells, arms) -> str:
    out = []
    for cell in cells:
        by = {a: sorted((r for r in results if r["cell"] == cell and r["arm"] == a), key=lambda r: r["seed"]) for a in arms}
        days = len(by[arms[0]][0]["per_day"])
        out.append(f"\n== parent: {cell}, mean over {len(by[arms[0]])} seeds ==")
        out.append(f"{'arm':8s}" + "".join(f"{k:>12s}" for k in KEYS) + f"{'burns':>8s}{'told':>6s}{'wrong':>7s}{'rounds':>8s}   (last day)")
        for a in arms:
            rs = by[a]
            out.append(f"{a:8s}" + "".join(f"{m(r['per_day'][-1][k] for r in rs):12.3f}" for k in KEYS)
                       + f"{m(r['burns'] for r in rs):8.1f}{m(r['commands'] for r in rs):6.1f}{m(r['wrong_commands'] for r in rs):7.1f}"
                       f"{m(r['rounds'] for r in rs):8.0f}")
        out.append("per day, mean:  follow_good / blind   (trust in a good word / in a bad one)")
        out.append(f"{'arm':8s}" + "".join(f"{'day ' + str(d):>12s}" for d in range(1, days + 1)))
        for a in arms:
            out.append(f"{a:8s}" + "".join(f"{m(r['per_day'][d]['follow_good'] for r in by[a]):6.2f}/{m(r['per_day'][d]['blind'] for r in by[a]):.2f}"
                                           for d in range(days)))
        out.append(f"paired difference to {arms[0]}, last day, mean (sd) [mean of the 3 last days]:")
        for a in arms[1:]:
            cells_ = []
            for k in ("follow_good", "blind", "discern", "problem_acc", "word_acc"):
                d = [x["per_day"][-1][k] - y["per_day"][-1][k] for x, y in zip(by[a], by[arms[0]])]
                d3 = [m(x["per_day"][i][k] - y["per_day"][i][k] for i in range(days - 3, days)) for x, y in zip(by[a], by[arms[0]])]
                cells_.append(f"{k} {m(d):+.3f}({st.pstdev(d):.3f})[{m(d3):+.3f}]")
            out.append(f"  {a:8s}" + "  ".join(cells_))
    return "\n".join(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", default="200-223")
    ap.add_argument("--days", type=int, default=8)
    ap.add_argument("--cells", nargs="+", default=["commander:0.25", "commander:0.5", "balanced", "commander:0.25>disc"])
    ap.add_argument("--arms", nargs="+", default=DEFAULT_ARMS)
    ap.add_argument("--jobs", type=int, default=12)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    tasks = [(s, c, arm, a.days) for c in a.cells for arm in a.arms for s in exp_ea.seeds_of(a.seeds)]
    t0 = time.time()
    with ProcessPoolExecutor(a.jobs) as ex:
        results = list(ex.map(job, tasks))
    from server.world.ontology import ONTOLOGY_PATH
    text = f"ontology: {ONTOLOGY_PATH.name}  seeds: {a.seeds}  days: {a.days}" + report(results, a.cells, a.arms)
    print(text)
    print(f"\nwall={time.time() - t0:.0f}s")
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps({"results": results}))
    Path(a.out).with_suffix(".txt").write_text(text + "\n")


if __name__ == "__main__":
    main()
