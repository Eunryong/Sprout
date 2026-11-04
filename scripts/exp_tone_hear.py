"""Does a child that HEARS the tone obey differently? (docs/WORLD.md 22.9)

22.8 found that repeating a lesson in a firm voice changes nothing when the child cannot tell the voice. Here the tone is a
token in the child's input (GROWING_TONE_INPUT=1): an instruction or a demonstration is said gently or firmly, and the child
learns, from the parent's behaviour alone, what that voice goes with. The parent's choices are the same in every arm; only
the tone differs, and when it is firm:

  gentle  never (the token is there but always GENTLE)      firm   always
  half    on a random half, whatever is right              wrong  exactly when the instruction is mistaken (bossy habit)
  right   exactly when the instruction is right (sure habit)

`deaf` = the same parent with the token switched off (run without GROWING_TONE_INPUT): the control for "a token is in the input".
On the 0-3 years world, 8 days, engine path. At the last day the held-out panel puts each instruction on the scene in no
tone / gently / firmly and measures: follow_good, blind (useless + harmful instructions followed), discern = good - blind.

  GROWING_TONE_INPUT=1 uv run python scripts/exp_tone_hear.py --out scripts/out/tone_hear.json
  uv run python scripts/exp_tone_hear.py --arms gentle --out scripts/out/tone_hear_deaf.json     # the deaf control
  uv run python scripts/exp_tone_hear.py --merge scripts/out/tone_hear.json scripts/out/tone_hear_deaf.json --out scripts/out/tone_hear_all.json
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

HEARS = bool(os.environ.get("GROWING_TONE_INPUT"))
ARMS = ["gentle", "firm", "half", "wrong", "right"]
HEARD = [("none", None), ("gentle", "gentle"), ("firm", "firm")]  # how the panel says the instruction


def parse(cell: str) -> tuple[str, float]:
    style, *rest = cell.split(":")
    return style, float(rest[0]) if rest else 0.25


def run(seed: int, cell: str, arm: str, days: int) -> dict:
    import torch
    torch.set_num_threads(1)

    from server.game.parents import play_day
    from server.game.session import Session
    from server.learning.evaluator import command_panel
    from server.world.ontology import PROBLEMS

    style, p_wrong = parse(cell)
    tmp = Path(tempfile.mkdtemp(prefix=f"th{seed}-"))
    try:
        s = Session.new(tmp / "s", seed=seed)
        rng = random.Random(f"{seed}:{style}")  # the parent's choices are the same in every arm
        for _ in range(days):
            play_day(s, style, 1, rng, 0.0, p_wrong, tones=(arm, "gentle"))
            s.end_day()
        topics = s.met_topics() & set(PROBLEMS)
        panels = {name: command_panel(s.model, s.quiz, topics, tone) for name, tone in HEARD}
        told = [e.payload for e in s.engine.log if e.type.value == "commanded"]
        firm = [p["quality"] for p in told if p["tone"] == "firm"]
        return {"seed": seed, "cell": cell, "arm": arm if HEARS else f"{arm}/deaf", "panels": panels,
                "commands": len(told), "firm_commands": len(firm), "firm_wrong": sum(q != "good" for q in firm),
                "wrong_commands": sum(p["quality"] != "good" for p in told)}
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
        by = {a: rs for a, rs in by.items() if rs}
        n = len(next(iter(by.values())))
        out.append(f"\n== parent: {cell}, mean over {n} seeds (last day) ==")
        out.append(f"{'arm':13s}{'alone':>7s}{'words':>7s} | {'blind: none':>11s}{'gentle':>8s}{'firm':>8s} | {'follow good: none':>17s}{'gentle':>8s}{'firm':>8s}"
                   f" | {'discern: gentle':>15s}{'firm':>8s} | {'told firm':>9s}{'..wrong':>8s}")
        for a, rs in by.items():
            P = lambda tone, k: m(r["panels"][tone][k] for r in rs)  # noqa: E731
            out.append(f"{a:13s}{P('none', 'problem_acc'):7.2f}{P('none', 'word_acc'):7.2f} | "
                       f"{P('none', 'blind'):11.2f}{P('gentle', 'blind'):8.2f}{P('firm', 'blind'):8.2f} | "
                       f"{P('none', 'follow_good'):17.2f}{P('gentle', 'follow_good'):8.2f}{P('firm', 'follow_good'):8.2f} | "
                       f"{P('gentle', 'discern'):15.2f}{P('firm', 'discern'):8.2f} | "
                       f"{m(r['firm_commands'] for r in rs):9.1f}{m(r['firm_wrong'] for r in rs):8.1f}")
        ref = "gentle" if "gentle" in by else next(iter(by))
        out.append(f"paired difference to {ref}, mean (sd): firm-voiced panel minus gentle-voiced panel, within each arm:")
        for a, rs in by.items():
            d_blind = [r["panels"]["firm"]["blind"] - r["panels"]["gentle"]["blind"] for r in rs]
            d_good = [r["panels"]["firm"]["follow_good"] - r["panels"]["gentle"]["follow_good"] for r in rs]
            out.append(f"  {a:13s} firm - gentle voice: blind {m(d_blind):+.3f}({st.pstdev(d_blind):.3f})  "
                       f"follow_good {m(d_good):+.3f}({st.pstdev(d_good):.3f})")
        if ref in by:
            out.append(f"vs {ref}, same voice (the panel says it gently): independence and obedience:")
            for a, rs in by.items():
                if a == ref:
                    continue
                cells = []
                for name, key, tone in (("alone", "problem_acc", "none"), ("blind", "blind", "gentle"), ("good", "follow_good", "gentle")):
                    d = [x["panels"][tone][key] - y["panels"][tone][key] for x, y in zip(rs, by[ref])]
                    cells.append(f"{name} {m(d):+.3f}({st.pstdev(d):.3f})")
                out.append(f"  {a:13s} " + "  ".join(cells))
    out.append("\ncolumns: alone = accuracy with no instruction; blind/follow good = share of useless+harmful / good instructions followed,"
               " the panel putting the instruction on the scene in no tone / gently / firmly; told firm = instructions said firmly "
               "(..wrong: of which mistaken).")
    return "\n".join(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", default="200-223")
    ap.add_argument("--days", type=int, default=8)
    ap.add_argument("--cells", nargs="+", default=["commander:0.25", "commander:0.5", "balanced"])
    ap.add_argument("--arms", nargs="+", default=ARMS)
    ap.add_argument("--jobs", type=int, default=12)
    ap.add_argument("--merge", nargs="*", default=[], help="result files to put in the same report (instead of running)")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    t0 = time.time()
    if a.merge:
        results = sum((json.loads(Path(f).read_text())["results"] for f in a.merge), [])
        cells = sorted({r["cell"] for r in results}, key=a.cells.index if all(c in a.cells for c in {r["cell"] for r in results}) else str)
        arms = [x for x in ARMS + [f"{y}/deaf" for y in ARMS] if any(r["arm"] == x for r in results)]
    else:
        tasks = [(s, c, arm, a.days) for c in a.cells for arm in a.arms for s in exp_ea.seeds_of(a.seeds)]
        with ProcessPoolExecutor(a.jobs) as ex:
            results = list(ex.map(job, tasks))
        cells, arms = a.cells, [x if HEARS else f"{x}/deaf" for x in a.arms]
    from server.world.ontology import ONTOLOGY_PATH
    text = f"ontology: {ONTOLOGY_PATH.name}  seeds: {a.seeds}  days: {a.days}  the child hears the tone: {HEARS}" + report(results, cells, arms)
    print(text)
    print(f"\nwall={time.time() - t0:.0f}s")
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps({"results": results}))
    Path(a.out).with_suffix(".txt").write_text(text + "\n")


if __name__ == "__main__":
    main()
