"""E-A on the real game path: the same four parents (+ reference), but every act is an engine event through `Session` with
the Director (template, via the mock client), cards, world memory and an attention budget. Compare with `exp_ea.py`
(the harness without an engine); the metrics are the same function (`learning.evaluator.command_panel`).

  uv run python scripts/exp_ea_engine.py --seeds 200-239 --jobs 12
"""
import argparse
import json
import os
import random
import shutil
import sys
import tempfile
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
os.environ.setdefault("GROWING_LLM", "mock")

import exp_ea  # noqa: E402
from server.game.parents import P_COMMAND_WRONG, STYLES  # noqa: E402


def run(seed: int, style: str, days: int = 6, stage1: int = 3, free: float = 0.0, salt: str = "",
        p_wrong: float = P_COMMAND_WRONG, wrong_kind: str = "any", panel_days: tuple = ()) -> dict:
    import torch
    torch.set_num_threads(1)
    from server.game.parents import play_day
    from server.game.session import Session
    from server.learning.evaluator import command_panel
    from server.world.ontology import PROBLEMS

    tmp = Path(tempfile.mkdtemp(prefix=f"eae{seed}-"))
    try:
        s = Session.new(tmp / "s", seed=seed)
        rng = random.Random(f"{seed}:{style}{salt}")  # the parent's own randomness
        rows = {}
        for day in range(1, days + 1):
            play_day(s, style, 1 if day <= stage1 else 2, rng, free, p_wrong, wrong_kind)
            if day in (stage1, days) or day in panel_days:
                rows[day] = s.panel()
                if day == stage1:  # the untrained child on the same quiz and the same topics
                    rows[0] = command_panel(s.model0, s.quiz, s.met_topics() & set(PROBLEMS))
            s.end_day()
        jr = s.journal()
        tot = lambda k, ds: sum(jr[d][k] for d in ds if d in jr)  # noqa: E731
        s1, s2 = range(1, stage1 + 1), range(stage1 + 1, days + 1)
        return {"seed": seed, "style": style, "rows": rows, "n_params": sum(p.numel() for p in s.model.parameters()),
                "examples": len(s.buffer), "words_known": len(s.state.words), "open_issues": len(s.state.open_issues()),
                "journal": {k: [tot(k, s1), tot(k, s2)] for k in ("talk", "show", "command", "offer", "praise", "correct", "wrong_shows")},
                "met": sorted(s.met_topics())}
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def job(a):
    return run(*a)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", default="200-239")
    ap.add_argument("--days", type=int, default=6)
    ap.add_argument("--stage1", type=int, default=3)
    ap.add_argument("--jobs", type=int, default=12)
    ap.add_argument("--free", type=float, default=0.0)
    ap.add_argument("--styles", default=",".join(STYLES))
    ap.add_argument("--salt", default="", help="another random stream for the parent (same child weights and quiz)")
    ap.add_argument("--out", default="scripts/out/ea_engine.json")
    a = ap.parse_args()
    tasks = [(s, st_, a.days, a.stage1, a.free, a.salt) for s in exp_ea.seeds_of(a.seeds) for st_ in a.styles.split(",")]
    t0 = time.time()
    with ProcessPoolExecutor(a.jobs) as ex:
        results = list(ex.map(job, tasks))
    text, summary = exp_ea.report(results, a.days, a.stage1)
    print(text)
    by = {s: [r for r in results if r["style"] == s] for s in a.styles.split(",")}
    print("\n== parenting journal, mean per run: [stage 1, stage 2] ==")
    for s, rs in by.items():
        print(f"{s:13s} " + "  ".join(f"{k} {sum(r['journal'][k][0] for r in rs) / len(rs):.1f}/{sum(r['journal'][k][1] for r in rs) / len(rs):.1f}"
                                      for k in ("talk", "show", "command", "offer", "praise", "correct", "wrong_shows"))
              + f"  open issues {sum(r['open_issues'] for r in rs) / len(rs):.1f}  words known {sum(r['words_known'] for r in rs) / len(rs):.1f}")
    print(f"\nparams={results[0]['n_params']} examples/run={results[0]['examples']} seeds={a.seeds} wall={time.time() - t0:.0f}s")
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps({"results": results, "summary": summary}))


if __name__ == "__main__":
    main()
