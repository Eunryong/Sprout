"""How much does the child try out? (docs/DIRECTION.md 6, curiosity)

A child is left alone (`leave`) scene after scene. Over the first 40 scenes we count what it tries and what comes of it:
  kinds    distinct things tried (verb x shape x material)      new    share of tries that were a thing never lived before
  worse    share of tries that turned out 'worse'                solved share that turned out 'solved'
  wait     share of tries that were WAIT (doing nothing)
and and the same for the second half only (new_late, solved_late: once it has learned something, does it still try new things?),
then, after 40 scenes, how well it knows the world (problem_acc: picks a solving option). Curiosity is GROWING_CURIOSITY (beta).

  uv run python scripts/exp_explore.py --betas 0 0.2 0.4 0.6 --out scripts/out/explore.json
"""
import argparse
import json
import os
import statistics as st
import subprocess
import sys
import tempfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
SCENES = 40
KEYS = ["kinds", "new", "new_late", "solved_late", "worse", "wait", "problem_acc"]


def one(seed: int) -> None:
    import shutil
    import torch
    torch.set_num_threads(1)
    from server.game.simple import SimpleGame
    from server.learning.codec import novelty_key
    from server.learning.learner import OnlineCfg
    from server.world.situation import observation
    tmp = Path(tempfile.mkdtemp(prefix="exp-"))
    try:
        g = SimpleGame.new(tmp / "g", seed=seed, ocfg=OnlineCfg())
        kinds, tries, new, worse, solved, wait, new_late, solved_late = set(), 0, 0, 0, 0, 0, 0, 0
        for i in range(SCENES):
            g.scene()
            obs = observation(g.s.state.current)
            t = g.play("leave")
            o = obs.options[t.acted]
            k = novelty_key(o)
            fresh = k not in kinds
            new += fresh
            new_late += fresh and i >= SCENES // 2
            solved_late += t.result == "solved" and i >= SCENES // 2
            kinds.add(k)
            tries += 1
            worse += t.result == "worse"
            solved += t.result == "solved"
            wait += o.verb == "WAIT"
        print(json.dumps({"kinds": len(kinds), "new": new / tries, "new_late": new_late / (SCENES // 2), "solved_late": solved_late / (SCENES // 2), "worse": worse / tries, "solved": solved / tries,
                          "wait": wait / tries, "problem_acc": g.s.panel()["problem_acc"]}))
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def run(beta: float, seed: int) -> dict:
    env = {**os.environ, "GROWING_LLM": "mock", "GROWING_CURIOSITY": str(beta)}
    r = subprocess.run([sys.executable, __file__, "--one", str(seed)], env=env, capture_output=True, text=True, cwd=ROOT)
    if r.returncode:
        raise RuntimeError(f"beta {beta}: {r.stderr[-400:]}")
    return {"beta": beta, "seed": seed, **json.loads(r.stdout.strip().splitlines()[-1])}


def main():
    if len(sys.argv) > 1 and sys.argv[1] == "--one":
        one(int(sys.argv[2]))
        return
    ap = argparse.ArgumentParser()
    ap.add_argument("--betas", nargs="*", type=float, default=[0, 0.2, 0.4, 0.6])
    ap.add_argument("--seeds", default="200-211")
    ap.add_argument("--jobs", type=int, default=12)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    lo, hi = a.seeds.split("-")
    tasks = [(b, sd) for b in a.betas for sd in range(int(lo), int(hi) + 1)]
    with ThreadPoolExecutor(a.jobs) as ex:
        res = list(ex.map(lambda t: run(*t), tasks))
    lines = [f"alone, {SCENES} scenes" + "".join(f"{k:>13s}" for k in KEYS)]
    for b in a.betas:
        rs = [r for r in res if r["beta"] == b]
        lines.append(f"  beta {b:<14}" + "".join(f"{st.mean(r[k] for r in rs):13.3f}" for k in KEYS))
    text = "\n".join(lines)
    print(text)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(res))
    Path(a.out).with_suffix(".txt").write_text(text + "\n")


if __name__ == "__main__":
    main()
