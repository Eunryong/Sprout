"""How sure is the child of what will happen, compared with how often it is so? (docs/WORLD.md 22.14, DIRECTION.md 6)

A child is raised scene by scene by a mixed parent (says or shows, 30% of the time something that does not solve the
problem). After 1, 3, 10, 30 and 64 scenes it is asked, on held-out scenes of the problems it has met, what will come of each
option. Measured against what the world's rules really give:
  nll    mean -log p(true result)                    conf   mean probability of its most likely result
  acc    share where the most likely result is true  ece    |confidence - accuracy|, 10 bins (0 = sure exactly as often as right)
  sat    share of options it is >= 99% sure of       solve  problem_acc (picks a solving option), blind (follows a harmful word)
Settings are env/args of one process each: GROWING_SMOOTH, --steps, --stop-loss.

  uv run python scripts/exp_calib.py --out scripts/out/calib.json
"""
import argparse
import json
import math
import os
import random
import statistics as st
import subprocess
import sys
import tempfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
CHECKS = [1, 3, 10, 30, 64]
KEYS = ["nll", "conf", "acc", "ece", "sat", "problem_acc", "blind"]


def one(seed: int, steps: int, stop_loss: float, lr: float) -> None:
    import shutil
    import torch
    torch.set_num_threads(1)
    from server.game.session import Session
    from server.learning.codec import RESULTS, encode_try, result_logits
    from server.learning.learner import OnlineCfg
    from server.world.ontology import outcome_label
    from server.world.situation import values
    from server.world.state import Phase
    tmp = Path(tempfile.mkdtemp(prefix="cal-"))
    try:
        s = Session.new(tmp / "s", seed=seed, ocfg=OnlineCfg(steps=steps, stop_loss=stop_loss, lr=lr))
        rng = random.Random(seed)

        def measure() -> dict:
            met = s.met_topics()
            items = [q for q in s.quiz if q.topic in met and len(q.obs.tags) == 1]
            rows = []  # (p of every result, true result index)
            s.model.eval()
            with torch.no_grad():
                for q in items:
                    p = torch.softmax(result_logits(s.model, [encode_try(q.obs, o, False) for o in q.obs.options]), dim=-1)
                    rows += [(p[i].tolist(), RESULTS.index(outcome_label(v))) for i, v in enumerate(q.values)]
            nll = st.mean(-math.log(max(p[t], 1e-9)) for p, t in rows)
            conf = [max(p) for p, _ in rows]
            hit = [max(range(4), key=p.__getitem__) == t for p, t in rows]
            ece = 0.0
            for b in range(10):
                idx = [i for i, c in enumerate(conf) if min(int(c * 10), 9) == b]
                if idx:
                    ece += len(idx) / len(rows) * abs(st.mean(conf[i] for i in idx) - st.mean(hit[i] for i in idx))
            panel = s.panel()
            return {"nll": nll, "conf": st.mean(conf), "acc": st.mean(hit), "ece": ece, "sat": st.mean(c >= 0.99 for c in conf),
                    "problem_acc": panel["problem_acc"], "blind": panel["blind"]}

        out, n = {}, 0
        while n < CHECKS[-1]:
            if s.state.phase == Phase.EVENING:
                s.end_day()
            s.draw_cards()
            s.pick_card(0)
            vals = values(s.state.current)
            good = [i for i, v in enumerate(vals) if v == max(vals)]
            other = [i for i in range(len(vals)) if i not in good]
            pick = rng.choice(other) if other and rng.random() < 0.3 else rng.choice(good)
            if rng.random() < 0.5:
                s.command(pick)
                s.try_child()
            else:
                s.show(pick)
            n += 1
            if n in CHECKS:
                out[n] = measure()
        print(json.dumps(out))
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def run(label: str, smooth: float, steps: int, stop_loss: float, lr: float, seed: int) -> dict:
    env = {**os.environ, "GROWING_LLM": "mock", "GROWING_SMOOTH": str(smooth)}
    r = subprocess.run([sys.executable, __file__, "--one", str(seed), str(steps), str(stop_loss), str(lr)], env=env, capture_output=True, text=True, cwd=ROOT)
    if r.returncode:
        raise RuntimeError(f"{label}: {r.stderr[-400:]}")
    return {"label": label, "seed": seed, "at": json.loads(r.stdout.strip().splitlines()[-1])}


def main():
    if len(sys.argv) > 1 and sys.argv[1] == "--one":
        one(int(sys.argv[2]), int(sys.argv[3]), float(sys.argv[4]), float(sys.argv[5]))
        return
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", default="200-211")
    ap.add_argument("--jobs", type=int, default=12)
    ap.add_argument("--out", required=True)
    ap.add_argument("--grid", nargs="*", default=["base|", "smooth0.1|smooth=0.1", "smooth0.2|smooth=0.2", "steps20|steps=20", "steps10|steps=10",
                                                  "stop0.4|stop_loss=0.4", "smooth0.1+steps20|smooth=0.1,steps=20"])
    a = ap.parse_args()
    lo, hi = a.seeds.split("-")
    seeds = range(int(lo), int(hi) + 1)
    tasks = []
    for g in a.grid:
        label, _, spec = g.partition("|")
        kv = dict(x.split("=") for x in spec.split(",") if x)
        tasks += [(label, float(kv.get("smooth", 0)), int(kv.get("steps", 60)), float(kv.get("stop_loss", 0)), float(kv.get("lr", 0.002)), sd) for sd in seeds]
    with ThreadPoolExecutor(a.jobs) as ex:
        results = list(ex.map(lambda t: run(*t), tasks))
    labels = list(dict.fromkeys(t[0] for t in tasks))
    lines = []
    for n in CHECKS:
        lines.append(f"\nafter {n} scenes" + "".join(f"{k:>13s}" for k in KEYS))
        for lab in labels:
            rs = [r["at"][str(n)] for r in results if r["label"] == lab]
            lines.append(f"  {lab:20s}" + "".join(f"{st.mean(r[k] for r in rs):13.3f}" for k in KEYS))
    text = "\n".join(lines)
    print(text)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(results))
    Path(a.out).with_suffix(".txt").write_text(text + "\n")


if __name__ == "__main__":
    main()
