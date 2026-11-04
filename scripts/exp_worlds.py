"""Are generated worlds learnable? A first acceptance gate (docs/WORLD.md 22.13).

Each world (a seeded template world from config/world_pools.yaml) is lived in by a child raised by the same balanced
parent for 8 days; the world sets what it teaches from a choice (its own burn / regret rounds). The reference is the
hand-written 0-3 world (onto_age03.yaml) raised the same way with the default game settings. The ontology is a module
global, so every run is a fresh process with GROWING_ONTOLOGY pointing at its world.

  uv run python scripts/exp_worlds.py --worlds 0-19 --out scripts/out/worlds.json
"""
import argparse
import json
import os
import random
import statistics as st
import subprocess
import sys
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
KEYS = ["problem_acc", "word_acc", "follow_good", "blind", "discern"]
REFERENCE = ROOT / "config" / "experiments" / "onto_age03.yaml"


def one(seed: int, style: str, days: int) -> None:
    """Child process: one child, one world (the one GROWING_ONTOLOGY names)."""
    import shutil
    import torch
    torch.set_num_threads(1)
    from server.game.parents import play_day
    from server.game.session import Session
    tmp = Path(tempfile.mkdtemp(prefix="wr-"))
    try:
        s = Session.new(tmp / "s", seed=seed)
        rng = random.Random(f"{seed}:{style}")
        for _ in range(days):
            play_day(s, style, 1, rng, 0.0)
            res = s.end_day()
        print(json.dumps({"panel": {k: res["panel"][k] for k in KEYS}, "burn_worse": s.game.burn_worse_rounds,
                          "burn_nothing": s.game.burn_nothing_rounds, "regret": s.game.regret_rounds,
                          "rounds": sum(e.strength for e in s.buffer), "burns": sum(e.mode == "burned" for e in s.buffer)}))
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def run(label: str, onto: Path, seed: int, style: str, days: int) -> dict:
    env = {**os.environ, "GROWING_ONTOLOGY": str(onto), "GROWING_LLM": "mock"}
    r = subprocess.run([sys.executable, __file__, "--one", str(seed), style, str(days)], env=env, capture_output=True, text=True, cwd=ROOT)
    if r.returncode:
        raise RuntimeError(f"{label}: {r.stderr[-400:]}")
    return {"label": label, "seed": seed, **json.loads(r.stdout.strip().splitlines()[-1])}


def main():
    if len(sys.argv) > 1 and sys.argv[1] == "--one":
        one(int(sys.argv[2]), sys.argv[3], int(sys.argv[4]))
        return
    ap = argparse.ArgumentParser()
    ap.add_argument("--worlds", default="0-19")
    ap.add_argument("--child-seeds", default="200-201")
    ap.add_argument("--reference-seeds", default="200-219")
    ap.add_argument("--style", default="balanced")
    ap.add_argument("--days", type=int, default=8)
    ap.add_argument("--jobs", type=int, default=12)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    from server.world.worldgen import sheet_hash, template_world, write_world
    rng = lambda spec: range(int(spec.split("-")[0]), int(spec.split("-")[-1]) + 1)  # noqa: E731
    tasks, hashes = [], {}
    root = Path(a.out).parent / "worlds"
    for w in rng(a.worlds):
        sheet = template_world(w)
        onto, _ = write_world(sheet, root / f"w{w}")
        hashes[f"w{w}"] = sheet_hash(sheet)
        tasks += [(f"w{w}", onto, s, a.style, a.days) for s in rng(a.child_seeds)]
    tasks += [("reference", REFERENCE, s, a.style, a.days) for s in rng(a.reference_seeds)]
    t0 = time.time()
    with ThreadPoolExecutor(a.jobs) as ex:
        results = list(ex.map(lambda t: run(*t), tasks))
    labels = [f"w{w}" for w in rng(a.worlds)]
    ref = [r for r in results if r["label"] == "reference"]
    m = lambda rs, k: st.mean(r["panel"][k] for r in rs)  # noqa: E731
    lines = [f"parent: {a.style}, {a.days} days; worlds {a.worlds} x child seeds {a.child_seeds}; reference = onto_age03.yaml x {len(ref)} child seeds",
             f"{'world':10s}{'hash':>14s}" + "".join(f"{k:>13s}" for k in KEYS) + f"{'burns':>7s}{'rounds':>8s}  learning(worse/nothing/regret)"]
    for lab in labels:
        rs = [r for r in results if r["label"] == lab]
        lines.append(f"{lab:10s}{hashes[lab]:>14s}" + "".join(f"{m(rs, k):13.3f}" for k in KEYS)
                     + f"{st.mean(r['burns'] for r in rs):7.1f}{st.mean(r['rounds'] for r in rs):8.0f}  {rs[0]['burn_worse']}/{rs[0]['burn_nothing']}/{rs[0]['regret']}")
    lines.append(f"{'reference':10s}{'':>14s}" + "".join(f"{m(ref, k):13.3f}" for k in KEYS) + f"{st.mean(r['burns'] for r in ref):7.1f}{st.mean(r['rounds'] for r in ref):8.0f}")
    per_world = {k: [m([r for r in results if r["label"] == lab], k) for lab in labels] for k in KEYS}
    lines.append("over worlds: " + "  ".join(f"{k} mean {st.mean(v):.3f} sd {st.pstdev(v):.3f} min {min(v):.3f} max {max(v):.3f}" for k, v in per_world.items()))
    text = "\n".join(lines)
    print(text)
    print(f"\nwall={time.time() - t0:.0f}s")
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps({"results": results, "hashes": hashes}))
    Path(a.out).with_suffix(".txt").write_text(text + "\n")


if __name__ == "__main__":
    main()
