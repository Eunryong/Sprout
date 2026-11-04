"""Do the child's curves hold in worlds the real Director made? (docs/DIRECTION.md 6, mock-experiment caveat)

The sheets in scripts/out/world_live.json were written by the real Director (scripts/exp_world_live.py; nothing is called here, the
key is not used). Each is written out as an ontology file, and a child is raised in it with the same two measurements as before:
  calibration (exp_calib.py --one, after 1/3/10/30/64 scenes): conf, acc, ece, sat, problem_acc, blind
  exploration (exp_explore.py --one, alone, 40 scenes) at curiosity beta 0 / 0.4 / 0.8
against the hand-written world (config/ontology.yaml) raised the same way. The ontology is a module global, so each run is its own process.

  uv run python scripts/exp_live.py --out scripts/out/live_curves.json
"""
import argparse
import json
import os
import statistics as st
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
CHECKS = ["1", "3", "10", "30", "64"]
CAL_KEYS = ["conf", "acc", "ece", "sat", "problem_acc", "blind"]
EXP_KEYS = ["kinds", "new_late", "solved_late", "worse", "problem_acc"]
BETAS = [0.0, 0.4, 0.8]


def materialise(src: Path, root: Path) -> dict[str, Path]:
    from server.world.worldgen import WorldSheet, write_world
    out = {}
    for i, r in enumerate(json.loads(src.read_text())):
        onto, _ = write_world(WorldSheet.model_validate(r["sheet"]), root / f"d{i}")
        out[f"d{i}"] = onto
    return out


def sub(script: str, args: list, onto: Path | None, extra: dict) -> dict:
    env = {**os.environ, "GROWING_LLM": "mock", **extra}
    if onto is not None:
        env["GROWING_ONTOLOGY"] = str(onto.resolve())
    r = subprocess.run([sys.executable, str(ROOT / "scripts" / script), "--one", *map(str, args)], env=env, capture_output=True, text=True, cwd=ROOT)
    if r.returncode:
        raise RuntimeError(f"{script} {args} {onto}: {r.stderr[-300:]}")
    return json.loads(r.stdout.strip().splitlines()[-1])


def job(kind: str, world: str, onto: Path | None, seed: int, beta: float) -> dict:
    if kind == "calib":  # the defaults: smoothing 0.15, 20 steps (config/model.yaml)
        return {"kind": kind, "world": world, "seed": seed, "beta": beta, "at": sub("exp_calib.py", [seed, 20, 0.0, 0.002], onto, {"GROWING_SMOOTH": "0.15"})}
    return {"kind": kind, "world": world, "seed": seed, "beta": beta, "at": sub("exp_explore.py", [seed], onto, {"GROWING_CURIOSITY": str(beta)})}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sheets", default=str(ROOT / "scripts/out/world_live.json"))
    ap.add_argument("--child-seeds", default="200-202")
    ap.add_argument("--hand-seeds", default="200-211")
    ap.add_argument("--jobs", type=int, default=12)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    rg = lambda s: range(int(s.split("-")[0]), int(s.split("-")[-1]) + 1)  # noqa: E731
    worlds = materialise(Path(a.sheets), Path(a.out).parent / "live_worlds")
    tasks = []
    for kind, betas in (("calib", [0.0]), ("explore", BETAS)):
        for b in betas:
            tasks += [(kind, "hand", None, s, b) for s in rg(a.hand_seeds)]
            tasks += [(kind, w, p, s, b) for w, p in worlds.items() for s in rg(a.child_seeds)]
    with ThreadPoolExecutor(a.jobs) as ex:
        res = list(ex.map(lambda t: job(*t), tasks))

    def mean(rs, f):
        return st.mean(f(r) for r in rs) if rs else float("nan")
    live = lambda rs: [r for r in rs if r["world"] != "hand"]  # noqa: E731
    hand = lambda rs: [r for r in rs if r["world"] == "hand"]  # noqa: E731
    cal = [r for r in res if r["kind"] == "calib"]
    lines = [f"{len(worlds)} worlds made by the real Director x seeds {a.child_seeds}; hand-written world x seeds {a.hand_seeds}", "",
             "calibration (smoothing 0.15, 20 steps)" + "".join(f"{k:>13s}" for k in CAL_KEYS) + f"{'conf-acc':>10s}"]
    for n in CHECKS:
        for lab, sel in (("hand", hand), ("live (mean of worlds)", live)):
            rs = sel(cal)
            lines.append(f"  after {n:>2s} scenes {lab:22s}" + "".join(f"{mean(rs, lambda r, k=k: r['at'][n][k]):13.3f}" for k in CAL_KEYS)
                         + f"{mean(rs, lambda r: r['at'][n]['conf'] - r['at'][n]['acc']):10.3f}")
    lines += ["", "per world, after 64 scenes:   conf-acc    ece   problem_acc   blind"]
    for w in worlds:
        rs = [r for r in cal if r["world"] == w]
        lines.append(f"  {w:6s}{'':16s}{mean(rs, lambda r: r['at']['64']['conf'] - r['at']['64']['acc']):8.3f}{mean(rs, lambda r: r['at']['64']['ece']):9.3f}"
                     f"{mean(rs, lambda r: r['at']['64']['problem_acc']):11.3f}{mean(rs, lambda r: r['at']['64']['blind']):9.3f}")
    lines += ["", "exploration, alone, 40 scenes" + "".join(f"{k:>13s}" for k in EXP_KEYS)]
    exp = [r for r in res if r["kind"] == "explore"]
    for b in BETAS:
        for lab, sel in (("hand", hand), ("live (mean of worlds)", live)):
            rs = sel([r for r in exp if r["beta"] == b])
            lines.append(f"  beta {b:<4} {lab:22s}" + "".join(f"{mean(rs, lambda r, k=k: r['at'][k]):13.3f}" for k in EXP_KEYS))
    text = "\n".join(lines)
    print(text)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(res))
    Path(a.out).with_suffix(".txt").write_text(text + "\n")


if __name__ == "__main__":
    main()
