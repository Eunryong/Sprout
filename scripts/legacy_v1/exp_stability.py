"""Stability sweep for online learning. Many seeds, configs in parallel, one summary line per config.

  uv run python scripts/exp_stability.py --seeds 100-109 --grid "base|" "cos|cosine=1" "decay|lr_decay_tau=10"

Metrics (all on the held-out quiz pool of words the child has met; greedy = argmax accuracy,
p = mean probability on the right slot):
  d3 g/p      mean greedy / p at day 3, with min over seeds
  d10 g       mean greedy at day 10, min over seeds, std over seeds
  drop        mean over seeds of (best p over days 4..N) - p at day N     -> late decline
  dips        mean over seeds of the summed day-to-day decreases of p from day 4 on -> wobble
"""
import argparse
import json
import statistics as st
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from dataclasses import fields
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def parse_seeds(s):
    out = []
    for part in s.split(","):
        if "-" in part:
            a, b = part.split("-")
            out += list(range(int(a), int(b) + 1))
        else:
            out.append(int(part))
    return out


def parse_val(v):
    for f in (int, float):
        try:
            return f(v)
        except ValueError:
            pass
    return {"true": True, "false": False}.get(v.lower(), v)


def parse_grid(item):
    name, _, kv = item.partition("|")
    kw = {}
    for pair in filter(None, kv.split(",")):
        k, v = pair.split("=")
        kw[k] = parse_val(v)
    return name, kw


def job(args):
    import torch
    torch.set_num_threads(1)
    import exp_task_a
    from server.learning.learner import OnlineCfg
    name, kw, seed, days, cond = args
    if cond == "session":
        return name, run_session(kw, seed, days)
    ofields = {f.name for f in fields(OnlineCfg)}
    ocfg = OnlineCfg(**{k: v for k, v in kw.items() if k in ofields})
    d = kw.get("d_model", 64)
    mcfg = {"d_model": d, "n_layers": kw.get("layers", 2), "n_heads": 4, "d_ff": 2 * d}
    r = exp_task_a.run(seed, cond, days, int(kw.get("per_day", 6)), mcfg, ocfg, 20, float(kw.get("p_show", 0.5)))
    return name, r


def run_session(kw, seed, days):
    """End-to-end: real Session (engine + Director with exposure balancing + quiz pool) driven by autoplay."""
    import random
    import tempfile
    from dataclasses import fields as dfields
    from server.config import GameCfg, load_configs
    from server.game.autoplay import play_day
    from server.game.session import Session
    from server.learning.learner import OnlineCfg
    _, mcfg, _ = load_configs()
    ofields = {f.name for f in dfields(OnlineCfg)}
    ocfg = OnlineCfg(**{k: v for k, v in kw.items() if k in ofields})
    game = GameCfg(per_day=int(kw.get("per_day", 6)))
    rows = []
    with tempfile.TemporaryDirectory() as d:
        s = Session.new(f"{d}/s", seed=seed, game=game, mcfg=mcfg, ocfg=ocfg)
        rng = random.Random(seed + 1)
        for day in range(1, days + 1):
            r = play_day(s, rng, float(kw.get("p_show", 0.5)))
            rows.append({"day": day, "n_examples": len(s.buffer), "greedy": r["pool_greedy"],
                         "p_correct": r["pool_p_correct"], "random": r["pool_random"]})
        secs = [e.payload["report"]["seconds"] for e in s.engine.log if e.type.value == "learned"]
    return {"seed": seed, "cond": "session", "rows": rows, "mean_latency": st.mean(secs), "max_latency": max(secs)}


def summarize(rs, days):
    def col(key, d):
        return [r["rows"][d - 1][key] for r in rs]
    drops, dips = [], []
    for r in rs:
        p = [x["p_correct"] for x in r["rows"]]
        drops.append(max(p[3:]) - p[-1])
        dips.append(sum(max(0.0, p[i - 1] - p[i]) for i in range(3, len(p))))
    d3g, d3p, dn = col("greedy", 3), col("p_correct", 3), col("greedy", days)
    return {
        "d3_g": st.mean(d3g), "d3_g_min": min(d3g), "d3_g_sd": st.pstdev(d3g), "d3_p": st.mean(d3p),
        f"d{days}_g": st.mean(dn), f"d{days}_g_min": min(dn), f"d{days}_g_sd": st.pstdev(dn),
        "drop": st.mean(drops), "dips": st.mean(dips),
        "lat": st.mean(r["mean_latency"] for r in rs), "lat_max": max(r["max_latency"] for r in rs),
        "n_ex_d3": rs[0]["rows"][2]["n_examples"],
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", default="100-109")
    ap.add_argument("--days", type=int, default=10)
    ap.add_argument("--cond", default="online", help="online | batch | none | narrow | session (real Session pipeline)")
    ap.add_argument("--jobs", type=int, default=8)
    ap.add_argument("--grid", nargs="+", required=True, help='"name|k=v,k=v" ...')
    ap.add_argument("--curve", action="store_true", help="also print mean greedy for every day")
    ap.add_argument("--out", default="")
    a = ap.parse_args()
    seeds, grid = parse_seeds(a.seeds), [parse_grid(g) for g in a.grid]
    tasks = [(n, kw, s, a.days, a.cond) for n, kw in grid for s in seeds]
    t0 = time.time()
    by = {n: [] for n, _ in grid}
    with ProcessPoolExecutor(a.jobs) as ex:
        for name, r in ex.map(job, tasks):
            by[name].append(r)
    hdr = f"{'config':26s} d3 greedy(mean/min/sd) p | d{a.days} greedy(mean/min/sd) | drop dips | lat(mean/max)"
    print(hdr)
    summ = {}
    for n, _ in grid:
        m = summarize(by[n], a.days)
        summ[n] = m
        k = f"d{a.days}_g"
        print(f"{n:26s} {m['d3_g']:.2f}/{m['d3_g_min']:.2f}/{m['d3_g_sd']:.2f} {m['d3_p']:.2f} | "
              f"{m[k]:.2f}/{m[k+'_min']:.2f}/{m[k+'_sd']:.2f} | {m['drop']:.3f} {m['dips']:.3f} | "
              f"{m['lat']:.2f}/{m['lat_max']:.2f}s  (n_ex@d3={m['n_ex_d3']})")
    if a.curve:
        for n, _ in grid:
            c = [st.mean(r["rows"][d]["greedy"] for r in by[n]) for d in range(a.days)]
            print(f"  {n:24s} greedy by day: " + " ".join(f"{x:.2f}" for x in c))
    print(f"seeds={seeds[0]}..{seeds[-1]} (n={len(seeds)}) wall={time.time()-t0:.0f}s")
    if a.out:
        Path(a.out).parent.mkdir(parents=True, exist_ok=True)
        Path(a.out).write_text(json.dumps({"args": vars(a), "summary": summ, "runs": by}, indent=1))


if __name__ == "__main__":
    main()
