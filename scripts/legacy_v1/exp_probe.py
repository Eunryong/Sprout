"""Does narrow teaching leave a visible bias in the contrast probe?

Conditions (real Session pipeline, synthetic perfect player, 6 situations/day):
  diverse         default Director: target color/size vary
  narrow          every requested object is RED and SMALL (the player only ever shows a red small one)
  narrow-noredraw same, with the distractor-redraw augmentation switched off
Reported per day (mean over seeds and words): quiz accuracy overall / on off-surface targets (not red+small),
probe pull per attribute (0 = indifferent, 0.5 = always the same value) and the probe's preference for RED.
"""
import argparse
import random
import statistics as st
import sys
import tempfile
from concurrent.futures import ProcessPoolExecutor
from dataclasses import replace
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

CONDS = {"diverse": {}, "narrow": {"narrow": True}, "narrow-noredraw": {"narrow": True, "redraw": 0.0}}


def job(args):
    import torch
    torch.set_num_threads(1)
    from server.config import GameCfg, load_configs
    from server.game.autoplay import play_day
    from server.game.session import Session
    from server.learning.evaluator import score
    name, seed, days = args
    opts = CONDS[name]
    _, mcfg, ocfg = load_configs()
    ocfg = replace(ocfg, redraw=opts.get("redraw", ocfg.redraw))
    rows = []
    with tempfile.TemporaryDirectory() as d:
        s = Session.new(f"{d}/s", seed=seed, game=GameCfg(per_day=6), mcfg=mcfg, ocfg=ocfg, client=None)
        if opts.get("narrow"):
            s.director.force_target_surface = ("red", "small")
        rng = random.Random(seed + 1)
        for day in range(1, days + 1):
            r = play_day(s, rng)
            m = s.mind()
            known = {s.state.words[w] for w in s.state.picked_words}
            pool = [q for q in s.quiz if q.obs.name_token in known]
            off = [q for q in pool if (q.color, q.size) != ("red", "small")]
            on = [q for q in pool if (q.color, q.size) == ("red", "small")]
            rows.append({
                "day": day, "acc": score(s.model, pool)["greedy"], "acc_off": score(s.model, off)["greedy"],
                "acc_on": score(s.model, on)["greedy"],
                **{f"pull_{a}": st.mean(e["pulls"][a] for e in m.values()) for a in ("color", "shape", "size", "material")},
                "red": st.mean(e["prefs"]["color"]["red"] for e in m.values()),
                "small": st.mean(e["prefs"]["size"]["small"] for e in m.values()),
                "top": [e["cues"][0][:2] for e in m.values()],
            })
    return name, seed, rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=20)
    ap.add_argument("--days", type=int, default=6)
    ap.add_argument("--jobs", type=int, default=16)
    ap.add_argument("--show-days", default="1,2,3,6")
    a = ap.parse_args()
    seeds = range(300, 300 + a.seeds)
    tasks = [(n, s, a.days) for n in CONDS for s in seeds]
    by = {n: [] for n in CONDS}
    with ProcessPoolExecutor(a.jobs) as ex:
        for n, s, rows in ex.map(job, tasks):
            by[n].append(rows)
    show = [int(x) for x in a.show_days.split(",")]
    cols = ["acc", "acc_off", "acc_on", "pull_color", "pull_shape", "pull_material", "pull_size", "red", "small"]
    print(f"{'cond':16s} day " + " ".join(f"{c:>13s}" for c in cols))
    for n in CONDS:
        for d in show:
            print(f"{n:16s} {d:3d} " + " ".join(f"{st.mean(r[d-1][c] for r in by[n]):13.2f}" for c in cols))
    print("\nMost common strongest cue per word at the last day (attribute,value):")
    last = a.days
    for n in CONDS:
        cues = [tuple(c) for r in by[n] for c in r[last - 1]["top"]]
        tally = {}
        for c in cues:
            tally[c] = tally.get(c, 0) + 1
        top = sorted(tally.items(), key=lambda kv: -kv[1])[:5]
        print(f"  {n:16s} " + ", ".join(f"{c[0]}={c[1]} x{k}" for c, k in top), f"(of {len(cues)})")


if __name__ == "__main__":
    main()
