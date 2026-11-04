"""P2 gate: the REAL game stack (engine v2 + template Director through the Mock client + online learner + quiz) played
by a synthetic player over many seeds. Unlike the P1 harness this goes through Session, so it measures what a player
will actually get: word requests only as rare follow-ups, follow-up problems, unlocked places, 3-layer model.

  uv run python scripts/exp_gate.py --seeds 200-239 --days 10 --jobs 16 --grid "pd8|per_day=8" "pd10|per_day=10"
  uv run python scripts/exp_gate.py --seeds 200-239 --days 10 --cond none --grid "control|per_day=8"   # learning off
"""
import argparse
import json
import random
import shutil
import statistics as st
import sys
import tempfile
import time
from concurrent.futures import ProcessPoolExecutor
from dataclasses import fields
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

GROUPS = ("problems", "requests", "combos")


def group_of(topic: str) -> str:
    if topic.startswith("REQUEST"):
        return "requests"
    return "combos" if "+" in topic else "problems"


def run(seed: int, cond: str, days: int, kw: dict):
    import torch
    torch.set_num_threads(1)
    from server.config import GameCfg, load_configs
    from server.game.session import Session
    from server.learning.evaluator import score, tag_reliance
    from server.learning.learner import OnlineCfg
    from server.learning.model import ModelCfg
    from server.world.events import EndDay, RecordQuiz, Skip
    from server.world.state import Phase

    g, m, o = load_configs()
    game = GameCfg(**{**g.__dict__, **{k: v for k, v in kw.items() if k in {f.name for f in fields(GameCfg)}}})
    mcfg = ModelCfg(**{**m.to_dict(), **{k: v for k, v in kw.items() if k in {f.name for f in fields(ModelCfg)}}})
    ocfg = OnlineCfg(**{**o.__dict__, **{k: v for k, v in kw.items() if k in {f.name for f in fields(OnlineCfg)}}})
    tmp = Path(tempfile.mkdtemp(prefix=f"gate{seed}-"))
    try:
        s = Session.new(tmp / "s", seed=seed, game=game, mcfg=mcfg, ocfg=ocfg)
        rng = random.Random(seed + 1)
        p_show = float(kw.get("p_show", 0.5))
        p_mistake = float(kw.get("p_mistake", 0.0))  # how often the player carries out a non-best option
        rows, first = [], True
        for day in range(1, days + 1):
            while s.state.phase != Phase.EVENING:
                if s.state.phase == Phase.CARDS:
                    s.draw_cards()
                    s.pick_card(rng.randrange(2))
                best = s.debug_best()
                if cond == "none":  # learning off: the same play, but the child never changes
                    s.engine.submit(Skip(action_id=f"skip{len(s.engine.log)}"))
                    continue
                n_opts = len(s.state.current.options)
                bad = [i for i in range(n_opts) if i not in best]
                pick = rng.choice(bad) if bad and rng.random() < p_mistake else rng.choice(best)  # a slip of the player
                if (first and day == 1) or rng.random() < p_show:
                    s.show(pick)
                else:
                    r = s.try_child()
                    if r.was_right:
                        s.praise()
                    else:
                        s.correct(pick if pick != r.option else rng.choice([b for b in best]))  # cannot 'correct' to the same
                first = False
            met = s.met_topics()
            row = {"day": day}
            for grp in GROUPS:
                items = [q for q in s.quiz if q.topic in met and group_of(q.topic) == grp]
                if items:
                    row[grp] = score(s.model, items)["optimal"]
                    row[grp + "_untrained"] = score(s.model0, items)["optimal"]
                    row[grp + "_random"] = score(s.model0, items)["random"]
            row["by_topic"] = {t: score(s.model, [q for q in s.quiz if q.topic == t])["optimal"]
                               for t in sorted(met) if group_of(t) == "problems"}  # single problems only
            row["age_months"] = s.view().age_months
            row["situations"] = sum(s.state.topic_counts.values())
            s.engine.submit(RecordQuiz(action_id=f"q{day}", result={}))
            s.engine.submit(EndDay(action_id=f"e{day}"))
            rows.append(row)
        picked = s.state.topic_counts
        lat = [e.payload["report"]["seconds"] for e in s.engine.log if e.type.value == "learned"]
        tr = tag_reliance(s.model, [q for q in s.quiz if q.topic in s.met_topics()])
        followups = sum(1 for e in s.engine.log if e.type.value == "cards_offered"
                        for c in e.payload["situations"] if c["links_to"])
        return {"seed": seed, "rows": rows, "topic_counts": picked, "n_params": sum(p.numel() for p in s.model.parameters()),
                "mean_latency": st.mean(lat) if lat else 0.0, "max_latency": max(lat) if lat else 0.0,
                "tag_reliance": tr, "followup_cards": followups, "words_known": len(s.state.words),
                "open_issues": len(s.state.open_issues()), "unlocked": list(s.state.unlocked)}
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def job(a):
    name, kw, seed, days, cond = a
    return name, run(seed, cond, days, kw)


def parse_val(v):
    for f in (int, float):
        try:
            return f(v)
        except ValueError:
            pass
    return v


def seeds_of(s):
    out = []
    for part in s.split(","):
        if "-" in part:
            a, b = part.split("-")
            out += range(int(a), int(b) + 1)
        else:
            out.append(int(part))
    return out


def show_opening(by, grid, D):
    """With an age schedule: held-out accuracy of each problem at the end of the day it opened and the days after (k = days
    since opening), and at the last day. Only seeds that met the topic count."""
    from server.world.ontology import PROBLEMS, opened_on
    if all(opened_on(t) == 1 for t in PROBLEMS):
        return
    for n, _ in grid:
        rs = by[n]
        print(f"\n== {n}: accuracy by days since the problem opened (k=0 is the end of its first day), last day D={D} ==")
        print(f"{'problem':10s} opens  " + "".join(f"{'k=' + str(k):>8s}" for k in range(4)) + f"{'day ' + str(D):>9s}{'peak':>7s}{'met':>5s}")
        for t in PROBLEMS:
            o = opened_on(t)
            cols = []
            for k in range(4):
                d = o + k
                v = [r["rows"][d - 1]["by_topic"][t] for r in rs if d <= D and t in r["rows"][d - 1].get("by_topic", {})]
                cols.append(f"{st.mean(v):8.2f}" if v else f"{'-':>8s}")
            last = [r["rows"][D - 1]["by_topic"][t] for r in rs if t in r["rows"][D - 1].get("by_topic", {})]
            peak = [max(r["rows"][d]["by_topic"].get(t, 0) for d in range(o - 1, D)) for r in rs if t in r["rows"][D - 1].get("by_topic", {})]
            print(f"{t:10s} day {o}  " + "".join(cols) + (f"{st.mean(last):9.2f}{st.mean(peak):7.2f}{len(last):5d}" if last else ""))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", default="100-119")
    ap.add_argument("--days", type=int, default=10)
    ap.add_argument("--cond", default="online")
    ap.add_argument("--jobs", type=int, default=16)
    ap.add_argument("--grid", nargs="+", required=True)
    ap.add_argument("--out", default="")
    a = ap.parse_args()
    grid = []
    for g in a.grid:
        name, _, kv = g.partition("|")
        grid.append((name, {k: parse_val(v) for k, v in (p.split("=") for p in kv.split(",") if p)}))
    seeds = seeds_of(a.seeds)
    tasks = [(n, kw, s, a.days, a.cond) for n, kw in grid for s in seeds]
    t0 = time.time()
    by = {n: [] for n, _ in grid}
    with ProcessPoolExecutor(a.jobs) as ex:
        for n, r in ex.map(job, tasks):
            by[n].append(r)
    D = a.days
    summ = {}
    for n, _ in grid:
        rs = by[n]
        print(f"\n== {n} ==  n={len(rs)} seeds, params={rs[0]['n_params']}, learn latency mean {st.mean(r['mean_latency'] for r in rs):.2f}s "
              f"max {max(r['max_latency'] for r in rs):.2f}s")
        print(f"{'group':9s} {'d3 mean/min/sd':16s} d3>=.8  d6 mean | d{D} mean/min | untrained@d3 chance@d3 | by day")
        summ[n] = {}
        for grp in GROUPS:
            col = lambda d, k=grp: [r["rows"][d - 1][k] for r in rs if k in r["rows"][d - 1]]
            d3, d6, dn = col(3), col(min(6, D)), col(D)
            if not d3:
                continue
            curve = [st.mean(x) for d in range(1, D + 1) if (x := col(d))]
            summ[n][grp] = {"d3": st.mean(d3), "d3_min": min(d3), "d3_ok": sum(x >= 0.8 for x in d3), "n": len(d3),
                            "dN": st.mean(dn), "curve": curve}
            print(f"{grp:9s} {st.mean(d3):.2f}/{min(d3):.2f}/{st.pstdev(d3):.2f}    {sum(x >= 0.8 for x in d3):3d}/{len(d3):<3d}  {st.mean(d6):.2f}   | "
                  f"{st.mean(dn):.2f}/{min(dn):.2f}      | {st.mean(col(3, grp + '_untrained')):.2f}          {st.mean(col(3, grp + '_random')):.2f}      | "
                  + " ".join(f"{x:.2f}" for x in curve))
        tc = [sum(v for k, v in r["topic_counts"].items() if k.startswith("REQUEST")) for r in rs]
        multi = [sum(v for k, v in r["topic_counts"].items() if "+" in k) for r in rs]
        total = [sum(r["topic_counts"].values()) for r in rs]
        print(f"situations picked per run {st.mean(total):.0f}: word requests {st.mean(tc):.1f}, two-problem {st.mean(multi):.1f}, "
              f"follow-up cards offered {st.mean(r['followup_cards'] for r in rs):.1f}, words known {st.mean(r['words_known'] for r in rs):.1f}, "
              f"open issues at end {st.mean(r['open_issues'] for r in rs):.1f}, yard unlocked {sum('yard' in r['unlocked'] for r in rs)}/{len(rs)}")
        tr = [r["tag_reliance"] for r in rs if r["tag_reliance"]["n"]]
        if tr:
            print(f"reads the tag: intact {st.mean(t['intact'] for t in tr):.2f} -> swapped {st.mean(t['swapped'] for t in tr):.2f}")
    show_opening(by, grid, D)
    print(f"seeds={seeds[0]}..{seeds[-1]} wall={time.time() - t0:.0f}s")
    if a.out:
        Path(a.out).parent.mkdir(parents=True, exist_ok=True)
        Path(a.out).write_text(json.dumps({"args": vars(a), "summary": summ}, indent=1))


if __name__ == "__main__":
    main()
