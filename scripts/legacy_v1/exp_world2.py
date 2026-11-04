"""P1 feasibility (WORLD.md section 10): can a randomly initialised tiny transformer learn situation -> best option
from a synthetic player's choices? Synthetic player = script that picks a best option by the rules table. The child
sees only tags / word token / option verbs + target attributes.

Topics (what a 'card' is about): THIRSTY, FLOOR_WET, DARK, COLD, VISITOR  +  REQUEST:<kind> x4 words (+ optional
two-problem situations). The Director stand-in offers the two least-practised topics and the player picks one at random.

  uv run python scripts/exp_world2.py --seeds 100-119 --grid "base|" "pd10|per_day=10"
"""
import argparse
import json
import random
import statistics as st
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from dataclasses import fields
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

WORDS = ["bobo", "mimi", "tata", "lulu"]
PROBLEMS = ["THIRSTY", "FLOOR_WET", "DARK", "COLD", "VISITOR"]


def run(seed, cond, days, per_day, mcfg, ocfg, quiz_per_topic, p_show, topics_mode, multi, d_model_unused=None):
    import copy
    from server.learning.codec2 import (
        MAX_LEN2, VOCAB2, Example2, augment2, make_quiz2, option_probs, score2, weighted_ce2,
    )
    from server.learning.learner import Learner
    from server.learning.model import ModelCfg, build_model, n_params
    from server.world.world2 import REQUEST_KINDS, make_situation2, observation, optimal, topic_of, values

    rng = random.Random(seed)
    ks = list(REQUEST_KINDS)
    rng.shuffle(ks)
    words = {k: (WORDS[i], f"NAME_{i:02d}") for i, k in enumerate(ks)}
    # topics: key -> (tags, word, token, kind)
    topics = {p: ([p], None, None, None) for p in PROBLEMS}
    if topics_mode == "all":
        for k in ks:
            topics[f"REQUEST:{k}"] = (["REQUEST"], words[k][0], words[k][1], k)
    elif topics_mode == "grouped":  # one REQUEST topic; the word is whichever has been practised least
        topics["REQUEST"] = (["REQUEST"], None, None, None)
    quiz_topics = [v for k, v in topics.items() if k != "REQUEST"]
    if topics_mode == "grouped":
        quiz_topics += [(["REQUEST"], words[k][0], words[k][1], k) for k in ks]
    if multi > 0:
        quiz_topics += [(["COLD", "DARK"], None, None, None), (["THIRSTY", "FLOOR_WET"], None, None, None)]
    quiz = make_quiz2(random.Random(seed + 10_000), quiz_topics, quiz_per_topic)
    forbidden = {q.hash for q in quiz}
    model0 = build_model(ModelCfg(**{**mcfg, "seed": seed, "vocab_size": len(VOCAB2), "max_len": MAX_LEN2}))
    model = copy.deepcopy(model0)
    learner = Learner(model, ocfg, ce_fn=weighted_ce2, aug_fn=augment2)
    buffer, counts, met = [], {k: 0 for k in topics}, set()
    req_counts = {k: 0 for k in ks}
    rows, lat = [], []
    ex_id = 0
    for day in range(1, days + 1):
        for i in range(per_day):
            cards = sorted(topics, key=lambda k: (counts[k], rng.random()))[:2]
            key = rng.choice(cards)
            counts[key] += 1
            tags, w, tok, kind = topics[key]
            if key == "REQUEST":
                kind = min(ks, key=lambda k: (req_counts[k], rng.random()))
                req_counts[kind] += 1
                w, tok = words[kind]
                met.add(f"REQUEST:{kind}")
            else:
                met.add(key)
            if multi > 0 and tags != ["REQUEST"] and rng.random() < multi:
                other = rng.choice([p for p in PROBLEMS if p != tags[0]])
                tags = [tags[0], other]
            sit = make_situation2(rng, tags, f"s{day}.{i}", w, tok, kind, forbidden)
            obs = observation(sit)
            best = optimal(sit)
            show = (day == 1 and i == 0) or rng.random() < p_show
            if show:
                ex = Example2(id=f"e{ex_id}", day=day, source="show", obs=obs, target_slot=rng.choice(best))
            else:
                pr = option_probs(model, obs)
                pick = rng.choices(range(len(pr)), weights=pr)[0]
                if pick in best:
                    ex = Example2(id=f"e{ex_id}", day=day, source="praise", obs=obs, target_slot=pick, weight=0.5)
                else:
                    ex = Example2(id=f"e{ex_id}", day=day, source="correct", obs=obs, target_slot=rng.choice(best),
                                  rejected_slot=pick)
            ex_id += 1
            buffer.append(ex)
            if cond == "online":
                lat.append(learner.learn(ex, buffer, rng).seconds)
        seen = [q for q in quiz if all(part in met for part in q.topic.split("+"))]  # only topics the child has met
        sc = score2(model, seen)
        by_topic = {}
        for t in {q.topic for q in seen}:
            by_topic[t] = score2(model, [q for q in seen if q.topic == t])["optimal"]
        rows.append({"day": day, "n_examples": len(buffer), "n_quiz": sc["n"], "greedy": sc["optimal"],
                     "p_correct": sc["p_opt"], "value": sc["value"], "random": sc["random"],
                     "by_topic": by_topic})
    # tag-blindness probe: same options, but the tag token is swapped for a different problem tag; scored against the
    # ORIGINAL truth. A child that reads the tag should now fail; a tag-blind shortcut would not notice.
    singles = [q for q in quiz if q.obs.tags[0] in PROBLEMS and len(q.obs.tags) == 1 and q.topic in met]
    swapped = [q.model_copy(update={"obs": q.obs.model_copy(update={"tags": [PROBLEMS[(PROBLEMS.index(q.obs.tags[0]) + 1) % 5]]})})
               for q in singles]
    rows[-1]["tagswap"] = score2(model, swapped)["optimal"] if swapped else float("nan")
    rows[-1]["tag_intact"] = score2(model, singles)["optimal"] if singles else float("nan")
    return {"seed": seed, "cond": cond, "rows": rows, "mean_latency": st.mean(lat) if lat else 0.0,
            "max_latency": max(lat) if lat else 0.0, "n_params": n_params(model)}


def parse_val(v):
    for f in (int, float):
        try:
            return f(v)
        except ValueError:
            pass
    return v


def job(a):
    import torch
    torch.set_num_threads(1)
    from server.learning.learner import OnlineCfg
    name, kw, seed, days, cond = a
    of = {f.name for f in fields(OnlineCfg)}
    ocfg = OnlineCfg(**{k: v for k, v in kw.items() if k in of})
    d = int(kw.get("d_model", 64))
    mcfg = {"d_model": d, "n_layers": int(kw.get("layers", 2)), "n_heads": 4, "d_ff": 2 * d}
    r = run(seed, cond, days, int(kw.get("per_day", 8)), mcfg, ocfg, 12, float(kw.get("p_show", 0.5)),
            kw.get("topics", "all"), float(kw.get("multi", 0.0)))
    return name, r


def seeds_of(s):
    out = []
    for part in s.split(","):
        if "-" in part:
            a, b = part.split("-")
            out += range(int(a), int(b) + 1)
        else:
            out.append(int(part))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", default="100-119")
    ap.add_argument("--days", type=int, default=10)
    ap.add_argument("--cond", default="online")
    ap.add_argument("--jobs", type=int, default=16)
    ap.add_argument("--grid", nargs="+", required=True)
    ap.add_argument("--curve", action="store_true")
    ap.add_argument("--topics-detail", type=int, default=0, help="print per-topic accuracy at this day")
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
    print(f"{'config':24s} d3 optimal(mean/min/sd)  d3>=.8 | d6 mean/min | d{D} mean/min/sd | drop | rand@d3 | lat")
    summ = {}
    for n, _ in grid:
        rs = by[n]
        g = lambda d, k="greedy": [r["rows"][d - 1][k] for r in rs]
        d3, d6, dn = g(3), g(min(6, D)), g(D)
        drops = [max(x["greedy"] for x in r["rows"][3:]) - r["rows"][-1]["greedy"] for r in rs] if D > 4 else [0]
        summ[n] = {"d3": st.mean(d3), "d3_min": min(d3), "d3_ok": sum(x >= 0.8 for x in d3) / len(d3),
                   "d6": st.mean(d6), f"d{D}": st.mean(dn)}
        print(f"{n:24s} {st.mean(d3):.2f}/{min(d3):.2f}/{st.pstdev(d3):.2f}  {sum(x>=0.8 for x in d3)}/{len(d3)}   | "
              f"{st.mean(d6):.2f}/{min(d6):.2f} | {st.mean(dn):.2f}/{min(dn):.2f}/{st.pstdev(dn):.2f} | {st.mean(drops):.3f} | "
              f"{st.mean(g(3,'random')):.2f} | {st.mean(r['mean_latency'] for r in rs):.2f}s/{max(r['max_latency'] for r in rs):.2f}s")
    if a.curve:
        for n, _ in grid:
            c = [st.mean(r["rows"][d]["greedy"] for r in by[n]) for d in range(D)]
            print(f"  {n:22s} optimal by day: " + " ".join(f"{x:.2f}" for x in c))
    if a.topics_detail:
        d = a.topics_detail
        for n, _ in grid:
            tt = {}
            for r in by[n]:
                for t, v in r["rows"][d - 1]["by_topic"].items():
                    tt.setdefault(t, []).append(v)
            print(f"  {n:22s} day {d} by topic: " + "  ".join(f"{t}={st.mean(v):.2f}" for t, v in sorted(tt.items())))
    for n, _ in grid:
        ti = [r["rows"][-1]["tag_intact"] for r in by[n]]
        ts = [r["rows"][-1]["tagswap"] for r in by[n]]
        print(f"  {n:22s} day {D}: problem-tag quiz, tag intact {st.mean(ti):.2f} | tag swapped to another tag {st.mean(ts):.2f}")
    print(f"seeds={seeds[0]}..{seeds[-1]} n={len(seeds)} wall={time.time()-t0:.0f}s")
    if a.out:
        Path(a.out).parent.mkdir(parents=True, exist_ok=True)
        Path(a.out).write_text(json.dumps({"args": vars(a), "summary": summ}, indent=1))


if __name__ == "__main__":
    main()
