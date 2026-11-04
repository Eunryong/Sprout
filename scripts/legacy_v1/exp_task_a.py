"""D1 gate: can a randomly initialised tiny transformer learn word -> kind from a synthetic player's
choices? The synthetic player is a script that always knows the right answer; it is NOT an LLM and the
child model never sees the kind label.

Conditions
  online  : learn right after each choice (the planned design)
  batch   : same data, one compute-matched training run at the end of each day
  none    : same interaction, weights never updated (learning-disabled control)
  narrow  : like online, but the player only ever shows RED/SMALL targets (tests the 'data diversity' premise)
"""
import argparse
import copy
import json
import random
import statistics as st
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from server.learning.dataset import Example  # noqa: E402
from server.learning.evaluator import make_quiz, score  # noqa: E402
from server.learning.learner import Learner, OnlineCfg  # noqa: E402
from server.learning.model import ModelCfg, build_model, n_params  # noqa: E402
from server.learning.policy import act  # noqa: E402
from server.world.catalog import KINDS  # noqa: E402
from server.world.scenegen import make_situation  # noqa: E402
from server.world.situation import answer_slots, build_observation  # noqa: E402

WORDS = ["bobo", "mimi", "tata", "lulu"]


def run(seed, cond, days, per_day, mcfg, ocfg, quiz_n, p_show=0.5):
    rng = random.Random(seed)
    kinds = list(KINDS)
    rng.shuffle(kinds)
    words = {k: (WORDS[i], f"NAME_{i:02d}") for i, k in enumerate(kinds)}
    quiz = make_quiz(random.Random(seed + 10_000), words, quiz_n)
    forbidden = {q.hash for q in quiz}
    model0 = build_model(ModelCfg(**{**mcfg, "seed": seed}))
    model = copy.deepcopy(model0)
    learner = Learner(model, ocfg)
    buffer: list[Example] = []
    narrow = cond == "narrow"
    seen_kinds: set[str] = set()
    rows, lat = [], []
    n_obj = 0
    ex_id = 0
    for day in range(1, days + 1):
        day_examples = []
        for i in range(per_day):
            kind = rng.choice(kinds)
            seen_kinds.add(kind)
            w, tok = words[kind]
            sit, objs = make_situation(rng, f"s{day}.{i}", n_obj, w, tok, kind, forbidden=forbidden,
                                       target_surface=("red", "small") if narrow else None)
            n_obj += len(objs)
            om = {o.id: o for o in objs}
            obs = build_observation(sit, om)
            right = answer_slots(sit, om)[0]
            show = (day == 1 and i == 0) or rng.random() < p_show
            if show:
                ex = Example(id=f"e{ex_id}", day=day, source="show", obs=obs, target_slot=right)
            else:
                a = act(model, obs, rng)  # the child's own sampled attempt
                if a.slot == right:
                    ex = Example(id=f"e{ex_id}", day=day, source="praise", obs=obs, target_slot=right, weight=0.5)
                else:
                    ex = Example(id=f"e{ex_id}", day=day, source="correct", obs=obs, target_slot=right,
                                 rejected_slot=a.slot)
            ex_id += 1
            buffer.append(ex)
            day_examples.append(ex)
            if cond in ("online", "narrow"):
                rep = learner.learn(ex, buffer, rng)
                lat.append(rep.seconds)
        if cond == "batch":
            rep = learner.learn_batch(buffer, rng, steps=ocfg.steps * len(day_examples))
            lat.append(rep.seconds)
        qs = [q for q in quiz if q.kind in seen_kinds]  # only words the child has met
        sc = score(model, qs)
        base = score(model0, qs)
        row = {"day": day, "n_examples": len(buffer), "n_quiz": sc["n"], **{k: round(v, 3) for k, v in sc.items() if k != "n"},
               "untrained_p": round(base["p_correct"], 3)}
        if narrow:
            off = [q for q in qs if (q.color, q.size) != ("red", "small")]
            row["p_correct_offsurface"] = round(score(model, off)["p_correct"], 3)
            row["n_off"] = len(off)
        rows.append(row)
    return {"seed": seed, "cond": cond, "rows": rows, "max_latency": max(lat) if lat else 0.0,
            "mean_latency": st.mean(lat) if lat else 0.0, "n_params": n_params(model)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", default="0,1,2")
    ap.add_argument("--conds", default="online,batch,none,narrow")
    ap.add_argument("--days", type=int, default=10)
    ap.add_argument("--per-day", type=int, default=6)
    ap.add_argument("--quiz-n", type=int, default=20, help="quiz items per kind")
    ap.add_argument("--lr", type=float, default=OnlineCfg.lr)
    ap.add_argument("--steps", type=int, default=OnlineCfg.steps)
    ap.add_argument("--replay", type=int, default=OnlineCfg.replay)
    ap.add_argument("--oversample", type=int, default=OnlineCfg.oversample)
    ap.add_argument("--d-model", type=int, default=ModelCfg.d_model)
    ap.add_argument("--layers", type=int, default=ModelCfg.n_layers)
    ap.add_argument("--out", default="")
    a = ap.parse_args()
    mcfg = {"d_model": a.d_model, "n_layers": a.layers, "n_heads": 4, "d_ff": a.d_model * 2}
    ocfg = OnlineCfg(lr=a.lr, steps=a.steps, replay=a.replay, oversample=a.oversample)
    results = []
    t0 = time.time()
    for cond in a.conds.split(","):
        for seed in map(int, a.seeds.split(",")):
            r = run(seed, cond, a.days, a.per_day, mcfg, ocfg, a.quiz_n)
            results.append(r)
            last = r["rows"]
            d3 = last[min(2, len(last) - 1)]
            print(f"{cond:7s} seed={seed} p@d3={d3['p_correct']:.3f} greedy@d3={d3['greedy']:.3f} "
                  f"p@d{a.days}={last[-1]['p_correct']:.3f} greedy@d{a.days}={last[-1]['greedy']:.3f} "
                  f"rand={last[-1]['random']:.3f} lat(mean/max)={r['mean_latency']:.2f}/{r['max_latency']:.2f}s", flush=True)
    print(f"params={results[0]['n_params']}  wall={time.time()-t0:.0f}s")
    if a.out:
        Path(a.out).parent.mkdir(parents=True, exist_ok=True)
        Path(a.out).write_text(json.dumps({"args": vars(a), "results": results}, indent=1))


if __name__ == "__main__":
    main()
