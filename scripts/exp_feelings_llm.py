"""muse-spark as part of the learning, three ways (all research only; the key is read from .env.local by the game's client and never printed):

  oracle   the model writes what a baby would feel for every (situation, response): how consistent is it with itself and with the
           hand-written table? Tables are saved to scripts/out/feel_llm_tables.json (R repeats of the same question).
  prior    (offline, in exp_feelings.py arms pre_llm*) a tiny child pretrained on those tables = innate common sense from the LLM.
  control  the model decides HOW the child is trained, scene by scene (steps, learning-rate scale, how much old experience to replay)
           from a summary of how learning is going. This is a teacher: the game's rule is that the Director never reacts to what the
           child learned, so this arm is a research measurement, not a feature.

  uv run python scripts/exp_feelings_llm.py oracle --repeats 5
  uv run python scripts/exp_feelings_llm.py control --seeds 300-303
"""
import argparse
import json
import random
import statistics as st
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import exp_feelings as X  # noqa: E402

from pydantic import BaseModel  # noqa: E402

TABLES = X.OUT / "feel_llm_tables.json"
SITUATION = {"COLD": "The baby is cold and shivering.", "BABY_HUNGRY": "The baby is hungry and fussing for milk.", "UPSET": "The baby is crying and cannot settle.",
             "BORED": "The baby is bored and fretful.", "DARK": "The baby is frightened in the dark."}
DO = {"COVER": "wraps the baby in the {k}", "HUG": "cuddles the baby with the {k}", "FEED": "feeds the baby with the {k}", "GIVE": "gives the baby the {k} to hold",
      "FETCH": "brings the baby the {k}", "READ": "shows the baby the {k}", "LIGHT": "switches on the {k} near the baby", "DRAW": "waves the {k} in front of the baby",
      "WAIT": "waits and does nothing"}
ORACLE = ("You are the world of a gentle baby-care simulation. For a baby in a situation and one response of the parent, say what the baby FEELS, "
          "using only these words: relieved (the need is met: warm, full, comfortable), cared (feels looked after and safe), eased (a little better, "
          "not all the way), unmet (the need is still there), upset (annoyed or distressed by this), ignored (feels nobody is helping). "
          "Give one or two words per response, as a real baby would feel it. Never say what the parent should do. "
          'Reply JSON only: {"answers": {"<response id>": ["word", ...]}} with EVERY response id you were given.')


class Answers(BaseModel):
    answers: dict[str, list[str]]


def client():
    from server.agents.factory import make_client
    return make_client({}, mode="live")


def ask_tag(cl, tag: str) -> dict:
    ids = {f"{v} {k}": DO[v].format(k=k) if k else DO["WAIT"] for v, k in X.CELLS}
    user = json.dumps({"situation": SITUATION[tag], "responses": ids}, ensure_ascii=False)
    for _ in range(3):
        try:
            out = cl.complete_json(agent="director", system=ORACLE, user=user, schema=Answers)
            ans = {k: [f for f in v if f in X.FEELS][:2] for k, v in out.answers.items() if k in ids}
            if len(ans) >= 0.9 * len(ids) and all(ans.values()):
                return ans
        except Exception as e:  # noqa: BLE001
            print("retry:", type(e).__name__, str(e)[:120], file=sys.stderr)
    return {}


def key(cell) -> str:
    return f"{cell[0]} {cell[1]}"


def oracle(a) -> None:
    cl = client()
    t0 = time.time()
    jobs = [(r, t) for r in range(a.repeats) for t in X.TAGS]
    with ThreadPoolExecutor(a.jobs) as ex:
        out = list(ex.map(lambda j: (j, ask_tag(cl, j[1])), jobs))
    tables = [{} for _ in range(a.repeats)]
    for (r, t), ans in out:
        tables[r][t] = ans
    TABLES.write_text(json.dumps(tables, ensure_ascii=False))
    hand = X.hand_table()

    def jac(x, y):
        x, y = set(x), set(y)
        return len(x & y) / len(x | y) if x | y else 1.0

    def ev(fs):
        return st.mean(X.VAL[X.FEELS.index(f)] for f in fs)
    self_j, hand_j, hand_ev, n = [], [], [], 0
    for t in X.TAGS:
        for c in X.CELLS:
            sets = [tables[r].get(t, {}).get(key(c)) for r in range(a.repeats)]
            sets = [s for s in sets if s]
            if len(sets) < 2:
                continue
            n += 1
            self_j.append(st.mean(jac(sets[i], sets[j]) for i in range(len(sets)) for j in range(i + 1, len(sets))))
            h = [X.FEELS[i] for i in (hand[t].get(c) if c in hand[t] else hand[t]["WAIT"] if c[0] == "WAIT" else hand[t]["_"])]
            hand_j.append(st.mean(jac(s, h) for s in sets))
            hand_ev.append((st.mean(ev(s) for s in sets), ev(h)))
    mx = [x for x, _ in hand_ev]
    my = [y for _, y in hand_ev]
    mean_x, mean_y = st.mean(mx), st.mean(my)
    corr = sum((x - mean_x) * (y - mean_y) for x, y in hand_ev) / (len(hand_ev) * st.pstdev(mx) * st.pstdev(my))
    print(json.dumps({"calls": len(jobs), "seconds": round(time.time() - t0), "cells": n, "self_agreement_jaccard": round(st.mean(self_j), 3),
                      "agreement_with_hand_table_jaccard": round(st.mean(hand_j), 3), "valence_correlation_with_hand": round(corr, 3)}))


CONTROL = ("You tune how a tiny neural network (a baby's mind, learning online one lived scene at a time) is trained. Before each scene is learned "
           "you are given a short summary of how learning is going and you choose: `steps` (gradient steps on this scene, 3 to 60), `replay` (how many "
           "older scenes are replayed with it, 0 to 32) and `lr_scale` (0.25 to 2.0). There are only a few scenes in a whole life (84), the first 12 "
           "come when the baby cannot act. You do not know what is right in the world; use only the summary. "
           'Reply JSON only: {"steps": int, "replay": int, "lr_scale": number}.')


class Plan(BaseModel):
    steps: int
    replay: int
    lr_scale: float


def make_hook(cl, log: list):
    def hook(summary: dict):
        t0 = time.time()
        try:
            p = cl.complete_json(agent="director", system=CONTROL, user=json.dumps(summary), schema=Plan)
            out = (min(60, max(3, p.steps)), min(32, max(0, p.replay)), min(2.0, max(0.25, p.lr_scale)))
        except Exception as e:  # noqa: BLE001
            print("control fallback:", type(e).__name__, str(e)[:100], file=sys.stderr)
            out = (20, 16, 1.0)
        log.append({"n": summary["experiences_so_far"], "surprise": summary["this_surprise"], "plan": out, "seconds": round(time.time() - t0, 1)})
        return out
    return hook


def control(a) -> None:
    cl = client()
    lo, hi = a.seeds.split("-")
    seeds = list(range(int(lo), int(hi) + 1))
    logs = {sd: [] for sd in seeds}

    def run(sd):
        return sd, X.one("ctrl_llm", sd, "hand", make_hook(cl, logs[sd]))
    with ThreadPoolExecutor(len(seeds)) as ex:
        rows = list(ex.map(run, seeds))
    (X.OUT / "feelings_ctrl_llm.json").write_text(json.dumps([{"arm": "ctrl_llm", "seed": sd, "res": r, "plans": logs[sd]} for sd, r in rows]))
    text = X.report(["ctrl_llm"], [("ctrl_llm", sd, r) for sd, r in rows], X.OUT / "feelings_ctrl_llm.txt")
    plans = [p for sd in seeds for p in logs[sd]]
    steps = [p["plan"][0] for p in plans]
    print(text)
    print(json.dumps({"calls": len(plans), "mean_call_seconds": round(st.mean(p["seconds"] for p in plans), 1), "steps_mean": round(st.mean(steps), 1),
                      "steps_first12": round(st.mean(p["plan"][0] for p in plans if p["n"] <= 12), 1), "steps_after12": round(st.mean(p["plan"][0] for p in plans if p["n"] > 12), 1),
                      "lr_scale_mean": round(st.mean(p["plan"][2] for p in plans), 2), "replay_mean": round(st.mean(p["plan"][1] for p in plans), 1)}))


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("control")
    c.add_argument("--seeds", default="300-305")
    o = sub.add_parser("oracle")
    o.add_argument("--repeats", type=int, default=5)
    o.add_argument("--jobs", type=int, default=5)
    a = ap.parse_args()
    if a.cmd == "oracle":
        oracle(a)
    else:
        control(a)


if __name__ == "__main__":
    main()
