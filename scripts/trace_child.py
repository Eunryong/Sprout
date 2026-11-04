"""Watch one child grow, scene by scene (so nothing about the research stays in the background).

For every scene it prints what the parent did (said something / showed something / nothing: the child decides alone), what the child expected of each
option BEFORE it acted, what it did, what came of it, and the fact it learned. Nothing is labelled right or wrong: the
numbers are the child's own expectations and the world's results.

  uv run python scripts/trace_child.py --scenes 12 --parent says --seed 1
  GROWING_ONTOLOGY=scripts/out/world_api77/ontology.yaml uv run python scripts/trace_child.py --scenes 12
"""
import argparse
import os
import random
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
os.environ.setdefault("GROWING_LLM", "mock")

from server.game.session import Session  # noqa: E402
from server.game.simple import expectations as simple_expectations  # noqa: E402
from server.learning.codec import RESULTS  # noqa: E402
from server.world.ontology import outcome_label  # noqa: E402
from server.world.situation import observation, values  # noqa: E402
from server.world.state import Phase  # noqa: E402


def expectations(s: Session, obs) -> list[list[float]]:
    return [[e[r] for r in RESULTS] for e in simple_expectations(s, obs)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scenes", type=int, default=12)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--parent", choices=["says", "shows", "alone", "mixed"], default="mixed", help="how the parent takes part in a scene")
    ap.add_argument("--p-wrong", type=float, default=0.3, help="how often what the parent says or shows does not solve the problem")
    a = ap.parse_args()
    rng = random.Random(a.seed)
    s = Session.new(tempfile.mkdtemp(prefix="trace-") + "/s", seed=a.seed)
    done = 0
    while done < a.scenes:
        if s.state.phase == Phase.EVENING:
            s.end_day()
        s.draw_cards()
        s.pick_card(0)
        cur = s.state.current
        vals = values(cur)
        how = rng.choice(["says", "shows"]) if a.parent == "mixed" else a.parent
        good = [i for i, v in enumerate(vals) if v == max(vals)]
        other = [i for i in range(len(vals)) if i not in good]
        pick = rng.choice(other) if other and rng.random() < a.p_wrong else rng.choice(good)
        view = s.view()
        print(f"\n=== scene {done + 1}: {view.narration}")
        if how == "says":
            s.command(pick)
        obs = observation(s.state.current)
        ex = expectations(s, obs)
        print("  options (child's expectation: solved/partial/nothing/worse)")
        for i, o in enumerate(view.options):
            mark = "  <- parent says" if how == "says" and i == pick else ""
            print(f"    {i}. {o.text:42s} " + " ".join(f"{x:.2f}" for x in ex[i]) + mark)
        if how == "shows":
            r = s.show(pick)
            print(f"  parent shows {pick}: {r.narration}  -> {r.outcome}")
        else:
            r = s.try_child()
            print(f"  child does {r.option}: {r.narration}  -> {r.outcome}  (picked with p={r.probs[r.option]:.2f})")
        f = s.buffer[-1].facts[0]
        print(f"  learned: [{f.option.verb}{' ' + f.option.target.shape + '/' + f.option.target.material if f.option.target else ''}]"
              f"{' (parent said so)' if f.said else ''} -> {f.result};  loss {r.learn.loss_before:.2f} -> {r.learn.loss_after:.2f}")
        done += 1


if __name__ == "__main__":
    main()
