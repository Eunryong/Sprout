"""How much does one exposure leave, and what does a stronger signal add? (docs/WORLD.md 22.10)

  uv run python scripts/exp_oneshot.py > scripts/out/oneshot.txt

A NEW problem taught once / twice / three times (separate learning rounds of 60 steps), or once with 120/180 steps.
p = probability of the taught option on the same scene, on other scenes of that problem, and after 10 lessons on OTHER problems."""
import copy
import os
import random
import statistics as st
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.environ.setdefault("GROWING_LLM", "mock")
os.environ["GROWING_ONTOLOGY"] = str(ROOT / "config" / "experiments" / "onto_age03.yaml")
os.environ["GROWING_NO_SCHEDULE"] = "1"

CONDS = {"1 round": (1, 60), "2 rounds": (2, 60), "3 rounds": (3, 60), "1 x 120 steps": (1, 120), "1 x 180 steps": (1, 180)}


def one(seed):
    import torch
    torch.set_num_threads(1)
    from server.config import load_configs
    from server.learning.codec import Example, option_probs
    from server.learning.learner import Learner
    from server.learning.model import build_model
    from server.world.generator import make_situation
    from server.world.ontology import PROBLEMS
    from server.world.situation import observation, values
    _, mcfg, ocfg = load_configs()
    rng = random.Random(seed)
    torch.manual_seed(seed)
    tags = PROBLEMS[:]
    rng.shuffle(tags)
    known, new = tags[:4], tags[4:]

    def lesson(tag, i):
        s = make_situation(rng, [tag])
        v = values(s)
        return Example(id=f"e{i}", day=1, source="show", obs=observation(s), chosen=max(range(len(v)), key=v.__getitem__))

    res = {}
    for prior in (0, 30):
        base = build_model(mcfg)
        buf, L = [], Learner(base, ocfg)
        for i in range(prior):
            ex = lesson(rng.choice(known), i)
            buf.append(ex)
            L.learn(ex, buf, rng)
        target_tag = new[0]
        target = lesson(target_tag, 900)
        others = [lesson(target_tag, 910 + i) for i in range(6)]
        extra = [lesson(rng.choice(known), 950 + i) for i in range(10)]
        for name, (rounds, steps) in CONDS.items():
            m = copy.deepcopy(base)
            Lm = Learner(m, ocfg)
            Lm.opt.load_state_dict(copy.deepcopy(L.opt.state_dict()))
            b = list(buf) + [target]
            p = lambda ex: option_probs(m, ex.obs)[ex.chosen]
            before = p(target)
            for r in range(rounds):
                Lm.learn(target, b, random.Random(seed + r), steps=steps)
            same, near = p(target), st.mean(p(o) for o in others)
            for e in extra:
                b.append(e)
                Lm.learn(e, b, random.Random(seed + 77), steps=60)
            res[(prior, name)] = (before, same, near, p(target))
    return res


if __name__ == "__main__":
    seeds = list(range(300, 340))
    with ProcessPoolExecutor(12) as ex:
        res = list(ex.map(one, seeds))
    for prior in (0, 30):
        print(f"\n== child with {prior} earlier lessons on OTHER problems; a new problem taught; mean probability of the taught option (chance ~0.25), {len(seeds)} seeds ==")
        print(f"{'':16s}{'before':>8s}{'same scene':>12s}{'other scenes':>14s}{'after 10 others':>17s}")
        for name in CONDS:
            c = [st.mean(r[(prior, name)][k] for r in res) for k in range(4)]
            print(f"{name:16s}{c[0]:8.2f}{c[1]:12.2f}{c[2]:14.2f}{c[3]:17.2f}")
