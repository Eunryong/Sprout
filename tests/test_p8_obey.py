"""Variants of the commanding parent and the -1 rules experiment world (docs/WORLD.md 21)."""
import json
import os
import random
import subprocess
import sys
from pathlib import Path

from server.game.parents import P_COMMAND_WRONG, pick_instruction
from server.world.ontology import TRAPS
from server.world.parenting import quality_buckets
from server.world.situation import values

from helpers import thirsty

ROOT = Path(__file__).resolve().parents[1]


def tell(s, n, **kw):
    rng = random.Random(0)
    v, verbs = values(s), [o.verb for o in s.options]
    return [pick_instruction(v, verbs, rng, **kw) for _ in range(n)], quality_buckets(v, verbs)


def test_a_parent_who_never_errs_always_tells_the_best_option():
    told, b = tell(thirsty(), 200, p_wrong=0.0)
    assert set(told) == set(b["good"])


def test_the_share_of_wrong_instructions_follows_p_wrong():
    for p in (0.25, 0.5, 0.75):
        told, b = tell(thirsty(), 4000, p_wrong=p)
        assert abs(sum(i not in b["good"] for i in told) / len(told) - p) < 0.03


def test_the_default_error_rate_is_the_one_of_the_first_experiments():
    assert P_COMMAND_WRONG == 0.25


def test_wrong_kind_picks_the_mistake_from_that_bucket_only():
    s = thirsty()  # 0 good | 1, 2 useless | 3 WAIT, harmful
    useless, b = tell(s, 300, p_wrong=1.0, wrong_kind="useless")
    harmful, _ = tell(s, 300, p_wrong=1.0, wrong_kind="harmful")
    anything, _ = tell(s, 600, p_wrong=1.0, wrong_kind="any")
    assert set(useless) == set(b["useless"]) and set(harmful) == set(b["harmful"])
    assert set(anything) == {1, 2, 3}


def test_an_empty_bucket_falls_back_to_any_wrong_option():
    s = thirsty()
    v, verbs = values(s), [o.verb for o in s.options]
    rng = random.Random(1)
    told = {pick_instruction(v[:3], verbs[:3], rng, 1.0, "harmful") for _ in range(100)}  # no WAIT on the table: nothing harmful
    assert told == {1, 2}


def test_the_shipped_world_has_no_traps_and_so_its_scenes_are_unchanged():
    assert all(not v for v in TRAPS.values())  # the trap branch is off: no extra random draws in the generator


def run_in_harm_world(code: str) -> str:
    env = {**os.environ, "GROWING_ONTOLOGY": str(ROOT / "config" / "experiments" / "onto_harm.yaml"), "PYTHONPATH": str(ROOT)}
    r = subprocess.run([sys.executable, "-c", code], env=env, capture_output=True, text=True, cwd=ROOT, timeout=120)
    assert r.returncode == 0, r.stderr
    return r.stdout.strip()


def test_negative_rules_make_an_active_option_harmful_in_the_experiment_world():
    out = run_in_harm_world("""
import json, random
from server.world.ontology import TRAPS
from server.world.generator import make_situation
from server.world.parenting import quality_buckets
from server.world.situation import Obj, option_value, values
cup = Obj(kind="cup", color="red", shape="cylinder", size="big", material="ceramic")
rng = random.Random(3)
active = 0
for i in range(200):
    s = make_situation(rng, [rng.choice(["THIRSTY", "FLOOR_WET", "DARK", "COLD", "VISITOR"])])
    v = values(s)
    active += any(x < 0 and o.verb != "WAIT" for o, x in zip(s.options, v))
    assert max(v) == 2 and min(v) <= max(v) - 1
print(json.dumps({"cover_cup": option_value(["THIRSTY"], None, "COVER", cup),
                  "cup_for_both": option_value(["THIRSTY", "DARK"], None, "COVER", cup),
                  "traps": TRAPS["DARK"], "active_share": active / 200}))
""")
    r = json.loads(out)
    assert r["cover_cup"] == -1 and r["cup_for_both"] == 0 and r["traps"] == ["COVER"]  # two problems: the better of the two applies
    assert r["active_share"] > 0.3


def test_an_active_harmful_option_is_a_worse_outcome_and_a_harmful_instruction():
    out = run_in_harm_world("""
import json, random
from server.world.generator import make_situation
from server.world.parenting import instruction_quality
from server.world.situation import outcome_of, values
rng = random.Random(5)
rows = []
for i in range(100):
    s = make_situation(rng, ["DARK"])
    for k, (o, x) in enumerate(zip(s.options, values(s))):
        if x < 0 and o.verb != "WAIT":
            rows.append([o.verb, outcome_of(s, k)[1], instruction_quality(s, k)])
print(json.dumps(rows))
""")
    rows = json.loads(out)
    assert rows and all(r == ["COVER", "worse", "harmful"] for r in rows)
