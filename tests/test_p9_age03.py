"""The 0-3 years experiment world (config/experiments/onto_age03.yaml, docs/WORLD.md section 22)."""
import json
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def in_age03_world(code: str) -> dict:
    env = {**os.environ, "GROWING_ONTOLOGY": str(ROOT / "config" / "experiments" / "onto_age03.yaml"),
           "GROWING_LLM": "mock", "PYTHONPATH": str(ROOT)}
    r = subprocess.run([sys.executable, "-c", code], env=env, capture_output=True, text=True, cwd=ROOT, timeout=180)
    assert r.returncode == 0, r.stderr
    return json.loads(r.stdout.strip().splitlines()[-1])


def test_the_age03_world_loads_and_the_glossary_covers_it():
    r = in_age03_world("""
import json
from server.agents import glossary
glossary.check_complete()
from server.config import GameCfg
from server.world.ontology import PROBLEMS, REQUEST_KINDS
print(json.dumps({"problems": PROBLEMS, "kinds_vs_words": len(REQUEST_KINDS) <= len(GameCfg().words)}))
""")
    assert r["kinds_vs_words"] and len(r["problems"]) == 8


def test_every_problem_makes_valid_scenes_with_a_solving_option_and_a_wrong_one():
    r = in_age03_world("""
import json, random
from server.world.generator import make_situation
from server.world.ontology import PROBLEMS
from server.world.situation import values
rng = random.Random(1)
bad = []
for t in PROBLEMS + ["REQUEST"]:
    for i in range(60):
        s = make_situation(rng, [t], "x", "house", "bobo", "NAME_00", "cup") if t == "REQUEST" else make_situation(rng, [t])
        v = values(s)
        if not (max(v) == 2 and min(v) < 2 and 3 <= len(s.options) <= 5):
            bad.append(t)
print(json.dumps({"bad": bad}))
""")
    assert r["bad"] == []


def test_the_new_things_are_not_told_apart_by_one_attribute():
    r = in_age03_world("""
import json
from server.world.ontology import KINDS
by_shape, by_mat = {}, {}
for k, s in KINDS.items():
    by_shape.setdefault(s.shape, []).append(k)
    by_mat.setdefault(s.material, []).append(k)
print(json.dumps({"shared_shape": sorted(v for v in by_shape.values() if len(v) > 1), "shared_mat": sorted(v for v in by_mat.values() if len(v) > 1)}))
""")
    assert ["cloth", "paper", "blanket"] in r["shared_shape"] and ["flashlight", "spoon"] in r["shared_mat"]


def test_the_schedule_opens_problems_in_three_bursts_and_ages_the_child_three_months_a_day():
    r = in_age03_world("""
import json
from server.world.ontology import age_months, open_problems, opened_on, PROBLEMS
print(json.dumps({"age": [age_months(d) for d in range(1, 9)],
                  "open": {d: len(open_problems(d)) for d in range(1, 9)},
                  "opened": {t: opened_on(t) for t in PROBLEMS}}))
""")
    assert r["age"] == [12, 15, 18, 21, 24, 27, 30, 33]
    assert [r["open"][str(d)] for d in range(1, 9)] == [3, 3, 5, 5, 8, 8, 8, 8]
    assert r["opened"] == {"THIRSTY": 1, "HUNGRY": 1, "UPSET": 1, "FLOOR_WET": 3, "MESSY": 3, "DARK": 5, "COLD": 5, "VISITOR": 5}


def test_the_engine_refuses_a_card_for_a_problem_that_is_not_open_yet():
    r = in_age03_world("""
import json, random, tempfile
from pathlib import Path
from server.game.session import Session
from server.world.engine import RuleError
from server.world.events import OfferCards
from server.world.generator import make_situation
rng = random.Random(0)
s = Session.new(Path(tempfile.mkdtemp()) / "s", seed=1)
bad = [make_situation(rng, ["DARK"], "sit_x_0"), make_situation(rng, ["THIRSTY"], "sit_x_1")]
try:
    s._submit(OfferCards(action_id="a", situations=bad))
    out = "accepted"
except RuleError as e:
    out = str(e)
print(json.dumps({"out": out}))
""")
    assert "cannot come up yet" in r["out"]


def test_the_template_director_only_offers_open_problems_and_eases_new_ones_in():
    r = in_age03_world("""
import json, random
from server.agents.director import template_proposal
from server.agents.schemas import DirectorContext
from server.world.ontology import age_months, open_problems
seen = {1: set(), 3: set(), 5: set()}
first_day_new = 0
for day in (1, 3, 5):
    for seed in range(60):
        ctx = DirectorContext(day=day, open_issues=[], nameable=[], requests_left=0, unlocked=["house"], start_index=0,
                              age_months=age_months(day), open_tags=open_problems(day))
        prop = template_proposal(ctx, random.Random(seed), set(), {})
        for c in prop.cards:
            seen[day] |= set(c.tags)
            if day == 3 and set(c.tags) & {"FLOOR_WET", "MESSY"}:
                first_day_new += 1
print(json.dumps({"seen": {d: sorted(v) for d, v in seen.items()}, "first_day_new": first_day_new}))
""")
    assert set(r["seen"]["1"]) <= {"THIRSTY", "HUNGRY", "UPSET"} and set(r["seen"]["3"]) <= {"THIRSTY", "HUNGRY", "UPSET", "FLOOR_WET", "MESSY"}
    assert "VISITOR" not in r["seen"]["3"] and "DARK" in r["seen"]["5"]
    assert 0 < r["first_day_new"] < 120  # eased in: fewer than every slot of the opening day


def test_an_escalation_to_a_problem_that_is_not_open_is_not_a_legal_follow_up():
    r = in_age03_world("""
import json
from server.world.state import allowed_followup_tags
print(json.dumps({"day3": allowed_followup_tags("FLOOR_WET", "worse", 3), "day5": allowed_followup_tags("FLOOR_WET", "worse", 5)}))
""")
    assert r["day3"] == [["FLOOR_WET"]] and r["day5"] == [["FLOOR_WET"], ["FLOOR_WET", "VISITOR"]]


def test_without_a_schedule_everything_is_open_from_day_one():
    out = subprocess.run([sys.executable, "-c", "from server.world.ontology import open_problems, PROBLEMS, age_months\n"
                          "print(open_problems(1) == PROBLEMS, age_months(1), age_months(9))"],
                         capture_output=True, text=True, cwd=ROOT, env={**os.environ, "PYTHONPATH": str(ROOT)}, timeout=60)
    assert out.stdout.strip() == "True 0 0", out.stderr
