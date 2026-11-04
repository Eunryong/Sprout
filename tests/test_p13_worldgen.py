"""Generated worlds: relations picked from authored pools, checked by the engine side (docs/WORLD.md 22.13)."""
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from server.config import GameCfg
from server.game.parents import play_day
from server.game.session import Session
from server.learning.learner import OnlineCfg
from server.world.events import RuleError
from server.world.ontology import RULES
from server.world.worldgen import (
    check_sheet, game_overrides, load_pools, read_sheet, sheet_hash, smoke, template_world, write_world,
)

import random

POOLS = load_pools()


def world(seed=1):
    return template_world(seed, POOLS)


def broken(seed=1, **change):
    return world(seed).model_copy(update=change)


def test_template_worlds_are_valid_and_the_same_seed_gives_the_same_world():
    hashes = set()
    for seed in range(30):
        w = world(seed)
        assert check_sheet(w, POOLS) == []
        assert world(seed) == w
        hashes.add(sheet_hash(w))
    assert len(hashes) >= 25  # worlds really differ


def test_every_problem_has_a_solving_relation_and_no_zero_rows():
    for seed in range(10):
        for rows in world(seed).rules.values():
            assert any(r[2] == 2 for r in rows) and all(r[2] != 0 for r in rows)


def test_a_relation_outside_the_pool_is_rejected():
    w = world()
    rules = {**w.rules, "THIRSTY": w.rules["THIRSTY"] + [["WIPE", "ball", 1]]}
    assert any("not a plausible relation" in e for e in check_sheet(broken(rules=rules), POOLS))


def test_a_value_the_pool_does_not_allow_is_rejected():
    w = world()
    rules = {**w.rules, "THIRSTY": [["DRINK", "cup", 2], ["COVER", "cup", 2]]}
    assert any("cannot be 2" in e for e in check_sheet(broken(rules=rules), POOLS))


def test_a_problem_nothing_solves_is_rejected():
    w = world()
    rules = {**w.rules, "DARK": [["GIVE", "flashlight", 1]]}
    assert any("DARK: nothing solves it" in e for e in check_sheet(broken(rules=rules), POOLS))


def test_a_word_that_could_never_be_introduced_is_rejected():
    w = world()
    rules = {t: [r for r in rows if r[1] != "paper" or r[2] < 1] for t, rows in w.rules.items()}  # paper helps nowhere
    req = ["paper"] + [k for k in w.request_kinds if k != "paper"][:3]
    errs = check_sheet(broken(rules=rules, request_kinds=req), POOLS)
    assert any("word for paper could never be introduced" in e for e in errs)


def test_two_problems_must_be_open_on_day_one():
    w = world()
    opens = {t: max(POOLS["problems"][t]["opens"]) for t in w.opens}  # only THIRSTY can come up at 12 months
    assert sum(m == 12 for m in opens.values()) == 1
    assert any("two problems must be open" in e for e in check_sheet(broken(opens=opens), POOLS))


def test_a_follow_up_cannot_come_before_the_problem_it_follows():
    pools = {**POOLS, "problems": {**POOLS["problems"], "VISITOR": {**POOLS["problems"]["VISITOR"], "opens": [12, 24]}}}
    w = template_world(1, pools)
    opens = {**w.opens, "FLOOR_WET": 24}
    assert not any("before it can come up" in e for e in check_sheet(w.model_copy(update={"escalation": {"FLOOR_WET": "VISITOR"}, "opens": {**opens, "VISITOR": 24}}), pools))
    bad = check_sheet(w.model_copy(update={"escalation": {"FLOOR_WET": "VISITOR"}, "opens": {**opens, "VISITOR": 12}}), pools)
    assert any("VISITOR would follow FLOOR_WET before it can come up" in e for e in bad)


def test_what_the_world_teaches_stays_in_range_and_own_burns_stay_off():
    w = world()
    assert any("outside" in e for e in check_sheet(broken(learning={**w.learning, "burn_worse_rounds": 9}), POOLS))
    assert any("outside" in e for e in check_sheet(broken(learning={**w.learning, "own_burn_worse_rounds": 1}), POOLS))
    assert any("unknown setting" in e for e in check_sheet(broken(learning={**w.learning, "teach_everything": 1}), POOLS))


def test_too_many_harmful_relations_are_rejected():
    pools = {**POOLS, "max_traps_per_world": 0}
    w = next(w for w in (template_world(s, POOLS) for s in range(40)) if any(r[2] < 0 for rows in w.rules.values() for r in rows))
    assert any("harmful relations" in e for e in check_sheet(w, pools))


def test_an_invalid_sheet_is_not_written(tmp_path):
    w = world()
    with pytest.raises(ValueError):
        write_world(broken(rules={**w.rules, "DARK": []}), tmp_path / "w", POOLS)
    assert not (tmp_path / "w" / "ontology.yaml").exists()


def test_a_written_world_loads_as_an_ontology_and_makes_scenes(tmp_path):
    for seed in (1, 2, 3, 4):
        onto, js = write_world(world(seed), tmp_path / f"w{seed}", POOLS)
        assert smoke(onto) == []
        assert json.loads(js.read_text())["hash"] == sheet_hash(world(seed))


def test_the_world_sets_what_it_teaches_and_a_hand_written_world_sets_nothing(tmp_path):
    onto, _ = write_world(world(3), tmp_path / "w", POOLS)
    over = game_overrides(onto)
    assert over["burn_worse_rounds"] == world(3).learning["burn_worse_rounds"] and over["days"] == 8
    assert game_overrides(Path("config/ontology.yaml")) == {} and read_sheet(Path("config/ontology.yaml")) is None
    assert set(over) <= set(GameCfg.__dataclass_fields__)


def small_session(tmp_path, **game):
    return Session.new(tmp_path / "s", seed=0, game=GameCfg(per_day=2, quiz_per_topic=3, **game), ocfg=OnlineCfg(steps=4))


def test_the_rules_are_not_shown_during_the_game(tmp_path):
    s = small_session(tmp_path, days=1)
    with pytest.raises(RuleError) as e:
        s.reveal()
    assert e.value.code == "not_over"


def test_after_the_last_day_the_rules_are_shown_with_how_well_the_child_learned_them(tmp_path):
    s = small_session(tmp_path, days=1)
    play_day(s, "balanced", 1, random.Random(0))
    s.end_day()
    r = s.reveal()
    assert len(r["rules"]) == sum(len(rows) for rows in RULES.values())
    assert {x["problem"] for x in r["rules"]} == set(RULES)
    for x in r["rules"]:
        assert x["p_pick"] is None or 0 <= x["p_pick"] <= 1
        assert x["scenes"] == 0 or x["chance"] > 0
    assert any(x["scenes"] > 0 for x in r["rules"]) and (tmp_path / "s" / "ontology.yaml").exists()


def test_a_whole_day_on_a_generated_world_ends_in_its_reveal(tmp_path):
    onto, _ = write_world(world(5), tmp_path / "w", POOLS)
    code = f"""
import random, json
from server.config import GameCfg
from server.game.parents import play_day
from server.game.session import Session
from server.learning.learner import OnlineCfg
from dataclasses import replace
s = Session.new({str(tmp_path / 's')!r}, seed=0, game=None, ocfg=OnlineCfg(steps=4))
s.game = replace(s.game, per_day=2, days=1)
play_day(s, 'balanced', 1, random.Random(0))
s.end_day()
r = s.reveal()
print(json.dumps({{"world": r["world"], "burn": s.game.burn_worse_rounds, "n": len(r["rules"])}}))
"""
    env = {**os.environ, "GROWING_ONTOLOGY": str(onto), "GROWING_LLM": "mock"}
    out = subprocess.run([sys.executable, "-c", code], env=env, capture_output=True, text=True, cwd=Path(__file__).resolve().parents[1])
    assert out.returncode == 0, out.stderr[-600:]
    got = json.loads(out.stdout.strip().splitlines()[-1])
    assert got["world"] == sheet_hash(world(5)) and got["burn"] == world(5).learning["burn_worse_rounds"] and got["n"] > 8
