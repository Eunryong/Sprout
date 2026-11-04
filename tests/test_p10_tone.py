"""A firm tone repeats a lesson (docs/WORLD.md 22.8)."""
import random

import pytest

from server.game.session import Session
from server.learning.codec import Example, data_hash
from server.learning.learner import OnlineCfg
from server.world.events import RuleError
from server.world.parenting import journal
from server.world.situation import observation

from helpers import thirsty


def new(tmp_path, name="s", **kw):
    return Session.new(tmp_path / name, seed=0, ocfg=OnlineCfg(steps=6), **kw)


def to_situation(s):
    s.draw_cards()
    s.pick_card(0)


def test_a_lesson_is_said_once_unless_the_tone_is_firm():
    ex = Example(id="a", day=1, source="show", obs=observation(thirsty()), chosen=0)
    assert ex.repeat == 1
    assert data_hash([ex]) == data_hash([ex.model_copy(update={"repeat": 1})])
    assert data_hash([ex]) != data_hash([ex.model_copy(update={"repeat": 3})])  # a different lesson is a different dataset


def test_a_firm_show_learns_the_lesson_repeatedly_and_a_gentle_one_once(tmp_path):
    reps = {}
    for tone in ("gentle", "firm"):
        s = new(tmp_path, tone)
        to_situation(s)
        r = s.show(s.debug_best()[0], tone=tone)
        reps[tone] = (s.buffer[-1].repeat, r.learn.batch_size)
    assert reps["gentle"][0] == 1 and reps["firm"][0] == 3
    assert reps["firm"][1] - reps["gentle"][1] == 2  # the batch holds three copies of the lesson instead of one


def test_the_number_of_repeats_is_a_game_setting(tmp_path):
    from server.config import GameCfg
    s = new(tmp_path, game=GameCfg(firm_repeat=2))
    to_situation(s)
    s.show(s.debug_best()[0], tone="firm")
    assert s.buffer[-1].repeat == 2


def test_the_tone_is_a_recorded_part_of_the_teaching_move_and_the_journal_counts_it(tmp_path):
    s = new(tmp_path)
    to_situation(s)
    s.show(s.debug_best()[0], tone="firm")
    shown = [e for e in s.engine.log if e.type.value == "shown"]
    assert all(e.payload["tone"] == "firm" for e in shown)
    assert journal(s.engine.log)[1]["firm"] == len(shown)


def test_an_unknown_tone_is_refused_before_anything_is_saved(tmp_path):
    s = new(tmp_path)
    to_situation(s)
    n = len(s.engine.log)
    with pytest.raises(RuleError):
        s.show(0, tone="shouting")
    assert len(s.engine.log) == n and len(s.buffer) == 0


def test_the_repeat_survives_saving_and_loading(tmp_path):
    s = new(tmp_path)
    to_situation(s)
    s.show(s.debug_best()[0], tone="firm")
    again = Session.load(tmp_path / "s")
    assert [e.repeat for e in again.buffer] == [3] and not again.warnings


def test_in_weight_mode_a_firm_lesson_is_heavier_not_repeated(tmp_path):
    from server.config import GameCfg
    s = new(tmp_path, game=GameCfg(firm_mode="weight", firm_repeat=3))
    to_situation(s)
    s.show(s.debug_best()[0], tone="firm")
    ex = s.buffer[-1]
    assert ex.weight == 3.0 and ex.repeat == 1


# ---- the child hears the tone (GROWING_TONE_INPUT=1, docs/WORLD.md 22.9) ----
import json
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def run_code(code: str, hear: bool) -> dict:
    env = {**os.environ, "PYTHONPATH": str(ROOT), "GROWING_LLM": "mock"}
    env.pop("GROWING_TONE_INPUT", None)
    if hear:
        env["GROWING_TONE_INPUT"] = "1"
    r = subprocess.run([sys.executable, "-c", code], env=env, capture_output=True, text=True, cwd=ROOT, timeout=120)
    assert r.returncode == 0, r.stderr
    return json.loads(r.stdout.strip().splitlines()[-1])


ENCODE = """
import json, sys
sys.path.insert(0, "tests")
from helpers import thirsty
from server.learning import codec
from server.world.situation import observation
s = thirsty()
s.command, s.command_tone = 0, "firm"
obs = observation(s)
ids = codec.encode(obs)
plain = codec.encode(obs.model_copy(update={"tone": None}))
print(json.dumps({"tone": obs.tone, "len_diff": len(ids) - len(plain), "same_ids": ids == plain,
                  "tokens": [codec.VOCAB[i] for i in ids if codec.VOCAB[i].startswith("TONE")], "max_len": codec.MAX_LEN, "vocab": len(codec.VOCAB)}))
"""


def test_the_tone_token_is_in_the_input_only_when_the_game_is_run_with_it():
    off, on = run_code(ENCODE, False), run_code(ENCODE, True)
    assert off["tone"] == "firm" and off["same_ids"] and off["tokens"] == []  # said firmly, but not heard
    assert on["len_diff"] == 1 and on["tokens"] == ["TONE_FIRM"] and on["max_len"] == off["max_len"] + 1 and on["vocab"] == off["vocab"] + 2


def test_a_command_carries_its_tone_to_the_childs_scene_and_a_plain_show_is_learned_in_its_tone():
    r = run_code("""
import json, tempfile
from pathlib import Path
from server.game.session import Session
from server.learning.learner import OnlineCfg
s = Session.new(Path(tempfile.mkdtemp()) / "s", seed=0, ocfg=OnlineCfg(steps=4))
s.draw_cards(); s.pick_card(0)
s.command(s.debug_best()[0], tone="firm")
told = [e.payload["tone"] for e in s.engine.log if e.type.value == "commanded"]
s.show(s.debug_best()[0], tone="firm")
s.draw_cards(); s.pick_card(0)
s.show(s.debug_best()[0], tone="gentle")
print(json.dumps({"told": told, "obs": [e.obs.tone for e in s.buffer], "cmd": [e.obs.cmd is not None for e in s.buffer]}))
""", True)
    assert r["told"] == ["firm"] and r["obs"] == ["firm", "gentle"] and r["cmd"] == [True, False]


def test_the_tone_is_not_part_of_the_scene_identity_so_the_quiz_stays_isolated():
    from server.world.situation import obs_hash
    obs = observation(thirsty())
    assert obs_hash(obs) == obs_hash(obs.model_copy(update={"tone": "firm"}))


def test_a_firm_parent_policy_is_a_habit_of_tone_not_of_knowledge(tmp_path):
    from server.game.parents import _tone
    s = new(tmp_path)
    assert _tone(s, "firm", True) == "firm" and _tone(s, "gentle", False) == "gentle"
    assert _tone(s, "wrong", False) == "firm" and _tone(s, "wrong", True) == "gentle"
    assert _tone(s, "right", True) == "firm" and _tone(s, "right", False) == "gentle"
    assert {_tone(s, "half", True) for _ in range(1)} <= {"firm", "gentle"}
