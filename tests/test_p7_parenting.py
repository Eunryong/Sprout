"""Talk / Command events, the attention budget and the parenting metrics on the real engine path (docs/WORLD.md 18-20)."""
import json
import random
import sys
from pathlib import Path

import pytest

from server.game.parents import STYLES, play_day
from server.game.session import Session
from server.learning.codec import encode, TOK
from server.learning.evaluator import command_panel
from server.learning.learner import OnlineCfg
from server.world.engine import Engine
from server.world.events import Command, EndDay, Event, EventType, PickCard, RecordQuiz, RuleError, Show, Talk, Try
from server.world.generator import make_situation
from server.world.parenting import classify_response, instruction_quality, journal, quality_buckets
from server.world.situation import observation, optimal, values
from server.world.state import Phase, fold

from helpers import cold, dark, request, thirsty, visitor
from test_engine import best, start, wrong

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import exp_ea  # noqa: E402


def talk_sit(sid="k1", kind="cup", word="bobo", token="NAME_00", seed=0):
    return make_situation(random.Random(seed), ["REQUEST"], sid, "house", word, token, kind)


def talk(e, aid="t1", **kw):
    sit = talk_sit(**kw)
    return e.submit(Talk(action_id=aid, situation=sit, option=optimal(sit)[0]))


def new(tmp_path, seed=0, name="s"):
    return Session.new(tmp_path / name, seed=seed, ocfg=OnlineCfg(steps=12))


# ---- quality and response are computed from the rules table ----
def test_quality_buckets_and_instruction_quality_follow_the_rules_table():
    s = thirsty()  # 0 DRINK cup = 2 | 1 DRINK spoon = 0 | 2 WIPE cup = 0 | 3 WAIT = -1
    b = quality_buckets(values(s), [o.verb for o in s.options])
    assert b["good"] == [0] and b["harmful"] == [3] and set(b["useless"]) == {1, 2}
    assert [instruction_quality(s, i) for i in range(4)] == ["good", "useless", "useless", "harmful"]


def test_response_classes():
    s = thirsty()
    assert classify_response(s, 0, 0) == "FOLLOW"
    assert classify_response(s, 1, 0) == "SELF_BEST"  # told something useless, did the best thing
    assert classify_response(s, 0, 3) == "WAIT"
    assert classify_response(s, 0, 1) == "OTHER"


# ---- Command ----
def test_command_is_recorded_with_its_quality_and_reaches_the_child_without_it():
    e = Engine()
    start(e)
    w = wrong(e)
    ev = e.submit(Command(action_id="c1", option=w))[0]
    assert ev.type == EventType.COMMANDED and ev.payload["quality"] in ("useless", "harmful", "near", "idle")
    assert e.state.current.command == w
    obs = observation(e.state.current)
    assert obs.cmd == obs.options[w]
    assert set(obs.cmd.model_dump()) == {"verb", "target"}  # no quality, no value, no reaction
    assert "quality" not in json.dumps(obs.model_dump())
    assert TOK["CMD"] in encode(obs)


def test_command_rules():
    e = Engine()
    with pytest.raises(RuleError, match="wrong_phase"):
        e.submit(Command(action_id="c0", option=0))
    start(e)
    with pytest.raises(RuleError, match="bad_option"):
        e.submit(Command(action_id="c1", option=99))
    e.submit(Command(action_id="c2", option=0))
    with pytest.raises(RuleError, match="already_commanded"):
        e.submit(Command(action_id="c3", option=1))
    assert e.submit(Command(action_id="c2", option=1)) == []  # idempotent re-send
    e.submit(Try(action_id="t", option=0))
    e2 = Engine()
    start(e2)
    e2.submit(Try(action_id="t", option=0))
    with pytest.raises(RuleError, match="wrong_phase"):
        e2.submit(Command(action_id="c", option=1))  # told after the child already chose and the scene is over


def test_try_after_command_classifies_the_response_and_a_plain_try_does_not():
    e = Engine()
    start(e)
    b = best(e)
    w = wrong(e)
    e.submit(Command(action_id="c1", option=w))
    ev = e.submit(Try(action_id="t1", option=b))[0]
    assert ev.payload["response"] == "SELF_BEST" and ev.payload["command"] == w
    e = Engine()
    start(e)
    assert "response" not in e.submit(Try(action_id="t1", option=best(e)))[0].payload


def test_command_does_not_cost_attention_and_the_situation_does():
    e = Engine(per_day=3)
    for i in range(3):
        start(e, aid=f"o{i}")
        e.submit(Command(action_id=f"c{i}", option=best(e)))
        assert e.state.spent == i
        e.submit(Show(action_id=f"s{i}", option=best(e)))
        assert e.state.spent == i + 1
    assert e.state.phase == Phase.EVENING


# ---- Talk ----
def test_talk_registers_the_word_teaches_and_spends_attention():
    e = Engine(per_day=8)
    ev = talk(e)[0]
    s = e.state
    assert ev.type == EventType.TALKED and ev.payload["was_right"] and ev.payload["outcome"] == "solved"
    assert s.words == {"bobo": "NAME_00"} and s.word_kinds == {"bobo": "cup"} and "cup" in s.nameable
    assert s.spent == 1 and s.situations_done == 1 and s.phase == Phase.CARDS and s.current is None
    assert s.topic_counts == {"REQUEST:cup": 1} and s.requests_today == 0  # the parent's own choice, not Grandma's request
    assert s.verb_counts == {"FETCH": 1}


def test_talk_is_limited_by_attention_not_by_the_daily_request_limit():
    e = Engine(per_day=3)
    for i, (k, w, t) in enumerate([("cup", "bobo", "NAME_00"), ("cloth", "mimi", "NAME_01"), ("paper", "tata", "NAME_02")]):
        talk(e, f"t{i}", sid=f"k{i}", kind=k, word=w, token=t, seed=i)
    assert e.state.phase == Phase.EVENING and e.state.spent == 3
    with pytest.raises(RuleError, match="wrong_phase"):
        talk(e, "t9", sid="k9", kind="ball", word="lulu", token="NAME_03")


def test_talk_cost_greater_than_what_is_left_is_refused():
    e = Engine(per_day=3, talk_cost=2)
    talk(e)
    assert e.state.spent == 2
    with pytest.raises(RuleError, match="no_attention"):
        talk(e, "t2", sid="k2", kind="cloth", word="mimi", token="NAME_01")
    start(e)  # one unit is still enough for an ordinary situation
    e.submit(Show(action_id="s", option=best(e)))
    assert e.state.phase == Phase.EVENING


def test_talk_keeps_unused_cards_until_the_day_is_full():
    e = Engine(per_day=2)
    start(e, aid="o1")
    e.submit(Show(action_id="s1", option=best(e)))
    from test_engine import offer
    offer(e, thirsty("a2"), visitor("v2"), "o2")
    talk(e)
    assert e.state.phase == Phase.EVENING and e.state.cards is None  # nothing carries over into the night


@pytest.mark.parametrize("bad", ["not a request", "reused id", "word clash", "token clash", "bad option", "not unlocked"])
def test_talk_validation(bad):
    e = Engine()
    talk(e)
    sit = talk_sit("k2", "cloth", "mimi", "NAME_01", seed=1)
    opt = optimal(sit)[0]
    if bad == "not a request":
        sit = thirsty("t9")
    elif bad == "reused id":
        sit = talk_sit("k1", "cloth", "mimi", "NAME_01", seed=1)
    elif bad == "word clash":
        sit = talk_sit("k2", "cloth", "bobo", "NAME_00", seed=1)  # bobo already means cup
    elif bad == "token clash":
        sit = talk_sit("k2", "cloth", "mimi", "NAME_00", seed=1)
    elif bad == "bad option":
        opt = 99
    elif bad == "not unlocked":
        sit = sit.model_copy(update={"location": "yard"})
    with pytest.raises(RuleError):
        e.submit(Talk(action_id="t2", situation=sit, option=opt))


def test_talk_cannot_collide_with_the_held_out_quiz():
    sit = talk_sit()
    from server.world.situation import obs_hash
    e = Engine(forbidden={obs_hash(observation(sit))})
    with pytest.raises(RuleError, match="held-out"):
        e.submit(Talk(action_id="t", situation=sit, option=optimal(sit)[0]))


def test_talk_inside_a_situation_is_refused():
    e = Engine()
    start(e)
    with pytest.raises(RuleError, match="wrong_phase"):
        talk(e)


def test_events_replay_to_the_same_state_and_old_logs_still_fold():
    e = Engine(per_day=5)
    talk(e)
    start(e, aid="o1")
    e.submit(Command(action_id="c1", option=wrong(e)))
    e.submit(Show(action_id="s1", option=wrong(e)))
    assert fold(e.log) == e.state
    old = [Event(**{**ev.model_dump(), "payload": {k: v for k, v in ev.payload.items() if k != "talk_cost"}}) for ev in e.log[:1]]
    assert fold(old).talk_cost == 1  # a log written before the budget existed


def test_default_flow_without_talk_spends_one_unit_per_situation_exactly_as_before():
    e = Engine(per_day=4)
    for i in range(4):
        start(e, aid=f"o{i}")
        e.submit(Show(action_id=f"s{i}", option=best(e)))
        assert e.state.spent == e.state.situations_done == i + 1
    assert e.state.phase == Phase.EVENING
    e.submit(RecordQuiz(action_id="q", result={}))
    e.submit(EndDay(action_id="d"))
    assert e.state.spent == 0 and e.state.day == 2


# ---- Session ----
def test_session_talk_teaches_the_word_with_only_the_name_token(tmp_path):
    s = new(tmp_path, seed=3)
    r = s.talk()
    ex = s.buffer[-1]
    assert ex.source == "talk" and ex.obs.name_token and ex.obs.tags == ["REQUEST"] and ex.obs.cmd is None
    assert r.was_right and s.state.spent == 1 and len(s.state.words) == 1
    d = ex.obs.model_dump()  # the child gets the NAME token and attributes, never the kind (a material may share its name)
    assert set(d) == {"tags", "name_token", "options", "cmd", "tone"} and d["tone"] in (None, "gentle", "firm")  # tone: how the parent said it
    assert all(set(o["target"] or ()) <= {"color", "shape", "size", "material"} for o in d["options"])
    assert s.save.read_examples()[-1] == ex
    assert sum(e.type.value == "learned" for e in s.engine.log) == 1


def test_session_talk_names_the_least_named_thing_first(tmp_path):
    s = new(tmp_path, seed=4)
    for _ in range(4):
        s.talk()
    assert sorted(s.state.word_kinds.values()) == sorted(s.word_map)  # four talks name four different things


def test_session_command_show_and_discerning_try(tmp_path):
    s = new(tmp_path, seed=5)
    s.draw_cards()
    s.pick_card(0)
    b = s.debug_best()[0]
    r = s.command(b)
    assert "quality" not in r.model_dump_json() and s.view().command == b
    s.show(b)
    assert s.buffer[-1].obs.cmd == s.buffer[-1].obs.options[b] and s.buffer[-1].source == "show"
    s.draw_cards()
    s.pick_card(0)
    told = s.debug_best()[0]
    s.command(told)
    r = s.try_child()
    assert r.followed == (r.option == told)
    assert s.buffer == s.save.read_examples()


def test_session_with_talk_and_commands_saves_and_loads(tmp_path):
    s = new(tmp_path, seed=6)
    play_day(s, "balanced", 1, random.Random(0))
    s.end_day()
    n = len(s.buffer)
    s2 = Session.load(tmp_path / "s")
    assert s2.warnings == [] and len(s2.buffer) == n and s2.state == s.state
    assert s2.journal() == s.journal()


def test_journal_counts_what_the_parent_did(tmp_path):
    s = new(tmp_path, seed=7)
    play_day(s, "commander", 1, random.Random(1))
    j = s.journal()[1]
    assert j["talk"] == STYLES["commander"] and j["command"] == 7 and j["show"] == 7 and j["offer"] == 0
    assert sum(j["command_quality"].values()) == 7
    assert s.state.spent == s.game.per_day and s.state.phase == Phase.EVENING
    # the journal is computed from the log alone
    assert journal(s.engine.log)[1] == j


@pytest.mark.parametrize("style", sorted(STYLES))
def test_each_style_fills_the_day_and_spends_only_what_it_has(tmp_path, style):
    s = new(tmp_path, seed=8)
    play_day(s, style, 1, random.Random(2))
    assert s.state.phase == Phase.EVENING and s.state.spent == s.game.per_day
    assert s.journal()[1]["talk"] == STYLES[style]  # all its talking happened


def test_styles_differ_only_in_what_the_parent_does(tmp_path):
    """Same learner, model and quiz for every style (docs/WORLD.md 18.10): a style is only a different set of acts."""
    a, b = new(tmp_path, seed=9, name="a"), new(tmp_path, seed=9, name="b")
    assert a.learner.cfg == b.learner.cfg
    assert all((x == y).all() for x, y in zip(a.model.state_dict().values(), b.model.state_dict().values()))
    assert [q.hash for q in a.quiz] == [q.hash for q in b.quiz]


def test_end_day_reports_the_panel_and_the_journal(tmp_path):
    s = new(tmp_path, seed=10)
    play_day(s, "chooser", 1, random.Random(3))
    r = s.end_day()
    assert r["journal"]["offer"] == 7
    assert {"problem_acc", "word_acc", "confidence", "hesitate", "obey", "blind", "discern", "refuse_best"} <= set(r["panel"])
    quiz_event = next(ev for ev in s.engine.log if ev.type == EventType.QUIZ_TAKEN)
    assert "panel" in quiz_event.payload["result"]


# ---- the metrics: one implementation for the harness and the session ----
def test_the_harness_and_the_session_use_the_same_panel(tmp_path):
    s = new(tmp_path, seed=11)
    play_day(s, "demonstrator", 1, random.Random(4))
    full, harness = command_panel(s.model, s.quiz), exp_ea.panel(s.model, s.quiz)
    assert full.keys() == harness.keys() and all(a == b or (a != a and b != b) for a, b in zip(full.values(), harness.values()))
    met = command_panel(s.model, s.quiz, {"COLD"})
    assert met["n_problem_items"] <= full["n_problem_items"] and met["n_word_items"] == full["n_word_items"]


def test_the_panel_only_reads_the_model(tmp_path):
    s = new(tmp_path, seed=12)
    play_day(s, "talker", 1, random.Random(5))
    before = {k: v.clone() for k, v in s.model.state_dict().items()}
    s.panel()
    assert all((before[k] == v).all() for k, v in s.model.state_dict().items())


def test_untrained_child_does_not_discern(tmp_path):
    s = new(tmp_path, seed=13)
    p = command_panel(s.model0, s.quiz)
    assert abs(p["discern"]) < 0.3 and p["hesitate"] > 0.8
