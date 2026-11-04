import random

import pytest

from server.world.engine import Engine
from server.world.events import (
    EndDay, Event, OfferCards, PickCard, RecordQuiz, RuleError, Show, Skip, Try,
)
from server.world.generator import make_situation
from server.world.situation import obs_hash, observation, optimal
from server.world.state import Phase, fold

from helpers import cold, dark, request, thirsty, visitor, wet


def offer(e, a, b, aid, teasers=()):
    return e.submit(OfferCards(action_id=aid, situations=[a, b], teasers=list(teasers)))


def start(e, a=None, b=None, aid="o1", pick=0):
    n = len(e.state.seen_ids)
    offer(e, a or thirsty(f"t{n}"), b or visitor(f"v{n}"), aid)
    e.submit(PickCard(action_id=f"p-{aid}", index=pick))


def best(e):
    return optimal(e.state.current)[0]


def wrong(e):
    cur = e.state.current
    return next(i for i in range(len(cur.options)) if i not in optimal(cur))


def test_idempotent():
    e = Engine()
    start(e)
    cmd = Show(action_id="s1", option=best(e))
    assert len(e.submit(cmd)) == 1
    n = len(e.log)
    assert e.submit(cmd) == [] and len(e.log) == n
    assert e.state.situations_done == 1


def test_fold_roundtrip():
    e = Engine(per_day=2)
    for i in range(2):
        start(e, thirsty(f"t{i}"), visitor(f"v{i}"), f"o{i}")
        e.submit(Show(action_id=f"s{i}", option=best(e)))
    assert e.state.phase == Phase.EVENING
    e.submit(RecordQuiz(action_id="q", result={"score": 0.5}))
    e.submit(EndDay(action_id="end"))
    assert e.state.day == 2
    rebuilt = Engine(log=[ev.model_copy() for ev in e.log])
    assert rebuilt.state == e.state == fold(e.log)
    again = fold([Event.model_validate_json(ev.model_dump_json()) for ev in e.log])  # survives a JSON round trip too
    assert again == e.state


def test_rule_errors():
    e = Engine(per_day=2)
    with pytest.raises(RuleError) as x:
        e.submit(PickCard(action_id="a", index=0))
    assert x.value.code == "no_cards"
    with pytest.raises(RuleError) as x:
        e.submit(Show(action_id="b", option=0))
    assert x.value.code == "wrong_phase"
    offer(e, thirsty("t0"), visitor("v0"), "o1")
    with pytest.raises(RuleError) as x:
        e.submit(PickCard(action_id="c", index=5))
    assert x.value.code == "bad_pick"
    e.submit(PickCard(action_id="c2", index=0))
    with pytest.raises(RuleError) as x:
        e.submit(Show(action_id="d", option=9))
    assert x.value.code == "bad_option"
    right = best(e)
    e.submit(Try(action_id="t1", option=right))  # the child's own act happens at once and ends the situation
    with pytest.raises(RuleError):
        e.submit(Skip(action_id="i"))  # nothing in progress now
    assert e.log[-1].payload["tag_values"] and e.state.situations_done == 1
    start(e, thirsty("t1"), visitor("v1"), "o2")
    e.submit(Try(action_id="t2", option=wrong(e)))
    assert e.state.phase == Phase.EVENING
    with pytest.raises(RuleError) as x:
        e.submit(EndDay(action_id="l"))
    assert x.value.code == "quiz_needed"


def test_events_record_what_the_rules_table_decided():
    e = Engine()
    start(e, wet("w0"), thirsty("t0"))
    ev = e.submit(Show(action_id="s", option=2))[0]  # WAIT on a wet floor
    p = ev.payload
    assert p["option"] == 2 and p["value"] == -1 and p["outcome"] == "worse" and p["was_right"] is False
    assert p["tag_values"] == {"FLOOR_WET": -1}
    start(e, wet("w1"), thirsty("t1"), "o2")
    p = e.submit(Show(action_id="s2", option=0))[0].payload
    assert p["value"] == 2 and p["outcome"] == "solved" and p["was_right"] is True
    start(e, wet("w2"), thirsty("t2"), "o3")
    p = e.submit(Show(action_id="s3", option=1))[0].payload  # paper: partial, not optimal
    assert p["outcome"] == "partial" and p["was_right"] is False


def test_unsolved_problems_become_issues_and_come_back():
    e = Engine()
    start(e, wet("w0"), thirsty("t0"))
    e.submit(Show(action_id="s", option=2))  # waited -> worse
    [issue] = e.state.open_issues()
    assert issue.id == "w0:FLOOR_WET" and issue.outcome == "worse"
    # a follow-up: same problem, or (it got worse) with its escalation
    offer(e, wet("w1", links_to=issue.id), thirsty("t1"), "o2")
    e.submit(PickCard(action_id="p2", index=0))
    e.submit(Show(action_id="s2", option=1))  # only a partial fix: still open, outcome updated
    assert e.state.open_issues()[0].outcome == "partial" and len(e.state.issues) == 1
    offer(e, wet("w2", links_to=issue.id), thirsty("t2"), "o3")
    e.submit(PickCard(action_id="p3", index=0))
    e.submit(Show(action_id="s3", option=0))  # solved -> the issue is closed
    assert e.state.open_issues() == []


def test_followup_rules():
    e = Engine()
    start(e, wet("w0"), thirsty("t0"))
    e.submit(Show(action_id="s", option=2))  # FLOOR_WET got worse
    iid = "w0:FLOOR_WET"
    with pytest.raises(RuleError):  # no such issue
        offer(e, wet("a", links_to="nope"), thirsty("b"), "o2")
    with pytest.raises(RuleError):  # a different problem cannot continue it
        offer(e, dark("a", links_to=iid), thirsty("b"), "o2")
    with pytest.raises(RuleError):  # two cards cannot continue the same issue
        offer(e, wet("a", links_to=iid), wet("b", links_to=iid), "o2")
    from helpers import sit
    both = sit("a", ["FLOOR_WET", "VISITOR"], [("cloth", "red", "big"), ("ball", "blue", "big"), ("cup", "green", "small")],
               [("WIPE", 0), ("GIVE", 1), ("WIPE", 2), ("WAIT", None)], links_to=iid)
    offer(e, both, thirsty("b"), "o2")  # worse + escalation FLOOR_WET -> VISITOR joins: allowed
    assert e.state.cards[0].tags == ["FLOOR_WET", "VISITOR"]
    # without "worse" the escalation is not allowed
    e2 = Engine()
    start(e2, wet("w0"), thirsty("t0"))
    e2.submit(Show(action_id="s", option=1))  # partial
    with pytest.raises(RuleError):
        offer(e2, both.model_copy(update={"id": "z"}), thirsty("b"), "o2")


def test_two_problem_situation_tracks_each_problem():
    from helpers import sit
    e = Engine()
    both = sit("a", ["FLOOR_WET", "VISITOR"], [("cloth", "red", "big"), ("ball", "blue", "big"), ("cup", "green", "small")],
               [("WIPE", 0), ("GIVE", 1), ("WAIT", None)])
    start(e, both, thirsty("b"))
    p = e.submit(Show(action_id="s", option=1))[0].payload  # give the ball: the visitor is fine, the floor stays wet
    assert p["tag_values"] == {"FLOOR_WET": 0, "VISITOR": 2}
    assert [(i.tag, i.outcome) for i in e.state.open_issues()] == [("FLOOR_WET", "nothing")]


def test_solving_dark_unlocks_the_yard():
    e = Engine()
    with pytest.raises(RuleError) as x:
        offer(e, thirsty("a", location="yard"), visitor("b"), "o1")
    assert "not unlocked" in str(x.value)
    start(e, dark("d0"), thirsty("t0"))
    e.submit(Show(action_id="s", option=best(e)))
    assert "yard" in e.state.unlocked
    offer(e, thirsty("a", location="yard"), visitor("b"), "o2")


def test_word_requests_are_sparse_and_follow_use():
    e = Engine(per_day=4)
    with pytest.raises(RuleError) as x:  # nothing was used yet: no word for the cup
        offer(e, request("r", "cup", "bobo", "NAME_00"), thirsty("t"), "o1")
    assert "not been used" in str(x.value)
    start(e, thirsty("t0"), visitor("v0"))
    e.submit(Show(action_id="s", option=best(e)))  # DRINK with the cup works -> the cup is nameable
    assert e.state.nameable == ["cup"]
    offer(e, request("r", "cup", "bobo", "NAME_00"), thirsty("t1"), "o2")
    assert e.state.words == {"bobo": "NAME_00"}
    e.submit(PickCard(action_id="p2", index=0))
    assert e.state.current_first is True
    e.submit(Show(action_id="s2", option=best(e)))
    # at most one request a day
    with pytest.raises(RuleError) as x:
        offer(e, request("r2", "cup", "bobo", "NAME_00"), thirsty("t2"), "o3")
    assert "limited" in str(x.value)
    # a new day lifts the limit; the word keeps its token, and is no longer a first use
    for i in range(2):
        start(e, thirsty(f"u{i}"), visitor(f"w{i}"), f"oo{i}")
        e.submit(Skip(action_id=f"sk{i}"))
    assert e.state.phase == Phase.EVENING
    e.submit(RecordQuiz(action_id="q", result={})); e.submit(EndDay(action_id="end"))
    offer(e, request("r3", "cup", "bobo", "NAME_00"), thirsty("t3"), "o4")
    e.submit(PickCard(action_id="p4", index=0))
    assert e.state.current_first is False


def test_word_consistency_is_enforced():
    e = Engine(per_day=8)
    start(e, thirsty("t0"), visitor("v0"))
    e.submit(Show(action_id="s", option=best(e)))  # cup nameable
    start(e, visitor("v1"), thirsty("t1"), "o2")
    e.submit(Show(action_id="s2", option=best(e)))  # ball nameable
    offer(e, request("r0", "cup", "bobo", "NAME_07"), thirsty("t2"), "o3")  # the engine fixes the token the Director chose
    assert e.state.words == {"bobo": "NAME_07"}
    e.submit(PickCard(action_id="p3", index=0)); e.submit(Skip(action_id="k3"))
    for i in range(6):
        if e.state.phase != Phase.CARDS:
            break
        start(e, thirsty(f"x{i}"), visitor(f"y{i}"), f"oo{i}")
        e.submit(Skip(action_id=f"skp{i}"))
    e.submit(RecordQuiz(action_id="q", result={})); e.submit(EndDay(action_id="end"))
    for bad in (request("a", "ball", "bobo", "NAME_07"),      # same word, another kind
                request("b", "cup", "mimi", "NAME_02"),       # same kind, another word
                request("c", "ball", "mimi", "NAME_07"),      # token already belongs to bobo
                request("d", "ball", "mimi", "NAME_99")):     # token outside the vocabulary
        with pytest.raises(RuleError):
            offer(e, bad, thirsty("t9" + bad.id), "oz" + bad.id)
    offer(e, request("e", "cup", "bobo", "NAME_07"), thirsty("tz"), "ok")  # same word, same kind, same token: fine


def test_engine_refuses_quiz_scenes_and_twins():
    e = Engine()
    t = thirsty("a")
    e.forbidden = {obs_hash(observation(t))}
    with pytest.raises(RuleError) as x:
        offer(e, t, visitor("b"), "o1")
    assert "quiz" in str(x.value)
    with pytest.raises(RuleError) as x:
        offer(e, visitor("a"), visitor("b"), "o2")
    assert "identical" in str(x.value)
    with pytest.raises(RuleError):  # one id twice
        offer(e, visitor("a"), cold("a"), "o3")
    offer(e, cold("a"), visitor("b"), "o4")
    assert e.state.cards and e.state.card_teasers == []
    e.submit(PickCard(action_id="p", index=0))
    e.submit(Skip(action_id="sk"))
    with pytest.raises(RuleError):  # an id that was used before
        offer(e, cold("a"), visitor("zz"), "o5")


def test_wrong_show_is_allowed_and_world_follows_it():
    e = Engine()
    start(e, thirsty("t0"), visitor("v0"))
    ev = e.submit(Show(action_id="s", option=wrong(e)))[0]
    assert ev.payload["was_right"] is False and ev.payload["value"] < 2
    assert e.state.situations_done == 1
    assert [i.tag for i in e.state.open_issues()] == ["THIRSTY"]


def test_generated_offers_are_accepted_by_the_engine():
    e, rng = Engine(per_day=50), random.Random(0)
    for i in range(10):
        a = make_situation(rng, ["COLD"], f"a{i}")
        b = make_situation(rng, ["DARK"], f"b{i}")
        e.submit(OfferCards(action_id=f"o{i}", situations=[a, b]))
        e.submit(PickCard(action_id=f"p{i}", index=i % 2))
        e.submit(Show(action_id=f"s{i}", option=best(e)))
    assert e.state.situations_done == 10 and e.state.topic_counts == {"COLD": 5, "DARK": 5}


def test_what_the_child_does_alone_changes_the_world_like_anything_else_done():
    e = Engine(per_day=2)
    start(e, thirsty("t1"), visitor("v1"), "o1")
    e.submit(Try(action_id="t1", option=wrong(e)))
    assert e.state.issues and e.state.issues[0].outcome in ("nothing", "worse")  # a problem left unsolved stays
    assert e.log[-1].type.value == "child_tried" and e.log[-1].payload["outcome"] in ("nothing", "worse", "partial")
