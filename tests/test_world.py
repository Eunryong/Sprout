import random

import pytest

from server.world.generator import make_situation
from server.world.ontology import KINDS, NEEDS, PROBLEMS, REQUEST_KINDS, RULES, TAGS, outcome_label
from server.world.situation import (
    BadSituation, Obj, obs_hash, observation, optimal, option_value, outcome_of, validate_situation,
    values,
)

from helpers import obj, sit, thirsty, wet

TOPICS = [(["THIRSTY"], None, None, None), (["FLOOR_WET"], None, None, None), (["DARK"], None, None, None),
          (["COLD"], None, None, None), (["VISITOR"], None, None, None), (["COLD", "DARK"], None, None, None),
          (["FLOOR_WET", "VISITOR"], None, None, None)] + [(["REQUEST"], f"w{i}", f"NAME_0{i}", k) for i, k in enumerate(REQUEST_KINDS)]


def sits(n=40):
    rng = random.Random(0)
    for tags, w, tok, kind in TOPICS:
        for i in range(n):
            yield make_situation(rng, tags, f"s{i}", "house", w, tok, kind)


def test_ontology_is_consistent():
    assert set(PROBLEMS) | {"REQUEST"} == set(TAGS)
    assert all(RULES[t][NEEDS[t]] == 2 for t in PROBLEMS)
    # kinds are told apart by (shape, material); cloth / paper / blanket share a shape, flashlight / spoon a material
    assert len({(k.shape, k.material) for k in KINDS.values()}) == len(KINDS)
    assert KINDS["cloth"].shape == KINDS["paper"].shape == KINDS["blanket"].shape
    assert KINDS["flashlight"].material == KINDS["spoon"].material


def test_rules_table():
    o = obj
    assert option_value(["THIRSTY"], None, "DRINK", o("cup")) == 2
    assert option_value(["THIRSTY"], None, "DRINK", o("spoon")) == 0
    assert option_value(["FLOOR_WET"], None, "WIPE", o("paper")) == 1
    assert option_value(["COLD"], None, "COVER", o("cloth")) == 1
    assert option_value(["DARK"], None, "WAIT", None) == 0 and option_value(["COLD"], None, "WAIT", None) == -1
    assert option_value(["REQUEST"], "ball", "FETCH", o("ball")) == 2
    assert option_value(["REQUEST"], "ball", "FETCH", o("cup")) == 0
    assert option_value(["COLD", "DARK"], None, "LIGHT", o("flashlight")) == 2  # either problem can be solved
    assert option_value(["COLD", "DARK"], None, "WAIT", None) == -1
    assert [outcome_label(v) for v in (2, 1, 0, -1)] == ["solved", "partial", "nothing", "worse"]


def test_outcome_of_handbuilt_situation():
    s = wet()
    assert values(s) == [2, 1, -1, 0]
    assert [outcome_of(s, i)[1] for i in range(4)] == ["solved", "partial", "worse", "nothing"]
    assert optimal(s) == [0]


def test_generator_invariants():
    for s in sits():
        validate_situation(s)
        v = values(s)
        assert 3 <= len(s.options) <= 5 and max(v) == 2 and min(v) < 2 and optimal(s)
        for t in s.tags:
            if t != "REQUEST":  # the right thing is always on the table
                verb, kind = NEEDS[t]
                assert any(o.verb == verb and s.objects[o.target].kind == kind for o in s.options if o.target is not None)


def test_child_observation_has_no_kind_value_or_narration():
    for s in list(sits(10))[:60]:
        obs = observation(s)
        d = obs.model_dump()
        assert set(d) == {"tags", "name_token", "options", "cmd", "tone"} and d["cmd"] is None and d["tone"] is None  # no parent instruction in an engine scene
        for o in d["options"]:
            assert set(o) == {"verb", "target"}
            assert o["target"] is None or set(o["target"]) == {"color", "shape", "size", "material"}
        assert (obs.name_token is not None) == (s.tags == ["REQUEST"])
        assert s.word is None or s.word not in obs.model_dump_json()  # Grandma's word is for the player; the child has a token


def test_hash_is_order_insensitive_and_forbidden_respected():
    s = next(sits(1))
    o = observation(s)
    rev = o.model_copy(update={"options": list(reversed(o.options))})
    assert obs_hash(o) == obs_hash(rev)
    rng = random.Random(1)
    for _ in range(50):
        t = make_situation(rng, ["THIRSTY"], "x", forbidden={obs_hash(o)})
        assert obs_hash(observation(t)) != obs_hash(o)


def test_validate_rejects_bad_situations():
    good = thirsty()
    validate_situation(good)
    cases = {
        "no solving option": sit("a", ["THIRSTY"], [("cup", "red", "big"), ("spoon", "red", "big"), ("ball", "red", "big")],
                                 [("DRINK", 1), ("DRINK", 2), ("WAIT", None)]),
        "all equally good": sit("b", ["DARK"], [("flashlight", "red", "big"), ("flashlight", "blue", "big"), ("flashlight", "green", "big")],
                                [("LIGHT", 0), ("LIGHT", 1), ("LIGHT", 2)]),
        "identical options": sit("c", ["THIRSTY"], [("cup", "red", "big"), ("spoon", "red", "big"), ("ball", "red", "big")],
                                 [("DRINK", 0), ("DRINK", 0), ("DRINK", 1)]),
        "WAIT with target": sit("d", ["THIRSTY"], [("cup", "red", "big"), ("spoon", "red", "big"), ("ball", "red", "big")],
                                [("DRINK", 0), ("DRINK", 1), ("WAIT", 0)]),
        "too few options": sit("e", ["THIRSTY"], [("cup", "red", "big"), ("spoon", "red", "big"), ("ball", "red", "big")],
                               [("DRINK", 0), ("DRINK", 1)]),
        "target out of range": sit("f", ["THIRSTY"], [("cup", "red", "big"), ("spoon", "red", "big"), ("ball", "red", "big")],
                                   [("DRINK", 0), ("DRINK", 1), ("DRINK", 7)]),
        "unknown verb": sit("g", ["THIRSTY"], [("cup", "red", "big"), ("spoon", "red", "big"), ("ball", "red", "big")],
                            [("DRINK", 0), ("DRINK", 1), ("DANCE", 2)]),
        "REQUEST mixed with a problem": good.model_copy(update={"tags": ["THIRSTY", "REQUEST"]}),
        "word on a problem": good.model_copy(update={"word": "bobo"}),
        "unknown location": good.model_copy(update={"location": "moon"}),
        "request without word": good.model_copy(update={"tags": ["REQUEST"]}),
    }
    for name, bad in cases.items():
        with pytest.raises(BadSituation):
            validate_situation(bad)
        assert name
    wrong_attrs = good.model_copy(update={"objects": [Obj(kind="cup", color="red", shape="flat", size="big", material="ceramic")] + good.objects[1:]})
    with pytest.raises(BadSituation):
        validate_situation(wrong_attrs)
