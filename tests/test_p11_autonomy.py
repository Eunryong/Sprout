"""Learning rounds by who chose, and replay that favours strong lessons (docs/WORLD.md 22.10)."""
import random
from collections import Counter

from server.config import GameCfg
from server.game.session import Session
from server.learning.codec import Example
from server.learning.learner import Learner, OnlineCfg
from server.learning.model import ModelCfg, build_model
from server.world.situation import observation

from helpers import thirsty


def new(tmp_path, **game):
    s = Session.new(tmp_path / "s", seed=0, game=GameCfg(**game), ocfg=OnlineCfg(steps=4))
    calls = []
    real = s.learner.learn
    s.learner.learn = lambda *a, **k: (calls.append(1), real(*a, **k))[1]
    return s, calls


def to_situation(s):
    s.draw_cards()
    s.pick_card(0)


def test_by_default_every_teaching_move_is_learned_once(tmp_path):
    s, calls = new(tmp_path)
    to_situation(s)
    s.command(s.debug_best()[0], tone="firm")
    s.show(s.debug_best()[0], tone="firm")
    to_situation(s)
    s.try_child()
    assert len(calls) == 2 and [e.strength for e in s.buffer] == [1.0, 1.0]


def test_a_firm_instruction_is_learned_in_more_rounds_than_a_gentle_one(tmp_path):
    s, calls = new(tmp_path, firm_rounds=2)
    to_situation(s)
    s.command(s.debug_best()[0], tone="firm")
    s.show(s.debug_best()[0], tone="firm")
    n_firm = len(calls)
    to_situation(s)
    s.show(s.debug_best()[0], tone="gentle")
    assert n_firm == 2 and len(calls) - n_firm == 1
    assert [(e.mode, e.strength) for e in s.buffer] == [("command_firm", 2.0), ("show_gentle", 1.0)]


def test_what_the_child_chose_itself_is_learned_in_more_rounds(tmp_path):
    s, calls = new(tmp_path, own_rounds=3)
    seen = set()
    for _ in range(8):
        if s.state.phase.value == "evening":
            break
        to_situation(s)
        before = len(calls)
        s.try_child()
        assert len(calls) - before == 3
        seen.add(s.buffer[-1].mode)
    assert seen == {"own_try"} and all(e.strength == 3.0 for e in s.buffer)


def test_a_stronger_lesson_is_one_learned_event_with_the_steps_of_all_its_rounds(tmp_path):
    s, _ = new(tmp_path, own_rounds=2)
    to_situation(s)
    res = s.try_child()
    learned = [e for e in s.engine.log if e.type.value == "learned"]
    assert len(learned) == 1 == len(s.buffer) and res.learn.steps == 8  # 2 rounds x 4 steps
    again = Session.load(tmp_path / "s")
    assert not again.warnings


def lessons(strengths, days):
    return [Example(id=f"e{i}", day=d, source="show", obs=observation(thirsty()), chosen=0, strength=s)
            for i, (s, d) in enumerate(zip(strengths, days))]


def draw(cfg, buf, new_day, n=400):
    L = Learner(build_model(ModelCfg()), cfg)
    rng = random.Random(0)
    new_ex = Example(id="new", day=new_day, source="show", obs=observation(thirsty()), chosen=0)
    c = Counter()
    for _ in range(n):
        c.update(e.id for e in L._replay(buf, new_ex, rng))
    return c


def test_uniform_replay_is_unchanged_by_default():
    buf = lessons([1.0] * 5 + [9.0] * 5, [1] * 10)
    cfg = OnlineCfg(replay=3)
    L = Learner(build_model(ModelCfg()), cfg)
    new_ex = Example(id="new", day=1, source="show", obs=observation(thirsty()), chosen=0)
    assert [e.id for e in L._replay(buf, new_ex, random.Random(5))] == [e.id for e in random.Random(5).sample(buf, 3)]


def test_replay_by_strength_rehearses_strong_lessons_more():
    buf = lessons([1.0] * 5 + [4.0] * 5, [1] * 10)
    c = draw(OnlineCfg(replay=3, replay_by_strength=True), buf, new_day=1)
    weak, strong = sum(c[f"e{i}"] for i in range(5)), sum(c[f"e{i}"] for i in range(5, 10))
    assert strong > 1.5 * weak


def test_replay_decay_lets_old_lessons_fade():
    buf = lessons([1.0] * 5 + [1.0] * 5, [1] * 5 + [6] * 5)
    c = draw(OnlineCfg(replay=3, replay_decay=0.6), buf, new_day=6)
    old, recent = sum(c[f"e{i}"] for i in range(5)), sum(c[f"e{i}"] for i in range(5, 10))
    assert recent > 3 * old


# ---- a wrong choice is learned by what it did (docs/WORLD.md 22.11) ----
def test_a_correction_gets_the_rounds_of_what_the_wrong_choice_did():
    g = GameCfg(own_rounds=2, partial_rounds=1, nothing_rounds=3, worse_rounds=5)
    assert [g.correction_rounds(o) for o in ("partial", "nothing", "worse")] == [1, 3, 5]
    assert GameCfg(own_rounds=2).correction_rounds("worse") == 2  # not set: as any self-made experience
    assert GameCfg().correction_rounds("worse") == 1  # off by default


def test_the_session_learns_each_wrong_choice_in_the_rounds_of_its_outcome(tmp_path, monkeypatch):
    import types

    from server.game import session as sess
    from server.world.situation import values

    s = Session.new(tmp_path / "s", seed=2, game=GameCfg(own_rounds=2, partial_rounds=1, nothing_rounds=3, worse_rounds=5),
                    ocfg=OnlineCfg(steps=2))
    wrong = {}

    def act(model, obs, rng, **kw):  # the child always makes its worst choice
        v = values(wrong["sit"])
        k = min(range(len(v)), key=v.__getitem__)
        return types.SimpleNamespace(option=k, probs=[1 / len(v)] * len(v))
    monkeypatch.setattr(sess, "act", act)
    expected = {"partial": 1.0, "nothing": 3.0, "worse": 5.0}
    seen = set()
    for _ in range(8):
        s.draw_cards()
        s.pick_card(0)
        wrong["sit"] = s.state.current
        s.try_child()
        ex = s.buffer[-1]
        assert ex.mode == "own_try" and ex.outcome in expected and ex.strength == expected[ex.outcome]
        seen.add(ex.outcome)
    assert len(seen) >= 2  # the scenes of the day do not all end alike
