import hashlib
import json
import random

import pytest
import torch

from server.agents.client import ClientError
from server.game.autoplay import play_day
from server.game.session import Session
from server.learning.codec import option_probs
from server.learning.evaluator import score
from server.learning.learner import OnlineCfg
from server.world.events import RuleError
from server.world.ontology import REQUESTS_PER_DAY
from server.world.situation import obs_hash, observation
from server.world.state import Phase

FAST = OnlineCfg(steps=12)


def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def new(tmp_path, seed=0, name="s", fast=True, **kw):
    if fast:
        kw.setdefault("ocfg", OnlineCfg(steps=12))
    return Session.new(tmp_path / name, seed=seed, **kw)


def to_situation(s, pick=0):
    s.draw_cards()
    s.pick_card(pick)


def wrong_option(s):
    cur = s.state.current
    return next(i for i in range(len(cur.options)) if i not in s.debug_best())


def test_quiz_not_in_train_and_ckpt0_immutable_and_weights_move(tmp_path):
    s = new(tmp_path, seed=3)
    h0 = sha(s.save.ckpt0)
    before = {k: v.clone() for k, v in s.model.state_dict().items()}
    rng = random.Random(0)
    for _ in range(2):
        play_day(s, rng)
    quiz_hashes = {q.hash for q in s.quiz}
    train_hashes = {obs_hash(e.obs) for e in s.buffer}
    assert len(s.buffer) == 16 and not (quiz_hashes & train_hashes)
    assert sha(s.save.ckpt0) == h0
    assert any(not torch.equal(before[k], v) for k, v in s.model.state_dict().items())
    assert sum(e.type.value == "learned" for e in s.engine.log) == len(s.buffer)


def test_every_situation_the_engine_ever_offered_avoids_the_quiz(tmp_path):
    s = new(tmp_path, seed=14)
    rng = random.Random(4)
    for _ in range(3):
        play_day(s, rng)
    offered = [c for e in s.engine.log if e.type.value == "cards_offered" for c in e.payload["situations"]]
    assert len(offered) >= 48
    from server.world.situation import Situation
    assert not ({obs_hash(observation(Situation(**c))) for c in offered} & {q.hash for q in s.quiz})


def test_reset_forgets(tmp_path):
    s = new(tmp_path, seed=4, fast=False)
    rng = random.Random(1)
    for _ in range(4):
        play_day(s, rng)
    pool = [q for q in s.quiz if q.topic in s.met_topics()]
    trained, reset = score(s.model, pool), score(s.model0, pool)
    assert trained["p_opt"] > reset["p_opt"] + 0.25, (trained, reset)


def test_resume_is_identical_and_continues(tmp_path):
    s = new(tmp_path, seed=5)
    rng = random.Random(2)
    play_day(s, rng)
    to_situation(s)
    s.show(s.debug_best()[0])  # leave the day half done
    q = s.quiz[0].obs
    p_before, state_before, n = option_probs(s.model, q), s.state.model_copy(deep=True), len(s.buffer)
    s2 = Session.load(tmp_path / "s")
    assert not s2.warnings
    assert s2.state == state_before and len(s2.buffer) == n
    assert option_probs(s2.model, q) == p_before
    assert s2.engine.forbidden == s.engine.forbidden
    # the optimizer state came back too: both sessions take the same next step
    s.draw_cards(); s2.draw_cards()
    assert s.state.cards == s2.state.cards
    s.pick_card(0); s2.pick_card(0)
    a = s.debug_best()[0]
    r1, r2 = s.show(a), s2.show(a)
    assert r1.learn.loss_after == pytest.approx(r2.learn.loss_after, rel=1e-5)
    play_day(s2, rng)  # and the loaded game can be played on


def test_load_refuses_a_save_from_another_ontology(tmp_path):
    s = new(tmp_path, seed=15)
    meta = json.loads(s.save.meta.read_text())
    meta["ontology"] = "0000"
    s.save.meta.write_text(json.dumps(meta))
    with pytest.raises(ValueError, match="ontology"):
        Session.load(tmp_path / "s")


def test_session_idempotent_show(tmp_path):
    s = new(tmp_path, seed=6)
    to_situation(s)
    a = s.debug_best()[0]
    assert not s.show(a, action_id="x1").duplicate
    n = len(s.buffer)
    assert s.show(a, action_id="x1").duplicate
    assert len(s.buffer) == n and s.state.situations_done == 1


def test_wrong_show_learned_as_is_and_world_reports_it(tmp_path):
    s = new(tmp_path, seed=7)
    to_situation(s)
    w = wrong_option(s)
    r = s.show(w)
    ex = s.buffer[-1]
    assert ex.chosen == w and ex.was_right is False and ex.source == "show"
    assert r.was_right is False and r.outcome in ("partial", "nothing", "worse") and r.value < 2
    assert s.save.read_examples()[-1] == ex
    assert [i.open for i in s.state.issues]  # the unsolved problem stays in the world


def test_letting_a_scene_pass_teaches_nothing_but_the_childs_own_act_does(tmp_path):
    s = new(tmp_path, seed=8)
    to_situation(s)
    s.skip()
    assert s.buffer == [] and s.state.situations_done == 1
    assert not any(e.type.value == "learned" for e in s.engine.log)
    to_situation(s)
    r = s.try_child()  # nobody takes part: the child chooses, the world answers, the child learns
    ex = s.buffer[-1]
    assert s.state.situations_done == 2 and ex.source == "try" and ex.chosen == r.option and ex.facts[0].result == r.outcome
    assert any(e.type.value == "learned" for e in s.engine.log)


def test_the_childs_own_act_is_one_example_and_cannot_be_followed_by_another_act_in_the_scene(tmp_path):
    s = new(tmp_path, seed=9)
    rng = random.Random(0)
    for _ in range(6):
        if s.state.phase == Phase.EVENING:
            s.end_day()
        to_situation(s, rng.randrange(2))
        s.try_child()
        with pytest.raises(RuleError):
            s.show(0)  # the scene is over
    assert {e.source for e in s.buffer} == {"try"} and all(e.weight == 1.0 and len(e.facts) == 1 for e in s.buffer)


def test_evening_gating_and_day_rollover(tmp_path):
    s = new(tmp_path, seed=10)
    with pytest.raises(RuleError):
        s.end_day()
    r = play_day(s, random.Random(3))
    assert r["day"] == 1 and s.state.day == 2 and s.state.phase == Phase.CARDS
    assert s.save.day_ckpt(1).exists()
    assert 0 < r["n"] <= 10 and 0 <= r["optimal"] <= 1 and set(r["by_topic"]) <= s.met_topics()
    assert s.state.requests_today == 0


def test_director_failure_does_not_stop_the_game(tmp_path):
    class Down:
        name = "down"

        def complete_json(self, **kw):
            raise ClientError("api down")
    s = new(tmp_path, seed=11, client=Down())
    to_situation(s)
    assert s.director.last_source == "template" and "api down" in s.director.last_error
    s.show(s.debug_best()[0])


def test_views_never_mix_child_and_player_knowledge(tmp_path):
    s = new(tmp_path, seed=12)
    s.draw_cards()
    v = s.view()
    assert v.phase == "cards" and len(v.previews) == 2 and all(p["teaser"] for p in v.previews)
    s.pick_card(0)
    v = s.view()
    assert v.options and all(o.text for o in v.options) and v.narration  # the player sees what each option is
    assert any(o.kind for o in v.options)  # player-facing kind labels exist here...
    assert "kind" not in observation(s.state.current).model_dump_json()  # ...and never in the child's input


def test_words_are_rare_and_follow_use(tmp_path):
    s = new(tmp_path, seed=16)
    rng = random.Random(7)
    for _ in range(5):
        play_day(s, rng)
    offered, picked_requests = None, {}
    for e in s.engine.log:
        if e.type.value == "cards_offered":
            offered = e.payload["situations"]
        elif e.type.value == "card_picked" and offered[e.payload["index"]]["tags"] == ["REQUEST"]:
            picked_requests[e.day] = picked_requests.get(e.day, 0) + 1
    assert all(n <= REQUESTS_PER_DAY for n in picked_requests.values())  # sparse by rule
    requests = [c for e in s.engine.log if e.type.value == "cards_offered" for c in e.payload["situations"]
                if c["tags"] == ["REQUEST"]]
    assert all(c["word_kind"] in s.state.nameable for c in requests)  # a word only for something that was used
    assert set(s.state.words.values()) <= {t for _, t in s.word_map.values()}
    assert s.state.requests_today == 0  # reset by the new day


def test_mind_report_and_snapshot_compare(tmp_path):
    s = new(tmp_path, seed=13)
    m0 = s.mind()
    assert m0["topics"] == {} and m0["tag_reliance"]["n"] == 0
    rng = random.Random(0)
    play_day(s, rng)
    play_day(s, rng)
    m = s.mind(compare_day=1)
    assert m["topics"] and all("before" in e for e in m["topics"].values())
    assert set(m["topics"]) == s.met_topics()
    assert all(0 <= e["accuracy"] <= 1 and e["practised"] > 0 for e in m["topics"].values())
    assert "tag_reliance_before" in m and m == s.mind(compare_day=1)  # deterministic
