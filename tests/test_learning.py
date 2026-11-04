import copy
import random

import torch

from server.learning.checkpoints import load, save
from server.learning.codec import (
    Fact, MAX_LEN, TOK, VOCAB, Example, augment, collate, encode, option_logits, option_probs, redraw_options, shuffle_options,
    weighted_ce,
)
from server.learning.evaluator import make_quiz, pick_daily, quiz_topics, score, tag_reliance
from server.learning.learner import Learner, OnlineCfg
from server.learning.model import ModelCfg, build_model, n_params
from server.world.generator import make_situation
from server.world.ontology import REQUEST_KINDS
from server.world.situation import obs_hash, observation, optimal

from test_world import sits

WM = {k: (f"w{i}", f"NAME_0{i}") for i, k in enumerate(REQUEST_KINDS)}


def make_ex(seed, tags=("THIRSTY",)):
    s = make_situation(random.Random(seed), list(tags), "s")
    obs, best = observation(s), optimal(s)[0]
    return Example(id=f"e{seed}", day=1, source="show", obs=obs, chosen=best, facts=[Fact(option=obs.options[best], result="solved")])


def test_encoding_and_masking():
    m = build_model(ModelCfg())
    for s in sits(8):
        ex = Example(id="e", day=1, source="show", obs=observation(s), chosen=optimal(s)[0])
        ids = encode(ex.obs)
        assert len(ids) <= MAX_LEN and max(ids) < len(VOCAB) and ids[-1] == TOK["ACT_PICK"]
        a, b, n, _, _ = collate([ex])
        lg = option_logits(m, a, b, n)[0]
        k = len(ex.obs.options)
        assert torch.isfinite(lg[:k]).all() and torch.isinf(lg[k:]).all()  # every existing option stays allowed
        p = option_probs(m, ex.obs)
        assert len(p) == k and abs(sum(p) - 1) < 1e-5


def test_default_model_is_three_layers():
    m = build_model(ModelCfg())
    assert m.cfg.n_layers == 3
    print("params", n_params(m))


def test_augmentation_keeps_the_taught_option():
    rng = random.Random(0)
    for s in sits(15):
        ex = Example(id="e", day=1, source="show", obs=observation(s), chosen=optimal(s)[0])
        chosen = ex.obs.options[ex.chosen]
        for p in (0.0, 1.0):
            a = augment(ex, rng, p)
            assert a.obs.options[a.chosen] == chosen and len(a.obs.options) == len(ex.obs.options)
        r = redraw_options(ex, rng)
        for i, o in enumerate(r.obs.options):
            if i != ex.chosen:
                assert o != chosen  # never a duplicate of what the player chose
        sh = shuffle_options(ex, rng)
        assert sorted(map(repr, sh.obs.options)) == sorted(map(repr, ex.obs.options))


def test_learning_lowers_loss_and_moves_weights():
    m = build_model(ModelCfg(seed=1))
    before = copy.deepcopy(m.state_dict())
    ex = make_ex(1)

    def avg_loss(model):  # average over option orders: one order can look good by luck at initialisation
        r = random.Random(5)
        with torch.no_grad():
            return sum(weighted_ce(model, [shuffle_options(ex, r)]).item() for _ in range(40)) / 40
    start = avg_loss(m)
    Learner(m, OnlineCfg(redraw=0.0)).learn(ex, [ex], random.Random(0), steps=240)
    assert avg_loss(m) < start * 0.6  # label smoothing keeps the loss above a floor (~0.48), so it cannot reach near zero
    assert any(not torch.equal(before[k], v) for k, v in m.state_dict().items())


def test_ckpt_roundtrip_and_immutable(tmp_path):
    m = build_model(ModelCfg())
    p = tmp_path / "ckpt_000.pt"
    save(m, p)
    ex = make_ex(2)
    Learner(m, OnlineCfg()).learn(ex, [ex], random.Random(0))
    m0, _ = load(p)
    assert option_probs(m0, ex.obs) != option_probs(m, ex.obs)  # ckpt_000 untouched by later learning
    try:
        save(m, p, overwrite=False)
        raise AssertionError("expected FileExistsError")
    except FileExistsError:
        pass


def test_quiz_scoring_isolation_and_baselines():
    topics = quiz_topics(WM, [["FLOOR_WET", "VISITOR"]])
    assert len(topics) == 5 + 4 + 1
    quiz = make_quiz(random.Random(3), topics, 5)
    assert len(quiz) == 50 and len({q.hash for q in quiz}) == 50
    m = build_model(ModelCfg(seed=1))
    sc = score(m, quiz)
    assert sc["n"] == 50 and 0 <= sc["optimal"] <= 1 and 0 < sc["random"] < 1
    # a model trained on a taught example is not tested on that scene (the generator avoids quiz hashes)
    forbid = {q.hash for q in quiz}
    rng = random.Random(0)
    for _ in range(100):
        s = make_situation(rng, ["THIRSTY"], "x", forbidden=forbid)
        assert obs_hash(observation(s)) not in forbid


def test_pick_daily_only_met_topics_and_spread():
    quiz = make_quiz(random.Random(1), quiz_topics(WM, []), 6)
    picks = pick_daily(quiz, {"THIRSTY", "DARK"}, random.Random(0), 6)
    assert len(picks) == 6 and {q.topic for q in picks} == {"THIRSTY", "DARK"}
    assert sum(q.topic == "THIRSTY" for q in picks) == 3
    assert pick_daily(quiz, set(), random.Random(0), 6) == []
    assert len(pick_daily(quiz, {"DARK"}, random.Random(0), 99)) == 6  # never more than exist


def test_tag_reliance_swaps_only_the_tag():
    quiz = make_quiz(random.Random(2), quiz_topics(WM, []), 4)
    m = build_model(ModelCfg())
    r = tag_reliance(m, quiz)
    assert r["n"] == 20 and 0 <= r["intact"] <= 1 and 0 <= r["swapped"] <= 1
