"""Parent instruction token (docs/WORLD.md section 18) and the E-A harness."""
import random
import sys
from pathlib import Path

import pytest

from server.learning.codec import MAX_LEN, TOK, VOCAB, Example, augment, encode, redraw_options
from server.world.generator import make_situation
from server.world.situation import obs_hash, observation, optimal, values

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import exp_ea  # noqa: E402


def scene(seed=0, tag="COLD"):
    return make_situation(random.Random(seed), [tag])


def test_no_instruction_encodes_exactly_as_before():
    obs = observation(scene())
    ids = encode(obs)
    assert TOK["CMD"] not in ids and len(ids) <= MAX_LEN
    assert ids[0] == TOK["[BOS]"] and ids[-1] == TOK["ACT_PICK"]


def test_instruction_is_the_verb_and_attributes_of_one_option():
    s = scene()
    obs = observation(s)
    opt = obs.options[1]
    with_cmd = obs.model_copy(update={"cmd": opt})
    ids = encode(with_cmd)
    i = ids.index(TOK["CMD"])
    assert ids[i + 1] == TOK[f"VERB_{opt.verb}"]
    if opt.target is not None:
        t = opt.target
        assert ids[i + 2:i + 6] == [TOK[f"COLOR_{t.color}"], TOK[f"SHAPE_{t.shape}"], TOK[f"SIZE_{t.size}"], TOK[f"MAT_{t.material}"]]
    assert len(ids) <= MAX_LEN
    assert len(encode(with_cmd)) > len(encode(obs))


def test_instruction_carries_no_quality_value_or_reaction():
    obs = observation(scene())
    dumped = obs.model_copy(update={"cmd": obs.options[0]}).model_dump()
    assert set(dumped["cmd"]) == {"verb", "target"}
    assert not any(w in " ".join(VOCAB).lower() for w in ("good", "useless", "harmful", "praise", "obey", "refuse"))


def test_quiz_isolation_ignores_the_instruction():
    obs = observation(scene())
    assert obs_hash(obs) == obs_hash(obs.model_copy(update={"cmd": obs.options[2]}))


@pytest.mark.parametrize("seed", range(20))
def test_augmentation_keeps_the_instructed_option_among_the_options(seed):
    rng = random.Random(seed)
    s = scene(seed, rng.choice(["COLD", "DARK", "THIRSTY", "VISITOR", "FLOOR_WET"]))
    obs = observation(s)
    idx = rng.randrange(len(obs.options))
    chosen = rng.choice([i for i in range(len(obs.options)) if i != idx] or [idx])
    ex = Example(id="x", day=1, source="correct", obs=obs.model_copy(update={"cmd": obs.options[idx]}), chosen=chosen)
    for _ in range(10):
        out = augment(ex, rng, 1.0)
        assert out.obs.cmd == ex.obs.cmd and out.obs.cmd in out.obs.options
        assert out.obs.options[out.chosen] == ex.obs.options[chosen]
    assert redraw_options(ex, rng).obs.cmd in redraw_options(ex, rng).obs.options


def test_instruction_quality_comes_from_the_rules_table():
    s = scene(3, "FLOOR_WET")
    v = values(s)
    q = exp_ea.kind_of_quality(v, s.options)
    assert set(q["good"]) == set(optimal(s))
    assert all(v[i] == 0 and s.options[i].verb != "WAIT" for i in q["useless"])
    assert all(v[i] < 0 for i in q["harmful"])
