"""The child learns what happens, not which option is right (docs/WORLD.md 22.14). The mode is a module-level switch
(GROWING_CONSEQUENCE, on by default; =0 is the older child), so each check runs in a process of its own."""
import json
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
HEAD = """
import json, random, tempfile
from server.config import GameCfg
from server.game.session import Session
from server.learning.codec import CONSEQUENCE, RESULTS, TOK, encode_try, option_probs
from server.learning.learner import OnlineCfg
from server.world.situation import observation, values
from server.world.ontology import outcome_label

def scene(steps=4, **game):
    s = Session.new(tempfile.mkdtemp() + "/s", seed=0, game=GameCfg(per_day=3, quiz_per_topic=2, **game), ocfg=OnlineCfg(steps=steps))
    s.draw_cards(); s.pick_card(0)
    return s, s.state.current, values(s.state.current)
"""


def run(body: str, flag: bool = True) -> dict:
    env = {**os.environ, "GROWING_LLM": "mock", "PYTHONPATH": str(ROOT)}
    env.pop("GROWING_UNTESTED", None)
    env["GROWING_CONSEQUENCE"] = "1" if flag else "0"
    out = subprocess.run([sys.executable, "-c", HEAD + body], env=env, capture_output=True, text=True, cwd=ROOT)
    assert out.returncode == 0, out.stderr[-800:]
    return json.loads(out.stdout.strip().splitlines()[-1])


def test_the_act_is_encoded_without_the_options_and_the_parents_word_is_one_token():
    got = run("""
s, cur, v = scene()
obs = observation(cur)
o = obs.options[0]
a, b = encode_try(obs, o, False), encode_try(obs, o, True)
print(json.dumps({"opts": any(t == TOK["OPT_0"] for t in a), "said_a": TOK["SAID"] in a, "said_b": TOK["SAID"] in b, "do": TOK["ACT_DO"] in a,
                  "same_otherwise": [t for t in b if t != TOK["SAID"]] == a}))
""")
    assert got == {"opts": False, "said_a": False, "said_b": True, "do": True, "same_otherwise": True}


def test_each_teaching_act_leaves_facts_of_what_happened_whatever_it_was():
    got = run("""
s, cur, v = scene()
worst = min(range(len(v)), key=v.__getitem__)
s.command(worst)
r = s.show(worst)
f = s.buffer[-1].facts
print(json.dumps({"n": len(f), "result": f[0].result, "label": outcome_label(v[worst]), "said": f[0].said, "negative": s.buffer[-1].negative}))
""")
    assert got["n"] == 1 and got["result"] == got["label"] and got["said"] is True and got["negative"] is False


def test_the_childs_own_act_is_learned_when_it_does_it_and_the_scene_is_over():
    got = run("""
s, cur, v = scene()
r = s.try_child()
f = s.buffer[-1].facts
print(json.dumps({"n": len(s.buffer), "facts": len(f), "result": f[0].result, "label": outcome_label(v[r.option]), "said": f[0].said,
                  "phase": s.state.phase.value, "done": s.state.situations_done}))
""")
    assert got["n"] == 1 and got["facts"] == 1 and got["result"] == got["label"] and got["said"] is False and got["done"] == 1


def test_following_the_parents_word_is_marked_in_the_childs_own_act():
    got = run("""
s, cur, v = scene()
s.command(0)
r = s.try_child()
print(json.dumps({"followed": r.option == 0, "said": s.buffer[-1].facts[0].said}))
""")
    assert got["followed"] == got["said"]


def test_what_the_child_has_seen_happen_is_what_it_expects_and_picks():
    got = run("""
s, cur, v = scene(steps=40)
obs = observation(cur)
good = max(range(len(v)), key=v.__getitem__)
bad = min(range(len(v)), key=v.__getitem__)
before = option_probs(s.model, obs)
if good == bad:
    print(json.dumps({"skip": True}))
else:
    s.show(good)
    after = option_probs(s.model, obs)
    print(json.dumps({"skip": False, "gain": after[good] - before[good]}))
""")
    assert got.get("skip") or got["gain"] > 0.2  # having seen it work, the child leans to it


def test_doing_as_the_parent_said_and_being_hurt_is_learned_longer():
    got = run("""
s, cur, v = scene(burn_worse_rounds=4, burn_nothing_rounds=3)
bad = [i for i, x in enumerate(v) if x < 0]
if not bad:
    bad = [i for i, x in enumerate(v) if x < max(v)]
s.command(bad[0])
s.show(bad[0])
print(json.dumps({"strength": s.buffer[-1].strength, "result": s.buffer[-1].facts[0].result}))
""")
    assert got["strength"] == (4.0 if got["result"] == "worse" else 3.0 if got["result"] == "nothing" else 1.0)
