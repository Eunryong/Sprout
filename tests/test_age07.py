"""The 0-7 years world (config/ontology_age07.yaml): the MVP of the web game. Each test runs in its own process: the ontology is a module global."""
import json
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def in_age07(code: str) -> dict:
    env = {**os.environ, "GROWING_ONTOLOGY": str(ROOT / "config" / "ontology_age07.yaml"), "GROWING_LLM": "mock", "PYTHONPATH": str(ROOT)}
    r = subprocess.run([sys.executable, "-c", code], env=env, capture_output=True, text=True, cwd=ROOT, timeout=400)
    assert r.returncode == 0, r.stderr[-800:]
    return json.loads(r.stdout.strip().splitlines()[-1])


def test_the_world_loads_and_every_prose_table_covers_it():
    r = in_age07("""
import json
from server.agents import glossary, narrator
glossary.check_complete()
from server.world.ontology import PROBLEMS, VERBS, REQUEST_KINDS, STAGES, age_months, GAME
from server.config import GameCfg
miss = [t for t in PROBLEMS if t not in narrator.PROBLEM_LINES or t not in narrator.STILL_LINES or t not in narrator.SOLVED_BY_TAG or t not in narrator.PROBLEM_VARIANTS]
miss += [v for v in VERBS if v not in narrator.VERB_PHRASES]
print(json.dumps({"problems": len(PROBLEMS), "miss": miss, "words": len(REQUEST_KINDS) <= len(GameCfg().words), "stages": [s["name"] for s in STAGES],
                  "end_age": age_months(GAME["days"] + 1)}))
""")
    assert r["problems"] == 15 and r["miss"] == [] and r["words"]
    assert r["stages"] == ["Baby", "Toddler", "Little kid", "Schoolchild"] and r["end_age"] == 84  # the story ends at seven


def test_every_problem_makes_valid_scenes():
    r = in_age07("""
import json, random
from server.world.generator import make_situation
from server.world.ontology import PROBLEMS
from server.world.situation import values
rng = random.Random(1)
bad = []
for t in PROBLEMS:
    for _ in range(60):
        s = make_situation(rng, [t]); v = values(s)
        if not (max(v) == 2 and min(v) < 2 and 3 <= len(s.options) <= 5):
            bad.append(t)
print(json.dumps({"bad": sorted(set(bad))}))
""")
    assert r["bad"] == []


def test_a_whole_life_from_birth_to_seven():
    r = in_age07("""
import json, tempfile, pathlib
from server.game.simple import SimpleGame
from server.learning.learner import OnlineCfg
from server.world.ontology import age_months, open_problems
g = SimpleGame.new(pathlib.Path(tempfile.mkdtemp()) / "g", seed=2, ocfg=OnlineCfg(steps=1))
first = g.scene()
baby_gate = ""
try:
    g.play("say", 0)
except ValueError as e:
    baby_gate = str(e)
tags_at, stages_first, n = [], [], 0
while not g.ended and n < 200:
    sc = g.scene()
    if g.ended:
        break
    from server.world.situation import observation
    tags_at.append((sc.age_months, list(g.s.state.current.tags)))
    if sc.stage_first:
        stages_first.append(sc.stage["name"])
    g.play("show", 0) if not sc.can_act or n % 2 else g.play("leave")
    n += 1
s = g.summary()
late_baby = [a for a, t in tags_at if "BABY_HUNGRY" in t and a >= 12]
early = [a for a, t in tags_at if ({"SCHOOL_MORNING", "FRIEND_SAD"} & set(t)) and a < 60]
try:
    g.play("show", 0); after_end = ""
except ValueError as e:
    after_end = str(e)
print(json.dumps({"first": [first.age_months, first.stage["name"], first.can_act], "gate": baby_gate, "n": n, "ended": g.ended, "stages_first": stages_first,
                  "late_baby": late_baby, "early": early, "summary_stages": [x["name"] for x in s.stages], "scenes": s.scenes, "after_end": after_end,
                  "moments": len(s.moments), "age": s.age}))
""")
    assert r["first"] == [0, "Baby", False] and "only show" in r["gate"]
    assert r["n"] == 84 and r["ended"] and r["scenes"] == 84 and r["age"] == "7 years"
    assert r["stages_first"] == ["Baby", "Toddler", "Little kid", "Schoolchild"]  # each stage is introduced once, when it begins
    assert r["late_baby"] == [] and r["early"] == []  # a baby's bottle does not come back, school troubles wait for school
    assert r["summary_stages"] == ["Baby", "Toddler", "Little kid", "Schoolchild"] and r["moments"] >= 8 and "over" in r["after_end"]


def test_the_baby_is_the_one_cared_for_and_nothing_is_done_for_grandma():
    r = in_age07("""
import json, tempfile, pathlib
from server.game.simple import SimpleGame
from server.learning.learner import OnlineCfg
from server.agents.director import ontology_payload
g = SimpleGame.new(pathlib.Path(tempfile.mkdtemp()) / "g", seed=1, ocfg=OnlineCfg(steps=1))
texts, babies = [], 0
while True:
    sc = g.scene()
    if sc.can_act:
        break
    texts += [sc.narration] + [o.text for o in sc.options]
    texts.append(g.play("show", 0).narration)
    babies += 1
toddler = [sc.narration] + [o.text for o in sc.options]
print(json.dumps({"babies": babies, "baby_text": texts, "toddler_text": toddler, "age": sc.age_months,
                  "baby_gloss": json.dumps({"p": {k: ontology_payload("baby")["problems"][k] for k in ("BABY_HUNGRY", "UPSET", "COLD")},
                                             "v": ontology_payload("baby")["verbs"]}), "plain_gloss": json.dumps(ontology_payload())}))
""")
    baby = " ".join(r["baby_text"]).lower()
    assert r["babies"] == 12 and r["age"] == 12
    assert "grandma" not in baby and "visitor" not in baby and "baby" in baby  # everything is about the baby
    assert "grandma" in r["plain_gloss"].lower() and "grandma" not in r["baby_gloss"].lower()  # the Director is told whose scene it is


def test_only_sensible_attempts_are_options_and_a_passed_trouble_is_not_continued():
    r = in_age07("""
import json, random
from server.world.generator import make_situation
from server.world.ontology import PROBLEMS, afforded, AFFORDANCES
from server.world.situation import Obj, OptionSpec, Situation, BadSituation, validate_situation
from server.world.state import allowed_followup_tags
rng = random.Random(7)
senseless = 0
for t in PROBLEMS:
    for _ in range(80):
        s = make_situation(rng, [t])
        validate_situation(s)
        senseless += sum(1 for o in s.options if o.target is not None and not afforded(o.verb, s.objects[o.target].kind))
cloth = Obj(kind="cloth", color="red", shape="flat", size="big", material="fabric")
teddy = Obj(kind="teddy", color="red", shape="soft", size="big", material="plush")
blanket = Obj(kind="blanket", color="red", shape="flat", size="big", material="wool")
bad = Situation(id="x", tags=["COLD"], location="house", objects=[blanket, cloth, teddy],
                options=[OptionSpec(verb="COVER", target=0), OptionSpec(verb="WASH", target=2), OptionSpec(verb="WAIT")])
try:
    validate_situation(bad); rejected = ""
except BadSituation as e:
    rejected = str(e)
print(json.dumps({"senseless": senseless, "rejected": rejected, "baby_followup_at_toddler": allowed_followup_tags("BABY_HUNGRY", "nothing", 3),
                  "baby_followup_at_baby": allowed_followup_tags("BABY_HUNGRY", "nothing", 1), "verbs": len(AFFORDANCES)}))
""")
    assert r["senseless"] == 0 and "WASH cannot be done with a teddy" in r["rejected"]
    assert r["baby_followup_at_toddler"] == [] and r["baby_followup_at_baby"] == [["BABY_HUNGRY"]]


def test_no_unmet_need_is_brought_back_by_force():
    r = in_age07("""
import json
from server.world.ontology import LINGERS
print(json.dumps({"lingers": LINGERS}))
""")
    assert r["lingers"] == []


def test_a_memory_tells_the_act_in_the_words_of_the_scene_it_was_lived_in():
    r = in_age07("""
import json, tempfile, pathlib
from server.game.simple import SimpleGame
from server.learning.learner import OnlineCfg
g = SimpleGame.new(pathlib.Path(tempfile.mkdtemp()) / "g", seed=1, ocfg=OnlineCfg(steps=1))
baby_acts, later = [], []
while True:
    sc = g.scene()
    if g.ended:
        break
    if sc.can_act:
        later += [m for m in (o.memory for o in sc.options) if m]
    t = g.play("show", 0) if not sc.can_act else g.play("leave")
    if not sc.can_act:
        baby_acts.append(t.remembers)
print(json.dumps({"baby": baby_acts, "later": later}))
""")
    assert not any("Grandma" in m for m in r["baby"])  # in the baby's life nothing is done for Grandma
    assert any("the baby" in m for m in r["later"])  # and later the child recalls it as it was: done to the baby


def test_a_babys_trouble_is_only_offered_things_a_person_would_try():
    r = in_age07("""
import json, random
from server.world.ontology import PLAUSIBLE, RULES
from server.world.generator import make_situation
rng, bad, n = random.Random(1), [], 0
for tag in ("BABY_HUNGRY", "UPSET", "COLD"):
    for _ in range(300):
        s = make_situation(rng, [tag])
        for o in s.options:
            if o.target is not None:
                pair = (o.verb, s.objects[o.target].kind)
                n += 1
                if pair not in PLAUSIBLE[tag] and pair not in RULES[tag]:
                    bad.append([tag, *pair])
print(json.dumps({"bad": bad[:5], "n": n}))
""")
    assert r["bad"] == [] and r["n"] > 2000


def test_milk_soothes_an_upset_baby_a_little():
    r = in_age07("""
import json
from server.world.ontology import RULES
print(json.dumps({"v": RULES["UPSET"][("FEED", "bottle")]}))
""")
    assert r["v"] == 1


def test_the_life_has_three_points_of_view_and_a_toddlers_trouble_is_its_own():
    r = in_age07("""
import json, random
from server import i18n
from server.agents import narrator
from server.world.ontology import voice_of
from server.world.generator import make_situation
voices = {m: voice_of(m) for m in (0, 11, 12, 35, 36, 59, 60, 83)}
out = {}
for lang in ("ko", "en"):
    i18n.set_lang(lang)
    bad = []
    for tag in ("THIRSTY", "COLD", "HUNGRY"):
        sit = make_situation(random.Random(3), [tag])
        texts = [narrator.intro(sit, False, (), "self")[0], narrator.after_words([tag], "solved", "self", "DRINK"), narrator.after_words([tag], "nothing", "self", "WAIT")]
        texts += [narrator.option_text(o.verb, sit.objects[o.target] if o.target is not None else None, "self") for o in sit.options]
        bad += [t for t in texts if "할머니" in t or "Grandma" in t]
    out[lang] = bad
print(json.dumps({"voices": voices, "bad": out}))
""")
    assert r["voices"] == {"0": "baby", "11": "baby", "12": "self", "35": "self", "36": "self", "59": "self", "60": "grandma", "83": "grandma"}
    assert r["bad"] == {"ko": [], "en": []}


def test_an_older_childs_trouble_never_offers_a_bottle_unless_a_rule_says_so():
    r = in_age07("""
import json, random
from server.world.ontology import RULES, OPEN_MONTH
from server.world.generator import make_situation
rng, bad = random.Random(2), 0
for tag in [t for t in RULES if OPEN_MONTH[t] >= 12]:
    for _ in range(100):
        s = make_situation(rng, [tag])
        for o in s.options:
            if o.target is not None and (o.verb == "FEED" or s.objects[o.target].kind == "bottle") and (o.verb, s.objects[o.target].kind) not in RULES[tag]:
                bad += 1
print(json.dumps({"bad": bad}))
""")
    assert r["bad"] == 0


def test_a_thing_that_is_no_part_of_the_answer_is_offered_only_once():
    r = in_age07("""
import json, random
from server.world.generator import make_situation
from server.world.ontology import NEEDS, PROBLEMS
bad = 0
for t in PROBLEMS:
    if t == "REQUEST": continue
    for s in range(120):
        sit = make_situation(random.Random(s), [t])
        good = NEEDS[t][1]
        wrong = [sit.objects[o.target].kind for o in sit.options if o.target is not None and sit.objects[o.target].kind != good]
        bad += len(wrong) - len(set(wrong))
print(json.dumps({"repeats": bad}))
""")
    assert r["repeats"] == 0


def test_age_three_has_two_steps_per_scene_with_speech_and_causal_memory():
    r = in_age07("""
import json, tempfile, pathlib
from server import i18n
from server.game.simple import SimpleGame
from server.learning.learner import OnlineCfg
i18n.set_lang("ko")
p = pathlib.Path(tempfile.mkdtemp()) / "two"
g = SimpleGame.new(p, seed=2, ocfg=OnlineCfg(steps=2), two_step=True)
while True:
    sc = g.scene()
    if sc.age_months >= 36:
        break
    g.play("show", 0)
lived_at_3 = g.lived
sc1 = g.scene()
t1 = g.play("leave", None)
sc2 = g.scene()
t2 = g.play("solo", 0)
sc3 = g.scene()
g2 = SimpleGame.load(p, two_step=True)
print(json.dumps({
    "lived_at_3": lived_at_3,
    "sc1": [sc1.number, sc1.step, sc1.step_total, all(bool(o.speech and o.text) for o in sc1.options)],
    "t1": [t1.step, t1.step_total, t1.has_next_step, bool(t1.child_said), "성공" not in t1.remembers and "실패" not in t1.remembers],
    "sc2": [sc2.number, sc2.step, sc2.step_total, all(bool(o.speech and o.text) for o in sc2.options)],
    "t2": [t2.step, t2.step_total, t2.has_next_step, bool(t2.child_said), "성공" not in t2.remembers and "실패" not in t2.remembers],
    "sc3_num": sc3.number,
    "lived_after": g.lived,
    "reloaded_lived": g2.lived,
}))
""")
    assert r["lived_at_3"] == 36
    assert r["sc1"] == [37, 1, 2, True]
    assert r["t1"] == [1, 2, True, True, True]
    assert r["sc2"] == [37, 2, 2, True]
    assert r["t2"] == [2, 2, False, True, True]
    assert r["sc3_num"] == 38 and r["lived_after"] == 37 and r["reloaded_lived"] == 37
