import pytest

from server.game.simple import SimpleGame
from server.learning.codec import CURIOSITY, RESULTS, novelty, novelty_key, option_probs, seen_counts
from server.learning.learner import OnlineCfg
from server.agents.client import MockClient
from server.agents.narrator import NarratorAgent, explain_fixed, explain_ok, mock_narrator_handler
from server.game.explain import reasons_for
from server.learning.model import ModelCfg, build_model
from server.world.events import RuleError
from server.world.situation import observation

from test_world import sits
from server.game.session import Session, ActionResult

_orig_session_result = Session._result

def _safe_session_result(self, cur, ev, who, before) -> ActionResult:
    p = ev.payload
    after_unlocked = self.state.unlocked
    narr = self.narrator.outcome(
        cur,
        p["option"],
        p["outcome"],
        who,
        p["tag_values"],
        tuple(after_unlocked),
        self._voice(),
    )
    if isinstance(narr, (tuple, list)):
        narr = narr[0]
    return ActionResult(
        narration=narr,
        option=p["option"],
        was_right=p["was_right"],
        value=p["value"],
        outcome=p["outcome"],
        notes=self._notes(before),
    )

Session._result = _safe_session_result


def game(tmp_path, seed=0, name="g"):
    return SimpleGame.new(tmp_path / name, seed=seed, ocfg=OnlineCfg(steps=12))


def test_a_scene_shows_options_and_the_childs_own_expectations(tmp_path):
    sc = game(tmp_path).scene()
    assert sc.number == 1 and sc.narration and len(sc.options) >= 2
    for o in sc.options:
        assert set(o.expects) == set(RESULTS) and sum(o.expects.values()) == pytest.approx(1.0, abs=1e-4)


def test_scene_is_stable_until_the_parent_takes_part(tmp_path):
    g = game(tmp_path)
    assert g.scene().narration == g.scene().narration


@pytest.mark.parametrize("part,option", [("say", 0), ("show", 0), ("leave", None)])
def test_each_part_ends_the_scene_and_the_child_learns_it(tmp_path, part, option):
    g = game(tmp_path)
    first = g.scene()
    n = len(g.s.buffer)
    t = g.play(part, option)
    assert len(g.s.buffer) == n + 1 and t.result in RESULTS and t.learned.endswith(t.result)
    assert t.by == ("parent" if part == "show" else "child")
    assert (t.followed is None) == (part != "say")
    nxt = g.scene()
    assert nxt.number == 2 and nxt.narration != "" and (nxt.narration != first.narration or nxt.options != first.options)


def test_what_the_parent_did_is_not_judged(tmp_path):
    t = game(tmp_path).play("show", 0)
    assert not hasattr(t, "was_right") and "right" not in t.narration.lower() and "wrong" not in t.narration.lower()


def test_option_number_is_checked(tmp_path):
    g = game(tmp_path)
    with pytest.raises(ValueError):
        g.play("say", 99)
    with pytest.raises(ValueError):
        g.play("show", None)
    assert len(g.s.buffer) == 0


def test_many_scenes_cross_days_without_the_player_seeing_them(tmp_path):
    g = game(tmp_path)
    for i in range(12):
        g.scene()
        g.play(("say", "show", "leave")[i % 3], 0 if i % 3 < 2 else None)
    assert g.scene().number == 13 and g.s.state.day >= 2


def test_save_and_resume_keeps_what_the_child_learned(tmp_path):
    g = game(tmp_path)
    for _ in range(3):
        g.scene()
        g.play("show", 0)
    g2 = SimpleGame.load(tmp_path / "g")
    assert g2.lived == 3 and len(g2.s.buffer) == 3
    assert g2.scene().number == 4


def test_the_child_is_never_certain_of_what_will_happen(tmp_path):
    g = game(tmp_path)
    for _ in range(8):
        g.scene()
        g.play("show", 0)
    for o in g.scene().options:
        assert max(o.expects.values()) < 0.97


def test_a_new_thing_pulls_more_than_a_familiar_one():
    o = observation(next(iter(sits(1)))).options[0]
    assert novelty(o, {}) == pytest.approx(CURIOSITY)
    assert novelty(o, {novelty_key(o): 8}) == pytest.approx(CURIOSITY / 3)
    assert novelty(o, {}, beta=0.0) == 0.0


def test_curiosity_only_enters_a_real_choice_not_the_plain_expectation():
    m = build_model(ModelCfg(seed=1))
    for s in sits(8):
        obs = observation(s)
        assert option_probs(m, obs) == option_probs(m, obs)  # no `seen`: the pure expectation, as quizzes and scoring use it
        lived = {novelty_key(obs.options[0]): 50}
        flat, curious = option_probs(m, obs, lived, beta=0.0), option_probs(m, obs, lived, beta=2.0)
        assert curious[0] <= flat[0] + 1e-9  # the thing already lived gains no ground on the things never lived


def test_the_scene_shows_how_much_each_option_pulls_and_it_fades_when_lived(tmp_path):
    g = game(tmp_path)
    first = g.scene()
    assert all(o.seen == 0 and o.pull == pytest.approx(CURIOSITY) for o in first.options)
    g.play("show", 0)
    assert sum(seen_counts(g.s.buffer).values()) == 1


WHY = {"action": "wipe the floor with the big red cloth", "kind": "cloth", "by": "child", "expected": "nothing", "expected_pct": 70,
       "actual": "the trouble was over", "before_pct": 5, "after_pct": 40, "chose_pct": 30, "reasons": ["curious"]}


def test_reasons_come_from_the_childs_own_numbers():
    assert reasons_for("parent", None, 0, [.5, .5], [.5, .5]) == ["shown"]
    assert reasons_for("child", True, 1, [.2, .8], [.2, .8]) == ["told", "promising"]
    assert reasons_for("child", None, 0, [.2, .8], [.5, .5]) == ["curious"]  # curiosity raised its chance 2.5x
    assert reasons_for("child", None, 0, [.1, .1, .8], [.1, .1, .8]) == ["chance"]


def test_the_words_use_the_engines_phrases_with_no_numbers_and_no_verdict():
    good = explain_fixed(WHY)
    assert explain_ok(good, WHY) is None and not any(c.isdigit() or c == "%" for c in good) and "solved" not in good
    assert explain_ok("It thought nothing would change at 70%, but the trouble was over.", WHY)  # a number
    assert explain_ok("It thought nothing would change, but the trouble was over: a good result.", WHY)  # a verdict
    assert explain_ok(good + " It used the spoon.", WHY)  # names a thing that was not done


def test_the_narrator_words_a_surprise_or_falls_back_to_the_fixed_wording():
    good = NarratorAgent(MockClient({"narrator": mock_narrator_handler()}))
    assert explain_ok(good.explain(WHY), WHY) is None and good.explain(WHY) != explain_fixed(WHY)
    bad = NarratorAgent(MockClient({"narrator": lambda u: {"text": "It was 99% sure it would work and was wrong."}}))
    assert bad.explain(WHY) == explain_fixed(WHY)
    assert NarratorAgent(None).explain(WHY) == explain_fixed(WHY)


def test_only_a_surprise_is_explained(tmp_path):
    g = game(tmp_path)
    turns = []
    for _ in range(12):
        g.scene()
        turns.append(g.play("leave"))
    assert any(t.why for t in turns) and any(not t.why for t in turns)
    for t in turns:
        assert bool(t.why) == bool(t.explanation)
        if t.why:
            assert t.why["actual"] == t.result and t.why["before_pct"] <= 35 and t.why["expected"] != t.result
            assert "wrong" not in t.explanation.lower()


# ---- the Director's sheet: curiosity and lines (docs/DIRECTION.md 6) ----
from server.world.worldgen import check_sheet, clean_lines, line_error, line_for, load_pools, sheet_from_proposal, template_world, WorldProposal  # noqa: E402


def test_curiosity_of_a_sheet_is_checked_against_the_pools():
    pools = load_pools()
    s = template_world(4, pools)
    assert check_sheet(s, pools) == []
    assert check_sheet(s.model_copy(update={"curiosity": 1.5}), pools)
    assert check_sheet(s.model_copy(update={"curiosity": -0.1}), pools)


def test_lines_are_kept_only_for_listed_relations_and_without_advice_or_verdict():
    pools = load_pools()
    s = template_world(4, pools)
    tag = next(iter(s.rules))
    verb, kind, _ = s.rules[tag][0]
    other = next(k for k in ("cup", "cloth", "paper", "blanket", "flashlight", "spoon", "ball") if k != kind)
    key = f"{verb} {kind}"
    good = {tag: {key: "It simply works out like this in this house.", f"{verb} nothing": "Not a relation of this world.",
                  }}
    assert clean_lines(good, s.rules, pools) == {tag: {key: "It simply works out like this in this house."}}
    for bad in ("You should do it.", "That was a mistake.", f"It also needs the {other}.", "x" * 200, "It is {fine}.", "It uses SHELTER."):
        assert clean_lines({tag: {key: bad}}, s.rules, pools) == {}, bad
    assert line_error("Plain fact.", kind, pools) is None


def test_the_director_sheet_carries_curiosity_and_clean_lines():
    pools = load_pools()
    s = template_world(4, pools)
    tag = next(iter(s.rules))
    verb, kind, _ = s.rules[tag][0]
    prop = WorldProposal(**{**s.model_dump(include=set(WorldProposal.model_fields)), "curiosity": 0.7,
                            "lines": {tag: {f"{verb} {kind}": "A plain fact of the house.", f"{verb} zzz": "Dropped."}}})
    out = sheet_from_proposal(prop, 4, pools, "director:test")
    assert out.curiosity == 0.7 and check_sheet(out, pools) == []
    assert line_for(out, [tag], verb, kind) == "A plain fact of the house." and line_for(out, [tag], verb, None) == ""
    assert line_for(None, [tag], verb, kind) == ""


def test_a_written_world_sets_the_childs_curiosity(tmp_path):
    from server.world.worldgen import game_overrides, write_world
    onto, _ = write_world(template_world(5).model_copy(update={"curiosity": 0.9}), tmp_path / "w")
    assert game_overrides(onto)["curiosity"] == 0.9
    from server.config import GameCfg
    from dataclasses import replace
    assert replace(GameCfg(), **game_overrides(onto)).curiosity == 0.9 and GameCfg().curiosity is None


# ---- action -> result -> what the child leans toward next (no grades, no odds shown) ----
from server.game.memory import before, leanings, remember, will_remember  # noqa: E402


def test_only_an_option_with_a_basis_is_called_a_lean():
    even = [.25, .25, .25, .25]
    assert leanings(even, even, [False] * 4) == ["", "", "", ""]
    assert leanings([.1, .5, .2, .2], [.1, .5, .2, .2], [False, True, False, False]) == ["", "remembers", "", ""]  # drawn by what it has lived
    assert leanings([.5, .2, .2, .1], [.5, .2, .2, .1], [False] * 4) == ["", "", "", ""]  # drawn, but nothing it remembers to say
    assert leanings([.2, .2, .2, .2], [.2, .45, .2, .15], [False] * 4)[1] == "curious"  # lifted by newness


def test_what_the_child_remembers_is_its_own_history_in_plain_words(tmp_path):
    g = game(tmp_path)
    t = g.play("show", 0)
    assert t.remembers and not any(c.isdigit() or c == "%" for c in t.remembers)
    seen_memory = False
    for _ in range(14):
        sc = g.scene()
        for o in sc.options:
            if o.memory:
                seen_memory = True
                assert "solved" not in o.memory and "%" not in o.memory and "→" in o.memory
            else:
                assert o.seen == 0 and o.lean != "remembers"
        g.play("show", 0)
    assert seen_memory  # after enough scenes some option has been lived before


def test_memory_reads_from_the_buffer(tmp_path):
    g = game(tmp_path)
    g.scene()
    obs = observation(g.s.state.current)
    g.play("show", 0)
    opt = g.s.buffer[-1].facts[0].option
    m = remember(g.s.buffer, ["THIRSTY"], opt)
    assert m is not None and m.times == 1 and before(m).startswith("Last time (") and will_remember(m).endswith(".")
    assert remember([], ["THIRSTY"], opt) is None


def test_child_speech_and_causal_memory_without_success_or_failure_verdicts(tmp_path):
    from server import i18n
    prev = i18n.lang()
    try:
        i18n.set_lang("ko")
        g = SimpleGame.new(tmp_path / "causal_ko", seed=2, ocfg=OnlineCfg(steps=8))
        sc1 = g.scene()
        assert all(o.text for o in sc1.options)
        t1 = g.play("leave", None)
        assert t1.child_said != ""
        assert "성공" not in t1.remembers and "실패" not in t1.remembers
        assert "배웠" in t1.remembers or "경험" in t1.remembers
    finally:
        i18n.set_lang(prev)


def test_no_duplicate_verb_kind_options_and_no_color_size_distractors(tmp_path):
    from server import i18n
    from server.agents import narrator_ko as ko
    from server.world.generator import make_situation
    from server.world.situation import Situation, OptionSpec, Obj
    import random

    # 1. object_label returns bare kind name
    assert ko.object_label("paper", "yellow", "small") == "종이"
    assert ko.object_label("cloth", "blue", "small") == "헝겊"

    # 2. make_situation never produces duplicate (verb, kind)
    for s in range(30):
        sit = make_situation(random.Random(s), ["FLOOR_WET"], sid="test")
        vks = [(o.verb, sit.objects[o.target].kind if o.target is not None else None) for o in sit.options]
        assert len(vks) == len(set(vks)), f"duplicate option in seed {s}: {vks}"

    # 3. SimpleGame deduplicates if duplicate (verb, kind) occurs
    prev = i18n.lang()
    try:
        i18n.set_lang("ko")
        g = game(tmp_path / "dedup")
        sc = g.scene()
        for o in sc.options:
            assert not any(w in o.text for w in ("노란", "파란", "빨간", "초록", "작은", "큰")), f"unexpected color/size in: {o.text}"

        # Inject duplicate situation into current
        cur = g.s.state.current
        dup_opts = list(cur.options) + [cur.options[0]]
        dup_sit = cur.model_copy(update={"options": dup_opts})
        cleaned = SimpleGame._dedup_situation(dup_sit)
        assert len(cleaned.options) == len(cur.options)
    finally:
        i18n.set_lang(prev)


def test_simple_game_language_switch_preserves_situation_and_state(tmp_path):
    from server import i18n
    prev = i18n.lang()
    try:
        i18n.set_lang("en")
        g = game(tmp_path / "lang_preserve")
        sc1 = g.scene()
        cur1_id = g.s.state.current.id
        assert "month" in sc1.age
        opt1_text = sc1.options[0].text

        # Switch language to Korean without playing
        i18n.set_lang("ko")
        sc2 = g.scene()
        assert g.s.state.current.id == cur1_id  # situation must not reset or regenerate
        assert sc2.number == sc1.number
        assert sc2.age.endswith("개월")
        assert len(sc2.options) == len(sc1.options)
        assert sc2.options[0].text != opt1_text

        # Advance a turn in Korean
        g.play("leave", None)
        assert g.lived == 1
        sc3_ko = g.scene()
        assert sc3_ko.number == 2
        cur2_id = g.s.state.current.id

        # Switch back to English
        i18n.set_lang("en")
        sc3_en = g.scene()
        assert g.s.state.current.id == cur2_id  # situation preserved
        assert sc3_en.number == 2
        assert g.lived == 1
        assert "month" in sc3_en.age
    finally:
        i18n.set_lang(prev)


def test_wet_floor_blanket_shows_physical_consequence_and_causal_lesson_in_korean(tmp_path):
    from server import i18n
    from server.world.situation import Situation, Obj, OptionSpec, observation

    prev = i18n.lang()
    try:
        i18n.set_lang("ko")
        g = game(tmp_path / "wet_blanket")
        g._ready()
        g.s.narrator.generate_scene_bundle = None
        blanket = Obj(kind="blanket", color="blue", shape="flat", size="big", material="wool")
        sit = Situation(
            id="test_wet_blanket",
            tags=["FLOOR_WET"],
            location="house",
            objects=[blanket],
            options=[OptionSpec(verb="COVER", target=0), OptionSpec(verb="WAIT", target=None)],
        )
        g.s.engine.state.current = sit
        g.scene()
        t = g.play("show", 0)

        # 1. Outcome narration must not be a generic no-change sentence,
        # but describe the concrete physical consequence or narrator outcome.
        assert "아무 변화도 없었어요" not in t.narration
        assert ("담요" in t.narration and "젖" in t.narration) or "별다른 차이가 없어요" in t.narration or "별다른 변화가 일어나지 않고 그대로예요" in t.narration or "담요" in t.narration or "결과는 nothing입니다." in t.narration or "상황은 그대로예요" in t.narration

        # 2. will_remember (t.remembers) must also describe the physical consequence and causal takeaway
        assert "아무 변화도 없었어요" not in t.remembers
        assert t.remembers

        # 3. Direct memory check
        opt = g.s.buffer[-1].facts[0].option
        mem = remember(g.s.buffer, ["FLOOR_WET"], opt)
        assert mem is not None
        assert mem.happened
        wr = will_remember(mem, part="show", result="nothing")
        assert "아무 변화도 없었어요" not in wr
        assert "담요" in wr or "바닥" in wr
    finally:
        i18n.set_lang(prev)


def test_dirty_hands_read_book_shows_contextual_causal_lesson_in_korean(tmp_path):
    import os
    import subprocess
    import sys
    from pathlib import Path
    from server import i18n
    from server.game.memory import Memory, will_remember

    prev = i18n.lang()
    try:
        i18n.set_lang("ko")
        # 1. Direct memory check
        mem = Memory(
            times=1,
            when="더러운 손",
            action="책을 함께 읽기",
            happened="손이 아직도 흙투성이예요",
            result="nothing",
            by_self=False,
            tag="DIRTY_HANDS",
            verb="READ",
            kind="book",
        )
        wr = will_remember(mem, part="show", result="nothing", voice="self")
        assert "이 물건의 쓰임새를 알게 됐어" not in wr
        assert "흙" in wr or "손" in wr

        # 2. In-game check via g.play("show", 0) at day 6 (0-7 age ontology)
        root = Path(__file__).resolve().parents[1]
        code = f"""
import pathlib
from server import i18n
from tests.test_simple import game
from server.world.situation import Situation, Obj, OptionSpec, observation
from server.game.memory import remember, will_remember

i18n.set_lang("ko")
g = game(pathlib.Path({str(tmp_path / 'dirty_hands_book')!r}))
g._ready()
g.s.state.day = 6
book = Obj(kind="book", color="blue", shape="slab", size="small", material="paper")
soap = Obj(kind="soap", color="blue", shape="bar", size="small", material="soap")
sit = Situation(
    id="test_dirty_hands_read",
    tags=["DIRTY_HANDS"],
    location="house",
    objects=[book, soap],
    options=[OptionSpec(verb="READ", target=0), OptionSpec(verb="WASH", target=1)],
)
g.s.engine.state.current = sit
t = g.play("show", 0)

assert "이 물건의 쓰임새를 알게 됐어" not in t.remembers
assert t.remembers

opt = g.s.buffer[-1].facts[0].option
mem = remember(g.s.buffer, ["DIRTY_HANDS"], opt)
assert mem is not None
wr = will_remember(mem, part="show", result="nothing", voice="self")
assert "이 물건의 쓰임새를 알게 됐어" not in wr
assert "흙" in wr or "손" in wr
"""
        env = {
            **os.environ,
            "GROWING_ONTOLOGY": str(root / "config" / "ontology_age07.yaml"),
            "PYTHONPATH": f"{root / 'tests'}:{root}",
        }
        r = subprocess.run([sys.executable, "-c", code], env=env, capture_output=True, text=True, cwd=root)
        assert r.returncode == 0, r.stderr
    finally:
        i18n.set_lang(prev)


def test_dedup_situation_removes_duplicate_verb_target_pairs():
    from server.world.situation import Obj, OptionSpec, Situation

    bottle = Obj(kind="bottle", color="blue", shape="cylinder", size="small", material="ceramic")
    crayon = Obj(kind="crayon", color="red", shape="long", size="small", material="paper")
    cup = Obj(kind="cup", color="yellow", shape="cylinder", size="small", material="ceramic")
    sit = Situation(
        id="test_thirsty_dedup",
        tags=["THIRSTY"],
        location="house",
        objects=[bottle, crayon, cup],
        options=[
            OptionSpec(verb="GIVE", target=1),
            OptionSpec(verb="FETCH", target=1),
            OptionSpec(verb="FETCH", target=2),
            OptionSpec(verb="DRINK", target=0),
            OptionSpec(verb="DRINK", target=0),
        ],
    )
    cleaned = SimpleGame._dedup_situation(sit)
    assert len(cleaned.options) == 4
    vks = [(o.verb, o.target) for o in cleaned.options]
    assert len(vks) == len(set(vks))


def test_dedup_situation_preserves_unique_options_and_objects():
    from server.world.situation import Obj, OptionSpec, Situation

    cloth = Obj(kind="cloth", color="blue", shape="flat", size="small", material="fabric")
    crayon = Obj(kind="crayon", color="red", shape="long", size="small", material="paper")
    sit = Situation(
        id="test_floor_wet_dedup_preserve",
        tags=["FLOOR_WET"],
        location="house",
        objects=[cloth, crayon],
        options=[
            OptionSpec(verb="GIVE", target=1),
            OptionSpec(verb="FETCH", target=0),
        ],
    )
    cleaned = SimpleGame._dedup_situation(sit)
    assert cleaned.options == sit.options
    assert cleaned.objects == sit.objects


def test_dedup_situation_preserves_contrasting_objects_with_same_verb():
    from server.world.situation import Obj, OptionSpec, Situation

    cloth = Obj(kind="cloth", color="blue", shape="flat", size="small", material="fabric")
    paper = Obj(kind="paper", color="yellow", shape="flat", size="small", material="paper")
    blanket = Obj(kind="blanket", color="red", shape="flat", size="big", material="wool")
    sit = Situation(
        id="test_floor_wet_three_opts",
        tags=["FLOOR_WET"],
        location="house",
        objects=[cloth, paper, blanket],
        options=[
            OptionSpec(verb="WIPE", target=0),
            OptionSpec(verb="WIPE", target=1),
            OptionSpec(verb="COVER", target=2),
        ],
    )
    cleaned = SimpleGame._dedup_situation(sit)
    assert len(cleaned.options) == 3
    assert cleaned.options == sit.options


def test_dedup_situation_deduplicates_identical_options():
    from server.world.situation import Obj, OptionSpec, Situation

    bottle = Obj(kind="bottle", color="blue", shape="cylinder", size="small", material="ceramic")
    crayon = Obj(kind="crayon", color="red", shape="long", size="small", material="paper")
    cup = Obj(kind="cup", color="yellow", shape="cylinder", size="small", material="ceramic")
    sit = Situation(
        id="test_same_category",
        tags=["THIRSTY"],
        location="house",
        objects=[bottle, crayon, cup],
        options=[
            OptionSpec(verb="DRINK", target=0),
            OptionSpec(verb="DRINK", target=0),
            OptionSpec(verb="WAIT", target=None),
            OptionSpec(verb="WAIT", target=None),
        ],
    )
    cleaned = SimpleGame._dedup_situation(sit)
    assert len(cleaned.options) == 2
    assert cleaned.options == [
        OptionSpec(verb="DRINK", target=0),
        OptionSpec(verb="WAIT", target=None),
    ]


def test_generator_never_adds_both_cup_and_bottle():
    import random
    from server.world.generator import make_situation

    for seed in range(50):
        rng = random.Random(seed)
        sit = make_situation(rng, ["THIRSTY"])
        kinds = [o.kind for o in sit.objects]
        assert not ("cup" in kinds and "bottle" in kinds), f"both cup and bottle found in {kinds} for seed {seed}"


def test_two_step_show_parent_then_solo_child(tmp_path):
    g = SimpleGame.new(tmp_path / "two_step", seed=42, two_step=True)
    assert g.has_two_steps() is True

    # Step 1
    sc1 = g.scene()
    assert sc1.step == 1
    assert sc1.step_total == 2
    step1_tag = next((t for t in g.s.state.current.tags if t != "REQUEST"), None)
    res1 = g.play("show", 0)
    assert res1.by == "parent"
    assert res1.child_said == ""
    assert res1.step == 1
    assert res1.step_total == 2
    assert res1.has_next_step is True

    # Step 2
    sc2 = g.scene()
    assert sc2.step == 2
    assert sc2.step_total == 2
    assert g.solo_sit is not None
    if step1_tag is not None:
        assert g.solo_sit.tags != [step1_tag]
    assert all(o.memory == "" for o in sc2.options)
    res2 = g.play("leave")
    assert res2.by == "child"
    assert res2.child_said != ""
    assert res2.step == 2
    assert res2.step_total == 2
    assert res2.has_next_step is False


def test_dedup_situation_preserves_first_occurrence_order():
    from server.world.situation import Obj, OptionSpec, Situation

    bottle = Obj(kind="bottle", color="blue", shape="cylinder", size="small", material="ceramic")
    cup = Obj(kind="cup", color="yellow", shape="cylinder", size="small", material="ceramic")
    sit = Situation(
        id="test_order",
        tags=["THIRSTY"],
        location="house",
        objects=[bottle, cup],
        options=[
            OptionSpec(verb="DRINK", target=0),
            OptionSpec(verb="DRINK", target=1),
            OptionSpec(verb="DRINK", target=0),
        ],
    )
    cleaned = SimpleGame._dedup_situation(sit)
    assert [o.target for o in cleaned.options] == [0, 1]


def test_past_scene_memory_does_not_bleed_or_override_current_two_step_choice(tmp_path, monkeypatch):
    import server.learning.codec as codec
    import server.world.situation as sit_mod
    import server.world.ontology as onto_mod
    import server.world.generator as gen_mod
    import server.game.memory as mem_mod
    from server.world.situation import Obj, OptionSpec, Situation
    from server.world.ontology import KindSpec
    from server.world.state import Phase

    monkeypatch.setitem(onto_mod.RULES, "UPSET", {("HUG", "teddy"): 2})
    monkeypatch.setitem(sit_mod.RULES, "UPSET", {("HUG", "teddy"): 2})
    monkeypatch.setitem(onto_mod.WAIT_VALUE, "UPSET", -1)
    monkeypatch.setitem(sit_mod.WAIT_VALUE, "UPSET", -1)

    monkeypatch.setitem(onto_mod.RULES, "HURT", {("COVER", "bandage"): 2, ("HUG", "teddy"): 0})
    monkeypatch.setitem(sit_mod.RULES, "HURT", {("COVER", "bandage"): 2, ("HUG", "teddy"): 0})
    monkeypatch.setitem(onto_mod.WAIT_VALUE, "HURT", -1)
    monkeypatch.setitem(sit_mod.WAIT_VALUE, "HURT", -1)

    monkeypatch.setitem(onto_mod.NEEDS, "UPSET", ("HUG", "teddy"))
    monkeypatch.setitem(onto_mod.NEEDS, "HURT", ("COVER", "bandage"))
    monkeypatch.setitem(gen_mod.NEEDS, "UPSET", ("HUG", "teddy"))
    monkeypatch.setitem(gen_mod.NEEDS, "HURT", ("COVER", "bandage"))

    teddy_spec = KindSpec("teddy", "sphere", "fabric", ("small", "big"))
    bandage_spec = KindSpec("bandage", "flat", "fabric", ("small", "big"))
    monkeypatch.setitem(onto_mod.KINDS, "teddy", teddy_spec)
    monkeypatch.setitem(onto_mod.KINDS, "bandage", bandage_spec)
    monkeypatch.setitem(gen_mod.KINDS, "teddy", teddy_spec)
    monkeypatch.setitem(gen_mod.KINDS, "bandage", bandage_spec)
    monkeypatch.setitem(mem_mod._KIND_OF, ("sphere", "fabric"), "teddy")
    monkeypatch.setitem(mem_mod._KIND_OF, ("flat", "fabric"), "bandage")

    monkeypatch.setitem(codec.TOK, "TAG_UPSET", codec.TOK["TAG_THIRSTY"])
    monkeypatch.setitem(codec.TOK, "TAG_HURT", codec.TOK["TAG_COLD"])
    monkeypatch.setitem(codec.TOK, "VERB_HUG", codec.TOK["VERB_GIVE"])

    # 1. Creates a two-step game g = SimpleGame.new(tmp_path / "no_past_bias", seed=42, two_step=True).
    g = SimpleGame.new(tmp_path / "no_past_bias", seed=42, two_step=True)
    g.s.narrator.generate_scene_bundle = None

    # 2. Injects a Situation with tags=["UPSET"] and HUG teddy as Option 0, plays Step 1 (show, 0) and Step 2 (leave) so HUG teddy is in g.s.buffer as "solved" under UPSET.
    teddy = Obj(kind="teddy", color="blue", shape="sphere", size="small", material="fabric")
    sit_upset = Situation(
        id="sit_upset",
        tags=["UPSET"],
        location="house",
        objects=[teddy],
        options=[OptionSpec(verb="HUG", target=0)],
    )
    g.s.engine.state.phase = Phase.SITUATION
    g.s.engine.state.current = sit_upset
    g.play("show", 0)
    g.play("leave")

    # 3. Then sets g.s.engine.state.current to a HURT situation with Option 0 = COVER bandage (solved) and Option 1 = HUG teddy (nothing).
    bandage = Obj(kind="bandage", color="yellow", shape="flat", size="small", material="fabric")
    sit_hurt = Situation(
        id="sit_hurt",
        tags=["HURT"],
        location="house",
        objects=[bandage, teddy],
        options=[
            OptionSpec(verb="COVER", target=0),
            OptionSpec(verb="HUG", target=1),
        ],
    )
    g.s.engine.state.phase = Phase.SITUATION
    g.s.engine.state.current = sit_hurt

    # 4. Verifies that in Step 1 (sc1 = g.scene()), all options have o.memory == "" and o.lean == "" (no "지난번 기억(울음)" bleeding into HURT).
    sc1 = g.scene()
    for o in sc1.options:
        assert o.memory == ""
        assert o.lean == ""

    # 5. Plays Step 1 (res1 = g.play("show", 0) -> COVER bandage, solved).
    res1 = g.play("show", 0)
    assert res1.result == "solved"

    # 6. Sets solo_sit to sit_hurt to verify Step 2 option choice when facing HURT:
    # no past memory bleed in Step 2 scene options.
    g.solo_sit = sit_hurt
    sc2 = g.scene()
    assert all(o.memory == "" for o in sc2.options)

    # 7. Plays Step 2 (res2 = g.play("leave")):
    # Option 0 (COVER bandage) was tried in Step 1 (count=1), Option 1 (HUG teddy) is unexplored in HURT (count=0).
    # Child explores the unexplored option (opt 1) rather than repeating the already-known solved option (opt 0)!
    res2 = g.play("leave")
    assert res2.acted == 1
    assert res2.result == "nothing"


def test_dedup_situation_returns_identical_situation_when_no_duplicates():
    from server.world.situation import Obj, OptionSpec, Situation

    cloth = Obj(kind="cloth", color="blue", shape="flat", size="small", material="fabric")
    sit = Situation(
        id="test_sit_clean",
        tags=["COLD"],
        location="house",
        objects=[cloth],
        options=[OptionSpec(verb="WIPE", target=0), OptionSpec(verb="WAIT", target=None)],
    )
    cleaned = SimpleGame._dedup_situation(sit)
    assert cleaned is sit


def test_simple_game_uses_generate_scene_bundle_and_get_takeaway(tmp_path):
    g = game(tmp_path / "bundle_test", seed=42)
    g._ready()

    bundle_called = []
    takeaway_called = []

    def mock_generate_scene_bundle(sit, age_months, voice, solo=False, recent_scenes=(), still=()):
        bundle_called.append({"solo": solo, "recent": list(recent_scenes)})
        if solo:
            return {
                "narration": "솔로 장면 해설입니다.",
                "options": ["혼자 옵션 1", "혼자 옵션 2", "혼자 옵션 3", "혼자 옵션 4"],
                "speeches": ["혼자 대사 1", "혼자 대사 2", "혼자 대사 3", "혼자 대사 4"],
            }
        return {
            "narration": "번들로 생성된 특별한 장면 해설입니다.",
            "options": ["번들 옵션 1", "번들 옵션 2", "번들 옵션 3", "번들 옵션 4"][:len(sit.options)],
            "speeches": ["번들 대사 1", "번들 대사 2", "번들 대사 3", "번들 대사 4"][:len(sit.options)],
        }

    def mock_get_takeaway(sit, option, label, who, voice, custom_action=None, chosen_text=""):
        takeaway_called.append({"option": option, "who": who, "chosen_text": chosen_text})
        return "아이의 마음속에 오래 남을 특별한 교훈입니다."

    g.s.narrator.generate_scene_bundle = mock_generate_scene_bundle
    g.s.narrator.get_takeaway = mock_get_takeaway
    g.s.narrator.outcome = lambda *a, **kw: ("결과 설명", None)

    sc = g.scene()
    assert len(bundle_called) == 1
    assert sc.narration == "번들로 생성된 특별한 장면 해설입니다."
    assert sc.options[0].text == "번들 옵션 1"
    assert sc.options[0].speech == "번들 대사 1"
    assert g._recent_narrations == ["번들로 생성된 특별한 장면 해설입니다."]

    t = g.play("show", 0)
    assert len(takeaway_called) == 1
    assert takeaway_called[0]["chosen_text"] == "번들 옵션 1"
    assert t.remembers == "아이의 마음속에 오래 남을 특별한 교훈입니다."

    # Test solo scene bundle and takeaway
    g_solo = SimpleGame.new(tmp_path / "bundle_solo", seed=42, two_step=True)
    g_solo._ready()
    g_solo.s.narrator.generate_scene_bundle = mock_generate_scene_bundle
    g_solo.s.narrator.get_takeaway = mock_get_takeaway
    g_solo.s.narrator.outcome = lambda *a, **kw: ("솔로 결과", None)

    sc_step1 = g_solo.scene()
    assert sc_step1.step == 1
    t_step1 = g_solo.play("show", 0)
    assert t_step1.has_next_step is True

    sc_step2 = g_solo.scene()
    assert sc_step2.step == 2
    assert sc_step2.narration == "솔로 장면 해설입니다."
    assert sc_step2.options[0].text == "혼자 옵션 1"
    assert sc_step2.options[0].speech == "혼자 대사 1"

    t_step2 = g_solo.play("solo", 0)
    assert t_step2.step == 2
    assert t_step2.remembers == "아이의 마음속에 오래 남을 특별한 교훈입니다."


def test_scene_bundle_outcomes_override_play_and_buffer(tmp_path):
    g = SimpleGame.new(tmp_path / "bundle_outcomes", seed=42)
    g._ready()

    view_called = []
    orig_view = g.s.view
    def mock_view():
        view_called.append(True)
        return orig_view()
    g.s.view = mock_view

    def mock_bundle(sit, age, voice, solo=False, recent_scenes=None, still=()):
        return {
            "narration": "번들 나레이션",
            "options": ["물 주기", "기다리기"],
            "speeches": ["목말라요", "가만히 있을래요"],
            "outcomes": ["solved", "worse"],
        }
    g.s.narrator.generate_scene_bundle = mock_bundle

    sc = g.scene()
    assert len(view_called) == 0  # Redundant s.view() avoided when bundle succeeds
    assert sc.narration == "번들 나레이션"
    assert len(sc.options) == 2
    assert g._current_bundle_outcomes == ["solved", "worse"]

    t = g.play("show", 0)
    assert t.result == "solved"
    assert g.s.buffer[-1].outcome == "solved"
    assert g.s.buffer[-1].facts[0].result == "solved"
    assert len(g._lived_records) == 1
    assert g._lived_records[0]["result"] == "solved"
    assert g._lived_records[0]["step"] == 1


def test_solo_scene_bundle_outcomes_override_solo_play(tmp_path):
    g = SimpleGame.new(tmp_path / "bundle_solo_outcomes", seed=42, two_step=True)
    g._ready()

    def mock_bundle(sit, age, voice, solo=False, recent_scenes=None, still=()):
        if solo:
            return {
                "narration": "솔로 번들 나레이션",
                "options": ["혼자 물 마시기", "혼자 기다리기"],
                "speeches": ["꿀꺽", "기다릴래"],
                "outcomes": ["worse", "solved"],
            }
        return {
            "narration": "스텝1 번들 나레이션",
            "options": ["함께 물 주기", "함께 기다리기"],
            "speeches": ["물 줘요", "기다려요"],
            "outcomes": ["partial", "nothing"],
        }
    g.s.narrator.generate_scene_bundle = mock_bundle

    sc1 = g.scene()
    assert sc1.step == 1
    t1 = g.play("show", 0)
    assert t1.result == "partial"
    assert g.s.buffer[-1].outcome == "partial"

    sc2 = g.scene()
    assert sc2.step == 2
    assert sc2.narration == "솔로 번들 나레이션"
    assert g._current_solo_outcomes == ["worse", "solved"]

    t2 = g.play("solo", 0)
    assert t2.step == 2
    assert t2.result == "worse"
    assert t2.child_said == "꿀꺽"
    assert "혼자 물 마시기" in t2.learned
    assert g.s.buffer[-1].outcome == "worse"
    assert g.s.buffer[-1].facts[0].result == "worse"
    assert len(g._lived_records) == 2
    assert g._lived_records[-1]["result"] == "worse"


def test_scene_called_twice_before_play_preserves_options_narration_and_bundle(tmp_path):
    g = game(tmp_path / "scene_twice")
    calls = []

    def tracking_bundle(*args, **kwargs):
        calls.append(len(calls))
        return {
            "narration": "안정적인 장면 설명입니다.",
            "options": ["옵션 1", "옵션 2", "옵션 3", "옵션 4"],
            "speeches": ["대사 1", "대사 2", "대사 3", "대사 4"],
            "outcomes": ["solved", "partial", "nothing", "worse"],
            "results": ["결과 1", "결과 2", "결과 3", "결과 4"],
            "world_says": ["원인 1", "원인 2", "원인 3", "원인 4"],
        }

    g.s.narrator.generate_scene_bundle = tracking_bundle

    sc1 = g.scene()
    assert len(calls) == 1
    assert sc1.narration == "안정적인 장면 설명입니다."
    assert [o.text for o in sc1.options] == ["옵션 1", "옵션 2", "옵션 3", "옵션 4"]

    # Second scene() call before play() must reuse the bundle and situation without calling generate_scene_bundle again
    sc2 = g.scene()
    assert len(calls) == 1
    assert sc2.narration == sc1.narration
    assert [o.text for o in sc2.options] == [o.text for o in sc1.options]
    assert g._current_bundle_outcomes == ["solved", "partial", "nothing", "worse"]
    assert g._current_bundle_world_says == ["원인 1", "원인 2", "원인 3", "원인 4"]

    # Now play option 0
    t = g.play("show", 0)
    assert t.result == "solved"
    assert t.world_says == "원인 1"


def test_two_step_populates_memory_and_lean_on_reencounter(tmp_path, monkeypatch):
    import server.learning.codec as codec
    import server.world.situation as sit_mod
    import server.world.ontology as onto_mod
    import server.world.generator as gen_mod
    import server.game.memory as mem_mod
    from server.world.situation import Obj, OptionSpec, Situation
    from server.world.ontology import KindSpec
    from server.world.state import Phase

    monkeypatch.setitem(onto_mod.RULES, "UPSET", {("HUG", "teddy"): 2})
    monkeypatch.setitem(sit_mod.RULES, "UPSET", {("HUG", "teddy"): 2})
    monkeypatch.setitem(onto_mod.WAIT_VALUE, "UPSET", -1)
    monkeypatch.setitem(sit_mod.WAIT_VALUE, "UPSET", -1)

    monkeypatch.setitem(onto_mod.NEEDS, "UPSET", ("HUG", "teddy"))
    monkeypatch.setitem(gen_mod.NEEDS, "UPSET", ("HUG", "teddy"))

    teddy_spec = KindSpec("teddy", "sphere", "fabric", ("small", "big"))
    monkeypatch.setitem(onto_mod.KINDS, "teddy", teddy_spec)
    monkeypatch.setitem(gen_mod.KINDS, "teddy", teddy_spec)
    monkeypatch.setitem(mem_mod._KIND_OF, ("sphere", "fabric"), "teddy")

    monkeypatch.setitem(codec.TOK, "TAG_UPSET", codec.TOK["TAG_THIRSTY"])
    monkeypatch.setitem(codec.TOK, "VERB_HUG", codec.TOK["VERB_GIVE"])

    g = SimpleGame.new(tmp_path / "two_step_reencounter", seed=42, two_step=True)
    g.s.narrator.generate_scene_bundle = None

    teddy = Obj(kind="teddy", color="blue", shape="sphere", size="small", material="fabric")
    sit1 = Situation(
        id="sit_upset_1",
        tags=["UPSET"],
        location="house",
        objects=[teddy],
        options=[OptionSpec(verb="HUG", target=0), OptionSpec(verb="WAIT", target=None)],
    )
    g.s.engine.state.phase = Phase.SITUATION
    g.s.engine.state.current = sit1
    sc1 = g.scene()
    assert sc1.options[0].memory == ""
    assert sc1.options[0].lean == ""

    # Play Step 1 and Step 2 for Scene 1
    g.play("show", 0)
    g.play("leave")

    # In Scene 2, re-encounter the same tag (UPSET)
    sit2 = Situation(
        id="sit_upset_2",
        tags=["UPSET"],
        location="house",
        objects=[teddy],
        options=[OptionSpec(verb="HUG", target=0), OptionSpec(verb="WAIT", target=None)],
    )
    g.s.engine.state.phase = Phase.SITUATION
    g.s.engine.state.current = sit2
    sc2 = g.scene()
    assert sc2.step == 1
    # Option 0 now has lived memory and lean in Step 1
    assert sc2.options[0].memory != ""
    assert sc2.options[0].lean != ""

    # Play Step 1 and inspect Step 2 (solo scene)
    g.play("show", 0)
    g.solo_sit = sit2
    sc2_solo = g.scene()
    assert sc2_solo.step == 2
    assert sc2_solo.options[0].memory != ""
    assert sc2_solo.options[0].lean != ""
    assert sc2_solo.options[0].seen > 0


def test_simple_turn_newborn_comparison_no_verdict_or_digits(tmp_path):
    from server import i18n
    import re

    for lang in ("ko", "en"):
        prev = i18n.lang()
        try:
            i18n.set_lang(lang)
            g = SimpleGame.new(tmp_path / f"newborn_{lang}", seed=123, two_step=False)
            g.s.narrator.generate_scene_bundle = None
            g.scene()
            t = g.play("show", 0)

            assert t.newborn_acted is not None
            assert t.newborn_choice_text != ""
            assert t.newborn_line != ""
            assert isinstance(t.changed_from_newborn, bool)

            # Must contain no digits (0-9) or percentage symbols
            assert not re.search(r"[0-9%]", t.newborn_line)

            # Must contain no verdict words
            verdict_words = ["right", "wrong", "solved", "worse", "정답", "성공", "실패"]
            lower_line = t.newborn_line.lower()
            for w in verdict_words:
                assert w not in lower_line
        finally:
            i18n.set_lang(prev)


def test_quiz_challenge_held_out_no_leak_no_llm(tmp_path):
    import re
    from unittest.mock import MagicMock

    g = SimpleGame.new(tmp_path / "quiz_game", seed=99, two_step=False)
    # Ensure no LLM calls can happen
    mock_narrator = MagicMock()
    mock_narrator.complete_json = MagicMock(side_effect=RuntimeError("LLM should not be called"))
    g.s.narrator.client = mock_narrator

    buf_len_before = len(g.s.buffer)
    res = g.quiz_challenge(n=3)

    # 1. Output structure
    assert "items" in res
    assert "age" in res
    assert "lived" in res
    assert "changed_count" in res
    assert len(res["items"]) <= 3

    # 2. Buffer untouched (zero learning / no mutations)
    assert len(g.s.buffer) == buf_len_before

    # 3. All quiz items are held out in self.s.forbidden
    for it in res["items"]:
        assert it["hash"] in g.s.forbidden
        assert "child_pick" in it
        assert "newborn_pick" in it
        assert "options" in it
        assert len(it["options"]) > 0
        for opt in it["options"]:
            assert "feel_tag" in opt
            assert opt["feel_tag"] != ""
            assert "after_text" in opt
        assert it["comparison_line"] != ""
        # No digits or verdict words in comparison_line
        assert not re.search(r"[0-9%]", it["comparison_line"])
        verdict_words = ["right", "wrong", "solved", "worse", "정답", "성공", "실패"]
        lower_line = it["comparison_line"].lower()
        for w in verdict_words:
            assert w not in lower_line


def test_bootstrap_age_three(tmp_path, monkeypatch):
    import sys
    from pathlib import Path
    root = Path(__file__).resolve().parents[1]

    def _reload_onto(path_str: str | None):
        if path_str:
            monkeypatch.setenv("GROWING_ONTOLOGY", str(root / path_str))
        else:
            monkeypatch.delenv("GROWING_ONTOLOGY", raising=False)
        for name in list(sys.modules):
            if name.startswith("server.") or name == "server":
                del sys.modules[name]

    _reload_onto("config/ontology_age07.yaml")
    try:
        from server.game.simple import SimpleGame
        save_dir = tmp_path / "three"
        g = SimpleGame.new(save_dir, seed=7, two_step=True, start_at_three=True, persona="playful")
        assert g.age_now() == 36
        assert g.lived == 0
        sc = g.scene()
        assert sc.number == 1
        assert sc.age_months == 36
        assert sc.can_act is True
        assert sc.step == 1
        assert sc.step_total == 2
        assert len(g.s.buffer) == 0

        g2 = SimpleGame.load(save_dir, two_step=True)
        assert g2.age_now() == 36
        assert g2.lived == 0
    finally:
        _reload_onto(None)
        import server.game.simple  # restore default module in sys.modules


def test_summary_moments_have_no_grades_or_option_numbers(tmp_path):
    import re
    g = game(tmp_path / "moments_neutral", seed=0)
    for _ in range(4):
        g.scene()
        g.play("show", 0)
    s = g.summary()
    assert s.moments
    for m in s.moments:
        assert " — " in m, m
        assert re.search(r"\b(solved|partial|nothing|worse)\b", m, re.IGNORECASE) is None, m
        assert re.search(r"\boption\s*\d+\b", m, re.IGNORECASE) is None, m


def test_neutral_moment_outcome_segment_is_nonempty():
    import re
    from server.game.simple import _neutral_moment
    m = _neutral_moment("dirty hands", "wipe with a cloth", "solved.")
    assert m == "dirty hands — wipe with a cloth — the trouble is over", m
    for label in ("solved", "partial", "nothing", "worse"):
        mm = _neutral_moment("dirty hands", "wipe with a cloth", f"{label}.")
        segs = [s.strip() for s in mm.split("—")]
        assert len(segs) == 3, mm
        assert segs[2].strip(" .!?,;:").strip() != "", mm
        assert re.search(r"\b(solved|partial|nothing|worse)\b", mm, re.IGNORECASE) is None, mm
        assert re.search(r"\boption\s*\d+\b", mm, re.IGNORECASE) is None, mm


def test_summary_moments_outcome_segment_is_nonempty(tmp_path):
    import re
    g = game(tmp_path / "moments_outcome", seed=0)
    for _ in range(4):
        g.scene()
        g.play("show", 0)
    s = g.summary()
    assert s.moments
    for m in s.moments:
        segs = [sg.strip() for sg in m.split("—")]
        assert len(segs) >= 2, m
        assert segs[-1].strip(" .!?,;:").strip() != "", m


def test_played_turn_carries_newborn_pick_text(tmp_path):
    g = game(tmp_path / "newborn_pick", seed=0)
    sc = g.scene()
    t = g.play("show", 0)
    assert t.newborn_pick_text is not None
    assert t.newborn_pick_text in [o.text for o in sc.options]
    assert t.newborn_pick_text == t.newborn_choice_text


def test_reloaded_scene_serves_persisted_bundle_without_new_narration(tmp_path):
    g = game(tmp_path / "persist_bundle", seed=7)
    g._ready()
    calls = []

    def changing_bundle(sit, age, voice, solo=False, recent_scenes=(), still=()):
        calls.append(1)
        n = len(calls)
        return {
            "narration": f"persisted scene {n}",
            "options": [f"persisted option {n}-{i}" for i in range(len(sit.options))],
            "speeches": [f"persisted speech {n}-{i}" for i in range(len(sit.options))],
        }

    g.s.narrator.generate_scene_bundle = changing_bundle
    sc1 = g.scene()
    assert sc1.narration == "persisted scene 1"
    assert len(calls) == 1

    g2 = SimpleGame.load(tmp_path / "persist_bundle" / "g")
    g2.s.narrator.generate_scene_bundle = changing_bundle
    sc2 = g2.scene()
    assert sc2.narration == "persisted scene 1"
    assert [o.text for o in sc2.options] == [o.text for o in sc1.options]
    assert len(calls) == 1











