"""P3: Director v2 (prompt, repair loop, curriculum, teaser guard) and Narrator v2 (LLM prose with guards)."""
import json
import random

import pytest

from server.agents import glossary
from server.agents.client import ClientError, MockClient, RouterClient
from server.agents.director import (
    SYSTEM, DirectorAgent, ProposalError, build_offer, clean_teaser, mock_director_handler, template_proposal,
)
from server.agents.guards import kinds_in, outcome_ok, text_ok
from server.agents.narrator import NarrationOut, NarratorAgent, mock_narrator_handler, outcome as fixed_outcome
from server.agents.schemas import DirectorContext, DirectorProposal
from server.world.engine import Engine
from server.world.events import EndDay, OfferCards, PickCard, RecordQuiz, Show
from server.world.ontology import KINDS, LOCATIONS, PROBLEMS, VERBS
from server.world.situation import optimal, topic_of

from helpers import request, thirsty, visitor, wet
from test_agents import WM, Boom, Fixed, bad_proposals, ctx, good


def test_glossary_covers_the_ontology():
    glossary.check_complete()


# ---- curriculum -------------------------------------------------------------------------------------------------
def test_state_folds_review_recent_and_theme():
    e = Engine(per_day=2)
    e.submit(OfferCards(action_id="o", situations=[wet("w"), thirsty("t")], theme="Rainy day"))
    assert e.state.day_theme == "Rainy day"
    e.submit(PickCard(action_id="p", index=0))
    e.submit(Show(action_id="s", option=optimal(e.state.current)[0]))
    e.submit(OfferCards(action_id="o2", situations=[thirsty("t1"), visitor("v")]))
    e.submit(PickCard(action_id="p2", index=0))
    assert e.state.recent_topics == ["FLOOR_WET", "THIRSTY"]
    e.submit(Show(action_id="s2", option=optimal(e.state.current)[0]))
    e.submit(RecordQuiz(action_id="q", result={"by_topic": {"THIRSTY": 0.9, "FLOOR_WET": 0.2, "DARK": 0.5}}))
    assert e.state.review_topics == ["FLOOR_WET", "DARK"]  # weakest first, only the ones under 0.6
    c = DirectorContext.from_state(e.state, 1)
    assert c.recent_topics == ["FLOOR_WET", "THIRSTY"] and not hasattr(c, "review_topics")  # the quiz result never reaches the Director


def test_template_avoids_what_just_happened():
    recent = {template_proposal(ctx(recent_topics=["COLD", "DARK"]), random.Random(s), set(), WM).cards[0].tags[0] for s in range(40)}
    assert not recent & {"COLD", "DARK"}


def test_two_cards_never_share_a_topic():
    c = ctx(day=2, nameable=["cup"],
            open_issues=[{"id": "w0:FLOOR_WET", "tag": "FLOOR_WET", "outcome": "worse", "location": "house"}])
    for seed in range(120):
        sits = build_offer(template_proposal(c, random.Random(seed), set(), WM), WM, 0, set())
        assert topic_of(sits[0]) != topic_of(sits[1])
    twin = good().model_copy(deep=True)
    twin.cards[1] = twin.cards[0].model_copy(update={"objects": list(reversed(twin.cards[0].objects)),
                                                     "options": list(reversed(twin.cards[0].options))})
    with pytest.raises(ProposalError):
        build_offer(twin, WM, 0, set())


# ---- prompt -----------------------------------------------------------------------------------------------------
def test_prompt_has_glossary_and_no_rules():
    seen = {}

    class Spy:
        name = "spy"

        def complete_json(self, *, agent, system, user, schema):
            seen["u"] = json.loads(user)
            seen["s"] = system
            return good()
    DirectorAgent(Spy(), WM).offer(ctx(recent_topics=["COLD"]), random.Random(0), set())
    onto = seen["u"]["ontology"]
    assert set(onto["problems"]) == set(PROBLEMS) and set(onto["verbs"]) == set(VERBS) and set(onto["kinds"]) == set(KINDS)
    assert set(onto["locations"]) == set(LOCATIONS)
    blob = (seen["s"] + json.dumps(seen["u"])).lower()
    for banned in ("value", "optimal", "quiz", "checkpoint", "probs", "reward", "correct answer", "rules"):
        assert banned not in blob, banned
    # the glossary says what things are, never which kind goes with which problem
    for tag, text in onto["problems"].items():
        assert not any(k in text.lower() for k in KINDS), tag
    assert "previous_error" not in seen["u"]
    assert "never name any object" in SYSTEM


# ---- repair loop ------------------------------------------------------------------------------------------------
class Scripted:
    """Answers from a list; records what it was asked."""
    name = "scripted"

    def __init__(self, answers):
        self.answers, self.asked = list(answers), []

    def complete_json(self, *, agent, system, user, schema):
        self.asked.append(json.loads(user))
        a = self.answers.pop(0)
        if isinstance(a, Exception):
            raise a
        return a


def test_repair_loop_fixes_one_bad_answer_and_feeds_back_the_reason():
    bad = bad_proposals()["nothing solves it"]
    c = Scripted([bad, good()])
    agent = DirectorAgent(c, WM)
    sits = agent.offer(ctx(), random.Random(0), set())
    assert len(sits) == 2 and agent.last_source == "scripted" and agent.last_error is None
    assert "previous_error" not in c.asked[0]
    assert "no option solves" in c.asked[1]["previous_error"] and c.asked[1]["previous_answer"]
    s = agent.stats.summary()
    assert (s["calls"], s["ok"], s["repaired"], s["fallback"]) == (1, 1, 1, 0)


def test_two_bad_answers_fall_back_to_the_template():
    c = Scripted([bad_proposals()["unknown tag"], ClientError("down"), good()])
    agent = DirectorAgent(c, WM)
    sits = agent.offer(ctx(), random.Random(0), set())
    assert agent.last_source == "template" and agent.last_error and len(c.asked) == 2 and len(sits) == 2
    assert agent.stats.summary()["fallback"] == 1
    assert len(c.answers) == 1  # no third try


def test_no_client_is_counted_apart_from_failures():
    agent = DirectorAgent(None, WM)
    agent.offer(ctx(), random.Random(0), set())
    s = agent.stats.summary()
    assert s["no_client"] == 1 and s["fallback"] == 0 and s["calls"] == 0


def test_on_call_hook_gets_one_row_per_offer():
    rows = []
    agent = DirectorAgent(Scripted([bad_proposals()["one card"], good()]), WM, on_call=rows.append)
    agent.offer(ctx(), random.Random(0), set())
    assert len(rows) == 1 and rows[0]["agent"] == "director" and rows[0]["ok"] and rows[0]["attempts"] == 2


# ---- teasers and themes -----------------------------------------------------------------------------------------
def test_text_guard_cases():
    assert text_ok("Grandma coughs a little.", max_len=80) is None
    assert text_ok("", max_len=80) == "empty"
    assert text_ok("x" * 100, max_len=80) == "too long"
    assert "names" in text_ok("Look at the cup!", max_len=80)
    assert "names" in text_ok("Two cups wait.", max_len=80)
    assert text_ok("Take the cup's handle.", max_len=80, allowed_kinds={"cup"}) is None
    assert "internal" in text_ok("Pick DRINK now", max_len=80)
    assert "internal" in text_ok("It is NAME_00", max_len=80)
    assert "answer" in text_ok("The right answer is left.", max_len=80)
    assert text_ok("<b>bold</b>", max_len=80) and text_ok("line\nbreak", max_len=80)
    assert "place" in text_ok("Off to the yard!", max_len=80, allowed_places={"house"})
    assert text_ok("Off to the yard!", max_len=80, allowed_places={"house", "yard"}) is None
    assert text_ok("Grandma asks for the bobo", max_len=80, must_contain="bobo") is None
    assert text_ok("Grandma asks for something", max_len=80, must_contain="bobo")
    assert kinds_in("A blanket and two flashlights") == {"blanket", "flashlight"}


def test_outcome_guard_cases():
    assert outcome_ok("That solves it!", "solved") is None
    assert outcome_ok("It gets worse.", "solved")
    assert outcome_ok("That solves it!", "worse") and outcome_ok("It gets worse.", "nothing")
    assert outcome_ok("It gets worse.", "worse") is None
    assert outcome_ok("Perfect!", "partial")


def test_director_drops_unsafe_teasers_but_keeps_the_cards():
    p = good().model_copy(deep=True)
    p.cards[0].teaser = "Grab the cup quickly!"
    p.cards[1].teaser = "A long and quiet morning."
    p.theme = "The right answer is cups"
    agent = DirectorAgent(Fixed("f", p), WM)
    sits = agent.offer(ctx(), random.Random(0), set())
    assert agent.last_source == "f" and len(sits) == 2
    assert agent.last_teasers == ["", "A long and quiet morning."] and agent.last_theme == "" and agent.teasers_dropped == 1
    assert clean_teaser("Grandma sighs.") == "Grandma sighs." and clean_teaser("x" * 500) is None


def test_mock_director_writes_safe_teasers_and_themes():
    agent = DirectorAgent(MockClient({"director": mock_director_handler(WM)}), WM)
    for seed in range(60):
        agent.offer(ctx(day=1 + seed % 3, nameable=["cup"]), random.Random(seed), set())
        assert all(agent.last_teasers) and agent.last_theme
    assert agent.teasers_dropped == 0


# ---- narrator ---------------------------------------------------------------------------------------------------
def narr(answers):
    class N:
        name = "n"

        def __init__(self):
            self.asked = []

        def complete_json(self, *, agent, system, user, schema):
            self.asked.append(json.loads(user))
            a = answers(json.loads(user)) if callable(answers) else answers
            if isinstance(a, Exception):
                raise a
            return NarrationOut(text=a)
    return N()


def test_narrator_without_client_is_the_fixed_phrase():
    w = wet("sit_1")
    n = NarratorAgent(None)
    assert n.intro(w, False)[0] == "Oh no, water is all over the floor!"
    assert n.outcome(w, 0, "solved", "player", {"FLOOR_WET": 2}) == fixed_outcome(w, 0, "solved", "player", {"FLOOR_WET": 2})
    assert n.stats.summary()["no_client"] == 2


def test_narrator_accepts_good_prose_and_caches_it():
    w = wet("sit_1")
    c = narr("A puddle is spreading across the floor.")
    n = NarratorAgent(c)
    t1, _ = n.intro(w, False)
    t2, _ = n.intro(w, False)
    assert t1 == t2 == "A puddle is spreading across the floor." and len(c.asked) == 1
    p = c.asked[0]
    assert p["task"] == "intro" and p["problems"][0]["tag"] == "FLOOR_WET"
    blob = json.dumps(p).lower()  # nothing about the objects or options reaches the intro prompt
    assert not any(k in blob for k in KINDS) and "option" not in blob and "wipe" not in blob


@pytest.mark.parametrize("text", ["Grab the cloth!", "It is DRINK time", "", "x" * 400, "WAIT {here}", "The right answer is near."])
def test_narrator_bad_intro_falls_back(text):
    w = wet("sit_1")
    n = NarratorAgent(narr(text))
    assert n.intro(w, False)[0] == "Oh no, water is all over the floor!"
    assert n.stats.summary()["fallback"] == 1 and n.last_error


def test_narrator_intro_failure_is_not_fatal():
    n = NarratorAgent(narr(ClientError("down")))
    assert n.intro(wet("sit_1"), False)[0].startswith("Oh no")


def test_narrator_still_flag_and_request_word():
    w = wet("sit_1")
    c = narr(lambda p: "The puddle is still there." if p["problems"][0]["continues"] else "Water everywhere.")
    n = NarratorAgent(c)
    assert n.intro(w, False, still=("FLOOR_WET",))[0] == "The puddle is still there."
    r = request("sit_2", "cup", "bobo", "NAME_00")
    asked_word = NarratorAgent(narr("Grandma wants something."))
    assert "bobo" in asked_word.intro(r, True)[0]  # the line did not contain the word: fixed text used
    ok = NarratorAgent(narr('Grandma smiles: "Bobo, please!"'))
    assert ok.intro(r, True)[0] == 'Grandma smiles: "Bobo, please!"'


def test_narrator_outcome_must_agree_with_the_engine():
    w = wet("sit_1")  # option 0 wipes with the cloth
    fixed = fixed_outcome(w, 0, "solved", "player", {"FLOOR_WET": 2})
    for bad in ["It gets worse.", "The floor is dry thanks to the spoon.", "Off to the yard to celebrate!", "It fails."]:
        assert NarratorAgent(narr(bad)).outcome(w, 0, "solved", "player", {"FLOOR_WET": 2}) == fixed
    good_text = NarratorAgent(narr("The floor gleams again."))
    assert good_text.outcome(w, 0, "solved", "player", {"FLOOR_WET": 2}) == "You wipe the floor with the big red cloth. The floor gleams again."
    # the wording of what was done is always the engine's, not the LLM's
    assert NarratorAgent(narr("The cloth does its job.")).outcome(w, 0, "solved", "player").startswith("You wipe the floor")
    # the chosen kind may be named, nothing else; the yard only once it is open
    assert NarratorAgent(narr("The cloth soaks it up.")).outcome(w, 0, "solved", "player").endswith("soaks it up.")
    assert "yard" in NarratorAgent(narr("Time for the yard!")).outcome(w, 0, "solved", "player", unlocked=("house", "yard"))
    assert NarratorAgent(narr("Time for the yard!")).outcome(w, 0, "solved", "player", {"FLOOR_WET": 2}, unlocked=("house",)) == fixed


def test_mock_narrator_path_is_valid_for_every_label_and_tag():
    n = NarratorAgent(MockClient({"narrator": mock_narrator_handler()}))
    labels = {"solved": 0, "partial": 1, "nothing": 1, "worse": 2}
    for i, mk in enumerate((wet, thirsty, visitor)):
        sit = mk(f"sit_{i}")
        for label, opt in labels.items():
            n.outcome(sit, opt, label, "player", {t: 2 if label == "solved" else 0 for t in sit.tags})
        n.intro(sit, False)
        n.intro(sit, False, still=tuple(sit.tags))
    r = request("sit_9", "cup", "bobo", "NAME_00")
    n.intro(r, True)
    for label, opt in (("solved", 0), ("nothing", 1)):
        n.outcome(r, opt, label, "child")
    s = n.stats.summary()
    assert s["fallback"] == 0 and s["ok"] == s["calls"] > 20, n.last_error


# ---- session end to end ----------------------------------------------------------------------------------------
def test_session_runs_with_llm_agents_and_logs_calls(tmp_path):
    from server.game.session import Session
    from server.world.state import Phase
    s = Session.new(tmp_path / "s", seed=3)
    rng = random.Random(0)
    while s.state.phase != Phase.EVENING:
        if s.state.phase == Phase.CARDS:
            s.draw_cards()
            v = s.view()
            assert v.theme and all(p["teaser"] for p in v.previews)
            s.pick_card(rng.randrange(2))
        v = s.view()
        assert v.narration and v.options
        r = s.show(s.debug_best()[0])
        assert r.narration.startswith("You ")
    st = s.llm_stats()
    assert st["director"]["fallback"] == 0 and st["director"]["ok"] == st["director"]["calls"] == s.game.per_day
    assert st["narrator"]["fallback"] == 0 and st["narrator"]["calls"] > 0
    rows = [json.loads(x) for x in s.save.llm_log.read_text().splitlines()]
    assert {r["agent"] for r in rows} == {"director", "narrator"} and all(r["ok"] for r in rows)
    s.end_day()
    s.draw_cards()  # the next morning: the curriculum input is built from the evening quiz
    assert set(s.state.review_topics) <= set(s.quiz_topic_names())


def test_session_with_a_broken_llm_still_plays(tmp_path):
    from server.game.session import Session
    from server.world.state import Phase
    s = Session.new(tmp_path / "s", seed=4, client=RouterClient([Boom()]))
    rng = random.Random(0)
    while s.state.phase != Phase.EVENING:
        if s.state.phase == Phase.CARDS:
            s.draw_cards()
            s.pick_card(rng.randrange(2))
        assert s.view().narration
        assert s.show(s.debug_best()[0]).narration
    st = s.llm_stats()
    assert st["director"]["ok"] == 0 and st["director"]["fallback"] == st["director"]["calls"] > 0
    assert st["narrator"]["ok"] == 0 and st["narrator"]["fallback"] > 0


def test_child_input_is_unchanged_by_the_prose():
    """Whatever the narrator writes, the child's observation (and so the training data) is identical."""
    from server.world.situation import obs_hash, observation
    w = wet("sit_1")
    before = obs_hash(observation(w))
    NarratorAgent(narr("A puddle is spreading across the floor.")).intro(w, False)
    assert obs_hash(observation(w)) == before
    import server.learning.codec as codec
    src = open(codec.__file__).read()
    assert "narrat" not in src.lower() and "teaser" not in src.lower() and "theme" not in src.lower()
