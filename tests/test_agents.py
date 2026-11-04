import ast
import json
import random
from pathlib import Path

import pytest

from server.agents.client import AllClientsFailed, ClientError, MockClient, RouterClient
from server.agents.director import SYSTEM, DirectorAgent, ProposalError, build_offer, mock_director_handler, template_proposal
from server.agents.narrator import intention, intro, outcome, refs_ok, world_note
from server.agents.schemas import CardProposal, DirectorContext, DirectorProposal, ObjSpec, OptSpec
from server.world.engine import Engine
from server.world.events import OfferCards, PickCard, Show
from server.world.situation import obs_hash, observation, optimal

from helpers import request, thirsty, visitor, wet

WM = {"cup": ("bobo", "NAME_00"), "cloth": ("mimi", "NAME_01"), "paper": ("tata", "NAME_02"), "ball": ("lulu", "NAME_03")}


def ctx(**kw):
    base = dict(day=1, open_issues=[], nameable=[], requests_left=1, unlocked=["house"], start_index=0)
    return DirectorContext(**(base | kw))


class Boom:
    name = "boom"

    def complete_json(self, **kw):
        raise ClientError("down")


class Fixed:
    def __init__(self, name, prop):
        self.name, self.prop = name, prop

    def complete_json(self, **kw):
        return self.prop


def good():
    return template_proposal(ctx(), random.Random(0), set(), WM)


def test_router_fallback_and_all_fail():
    r = RouterClient([Boom(), Fixed("second", {"ok": 1})])
    assert r.complete_json(agent="x", system="", user="{}", schema=dict) == {"ok": 1}
    assert r.last_used == "second" and len(r.errors) == 1
    with pytest.raises(AllClientsFailed):
        RouterClient([Boom(), Boom()]).complete_json(agent="x", system="", user="{}", schema=dict)


def bad_proposals():
    g = good()

    def tweak(f):
        p = g.model_copy(deep=True)
        f(p.cards[0])
        return p
    solves_nothing = tweak(lambda c: setattr(c, "options", [o for o in c.options if o.verb == "WAIT"] + [
        OptSpec(verb="DRINK", target=0), OptSpec(verb="DRINK", target=1), OptSpec(verb="DRINK", target=2)]))
    return {
        "unknown tag": tweak(lambda c: setattr(c, "tags", ["DRAGON"])),
        "unknown kind": tweak(lambda c: c.objects.append(ObjSpec(kind="dragon", color="red", size="big"))),
        "unknown colour": tweak(lambda c: setattr(c.objects[0], "color", "plaid")),
        "unknown verb": tweak(lambda c: c.options.append(OptSpec(verb="DANCE", target=0))),
        "target out of range": tweak(lambda c: setattr(c.options[0], "target", 9)),
        "WAIT with a target": tweak(lambda c: c.options.append(OptSpec(verb="WAIT", target=0))),
        "too few options": tweak(lambda c: setattr(c, "options", c.options[:2])),
        "too few objects": tweak(lambda c: setattr(c, "objects", c.objects[:2])),
        "twin cards": DirectorProposal(cards=[g.cards[0], g.cards[0]]),
        "one card": DirectorProposal(cards=g.cards[:1]),
        "nothing solves it": solves_nothing,
        "unknown location": tweak(lambda c: setattr(c, "location", "moon")),
        "request of an unnameable kind": tweak(lambda c: (setattr(c, "tags", ["REQUEST"]), setattr(c, "word_kind", "spoon"))),
    }


def test_build_offer_rejects_bad_proposals_and_engine_accepts_good():
    for name, bad in bad_proposals().items():
        with pytest.raises(ProposalError):
            build_offer(bad, WM, 0, set())
        assert name
    sits = build_offer(good(), WM, 0, set())
    e = Engine()
    e.submit(OfferCards(action_id="o", situations=sits))
    assert e.state.cards and len(e.state.cards) == 2


@pytest.mark.parametrize("name", list(bad_proposals()))
def test_director_invalid_fallback(name):
    agent = DirectorAgent(Fixed("evil", bad_proposals()[name]), WM)
    sits = agent.offer(ctx(), random.Random(1), set())
    assert agent.last_source == "template" and agent.last_error
    Engine().submit(OfferCards(action_id="o", situations=sits))  # the engine accepts the fallback


def test_director_rejects_quiz_collision():
    sits = build_offer(good(), WM, 0, set())
    banned = {obs_hash(observation(x)) for x in sits}
    with pytest.raises(ProposalError):
        build_offer(good(), WM, 0, banned)
    agent = DirectorAgent(Fixed("same", good()), WM)
    out = agent.offer(ctx(), random.Random(0), banned)
    assert agent.last_source == "template" and not ({obs_hash(observation(x)) for x in out} & banned)


def test_mock_client_drives_director_end_to_end():
    client = RouterClient([Boom(), MockClient({"director": mock_director_handler(WM)})])
    agent = DirectorAgent(client, WM)
    sits = agent.offer(ctx(), random.Random(3), set())
    assert agent.last_source == "mock" and len(sits) == 2 and agent.last_error is None


def test_director_prompt_carries_no_answers():
    seen = {}

    class Spy:
        name = "spy"

        def complete_json(self, *, agent, system, user, schema):
            seen["system"], seen["user"] = system, user
            return good()
    DirectorAgent(Spy(), WM).offer(ctx(recent_topics=["COLD"]), random.Random(0), set())
    payload = json.loads(seen["user"])
    text = (seen["system"] + seen["user"]).lower()
    for banned in ("value", "optimal", "quiz", "checkpoint", "probs", "reward", "correct answer"):
        assert banned not in text, banned
    assert set(payload["context"]) == {"day", "open_issues", "nameable", "requests_left", "unlocked", "recent_topics", "start_index",
                                  "age_months", "open_tags", "voice", "must_continue"}  # the world and the day: nothing about what the child practised or got wrong


def test_the_director_is_told_nothing_of_the_childs_learning():
    assert not {"topic_counts", "review_topics"} & set(DirectorContext.model_fields)
    text = SYSTEM.lower()
    for word in ("curriculum", "practice", "teacher", "review_topics", "topic_counts"):
        assert word not in text, word


def test_template_scenes_follow_the_worlds_flow_not_a_need_to_practise():
    first = {template_proposal(ctx(), random.Random(s), set(), WM).cards[0].tags[0] for s in range(80)}
    assert len(first) >= 4  # the opening trouble varies with the world's own pace: no problem is favoured


def test_template_followups_and_requests_are_always_engine_valid():
    e = Engine()
    e.submit(OfferCards(action_id="o", situations=[wet("w0"), thirsty("t0")]))
    e.submit(PickCard(action_id="p", index=0))
    e.submit(Show(action_id="s", option=2))  # waited on a wet floor: it got worse
    e.submit(OfferCards(action_id="o2", situations=[thirsty("t1"), visitor("v1")]))
    e.submit(PickCard(action_id="p2", index=0))
    e.submit(Show(action_id="s2", option=optimal(e.state.current)[0]))  # drank from the cup: nameable
    c = DirectorContext.from_state(e.state, 1)
    assert c.open_issues and c.nameable == ["cup"] and c.requests_left == 1
    kinds = set()
    for seed in range(80):
        prop = template_proposal(c, random.Random(seed), set(), WM)
        sits = build_offer(prop, WM, c.start_index, set())
        Engine(log=e.log).submit(OfferCards(action_id=f"x{seed}", situations=sits))
        for s in sits:
            kinds.add("followup" if s.links_to else "request" if s.tags == ["REQUEST"] else "problem")
            if s.links_to:
                assert s.links_to == "w0:FLOOR_WET" and s.tags in (["FLOOR_WET"], ["FLOOR_WET", "VISITOR"])
    assert kinds == {"followup", "request", "problem"}
    # no request once the day's limit is used
    none_left = c.model_copy(update={"requests_left": 0})
    assert all(s.tags != ["REQUEST"] for seed in range(40)
               for s in build_offer(template_proposal(none_left, random.Random(seed), set(), WM), WM, 0, set()))


def test_narrator_phrases_and_refs():
    w = wet("sit_1")
    text, hint = intro(w, False)
    assert "water" in text and hint is None
    assert "still wet" in intro(w, False, still=("FLOOR_WET",))[0]
    r = request("sit_2", "cup", "bobo", "NAME_00")
    text, hint = intro(r, True)
    assert "bobo" in text and "cup" in hint and intro(r, False)[1] is None
    assert intention(w, 0, "player") == "You wipe the floor with the big red cloth."
    assert "The child decides to wait and do nothing." == intention(w, 2, "child")
    assert "That solves it" in outcome(w, 0, "solved", "player", {"FLOOR_WET": 2})
    assert "gets worse" in outcome(w, 2, "worse", "player")
    assert "not the bobo" in outcome(r, 1, "nothing", "player")
    assert "that is the bobo" in outcome(r, 0, "solved", "child")
    assert "yard" in world_note(["yard"], [], []) and world_note([], [], []) == ""
    assert refs_ok("You bring the cup.", {"cup", "ball"})
    assert not refs_ok("You bring the cloth.", {"cup", "ball"})


def test_import_boundary_learning_never_imports_agents_or_game():
    root = Path(__file__).resolve().parents[1] / "server" / "learning"
    for f in root.glob("*.py"):
        for node in ast.walk(ast.parse(f.read_text())):
            names = []
            if isinstance(node, ast.Import):
                names = [a.name for a in node.names]
            elif isinstance(node, ast.ImportFrom):
                names = [node.module or ""]
            assert not any(n.startswith("server.agents") or n.startswith("server.game") for n in names), f


def test_training_code_never_touches_the_rules_table():
    """The child's code (encoding, loss, augmentation, learner, policy, model) must not know which option is good.
    Only the evaluator may (it scores held-out questions); it never trains."""
    forbidden = {"RULES", "NEEDS", "WAIT_VALUE", "values", "optimal", "option_value", "outcome_of", "REQUEST_VALUE"}
    root = Path(__file__).resolve().parents[1] / "server" / "learning"
    for name in ("codec.py", "learner.py", "policy.py", "model.py", "checkpoints.py"):
        tree = ast.parse((root / name).read_text())
        used = {n.id for n in ast.walk(tree) if isinstance(n, ast.Name)}
        imported = {a.name for n in ast.walk(tree) if isinstance(n, ast.ImportFrom) for a in n.names}
        assert not (forbidden & (used | imported)), (name, forbidden & (used | imported))
