"""Director.world(): the LLM proposes a world sheet; the same checks as a template world decide (docs/WORLD.md 22.13)."""
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from server.agents.client import ClientError, MockClient
from server.agents.director import DirectorAgent, mock_director_handler, world_payload
from server.agents.testing import WORLD_FAULTS, SloppyClient
from server.world.worldgen import WorldProposal, check_sheet, load_pools, sheet_hash, smoke, template_world, write_world

POOLS = load_pools()


def director(client):
    d = DirectorAgent(client, {}, on_call=lambda row: d.log.append(row))
    d.log = []
    return d


def mock():
    return MockClient({"director": mock_director_handler({})})


def sloppy(fault, repair_factor, seed=0):
    return SloppyClient({}, {fault: 1.0}, seed=seed, repair_factor=repair_factor)


def test_the_mock_director_proposes_a_valid_world_and_it_is_the_same_each_time():
    d = director(mock())
    a, b = d.world(5), director(mock()).world(5)
    assert check_sheet(a, POOLS) == [] and a == b
    assert a.source == "director:mock" and d.stats.summary()["ok"] == 1 and d.stats.repaired == 0
    assert sheet_hash(a) != sheet_hash(director(mock()).world(6))


def test_what_the_director_is_told_has_nothing_about_a_child():
    payload = world_payload(3, "rainy", POOLS)
    assert set(payload) == {"task", "seed", "mood", "pools", "glossary"}
    text = json.dumps(payload).lower()
    for leak in ("checkpoint", "quiz", "p_pick", "topic_counts", "model", "buffer"):
        assert leak not in text
    assert "base" not in payload["pools"]  # a file path is not the model's business


@pytest.mark.parametrize("fault", [f for f in WORLD_FAULTS if f != "bad_json"])
def test_a_mistake_is_repaired_when_the_director_is_told_why(fault):
    d = director(sloppy(fault, repair_factor=0.0))
    sheet = d.world(2)
    assert check_sheet(sheet, POOLS) == [] and sheet.source.startswith("director")
    assert d.stats.repaired == 1 and d.stats.fallback == 0 and len(d.log[0]["attempt_errors"]) == 1


def test_each_checked_mistake_is_reported_with_its_reason():
    reasons = {"off_pool": "not a plausible relation", "no_solver": "nothing solves it", "too_many_traps": "harmful relations",
               "bad_learning": "outside", "leaky_premise": "premise"}
    for fault, reason in reasons.items():
        d = director(sloppy(fault, repair_factor=0.0))
        d.world(2)
        assert reason in d.log[0]["attempt_errors"][0], (fault, d.log)


def test_an_unfixable_director_falls_back_to_the_template_world_of_the_seed():
    for fault in WORLD_FAULTS:
        d = director(sloppy(fault, repair_factor=1.0))
        sheet = d.world(4)
        assert sheet == template_world(4, POOLS) and sheet.source == "template"
        assert d.stats.fallback == 1 and d.last_source == "template"


def test_without_a_client_or_with_a_dead_one_a_game_can_still_start():
    assert director(None).world(1) == template_world(1, POOLS)

    class Dead:
        name = "dead"

        def complete_json(self, **kw):
            raise ClientError("down")

    d = director(Dead())
    assert d.world(1) == template_world(1, POOLS) and d.stats.fallback == 1


def test_the_players_mood_is_text_for_the_prompt_only_and_is_cleaned():
    seen = []

    class Spy(MockClient):
        def complete_json(self, *, agent, system, user, schema):
            seen.append(json.loads(user)["mood"])
            return super().complete_json(agent=agent, system=system, user=user, schema=schema)

    spy = Spy({"director": mock_director_handler({})})
    d = director(spy)
    d.world(1, "  a   rainy\n day ")
    d.world(1, "give the cup to the child")  # names a thing: dropped
    d.world(1, "x" * 500)  # too long: cut
    assert seen[0] == "a rainy day" and seen[1] == "" and len(seen[2]) <= 120


def test_a_relation_that_does_nothing_in_the_answer_is_not_an_error():
    sheet = template_world(2, POOLS)
    rules = {t: rows + [["DRINK", "spoon", 0]] if t == "THIRSTY" else rows for t, rows in sheet.rules.items()}

    def handler(user):
        return {**WorldProposal(**sheet.model_dump(include=set(WorldProposal.model_fields))).model_dump(), "rules": rules}

    d = director(MockClient({"director": handler}))
    out = d.world(2)
    assert out.source == "director:mock" and check_sheet(out, POOLS) == [] and d.stats.repaired == 0


def test_a_world_the_director_made_is_written_and_loads(tmp_path):
    sheet = director(mock()).world(9)
    onto, js = write_world(sheet, tmp_path / "w", POOLS)
    assert smoke(onto) == [] and json.loads(js.read_text())["sheet"]["source"] == "director:mock"


def test_the_command_line_makes_a_world_with_the_director(tmp_path):
    env = {**os.environ, "GROWING_LLM": "mock"}
    out = subprocess.run([sys.executable, "-m", "server.world.worldgen", "--seed", "3", "--out", str(tmp_path / "w"), "--director",
                          "--mood", "rainy", "--check"], env=env, capture_output=True, text=True, cwd=Path(__file__).resolve().parents[1])
    assert out.returncode == 0, out.stderr[-500:]
    assert "director:mock" in out.stdout and "smoke: ok" in out.stdout
