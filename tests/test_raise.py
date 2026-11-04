"""A child raised to three by an LLM parent (muse-spark in the game, a stand-in here) and taken over by the player."""
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from server.agents.client import MockClient
from server.agents.factory import make_client
from server.agents.parent import ParentAgent
from server.game.raise_child import raise_child
from server.web.app import create_app

ROOT = Path(__file__).resolve().parents[1]


class Scene:
    def __init__(self, can_act=True, n=3):
        self.can_act, self.age, self.narration = can_act, "1 year", "Grandma is cold."
        self.options = [type("O", (), {"index": i, "text": f"option {i}", "memory": "", "lean": ""})() for i in range(n)]


def parent(answer, **kw):
    return ParentAgent(MockClient({"parent": lambda user: answer}), **kw)


def test_the_parent_acts_once_and_a_baby_can_only_be_shown():
    d = parent({"part": "say", "option": 2, "reason": "x"}).decide(Scene(), [])
    assert (d.part, d.option) == ("say", 2)
    d = parent({"part": "leave", "option": 1, "reason": "x"}).decide(Scene(), [])
    assert (d.part, d.option) == ("leave", None)
    d = parent({"part": "leave", "option": 1, "reason": "x"}).decide(Scene(can_act=False), [])
    assert (d.part, d.option) == ("show", 1)  # a baby cannot choose: the most a parent can do is show


@pytest.mark.parametrize("answer", [{"part": "show", "option": 9}, {"part": "show", "option": None}, {"part": "dance", "option": 0}])
def test_a_parent_that_cannot_decide_falls_back_to_something_a_tired_parent_would_do(answer):
    a = parent(answer)
    d = a.decide(Scene(can_act=False), [])
    assert d.part == "show" and 0 <= d.option < 3 and a.stats.fallback == 1
    assert parent(answer).decide(Scene(), []).part == "leave"
    assert ParentAgent(None).decide(Scene(), []).part == "leave"


def test_the_parent_is_told_only_what_a_player_reads():
    seen = {}

    class Spy:
        name = "spy"

        def complete_json(self, *, agent, system, user, schema):
            seen["user"], seen["system"], seen["agent"] = user, system, agent
            return schema(part="show", option=0)
    ParentAgent(Spy(), "busy").decide(Scene(), ["show: x -> y"])
    text = (seen["user"] + seen["system"]).lower()
    for banned in ("value", "optimal", "rule", "solved", "reward", "correct answer"):
        assert banned not in text, banned
    assert seen["agent"] == "parent" and "busy" in seen["user"] or "brisk" in seen["user"]


def test_a_child_is_raised_by_the_parent_and_the_player_takes_over(tmp_path):
    kids = tmp_path / "kids"
    kids.mkdir()
    info = raise_child(kids / "mina", seed=3, persona="warm", client=make_client({}, mode="mock"), max_scenes=4)
    assert info["scenes"] == 4 and info["parent_calls"] == 4 and info["showed"] + info["told"] + info["left"] == 4
    assert (kids / "mina" / "parent_log.jsonl").read_text().count("\n") == 4
    c = TestClient(create_app(tmp_path / "current", seed=1, prefetch=False, children_dir=kids))
    assert [k["name"] for k in c.get("/api/children").json()] == ["mina"]
    assert c.post("/api/new", json={"child": "nobody"}).status_code == 404
    assert c.post("/api/new", json={"child": "../kids/mina"}).status_code == 404
    assert c.post("/api/new", json={"child": "mina"}).status_code == 200
    i = c.get("/api/info").json()
    assert i["lived"] == 4 and i["raised"]["name"] == "mina" and i["raised"]["persona"] == "warm"  # the album is for the moment of taking over
    assert c.get("/api/scene").json()["number"] == 5
    c.post("/api/play", json={"part": "show", "option": 0})
    assert c.get("/api/info").json()["raised"] is None
    assert len([p for p in tmp_path.iterdir() if "-old-" in p.name]) == 1  # the child it replaced is kept


def test_raising_in_the_0_7_world_ends_at_three_with_the_stage_and_the_baby_years(tmp_path):
    env = {**os.environ, "GROWING_LLM": "mock", "PYTHONPATH": str(ROOT)}
    r = subprocess.run([sys.executable, "-m", "server.game.raise_child", "--seed", "5", "--persona", "playful", "--out", str(tmp_path / "k"), "--mock"],
                       env=env, capture_output=True, text=True, cwd=ROOT, timeout=400)
    assert r.returncode == 0, r.stderr[-600:]
    info = json.loads(r.stdout.strip().splitlines()[-1])
    assert info["months"] == 36 and info["scenes"] == 36 and info["persona"] == "playful"
    log = [json.loads(x) for x in (tmp_path / "k" / "parent_log.jsonl").read_text().splitlines()]
    assert all(x["part"] == "show" for x in log if x["age_months"] < 12)  # a baby can only be shown
    assert (tmp_path / "k" / "ontology.yaml").read_text().count("BABY_HUNGRY") > 0  # the child grew up in the 0-7 years world


def test_a_new_child_in_a_new_world_is_raised_to_three_in_the_background(tmp_path):
    import time
    kids = tmp_path / "saves" / "children"
    c = TestClient(create_app(tmp_path / "current", seed=1, prefetch=False, children_dir=kids))
    assert c.get("/api/raising").json() == {"state": "none"}
    name = c.post("/api/raise").json()["name"]
    assert c.post("/api/raise").json()["name"] == name  # one at a time while it runs
    for _ in range(300):
        if c.get("/api/raising").json()["state"] != "running":
            break
        time.sleep(0.5)
    assert c.get("/api/raising").json()["state"] == "done"
    kid = [k for k in c.get("/api/children").json() if k["name"] == name][0]
    assert kid["months"] == 36 and kid["world"]["random"]
    import re
    import shutil
    shutil.rmtree(Path("saves/worlds") / ("h" + re.sub(r"\D", "", name)), ignore_errors=True)  # the world it drew belongs to this test only
