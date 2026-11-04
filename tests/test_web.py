import pytest
from fastapi.testclient import TestClient

from server.web.app import create_app


@pytest.fixture
def client(tmp_path):
    return TestClient(create_app(tmp_path / "current", seed=1, prefetch=False))


def test_the_page_is_served(client):
    r = client.get("/")
    assert r.status_code == 200 and "Sprout" in r.text and "/api/scene" in r.text
    assert "renderQaHistory" in r.text and "readAndClearInput" in r.text and "qaOpen" in r.text


def test_a_scene_has_options_with_the_childs_expectations(client):
    sc = client.get("/api/scene").json()
    assert sc["number"] == 1 and sc["narration"] and len(sc["options"]) >= 2
    for o in sc["options"]:
        assert set(o["expects"]) == {"solved", "partial", "nothing", "worse"} and abs(sum(o["expects"].values()) - 1) < 1e-3
    assert client.get("/api/scene").json()["narration"] == sc["narration"]  # the scene stays until the parent takes part


@pytest.mark.parametrize("body", [{"part": "say", "option": 0}, {"part": "show", "option": 0}, {"part": "leave"}])
def test_each_part_is_played_and_the_next_scene_follows(client, body):
    client.get("/api/scene")
    t = client.post("/api/play", json=body).json()
    assert t["result"] in {"solved", "partial", "nothing", "worse"} and t["learned"].endswith(t["result"]) and t["narration"]
    assert client.get("/api/scene").json()["number"] == 2
    assert client.get("/api/info").json()["lived"] == 1


def test_a_bad_option_is_refused_and_nothing_is_learned(client):
    assert client.post("/api/play", json={"part": "say", "option": 99}).status_code == 400
    assert client.post("/api/play", json={"part": "show"}).status_code == 400
    assert client.post("/api/play", json={"part": "judge", "option": 0}).status_code == 422
    assert client.get("/api/info").json()["lived"] == 0


def test_a_game_is_resumed_and_a_new_one_keeps_the_old(tmp_path):
    d = tmp_path / "current"
    c1 = TestClient(create_app(d, seed=1, prefetch=False))
    c1.post("/api/play", json={"part": "show", "option": 0})
    assert TestClient(create_app(d, seed=1, prefetch=False)).get("/api/info").json()["lived"] == 1  # resumed from disk
    c1.post("/api/new", json={"seed": 7})
    assert c1.get("/api/info").json() == {**c1.get("/api/info").json(), "lived": 0, "seed": 7}
    assert len([p for p in tmp_path.iterdir() if "-old-" in p.name]) == 1  # the old child is still there


def test_info_says_who_wrote_what(client):
    client.get("/api/scene")
    i = client.get("/api/info").json()
    assert i["mode"] == "mock" and i["world"].startswith("hand-written")
    assert i["director"]["source"] and set(i["narrator"]) == {"calls", "ok", "fallback"}


def test_a_made_world_is_written_as_an_ontology_file_and_the_game_runs_in_it(tmp_path):
    import json
    import os
    import subprocess
    import sys
    env = {**os.environ, "GROWING_LLM": "mock"}
    r = subprocess.run([sys.executable, "-m", "server.web.world", "--seed", "3", "--out", str(tmp_path / "w")], env=env, capture_output=True, text=True)
    assert r.returncode == 0, r.stderr[-400:]
    made = json.loads(r.stdout.strip().splitlines()[-1])
    assert made["source"] and (tmp_path / "w").exists()
    code = ("import sys; from fastapi.testclient import TestClient; from server.web.app import create_app; "
            "c = TestClient(create_app(sys.argv[1], seed=1)); "
            "t = c.post('/api/play', json={'part': 'show', 'option': 0}).json(); print(c.get('/api/info').json()['world'], t['result'])")
    g = subprocess.run([sys.executable, "-c", code, str(tmp_path / "g")], env={**env, "GROWING_ONTOLOGY": made["path"]}, capture_output=True, text=True)
    assert g.returncode == 0, g.stderr[-400:]
    assert g.stdout.split()[0] == made["source"]  # the game says it grew up in the made world


def test_the_next_scene_is_made_while_the_result_is_read(tmp_path):
    app = create_app(tmp_path / "current", seed=1)  # prefetch on
    c = TestClient(app)
    app.state.prefetch.join(timeout=60)  # the first scene was made at start
    assert app.state.holder.game.s.state.current is not None
    c.post("/api/play", json={"part": "show", "option": 0})
    app.state.prefetch.join(timeout=60)
    h = app.state.holder
    assert h.game.s.state.current is not None and h.game.lived == 1  # the next scene is already under way
    assert c.get("/api/scene").json()["number"] == 2


def test_a_new_child_still_starts_when_the_world_lives_inside_the_save(tmp_path):
    """The server resumed a save and runs in the world file kept inside it; starting over moves that folder aside."""
    import os, shutil, subprocess, sys
    from pathlib import Path
    root = Path(__file__).resolve().parents[1]
    save = tmp_path / "current"
    save.mkdir()
    shutil.copy(root / "config" / "ontology_age07.yaml", save / "ontology.yaml")
    code = f"""
from fastapi.testclient import TestClient
from server.web.app import create_app
c = TestClient(create_app({str(save)!r}, seed=1, prefetch=False))
c.get('/api/scene')
r = c.post('/api/new', json={{}})
print(r.status_code, c.get('/api/scene').status_code)
"""
    env = {**os.environ, "GROWING_ONTOLOGY": str(save / "ontology.yaml"), "GROWING_LLM": "mock", "PYTHONPATH": str(root)}
    r = subprocess.run([sys.executable, "-c", code], env=env, capture_output=True, text=True, cwd=root, timeout=300)
    assert r.returncode == 0 and r.stdout.split()[-2:] == ["200", "200"], r.stderr[-600:]
    assert (save / "ontology.yaml").exists() and len(list(tmp_path.glob("current-old-*"))) == 1


def test_stale_ontology_save_recovers_to_new_game(tmp_path):
    from server.web.app import Holder
    save = tmp_path / "stale"
    save.mkdir()
    (save / "meta.json").write_text('{"ontology_hash": "deadbeef"}')
    h = Holder(save, seed=42)
    assert h.game is not None
    assert h.game.s.seed == 42
    assert (save / "meta.json").exists()


def test_language_switch_preserves_game_state_and_updates_scene(client):
    from server import i18n
    prev = i18n.lang()
    try:
        # Start in EN: scene 1
        sc1 = client.get("/api/scene").json()
        assert sc1["number"] == 1
        assert "month" in sc1["age"]
        assert len(sc1["options"]) >= 2
        en_opt_text = sc1["options"][0]["text"]

        # Advance 1 turn
        client.post("/api/play", json={"part": "show", "option": 0})
        assert client.get("/api/info").json()["lived"] == 1

        # Scene 2 in EN
        sc2_en = client.get("/api/scene").json()
        assert sc2_en["number"] == 2
        assert "month" in sc2_en["age"]
        num_opts = len(sc2_en["options"])

        # Switch to KO: must return {"lang": "ko"} and not reset game
        r = client.post("/api/lang", json={"lang": "ko"}).json()
        assert r["lang"] == "ko"
        assert client.get("/api/info").json()["lang"] == "ko"
        assert client.get("/api/info").json()["lived"] == 1  # lived count unchanged

        # Fetch scene in KO: same turn/scene number, Korean wording
        sc2_ko = client.get("/api/scene").json()
        assert sc2_ko["number"] == 2
        assert sc2_ko["age"].endswith("개월")
        assert len(sc2_ko["options"]) == num_opts
        assert sc2_ko["options"][0]["text"] != sc2_en["options"][0]["text"]

        # Switch back to EN: state still preserved
        client.post("/api/lang", json={"lang": "en"})
        sc2_en_again = client.get("/api/scene").json()
        assert sc2_en_again["number"] == 2
        assert client.get("/api/info").json()["lived"] == 1
        assert "month" in sc2_en_again["age"]
        assert sc2_en_again["options"][0]["text"] == sc2_en["options"][0]["text"]
    finally:
        i18n.set_lang(prev)


def test_ask_child_endpoint(client):
    r = client.post("/api/ask", json={"question": "엄마랑 함께한 것 중에 뭐가 제일 기억에 남아?"})
    assert r.status_code == 200
    data = r.json()
    assert data["question"] == "엄마랑 함께한 것 중에 뭐가 제일 기억에 남아?"
    assert isinstance(data["answer"], str) and len(data["answer"]) > 0
    assert "age" in data
    assert "stage" in data
    assert "scenes" in data
    assert "learned_count" in data
    assert "moments" in data
    assert "history" in data
    assert len(data["history"]) == 1


def test_qa_history_persistence_and_reset(client):
    info0 = client.get("/api/info").json()
    assert info0["qa_history"] == []
    ask_get0 = client.get("/api/ask").json()
    assert ask_get0["history"] == []

    r1 = client.post("/api/ask", json={"question": "첫 번째 질문이야"})
    assert r1.status_code == 200
    d1 = r1.json()
    assert len(d1["history"]) == 1
    assert d1["history"][0]["question"] == "첫 번째 질문이야"
    assert d1["history"][0]["answer"] == d1["answer"]
    assert d1["history"][0]["age"] == d1["age"]
    assert d1["history"][0]["stage"] == d1["stage"]
    assert d1["history"][0]["scenes"] == d1["scenes"]

    r2 = client.post("/api/ask", json={"question": "두 번째 질문이야"})
    assert r2.status_code == 200
    d2 = r2.json()
    assert len(d2["history"]) == 2
    assert d2["history"][1]["question"] == "두 번째 질문이야"

    # GET /api/info and GET /api/ask return persisted history across requests
    info1 = client.get("/api/info").json()
    assert len(info1["qa_history"]) == 2
    assert info1["qa_history"] == d2["history"]

    ask_get1 = client.get("/api/ask").json()
    assert len(ask_get1["history"]) == 2
    assert ask_get1["history"] == d2["history"]

    # POST /api/new rotates h.dir and starts with an empty qa_history
    new_res = client.post("/api/new", json={"seed": 42})
    assert new_res.status_code == 200

    info2 = client.get("/api/info").json()
    assert info2["qa_history"] == []

    ask_get2 = client.get("/api/ask").json()
    assert ask_get2["history"] == []


def test_play_with_custom_text(client):
    client.get("/api/scene")
    r = client.post("/api/play", json={"option_index": 0, "custom_text": "따뜻하게 안아주며 이야기한다"})
    assert r.status_code == 200
    data = r.json()
    assert data["result"] in {"solved", "partial", "nothing", "worse"}
    assert data["narration"]


def test_ask_child_does_not_treat_nothing_as_learned_habit(client):
    client.get("/api/scene")
    r_play = client.post("/api/play", json={"part": "show", "option": 0})
    assert r_play.status_code == 200

    holder = client.app.state.holder
    assert holder.game.s.buffer
    # Simulate a failed attempt where result was 'nothing'
    holder.game.s.buffer[-1].facts[0].result = "nothing"
    holder.game.s.buffer[-1].facts[0].takeaway = "doesn't solve it"

    r_ask = client.post("/api/ask", json={"question": "손이 더러울 때 어떻게 해야 해?"})
    assert r_ask.status_code == 200
    data = r_ask.json()

    # 'nothing' outcome must NOT be in learned_habits
    assert data["learned_habits"] == []
    assert data["learned_count"] == 0
    # The failed outcome MUST be captured in trial_memories
    assert len(data["trial_memories"]) == 1
    assert "해결되지 않는다는 걸 배움" in data["trial_memories"][0] or "doesn't solve it" in data["trial_memories"][0]
    assert len(data["moments"]) == 1
    assert data["answer"]


def test_new_child_with_persona_bootstraps_3yo_foundation(client):
    r_new = client.post("/api/new", json={"persona": "playful", "seed": 11})
    assert r_new.status_code == 200
    assert r_new.json()["seed"] == 11

    info = client.get("/api/info").json()
    assert info["lived"] == 0
    assert info["raised"] is None
    assert info["curiosity"] == 0.55

    r_ask = client.post("/api/ask", json={"question": "안녕? 뭐 하고 놀까?"})
    assert r_ask.status_code == 200
    ask_data = r_ask.json()
    assert ask_data["learned_count"] == 0
    assert ask_data["learned_habits"] == []
    assert isinstance(ask_data["answer"], str) and len(ask_data["answer"]) > 0





