"""P4 preparation: the real-client adapter (against a fake transport), the response cache, key handling, fault
injection, and an offline record -> replay round trip through the whole game."""
import json
import os
import random
import subprocess
import sys

import pytest

from server.agents.cache import CachingClient, cache_key
from server.agents.client import ClientError, MockClient
from server.agents.director import mock_director_handler
from server.agents.factory import LLMCfg, LLMConfigError, make_client
from server.agents.http_client import ChatJSONClient, extract_json
from server.agents.narrator import NarrationOut, mock_narrator_handler
from server.agents.schemas import DirectorProposal
from server.agents.secrets import load_dotenv, redact
from server.agents.stats import AgentStats
from server.agents.testing import SloppyClient
from server.game.autoplay import play_day
from server.game.session import Session

from test_agents import WM, ctx, good

KEY = "sk-test-1234567890abcdef"


def reply(content, status=200, usage=None):
    body = {"choices": [{"message": {"content": content}}]}
    if usage:
        body["usage"] = usage
    return status, json.dumps(body)


class Fake:
    """Scripted transport: records every request."""

    def __init__(self, *answers):
        self.answers, self.calls = list(answers), []

    def __call__(self, url, headers, body, timeout):
        self.calls.append((url, headers, json.loads(body), timeout))
        a = self.answers.pop(0) if len(self.answers) > 1 else self.answers[0]
        if isinstance(a, Exception):
            raise a
        return a


def client(transport, **kw):
    return ChatJSONClient(name="m", model="muse-x", base_url="https://api.example/v1/", api_key=KEY,
                          transport=transport, sleep=lambda s: None, **kw)


def ask(c, schema=NarrationOut):
    return c.complete_json(agent="narrator", system="sys", user='{"a": 1}', schema=schema)


# ---- http adapter -----------------------------------------------------------------------------------------------
def test_request_shape_and_parsing():
    t = Fake(reply('{"text": "hi"}', usage={"prompt_tokens": 12, "completion_tokens": 3}))
    c = client(t)
    assert ask(c).text == "hi"
    url, headers, body, _ = t.calls[0]
    assert url == "https://api.example/v1/chat/completions" and headers["Authorization"] == f"Bearer {KEY}"
    assert body["model"] == "muse-x" and body["messages"][1] == {"role": "user", "content": '{"a": 1}'}
    assert "JSON Schema" in body["messages"][0]["content"] and "text" in body["messages"][0]["content"]
    assert KEY not in json.dumps(body)  # the key is only ever in the header
    assert (c.tokens_in, c.tokens_out, c.requests) == (12, 3, 1)


@pytest.mark.parametrize("content", ['```json\n{"text": "hi"}\n```', 'Sure! {"text": "hi"} Hope that helps.', '  {"text":"hi"}  '])
def test_tolerates_fences_and_prose(content):
    assert ask(client(Fake(reply(content)))).text == "hi"


@pytest.mark.parametrize("content", ["no json at all", '{"text": ', "[1, 2]", '{"wrong": 1}', None])
def test_unusable_replies_are_client_errors_without_retry(content):
    t = Fake(reply(content))
    with pytest.raises(ClientError):
        ask(client(t))
    assert len(t.calls) == 1


def test_retries_transient_failures_then_succeeds():
    t = Fake((503, "busy"), (429, "slow down"), TimeoutError("t/o"), reply('{"text": "ok"}'))
    c = client(t, retries=3)
    assert ask(c).text == "ok" and len(t.calls) == 4 and c.requests == 4


def test_gives_up_after_retries_and_hard_errors_do_not_retry():
    t = Fake((500, "boom"))
    with pytest.raises(ClientError, match="gave up after 3 tries"):
        ask(client(t, retries=2))
    assert len(t.calls) == 3
    t = Fake((401, f"bad key {KEY}"))
    with pytest.raises(ClientError) as e:
        ask(client(t))
    assert len(t.calls) == 1 and "401" in str(e.value) and KEY not in str(e.value)


def test_missing_key_or_url_is_refused():
    with pytest.raises(ClientError):
        ChatJSONClient(name="m", model="x", base_url="", api_key=KEY)
    with pytest.raises(ClientError):
        ChatJSONClient(name="m", model="x", base_url="https://a", api_key="")


def test_json_mode_is_dropped_if_the_server_rejects_it():
    t = Fake((400, '{"error": "unknown field response_format"}'), reply('{"text": "ok"}'))
    c = client(t)
    assert ask(c).text == "ok" and not c.json_mode
    assert "response_format" in t.calls[0][2] and "response_format" not in t.calls[1][2]


def test_extract_json_edge_cases():
    assert extract_json('```\n{"a": 1}\n```') == {"a": 1}
    with pytest.raises(ClientError):
        extract_json("")


def test_director_proposal_roundtrips_through_the_wire():
    prop = good()
    got = ask(client(Fake(reply(prop.model_dump_json()))), DirectorProposal)
    assert got == prop


# ---- secrets ---------------------------------------------------------------------------------------------------
def test_dotenv_loads_without_overriding_and_redact_hides_values(tmp_path, monkeypatch):
    monkeypatch.delenv("MUSE_API_KEY", raising=False)
    monkeypatch.setenv("MUSE_BASE_URL", "kept")
    f = tmp_path / ".env.local"
    f.write_text(f'# comment\nMUSE_API_KEY="{KEY}"\nMUSE_BASE_URL=other\n\nBROKEN\n')
    assert load_dotenv(f) == ["MUSE_API_KEY"]  # names only
    assert os.environ["MUSE_API_KEY"] == KEY and os.environ["MUSE_BASE_URL"] == "kept"
    assert redact(f"failed with {KEY} in it") == "failed with *** in it"
    assert load_dotenv(tmp_path / "missing") == []
    monkeypatch.delenv("MUSE_API_KEY")


def test_the_key_never_reaches_stats_or_logs(monkeypatch):
    monkeypatch.setenv("MUSE_API_KEY", KEY)
    rows = []
    s = AgentStats("director", rows.append)
    s.record(source="m", ok=False, attempts=1, error=f"boom {KEY}", started=0.0)
    assert KEY not in json.dumps(rows) and KEY not in json.dumps(s.errors)


def test_repo_ignores_env_files_and_cache():
    ignore = open(os.path.join(os.path.dirname(__file__), "..", ".gitignore")).read().split()
    assert ".env.local" in ignore and "cache/" in ignore


# ---- cache -----------------------------------------------------------------------------------------------------
def test_cache_records_hits_persist_and_failures_are_not_stored(tmp_path):
    path = tmp_path / "c.jsonl"
    t = Fake(reply('{"text": "one"}'))
    c = CachingClient(client(t), path, "record")
    assert ask(c).text == "one" and c.misses == 1 and c.last_used == "m"
    assert ask(c).text == "one" and c.hits == 1 and len(t.calls) == 1 and c.last_used == "cache:m"
    again = CachingClient(None, path, "replay")  # a new process reads the file
    assert ask(again).text == "one" and again.hits == 1
    with pytest.raises(ClientError, match="no recorded answer"):
        again.complete_json(agent="narrator", system="sys", user="other", schema=NarrationOut)
    bad = CachingClient(client(Fake(reply("nope"))), tmp_path / "b.jsonl", "record")
    with pytest.raises(ClientError):
        ask(bad)
    assert not (tmp_path / "b.jsonl").exists()
    assert cache_key("a", "s", "u", NarrationOut) != cache_key("a", "s", "u", DirectorProposal)


# ---- factory ---------------------------------------------------------------------------------------------------
def test_factory_modes(tmp_path, monkeypatch):
    monkeypatch.delenv("MUSE_API_KEY", raising=False)
    monkeypatch.delenv("MUSE_BASE_URL", raising=False)
    nofile = tmp_path / "no-env"
    assert make_client(WM, LLMCfg(), env_file=nofile).complete_json(
        agent="narrator", system="", user=json.dumps({"task": "outcome", "seed": 1, "label": "nothing", "problems": []}),
        schema=NarrationOut).text
    with pytest.raises(LLMConfigError, match="API key"):
        make_client(WM, LLMCfg(models={"director": ["a"], "narrator": ["b"]}), mode="live", env_file=nofile)
    with pytest.raises(LLMConfigError, match="unknown llm mode"):
        make_client(WM, LLMCfg(), mode="loud", env_file=nofile)
    monkeypatch.setenv("MUSE_API_KEY", KEY)
    with pytest.raises(LLMConfigError, match="base URL"):
        make_client(WM, LLMCfg(models={"director": ["a"], "narrator": ["b"]}), mode="live", env_file=nofile)
    monkeypatch.setenv("MUSE_BASE_URL", "https://api.example/v1")
    with pytest.raises(LLMConfigError, match="no models"):
        make_client(WM, LLMCfg(models={"director": ["a"]}), mode="live", env_file=nofile)


# ---- fault injection -------------------------------------------------------------------------------------------
def test_sloppy_client_injects_what_it_is_told_and_is_deterministic():
    def run(seed):
        c = SloppyClient(WM, {"unknown_tag": 1.0}, seed=seed)
        out = c.complete_json(agent="director", system="", user=json.dumps(
            {"context": ctx().model_dump(), "seed": 4}), schema=DirectorProposal)
        return out, c.injected
    a, inj = run(1)
    assert a.cards[0].tags == ["DRAGON"] and inj == {"unknown_tag": 1}
    assert run(1)[0] == a
    clean = SloppyClient(WM, {}, seed=0).complete_json(agent="director", system="", user=json.dumps(
        {"context": ctx().model_dump(), "seed": 4}), schema=DirectorProposal)
    assert clean.cards[0].tags != ["DRAGON"]
    # a repair call is fixed far more often than a first call
    rep = SloppyClient(WM, {"unknown_tag": 0.5}, seed=0, repair_factor=0.0)
    for _ in range(20):
        r = rep.complete_json(agent="director", system="", user=json.dumps(
            {"context": ctx().model_dump(), "seed": 4, "previous_error": "x"}), schema=DirectorProposal)
        assert r.cards[0].tags != ["DRAGON"]


# ---- the whole game over the real adapter, offline -------------------------------------------------------------
def wire(word_map):
    """A fake model server: answers chat requests with what the mock agents would say."""
    handlers = {"director": mock_director_handler(word_map), "narrator": mock_narrator_handler()}

    def transport(url, headers, body, timeout):
        req = json.loads(body)
        user = json.loads(req["messages"][1]["content"])
        agent = "director" if "context" in user else "narrator"
        return reply(json.dumps(handlers[agent](user)), usage={"prompt_tokens": 100, "completion_tokens": 50})
    return transport


def live_cfg(cache):
    return LLMCfg(mode="live", cache=str(cache), models={"director": ["muse-a", "muse-b"], "narrator": ["muse-b"]})


def events_of(s):
    return [(e.type.value, json.dumps(e.payload, sort_keys=True)) for e in s.engine.log if e.type.value != "learned"]


def test_live_record_then_replay_gives_the_identical_game(tmp_path, monkeypatch):
    monkeypatch.setenv("MUSE_API_KEY", KEY)
    monkeypatch.setenv("MUSE_BASE_URL", "https://api.example/v1")
    cache = tmp_path / "cache.jsonl"
    probe = Session.new(tmp_path / "p", seed=7, client=None)
    live = make_client(probe.word_map, live_cfg(cache), mode="live", transport=wire(probe.word_map), env_file=tmp_path / "x")
    s1 = Session.new(tmp_path / "s1", seed=7, client=live)
    play_day(s1, random.Random(1))
    st = s1.llm_stats()
    assert st["director"]["ok"] == st["director"]["calls"] == s1.game.per_day and st["narrator"]["fallback"] == 0
    assert live.misses > 0 and cache.exists()
    # the key is nowhere on disk
    for f in (tmp_path / "s1").rglob("*"):
        if f.is_file() and f.suffix in {".jsonl", ".json"}:
            assert KEY not in f.read_text(), f
    assert KEY not in cache.read_text()

    replay = make_client(probe.word_map, live_cfg(cache), mode="replay", env_file=tmp_path / "x")
    s2 = Session.new(tmp_path / "s2", seed=7, client=replay)
    play_day(s2, random.Random(1))
    assert replay.misses == 0 and replay.hits > 0
    assert events_of(s1) == events_of(s2)  # same cards, same words, same game, with no network at all
    assert s1.view().narration == s2.view().narration


def test_replay_with_an_empty_cache_still_plays_on_fallbacks(tmp_path, monkeypatch):
    probe = Session.new(tmp_path / "p", seed=2, client=None)
    replay = make_client(probe.word_map, LLMCfg(cache=str(tmp_path / "none.jsonl")), mode="replay", env_file=tmp_path / "x")
    s = Session.new(tmp_path / "s", seed=2, client=replay)
    play_day(s, random.Random(1))
    st = s.llm_stats()
    assert st["director"]["ok"] == 0 and st["director"]["fallback"] == s.game.per_day and st["narrator"]["ok"] == 0


# ---- the measurement script ------------------------------------------------------------------------------------
def test_pass_rate_script_runs_and_sees_injected_faults(tmp_path):
    out = tmp_path / "r.json"
    p = subprocess.run([sys.executable, "scripts/exp_llm.py", "--client", "sloppy", "--rates", "no_solver=0.5,names_kind=0.5",
                        "--seeds", "0-1", "--days", "1", "--jobs", "2", "--out", str(out)],
                       capture_output=True, text=True, cwd=os.path.join(os.path.dirname(__file__), ".."))
    assert p.returncode == 0, p.stderr[-800:]
    data = json.loads(out.read_text())
    s = data["summary"]
    assert s["director_calls"] == 16 and 0 < s["director_first_try"] < 1 and s["narrator_ok"] < 1
    assert "no option solves" in p.stdout and "names" in p.stdout and "acceptance (proposed)" in p.stdout
