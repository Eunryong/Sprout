"""Findings of the live smoke test (reasoning models spend max_tokens on thinking; the key sits in .env) and the
decoy mode of the generator that the world/choice research (docs/WORLD.md section 16) used."""
import json
import random

import pytest

from server.agents import secrets
from server.agents.client import ClientError
from server.agents.factory import LLMCfg, make_client
from server.agents.narrator import NarrationOut
from server.world.generator import make_situation
from server.world.ontology import NEEDS, PROBLEMS
from server.world.situation import observation, validate_situation, values

from test_p4_prep import KEY, Fake, ask, client, reply
from test_agents import WM


def body(content, finish="stop", reasoning=0, completion=10):
    return 200, json.dumps({"choices": [{"message": {"content": content}, "finish_reason": finish}],
                            "usage": {"prompt_tokens": 5, "completion_tokens": completion,
                                      "completion_tokens_details": {"reasoning_tokens": reasoning}}})


# ---- live findings ----------------------------------------------------------------------------------------------
def test_empty_answer_after_length_stop_names_the_reasoning_budget():
    c = client(Fake(body("", finish="length", reasoning=198, completion=200)), max_tokens=200)
    with pytest.raises(ClientError, match="used up by reasoning"):
        ask(c)
    assert c.tokens_reasoning == 198


def test_reasoning_tokens_are_counted_and_effort_is_sent_only_when_set():
    t = Fake(body('{"text": "hi"}', reasoning=7))
    assert ask(client(t, reasoning_effort="low")).text == "hi"
    assert t.calls[0][2]["reasoning_effort"] == "low"
    t2 = Fake(body('{"text": "hi"}'))
    ask(client(t2))
    assert "reasoning_effort" not in t2.calls[0][2]


def test_per_agent_params_reach_the_request(tmp_path, monkeypatch):
    monkeypatch.setenv("MUSE_API_KEY", KEY)
    monkeypatch.setenv("MUSE_BASE_URL", "https://api.example/v1")
    t = Fake(body('{"text": "hi"}'))
    cfg = LLMCfg(models={"director": ["d"], "narrator": ["n"]}, record=False,
                 params={"max_tokens": 1000},
                 agent_params={"narrator": {"max_tokens": 77, "reasoning_effort": "low"}})
    c = make_client(WM, cfg, mode="live", transport=t, env_file=tmp_path / "none")
    c.complete_json(agent="narrator", system="s", user="{}", schema=NarrationOut)
    sent = t.calls[0][2]
    assert sent["max_tokens"] == 77 and sent["reasoning_effort"] == "low" and sent["model"] == "n"


def test_dotenv_reads_dot_env_when_there_is_no_env_local(tmp_path, monkeypatch):
    monkeypatch.delenv("MUSE_API_KEY", raising=False)
    (tmp_path / ".env").write_text(f"MUSE_API_KEY={KEY}\n")
    monkeypatch.setattr(secrets, "ROOT", tmp_path)
    assert secrets.load_dotenv() == ["MUSE_API_KEY"]
    monkeypatch.delenv("MUSE_API_KEY")


def test_both_env_files_are_git_ignored():
    ignore = open(secrets.__file__.replace("server/agents/secrets.py", ".gitignore")).read().split()
    assert ".env" in ignore and ".env.local" in ignore


# ---- decoy mode -------------------------------------------------------------------------------------------------
def test_default_decoys_are_unchanged():
    a = make_situation(random.Random(5), ["COLD"])
    b = make_situation(random.Random(5), ["COLD"], decoys="random")
    assert a == b


def test_hard_decoys_stay_valid_and_include_other_problems_solutions():
    rng = random.Random(0)
    n = cross = 0
    for i in range(300):
        tags = [rng.choice(PROBLEMS)]
        s = make_situation(rng, tags, decoys="hard")
        validate_situation(s)
        assert max(values(s)) == 2
        n += 1
        for o in s.options:
            if o.target is not None and any((o.verb, s.objects[o.target].kind) == NEEDS[t] for t in PROBLEMS if t not in tags):
                cross += 1
                break
    assert cross / n > 0.8  # the cross decoy is the point of the mode (a truncated 5-option card may drop it)


def test_the_child_never_sees_which_option_is_a_decoy():
    s = make_situation(random.Random(3), ["DARK"], decoys="hard")
    dumped = observation(s).model_dump_json()
    for word in ("decoy", "cross", "lookalike", "value", "kind"):
        assert word not in dumped.lower()
