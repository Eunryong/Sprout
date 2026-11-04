"""Build the LLM client the game uses from config/llm.yaml (+ environment). Only place that reads the key."""
import os
from dataclasses import dataclass, field
from pathlib import Path

import yaml

from .cache import CachingClient
from .client import ClientError, LLMClient, MockClient, PerAgentClient, RouterClient
from .director import mock_director_handler
from .http_client import ChatJSONClient
from .narrator import mock_narrator_handler
from .parent import mock_parent_handler
from .secrets import ROOT, load_dotenv

LLM_CONFIG = ROOT / "config" / "llm.yaml"
MODES = ("mock", "live", "replay")


class LLMConfigError(Exception):
    pass


@dataclass
class LLMCfg:
    mode: str = "mock"
    record: bool = True
    cache: str = "cache/llm_cache.jsonl"
    api: dict = field(default_factory=lambda: {"base_url": "", "key_env": "MUSE_API_KEY"})
    models: dict = field(default_factory=dict)
    params: dict = field(default_factory=dict)
    agent_params: dict = field(default_factory=dict)  # per-agent overrides of `params`


def load_llm_cfg(path: Path = LLM_CONFIG) -> LLMCfg:
    raw = yaml.safe_load(path.read_text()) or {} if path.exists() else {}
    return LLMCfg(**raw)


def mock_client(word_map: dict) -> MockClient:
    return MockClient({"director": mock_director_handler(word_map), "narrator": mock_narrator_handler(), "parent": mock_parent_handler()})


def make_client(word_map: dict, cfg: LLMCfg | None = None, *, mode: str | None = None, transport=None,
                env_file: Path | None = None) -> LLMClient:
    """mock: stand-ins. live: the configured models (no mock behind them: the agents' own template / fixed-phrase
    fallbacks are the safety net, and failures stay visible in the stats). replay: recorded answers only."""
    load_dotenv(env_file)
    cfg = cfg or load_llm_cfg()
    mode = mode or os.environ.get("GROWING_LLM") or cfg.mode
    if mode not in MODES:
        raise LLMConfigError(f"unknown llm mode {mode!r} (use one of {MODES})")
    cache_path = ROOT / cfg.cache
    if mode == "mock":
        return RouterClient([mock_client(word_map)])
    if mode == "replay":
        return CachingClient(None, cache_path, "replay")
    key = os.environ.get(cfg.api.get("key_env", "MUSE_API_KEY"), "")
    base = os.environ.get("MUSE_BASE_URL") or cfg.api.get("base_url", "")
    if not key or not base:
        missing = " and ".join(n for n, v in (("the API key (" + cfg.api.get("key_env", "MUSE_API_KEY") + ")", key),
                                               ("the base URL (MUSE_BASE_URL or api.base_url)", base)) if not v)
        raise LLMConfigError(f"llm mode is 'live' but {missing} {"are" if " and " in missing else "is"} not set. Put it in .env.local (see .env.example) "
                             f"or use GROWING_LLM=mock.")
    base_params = {"timeout": 60, "temperature": 0.8, "max_tokens": 4000, "retries": 2, "json_mode": True} | cfg.params
    kw = {"transport": transport} if transport else {}

    def chain(agent: str) -> RouterClient:
        names = cfg.models.get(agent) or []
        if not names:
            raise LLMConfigError(f"no models configured for {agent!r} in llm.yaml")
        p = base_params | (cfg.agent_params.get(agent) or {})
        return RouterClient([ChatJSONClient(name=n, model=n, base_url=base, api_key=key, **p, **kw) for n in names])

    live: LLMClient = PerAgentClient({"director": chain("director"), "narrator": chain("narrator"), "parent": chain("director")})  # the parent thinks with the Director's models
    return CachingClient(live, cache_path, "record") if cfg.record else live
