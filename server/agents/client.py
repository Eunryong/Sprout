import json
from typing import Callable, Protocol

from pydantic import BaseModel


class ClientError(Exception):
    pass


class AllClientsFailed(ClientError):
    pass


class LLMClient(Protocol):
    name: str

    def complete_json(self, *, agent: str, system: str, user: str, schema: type[BaseModel]) -> BaseModel: ...


class MockClient:
    """No network. Handlers are plain functions: parsed user-JSON -> dict matching the schema."""

    name = "mock"

    def __init__(self, handlers: dict[str, Callable[[dict], dict]] | None = None):
        self.handlers = handlers or {}

    def complete_json(self, *, agent, system, user, schema):
        h = self.handlers.get(agent)
        if h is None:
            raise ClientError(f"mock has no handler for {agent!r}")
        return schema.model_validate(h(json.loads(user)))


class RouterClient:
    """Tries clients in order (e.g. muse-spark-1.3-contributor -> 1.2 -> partner open weights -> mock)."""

    name = "router"

    def __init__(self, chain: list[LLMClient]):
        self.chain = chain
        self.errors: list[str] = []
        self.last_used: str | None = None

    def complete_json(self, *, agent, system, user, schema):
        self.errors = []
        for c in self.chain:
            try:
                out = c.complete_json(agent=agent, system=system, user=user, schema=schema)
                self.last_used = c.name
                return out
            except Exception as e:  # noqa: BLE001 - any failure moves on to the next client
                self.errors.append(f"{c.name}: {e}")
        raise AllClientsFailed("; ".join(self.errors))


class PerAgentClient:
    """Routes each call to the client configured for that agent (director / narrator)."""

    name = "per-agent"

    def __init__(self, by_agent: dict[str, LLMClient], default: LLMClient | None = None):
        self.by_agent, self.default = by_agent, default
        self.last_used: str | None = None

    def complete_json(self, *, agent, system, user, schema):
        c = self.by_agent.get(agent, self.default)
        if c is None:
            raise ClientError(f"no client configured for {agent!r}")
        out = c.complete_json(agent=agent, system=system, user=user, schema=schema)
        self.last_used = getattr(c, "last_used", None) or c.name
        return out
