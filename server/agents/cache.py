"""Response cache for LLM calls. `record`: serve hits, call the model on a miss and store the usable answer.
`replay`: cache only (a miss is a ClientError), so a recorded run replays offline and identically, e.g. for the demo.
Keys cover agent + system + user + schema; failures are never stored."""
import hashlib
import json
from pathlib import Path

from pydantic import BaseModel

from .client import ClientError


def cache_key(agent: str, system: str, user: str, schema: type[BaseModel]) -> str:
    return hashlib.sha256(json.dumps([agent, system, user, schema.__name__]).encode()).hexdigest()


class CachingClient:
    def __init__(self, inner, path: str | Path, mode: str = "record"):
        assert mode in ("record", "replay"), mode
        if mode == "record" and inner is None:
            raise ValueError("record mode needs a client to call")
        self.inner, self.path, self.mode = inner, Path(path), mode
        self.name = f"cache({getattr(inner, 'name', 'none')})"
        self.last_used: str | None = None
        self.hits = self.misses = 0
        self.entries: dict[str, dict] = {}
        if self.path.exists():
            for line in self.path.read_text().splitlines():
                if line.strip():
                    row = json.loads(line)
                    self.entries[row["key"]] = row

    def complete_json(self, *, agent, system, user, schema):
        key = cache_key(agent, system, user, schema)
        row = self.entries.get(key)
        if row is not None:
            self.hits += 1
            self.last_used = "cache:" + row.get("source", "?")
            return schema.model_validate(row["response"])
        self.misses += 1
        if self.mode == "replay" or self.inner is None:
            raise ClientError("replay: no recorded answer for this request")
        out = self.inner.complete_json(agent=agent, system=system, user=user, schema=schema)
        source = getattr(self.inner, "last_used", None) or self.inner.name
        row = {"key": key, "agent": agent, "source": source, "response": out.model_dump(mode="json")}
        self.entries[key] = row
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.path, "a") as f:
            f.write(json.dumps(row) + "\n")
        self.last_used = source
        return out
