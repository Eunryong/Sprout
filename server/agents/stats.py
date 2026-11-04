import time

from .secrets import redact


class AgentStats:
    """Counts what P4 needs to report: how often the LLM's answer was usable, how often it needed a repair or a
    fallback, and how long it took."""

    def __init__(self, agent: str, on_call=None):
        self.agent, self.on_call = agent, on_call
        self.calls = self.ok = self.repaired = self.fallback = self.no_client = 0
        self.seconds = 0.0
        self.errors: list[str] = []

    def record(self, *, source: str, ok: bool, attempts: int, error: str | None, started: float,
               attempt_errors: list[str] | None = None) -> None:
        dt = time.perf_counter() - started
        self.seconds += dt
        if source == "none":
            self.no_client += 1
        else:
            self.calls += 1
            self.ok += ok
            self.repaired += ok and attempts > 1
            self.fallback += not ok
        error = redact(error)
        if error:
            self.errors = (self.errors + [error])[-20:]
        if self.on_call:
            self.on_call({"agent": self.agent, "source": source, "ok": ok, "attempts": attempts, "error": error,
                          "attempt_errors": [redact(e) for e in attempt_errors or []], "seconds": round(dt, 4)})

    def summary(self) -> dict:
        return {"agent": self.agent, "calls": self.calls, "ok": self.ok, "repaired": self.repaired,
                "fallback": self.fallback, "no_client": self.no_client,
                "mean_seconds": round(self.seconds / max(1, self.calls + self.no_client), 4)}
