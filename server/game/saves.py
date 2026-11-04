import json
import os
from pathlib import Path

from server.learning.codec import Example
from server.world.events import Event


class SaveDir:
    """saves/<id>/ : meta.json, events.jsonl (source of truth), dataset.jsonl, quiz.jsonl, ckpt/*.pt"""

    def __init__(self, root: str | Path):
        self.root = Path(root)
        self.events = self.root / "events.jsonl"
        self.dataset = self.root / "dataset.jsonl"
        self.quiz = self.root / "quiz.jsonl"
        self.meta = self.root / "meta.json"
        self.llm_log = self.root / "llm_log.jsonl"  # one line per Director/Narrator call: source, ok, attempts, seconds
        self.ckpt = self.root / "ckpt"
        self.ckpt0 = self.ckpt / "ckpt_000.pt"
        self.latest = self.ckpt / "latest.pt"

    def exists(self) -> bool:
        return self.meta.exists()

    def create(self) -> None:
        if self.meta.exists():
            raise FileExistsError(f"save already exists: {self.root}")
        self.ckpt.mkdir(parents=True, exist_ok=True)

    def day_ckpt(self, day: int) -> Path:
        return self.ckpt / f"day_{day:02d}.pt"

    @staticmethod
    def _append(path: Path, line: str) -> None:
        with open(path, "a") as f:
            f.write(line + "\n")
            f.flush()
            os.fsync(f.fileno())

    def append_event(self, ev: Event) -> None:
        self._append(self.events, ev.model_dump_json())

    def append_llm(self, row: dict) -> None:
        with open(self.llm_log, "a") as f:  # diagnostics only: no fsync, safe to lose
            f.write(json.dumps(row) + "\n")

    def append_example(self, ex: Example) -> None:
        self._append(self.dataset, ex.model_dump_json())

    def read_events(self) -> list[Event]:
        if not self.events.exists():
            return []
        return [Event.model_validate_json(x) for x in self.events.read_text().splitlines() if x]

    def read_examples(self) -> list[Example]:
        if not self.dataset.exists():
            return []
        return [Example.model_validate_json(x) for x in self.dataset.read_text().splitlines() if x]

    def write_meta(self, meta: dict) -> None:
        tmp = self.meta.with_name("meta.json.tmp")
        tmp.write_text(json.dumps(meta, indent=1))
        tmp.replace(self.meta)

    def read_meta(self) -> dict:
        return json.loads(self.meta.read_text())
