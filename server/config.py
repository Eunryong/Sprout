from dataclasses import dataclass, field
from pathlib import Path

import yaml

from server.learning.learner import OnlineCfg
from server.learning.model import ModelCfg

CONFIG_DIR = Path(__file__).resolve().parents[1] / "config"


@dataclass
class GameCfg:
    per_day: int = 8
    days: int = 8  # the game's last day; the rules of the world are shown after it (Session.reveal, docs/WORLD.md 22.13)
    quiz_per_topic: int = 12
    quiz_per_day: int = 10
    # Learning rounds a teaching move gets (docs/WORLD.md 22.10). 1 = off (as before). A round is one learning call.
    firm_rounds: int = 1  # an instruction or demonstration said firmly ("you must go to bed at ten")
    own_rounds: int = 1  # the child chose by itself and the parent answered (praise or correction): self-made experience
    # Rounds for a correction by what the child's own wrong choice DID (docs/WORLD.md 22.11); 0 = own_rounds
    partial_rounds: int = 0  # it helped a little
    nothing_rounds: int = 0  # nothing happened
    worse_rounds: int = 0  # it made things worse
    # Being burned by the parent's word (docs/WORLD.md 22.12): the child did as told and the result was bad. 0 = off (as before);
    # n = the experience is learned as something not to do here, in n rounds, instead of as a lesson to copy.
    burn_worse_rounds: int = 0  # it made things worse (hurt)
    burn_nothing_rounds: int = 0  # nothing came of it (not what the child expected)
    # The same for what the child chose itself, with no good word of the parent to have followed: "my choice, my consequence".
    own_burn_worse_rounds: int = 0
    own_burn_nothing_rounds: int = 0
    # The child did something else than a GOOD word of the parent and it went badly: "it would have been easier if I had listened".
    # The parent's word is learned in this many rounds. 0 = off.
    regret_rounds: int = 0
    firm_mode: str = "repeat"  # "repeat": the lesson is repeated at that moment; "weight": it stays 3x as heavy whenever it is replayed
    firm_repeat: int = 3  # times a lesson said in a firm tone is repeated (a gentle one: once); docs/WORLD.md 22.8
    curiosity: float | None = None  # the world's pull of the new (None: codec.CURIOSITY); a generated world sets it
    talk_cost: int = 1  # attention units a word demonstration costs (a situation costs 1); docs/WORLD.md 18.10
    words: list[str] = field(default_factory=lambda: ["bobo", "mimi", "tata", "lulu"])

    def correction_rounds(self, outcome: str) -> int:
        """Learning rounds of a correction, by what the child's own wrong choice did (partial | nothing | worse)."""
        return getattr(self, f"{outcome}_rounds", 0) or self.own_rounds

    def burn_rounds(self, outcome: str) -> int:
        """Rounds of the experience 'I did as I was told and it went badly' (0 = it does not count as a burn)."""
        return {"worse": self.burn_worse_rounds, "nothing": self.burn_nothing_rounds}.get(outcome, 0)

    def own_burn_rounds(self, outcome: str) -> int:
        """Rounds of the experience 'I chose this myself and it went badly' (0 = it does not count)."""
        return {"worse": self.own_burn_worse_rounds, "nothing": self.own_burn_nothing_rounds}.get(outcome, 0)


def _read(name: str) -> dict:
    p = CONFIG_DIR / name
    return yaml.safe_load(p.read_text()) or {} if p.exists() else {}


def load_configs() -> tuple[GameCfg, ModelCfg, OnlineCfg]:
    g, m = _read("game.yaml"), _read("model.yaml")
    return GameCfg(**g), ModelCfg(**m.get("model", {})), OnlineCfg(**m.get("online", {}))
