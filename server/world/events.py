from enum import StrEnum

from pydantic import BaseModel


class EventType(StrEnum):
    DAY_STARTED = "day_started"
    CARDS_OFFERED = "cards_offered"
    CARD_PICKED = "card_picked"
    SHOWN = "shown"
    CHILD_TRIED = "child_tried"
    SKIPPED = "skipped"
    TALKED = "talked"
    COMMANDED = "commanded"
    LEARNED = "learned"
    QUIZ_TAKEN = "quiz_taken"
    DAY_ENDED = "day_ended"


class Event(BaseModel):
    seq: int
    action_id: str
    day: int
    type: EventType
    payload: dict = {}


# ---- commands (player/session -> engine). Plain data; the engine never calls a model or an LLM. ----
class Cmd(BaseModel):
    action_id: str


class OfferCards(Cmd):
    situations: list  # list[Situation]; a Director proposal, validated by the engine
    teasers: list[str] = []  # player-facing flavour text per card; never part of a Situation, never read by the child
    theme: str = ""  # the Director's one-line "flow of the day" (player-facing only)


class PickCard(Cmd):
    index: int


TONES = ("gentle", "firm")  # how a teaching move is said (docs/WORLD.md 22.8): firm repeats the lesson, see Example.repeat


class Show(Cmd):
    option: int  # the player carries out this option; the child learns from the choice
    tone: str = "gentle"


class Try(Cmd):
    option: int  # chosen by the child model in the session
    probs: list[float] = []


class Talk(Cmd):
    """The parent names a thing while handing it over (docs/WORLD.md 18.10): a word-REQUEST demonstration the
    parent chose themselves, outside the Director's cards. `situation` is built by the session; the engine validates it."""
    situation: object  # Situation | dict
    option: int  # the option the parent demonstrates (the one that fetches the named kind)


class Command(Cmd):
    """The parent tells the child which option to take in the current situation. Any option can be told, wrong ones
    included (parents err). Its quality is computed by the engine from the rules table and never shown to the child."""
    option: int
    tone: str = "gentle"


class Skip(Cmd):
    pass


class RecordLearned(Cmd):
    report: dict


class RecordQuiz(Cmd):
    result: dict


class EndDay(Cmd):
    pass


class RuleError(Exception):
    def __init__(self, code: str, message: str = ""):
        super().__init__(f"{code}: {message}")
        self.code = code
