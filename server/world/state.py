from enum import StrEnum

from pydantic import BaseModel

from .events import Event, EventType
from .ontology import ESCALATION, NAMEABLE_MIN_VALUE, REQUEST_KINDS, START_LOCATIONS, UNLOCKS, open_problems
from .situation import Situation, optimal, topic_of


REVIEW_BELOW = 0.6  # a topic whose held-out accuracy at the last evening quiz is under this is offered for review
RECENT_TOPICS = 6
MAX_THEME = 120


class Phase(StrEnum):
    CARDS = "cards"
    SITUATION = "situation"
    EVENING = "evening"


class Issue(BaseModel):
    """A problem that was not solved. It stays in the world and can come back as a follow-up situation."""
    id: str  # "<situation id>:<tag>"
    tag: str
    location: str
    outcome: str  # partial | nothing | worse (how the last attempt ended)
    day: int
    open: bool = True


class WorldState(BaseModel):
    day: int = 1
    phase: Phase = Phase.CARDS
    per_day: int = 8  # attention units per day (docs/WORLD.md 18.10): one per situation, `talk_cost` per word demonstration
    talk_cost: int = 1
    spent: int = 0  # attention used today
    cards: list[Situation] | None = None
    card_teasers: list[str] = []
    day_theme: str = ""  # the Director's flow of the day (player-facing)
    recent_topics: list[str] = []  # last picked topics, oldest first
    review_topics: list[str] = []  # topics under REVIEW_BELOW at the last quiz, weakest first (curriculum input)
    current: Situation | None = None
    situations_done: int = 0
    quiz_taken: bool = False
    words: dict[str, str] = {}  # word -> NAME token (what the child could know)
    word_kinds: dict[str, str] = {}  # word -> kind (engine/player truth only)
    picked_words: set[str] = set()
    topic_counts: dict[str, int] = {}  # times each topic was the picked situation (Director balances coverage with this)
    verb_counts: dict[str, int] = {}  # times each verb was carried out
    issues: list[Issue] = []
    nameable: list[str] = []  # kinds that have been used successfully enough to be given a word
    unlocked: list[str] = list(START_LOCATIONS)
    requests_today: int = 0
    seen_ids: set[str] = set()
    current_first: bool = False  # first time this word is used in a picked situation (player hint)
    applied_action_ids: set[str] = set()
    last_seq: int = -1

    def open_issues(self) -> list[Issue]:
        return [i for i in self.issues if i.open]


def _spend(s: WorldState, units: int) -> None:
    s.spent += units
    s.situations_done += 1
    s.current = None
    if s.spent >= s.per_day:
        s.phase = Phase.EVENING
        s.cards, s.card_teasers = None, []  # unused cards do not carry over into the night
    else:
        s.phase = Phase.CARDS


def _finish(s: WorldState) -> None:
    _spend(s, 1)


def _carry_out(s: WorldState, option: int, tag_values: dict[str, int], day: int) -> None:
    """World consequences of the option that was carried out (decided by the rules table, recorded in the event)."""
    sit = s.current
    sit.chosen = option
    o = sit.options[option]
    s.verb_counts[o.verb] = s.verb_counts.get(o.verb, 0) + 1
    if o.target is not None and sit.objects[o.target].kind in REQUEST_KINDS:
        kind = sit.objects[o.target].kind
        if max(tag_values.values()) >= NAMEABLE_MIN_VALUE and kind not in s.nameable:
            s.nameable.append(kind)
    for tag, v in tag_values.items():
        if tag == "REQUEST":
            continue
        iid = f"{sit.id}:{tag}"
        if v >= 2:
            if tag in UNLOCKS and UNLOCKS[tag] not in s.unlocked:
                s.unlocked.append(UNLOCKS[tag])
            for i in s.issues:  # the problem that this situation continued (same tag) is now solved
                if i.open and i.tag == tag and sit.links_to == i.id:
                    i.open = False
        else:
            label = "worse" if v < 0 else ("partial" if v == 1 else "nothing")
            linked = next((i for i in s.issues if i.id == sit.links_to and i.tag == tag), None)
            if linked is not None:
                linked.outcome = label  # still not solved
            else:
                s.issues.append(Issue(id=iid, tag=tag, location=sit.location, outcome=label, day=day))


def apply(state: WorldState, ev: Event) -> WorldState:
    """Pure state transition. State changes only happen here."""
    s = state.model_copy(deep=True)
    s.last_seq = ev.seq
    s.applied_action_ids.add(ev.action_id)
    t, p = ev.type, ev.payload
    if t == EventType.DAY_STARTED:
        s.per_day = p.get("per_day", s.per_day)
        s.talk_cost = p.get("talk_cost", s.talk_cost)
    elif t == EventType.CARDS_OFFERED:
        s.cards = [Situation(**c) for c in p["situations"]]
        s.card_teasers = list(p.get("teasers", []))
        s.day_theme = p.get("theme", "")
        for c in s.cards:
            s.seen_ids.add(c.id)
            if c.word:
                s.words.setdefault(c.word, c.word_token)
                s.word_kinds.setdefault(c.word, c.word_kind)
    elif t == EventType.CARD_PICKED:
        s.current = s.cards[p["index"]]
        s.current_first = bool(s.current.word) and s.current.word not in s.picked_words
        if s.current.word:
            s.picked_words.add(s.current.word)
            s.requests_today += 1
        topic = topic_of(s.current)
        s.topic_counts[topic] = s.topic_counts.get(topic, 0) + 1
        s.recent_topics = (s.recent_topics + [topic])[-RECENT_TOPICS:]
        s.cards = None
        s.card_teasers = []
        s.phase = Phase.SITUATION
    elif t == EventType.TALKED:
        sit = Situation(**p["situation"])
        s.seen_ids.add(sit.id)
        s.words.setdefault(sit.word, sit.word_token)
        s.word_kinds.setdefault(sit.word, sit.word_kind)
        s.picked_words.add(sit.word)
        topic = topic_of(sit)
        s.topic_counts[topic] = s.topic_counts.get(topic, 0) + 1
        s.recent_topics = (s.recent_topics + [topic])[-RECENT_TOPICS:]
        sit.resolution = "talk"
        s.current = sit
        _carry_out(s, p["option"], p["tag_values"], ev.day)
        _spend(s, s.talk_cost)
    elif t == EventType.COMMANDED:
        s.current.command = p["option"]
        s.current.command_tone = p.get("tone", "gentle")
    elif t == EventType.SHOWN:
        s.current.resolution = "show"
        _carry_out(s, p["option"], p["tag_values"], ev.day)
        _finish(s)
    elif t == EventType.CHILD_TRIED:
        s.current.tried = p["option"]
        s.current.resolution = "try"
        _carry_out(s, p["option"], p["tag_values"], ev.day)
        _finish(s)
    elif t == EventType.SKIPPED:
        s.current.resolution = "skip"
        _finish(s)
    elif t == EventType.QUIZ_TAKEN:
        s.quiz_taken = True
        by = p.get("result", {}).get("by_topic", {})
        s.review_topics = [t for t, a in sorted(by.items(), key=lambda kv: (kv[1], kv[0])) if a < REVIEW_BELOW]
    elif t == EventType.DAY_ENDED:
        s.day += 1
        s.situations_done = 0
        s.spent = 0
        s.quiz_taken = False
        s.requests_today = 0
        s.phase = Phase.CARDS
    # LEARNED: log only
    return s


def fold(events: list[Event]) -> WorldState:
    s = WorldState()
    for e in events:
        s = apply(s, e)
    return s


def current_optimal(s: WorldState) -> list[int]:
    return optimal(s.current)


def allowed_followup_tags(tag: str, outcome: str, day: int | None = None) -> list[list[str]]:
    """Tag sets a follow-up of an issue may use: the same problem, or (if it got worse) with its escalation, which must
    be open at this age (`day`; None = no schedule check). A trouble that no longer comes up at this age (a baby's hunger, once the
    child is a toddler) is not continued: it has passed."""
    if day is not None and tag not in open_problems(day):
        return []
    sets = [[tag]]
    if outcome == "worse" and tag in ESCALATION and (day is None or ESCALATION[tag] in open_problems(day)):
        sets.append(sorted([tag, ESCALATION[tag]]))
    return sets
