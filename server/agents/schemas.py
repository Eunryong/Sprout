from pydantic import BaseModel

from server.world.ontology import LINGERS, REQUEST_FROM_MONTH, age_months, open_problems, voice_of
from server.world.state import WorldState


class ObjSpec(BaseModel):
    kind: str
    color: str
    size: str


class OptSpec(BaseModel):
    verb: str
    target: int | None = None  # index into the card's objects; null only for WAIT


class CardProposal(BaseModel):
    tags: list[str]
    location: str = "house"
    objects: list[ObjSpec]
    options: list[OptSpec]
    links_to: str | None = None  # id of an open issue this card continues
    word_kind: str | None = None  # REQUEST only: the kind Grandma names (the word itself comes from the save)
    teaser: str = ""  # player-facing flavour text. NEVER reaches the child, never decides anything


class DirectorProposal(BaseModel):
    theme: str = ""  # one line on the flow of the day (player-facing only)
    cards: list[CardProposal]


class DirectorContext(BaseModel):
    """Everything the Director may know: the world and the day (what just happened, what is unresolved, what the age and the places
    allow). Nothing about the child: no option values, no quiz, no model, no counts of what it has practised or got wrong, so a
    scene can only come out of the world's flow and never out of what the child needs to learn (WORLD.md 5.1, DIRECTION.md)."""
    day: int
    open_issues: list[dict]  # {id, tag, outcome, location}
    must_continue: list[dict] = []  # a need still unmet that does not go away by itself (the oldest one): a card has to continue it
    nameable: list[str]
    requests_left: int
    unlocked: list[str]
    recent_topics: list[str] = []  # topics of the last few picked situations, oldest first
    start_index: int  # first free situation number (ids are sit_<n>_<card>)
    age_months: int = 0  # how old the child is today (0 = no age schedule)
    open_tags: list[str] = []  # the problems that can come up at this age (empty = all of them)
    voice: str = "grandma"  # "baby": the baby is the one being cared for; "grandma": the child helps Grandma and others

    @classmethod
    def from_state(cls, s: WorldState, requests_per_day: int) -> "DirectorContext":
        return cls(day=s.day,
                   open_issues=[{"id": i.id, "tag": i.tag, "outcome": i.outcome, "location": i.location}
                                for i in s.open_issues() if i.tag in open_problems(s.day)],  # a trouble of an earlier age has passed
                   must_continue=[{"id": i.id, "tag": i.tag, "outcome": i.outcome, "location": i.location}
                                  for i in sorted(s.open_issues(), key=lambda i: i.day)
                                  if i.tag in LINGERS and i.tag in open_problems(s.day)][:1],
                   nameable=list(s.nameable), requests_left=max(0, requests_per_day - s.requests_today) if age_months(s.day) >= REQUEST_FROM_MONTH else 0,
                   unlocked=list(s.unlocked),
                   recent_topics=list(s.recent_topics), start_index=len(s.seen_ids),
                   age_months=age_months(s.day), open_tags=open_problems(s.day), voice=voice_of(age_months(s.day)))
