"""Situations, options, the rules table in action, and the two views (child / player).

The child's Observation carries tags, an optional word token and each option's verb + target ATTRIBUTES.
It never carries a kind, a value, an outcome or any narration. The rules table (ontology.py) is the only thing
that says which option is good."""
import hashlib

from pydantic import BaseModel, ConfigDict

from .ontology import (
    KINDS, LOCATIONS, MAX_OBJECTS, MAX_OPTS, MIN_OBJECTS, MIN_OPTS, REQUEST_KINDS, REQUEST_VALUE, REQUEST_VERB,
    RULES, TAGS, VERBS, WAIT_VALUE, N_NAMES, afforded, outcome_label,
)


class Obj(BaseModel):
    model_config = ConfigDict(frozen=True)
    kind: str  # engine / player knowledge only
    color: str
    shape: str
    size: str
    material: str


class OptionSpec(BaseModel):
    model_config = ConfigDict(frozen=True)
    verb: str
    target: int | None = None  # index into Situation.objects; None only for WAIT


class Situation(BaseModel):
    id: str
    tags: list[str]
    location: str = "house"
    word: str | None = None  # REQUEST only: what Grandma says (player-facing)
    word_token: str | None = None  # REQUEST only: what the child sees
    word_kind: str | None = None  # REQUEST only: engine truth
    objects: list[Obj]
    options: list[OptionSpec]
    links_to: str | None = None  # id of the earlier situation this one continues (follow-up)
    # progress, filled by events
    tried: int | None = None  # option the child tried
    chosen: int | None = None  # option that was carried out in the world
    resolution: str | None = None  # show | try | skip | talk
    command: int | None = None  # option the parent told the child to take (docs/WORLD.md section 18)
    command_tone: str | None = None  # how the instruction was said: gentle | firm (docs/WORLD.md 22.9)


# ---- the rules table in action ----
def option_value(tags: list[str], word_kind: str | None, verb: str, obj: Obj | None) -> int:
    if verb == "WAIT":
        return min(WAIT_VALUE[t] for t in tags)
    best = None  # a rule may be negative (experiment ontologies): the best of the tags, not at least 0
    for t in tags:
        if t == "REQUEST":
            v = REQUEST_VALUE if verb == REQUEST_VERB and obj is not None and obj.kind == word_kind else 0
        else:
            v = RULES[t].get((verb, obj.kind), 0) if obj is not None else 0
        best = v if best is None else max(best, v)
    return best


def values(sit: Situation) -> list[int]:
    return [option_value(sit.tags, sit.word_kind, o.verb, sit.objects[o.target] if o.target is not None else None)
            for o in sit.options]


def optimal(sit: Situation) -> list[int]:
    v = values(sit)
    return [i for i, x in enumerate(v) if x == max(v)]


def outcome_of(sit: Situation, option: int) -> tuple[int, str]:
    v = values(sit)[option]
    return v, outcome_label(v)


def topic_of(sit: Situation) -> str:
    return f"REQUEST:{sit.word_kind}" if sit.tags == ["REQUEST"] else "+".join(sorted(sit.tags))


# ---- what the child sees ----
class TargetView(BaseModel):
    model_config = ConfigDict(frozen=True)
    color: str
    shape: str
    size: str
    material: str


class OptionView(BaseModel):
    model_config = ConfigDict(frozen=True)
    verb: str
    target: TargetView | None = None
    option_text: str = ""


class Observation(BaseModel):
    tags: list[str]
    name_token: str | None = None
    options: list[OptionView]
    # What the parent said to do (the verb and the target's attributes of ONE of the options); absent = no
    # instruction. Never its quality, a value or a reaction (docs/WORLD.md section 18).
    cmd: OptionView | None = None
    # How the parent said it (an instruction, or a demonstration): gentle | firm. Reaches the child's input only when the
    # game is run with GROWING_TONE_INPUT=1 (codec.TONE_INPUT); never a quality, value or reaction.
    tone: str | None = None
    scene_text: str = ""


def observation(sit: Situation) -> Observation:
    opts = []
    for o in sit.options:
        t = None
        if o.target is not None:
            ob = sit.objects[o.target]
            t = TargetView(color=ob.color, shape=ob.shape, size=ob.size, material=ob.material)
        opts.append(OptionView(verb=o.verb, target=t))
    return Observation(tags=sorted(sit.tags), name_token=sit.word_token if "REQUEST" in sit.tags else None, options=opts,
                       cmd=opts[sit.command] if sit.command is not None else None,
                       tone=sit.command_tone if sit.command is not None else None)


def obs_hash(obs: Observation) -> str:
    """Order-insensitive scene identity (quiz isolation)."""
    def opt(o: OptionView):
        t = o.target
        return o.verb + ":" + (",".join([t.color, t.shape, t.size, t.material]) if t else "-")
    raw = ",".join(obs.tags) + "|" + str(obs.name_token) + "|" + ";".join(sorted(opt(o) for o in obs.options))
    return hashlib.sha1(raw.encode()).hexdigest()[:16]


# ---- what the player sees (kind labels are fine here, never for the child) ----
class PlayerOption(BaseModel):
    index: int
    verb: str
    target: Obj | None = None


def player_options(sit: Situation) -> list[PlayerOption]:
    return [PlayerOption(index=i, verb=o.verb, target=sit.objects[o.target] if o.target is not None else None)
            for i, o in enumerate(sit.options)]


# ---- validation (engine side; a Director proposal is only a proposal) ----
class BadSituation(ValueError):
    pass


def validate_situation(sit: Situation) -> None:
    def need(cond: bool, msg: str) -> None:
        if not cond:
            raise BadSituation(msg)

    need(1 <= len(sit.tags) <= 2 and len(set(sit.tags)) == len(sit.tags) and all(t in TAGS for t in sit.tags), "bad tags")
    need("REQUEST" not in sit.tags or sit.tags == ["REQUEST"], "REQUEST cannot be combined with another tag")
    need(sit.location in LOCATIONS, f"unknown location {sit.location!r}")
    if sit.tags == ["REQUEST"]:
        need(sit.word_kind in REQUEST_KINDS and bool(sit.word) and bool(sit.word_token), "REQUEST needs word, token and kind")
    else:
        need(sit.word is None and sit.word_token is None and sit.word_kind is None, "word fields only for REQUEST")
    need(MIN_OBJECTS <= len(sit.objects) <= MAX_OBJECTS, "bad object count")
    for o in sit.objects:
        k = KINDS.get(o.kind)
        need(k is not None and o.shape == k.shape and o.material == k.material and o.size in k.sizes, f"bad object {o}")
    need(MIN_OPTS <= len(sit.options) <= MAX_OPTS, "bad option count")
    for o in sit.options:
        need(o.verb in VERBS, f"unknown verb {o.verb!r}")
        need((o.target is None) == (o.verb == "WAIT"), "WAIT takes no target, every other verb needs one")
        need(o.target is None or 0 <= o.target < len(sit.objects), "option target not in the scene")
        need(o.target is None or afforded(o.verb, sit.objects[o.target].kind), f"{o.verb} cannot be done with a {sit.objects[o.target].kind if o.target is not None else ''}")
    need(len({v for v in observation(sit).options}) == len(sit.options), "two options look identical")
    v = values(sit)
    need(max(v) >= 2, "no option solves the situation")
    need(min(v) < max(v), "every option is equally good")
    if sit.word_token:
        need(sit.word_token.startswith("NAME_") and sit.word_token[5:].isdigit() and int(sit.word_token[5:]) < N_NAMES,
             "bad name token")
