"""Synthetic parents (docs/WORLD.md 18.10, 19) that raise the child through the real Session, so every parenting act is an
engine event with an attention cost. Scripts that know the engine truth: for experiments and demos, never for the child.

Stage 1 (the first days) is raised in one of these ways, stage 2 by the same discerning parent. A style differs from
another only in which parent actions happen (so which examples exist); the learner, the model and the quiz are the same."""
import random

from server.world.parenting import quality_buckets
from server.world.state import Phase

from .session import Session

STYLES = {"talker": 4, "commander": 1, "chooser": 1, "balanced": 2, "demonstrator": 1}  # naming units per day in stage 1
STAGE2_NAMING = 1
P_COMMAND_WRONG = 0.25  # share of a commander's instructions that are not the best option (parents err)
STAGE2_MIX = (("good", 0.5), ("useless", 0.25), ("harmful", 0.25))


def choose_card(s: Session, rng: random.Random) -> int:
    """The parent teaches the thing practised least so far, a problem before a word request, a single problem before a
    combination. (Reads the card previews only: tags, no hint of what is good.)"""
    def key(p):
        topic = "+".join(sorted(p["tags"]))
        return (p["tags"] == ["REQUEST"], len(p["tags"]) > 1, s.state.topic_counts.get(topic, 0), rng.random())
    return min(s.view().previews, key=key)["index"]


GENTLE = ("gentle", "gentle")  # (tone of a demonstration and an instruction, tone of a correction): docs/WORLD.md 22.8
TONE_POLICIES = ("gentle", "firm", "half", "wrong", "right")  # the first entry may also be a habit of tone, docs/WORLD.md 22.9


def _tone(s: Session, spec: str, right: bool) -> str:
    """The tone of one instruction or demonstration. `half`: firm on a random half; `wrong`: firm exactly when the parent
    is mistaken (a bossy habit); `right`: firm exactly when the parent is right (a sure habit). Its own random stream, so
    that the parent's other choices stay the same across tone policies."""
    if spec in ("gentle", "firm"):
        return spec
    firm = {"half": s._rng("tone").random() < 0.5, "wrong": not right, "right": right}[spec]
    return "firm" if firm else "gentle"


def _plain_show(s: Session, rng: random.Random, tones: tuple = GENTLE) -> None:
    s.show(rng.choice(s.debug_best()), tone=_tone(s, tones[0], True))


def pick_instruction(vals: list[int], verbs: list[str], rng: random.Random, p_wrong: float = P_COMMAND_WRONG,
                     wrong_kind: str = "any") -> int:
    """The option a commanding parent tells the child to take: the best one, except `p_wrong` of the time (parents err).
    `wrong_kind` says what the mistake looks like: any non-best option, or one of the useless / harmful ones (any non-best
    option if that bucket is empty on this scene). Shared by the harness and the engine path."""
    b = quality_buckets(vals, verbs)
    wrong = [i for i in range(len(vals)) if i not in b["good"]]
    pool = wrong if wrong_kind == "any" else (b[wrong_kind] or wrong)
    return rng.choice(pool) if pool and rng.random() < p_wrong else rng.choice(b["good"])


def _command_show(s: Session, rng: random.Random, p_wrong: float = P_COMMAND_WRONG, wrong_kind: str = "any",
                  tones: tuple = GENTLE) -> None:
    """'Do as I say': the child is told, and shown. Some of the time the parent is wrong."""
    cur = s.state.current
    idx = pick_instruction(_values(s), [o.verb for o in cur.options], rng, p_wrong, wrong_kind)
    tone = _tone(s, tones[0], idx in quality_buckets(_values(s), [o.verb for o in cur.options])["good"])
    s.command(idx, tone=tone)
    s.show(idx, tone=tone)


def _offer(s: Session, rng: random.Random, tones: tuple = GENTLE) -> None:
    """The parent leaves it to the child: it chooses and does it, and learns from what comes of it."""
    s.try_child()


def _discerning(s: Session, rng: random.Random, tones: tuple = GENTLE) -> None:
    """Stage 2 for every parent: say something to the child (half good, a quarter useless, a quarter harmful); the child then chooses."""
    cur = s.state.current
    by_q = quality_buckets(_values(s), [o.verb for o in cur.options])
    quality = rng.choices([q for q, _ in STAGE2_MIX], weights=[w for _, w in STAGE2_MIX])[0]
    if not by_q[quality]:
        quality = "good" if quality == "harmful" or not by_q["useless"] else "useless"
    s.command(rng.choice(by_q[quality]), tone=_tone(s, tones[0], quality == "good"))
    s.try_child()


def _values(s: Session) -> list[int]:
    from server.world.situation import values
    return values(s.state.current)


def play_day(s: Session, style: str, stage: int, rng: random.Random, free: float = 0.0,
             p_wrong: float = P_COMMAND_WRONG, wrong_kind: str = "any", tones: tuple = GENTLE) -> None:
    """One day of attention: `n_name` units of talking, the rest on problems, in a random order. `free` (stage 2): the
    share of problem units where the parent gives no instruction and lets the child choose. `p_wrong`, `wrong_kind`:
    how often and how a commanding parent errs in stage 1. `tones`: the tone of the parent's demonstrations and corrections."""
    n_name = STYLES[style] if stage == 1 else STAGE2_NAMING
    units = ["name"] * n_name + ["problem"] * (s.game.per_day - n_name)
    rng.shuffle(units)
    for u in units:
        st = s.state
        if st.phase == Phase.EVENING:
            break
        if u == "name" and st.spent + st.talk_cost <= st.per_day:
            s.talk()
            continue
        s.draw_cards()
        s.pick_card(choose_card(s, rng))
        if stage == 2:
            (_offer if rng.random() < free else _discerning)(s, rng, tones)
        elif style in ("talker", "demonstrator"):
            _plain_show(s, rng, tones)
        elif style == "commander":
            _command_show(s, rng, p_wrong, wrong_kind, tones)
        elif style == "chooser":
            _offer(s, rng, tones)
        else:
            rng.choice([_plain_show, lambda s_, r_, t_: _command_show(s_, r_, p_wrong, wrong_kind, t_), _offer])(s, rng, tones)
    while s.state.phase != Phase.EVENING:  # units were spent on a combo etc.: use the rest of the day as a plain demonstration
        s.draw_cards()
        s.pick_card(choose_card(s, rng))
        _plain_show(s, rng, tones)
