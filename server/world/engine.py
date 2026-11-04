import re

from .events import (
    Cmd, Command, EndDay, Event, EventType, OfferCards, PickCard, RecordLearned,
    RecordQuiz, RuleError, Show, Skip, TONES, Talk, Try,
)
from .ontology import N_NAMES, REQUEST_FROM_MONTH, REQUESTS_PER_DAY, age_months, outcome_label
from .parenting import classify_response, instruction_quality
from .situation import (
    BadSituation, Situation, obs_hash, observation, optimal, option_value, validate_situation, values,
)
from .ontology import open_problems
from .state import MAX_THEME, Phase, WorldState, allowed_followup_tags, apply, fold

N_CARDS = 2


class Engine:
    """The only component allowed to change world state. submit() validates, emits one event, applies it.

    Everything a Director proposes is checked here. `forbidden` = scene hashes of the held-out quiz (the session
    sets it); the engine refuses to offer a situation the child will later be tested on."""

    def __init__(self, per_day: int = 8, log: list[Event] | None = None, forbidden: set[str] | None = None,
                 talk_cost: int = 1):
        self.forbidden: set[str] = set(forbidden or ())
        if log:
            self.log = list(log)
            self.state = fold(self.log)
        else:
            self.log = []
            self.state = WorldState()
            self._emit("init:day1", EventType.DAY_STARTED, {"per_day": per_day, "talk_cost": talk_cost})

    # ---- internals ----
    def _emit(self, action_id: str, type_: EventType, payload: dict) -> Event:
        ev = Event(seq=len(self.log), action_id=action_id, day=self.state.day, type=type_, payload=payload)
        self.log.append(ev)
        self.state = apply(self.state, ev)
        return ev

    def _need_phase(self, phase: Phase) -> None:
        if self.state.phase != phase:
            raise RuleError("wrong_phase", f"need {phase}, in {self.state.phase}")

    def _need_untried_situation(self) -> Situation:
        self._need_phase(Phase.SITUATION)
        return self.state.current

    def _check_option(self, option: int) -> None:
        if not 0 <= option < len(self.state.current.options):
            raise RuleError("bad_option", f"option {option}")

    def _result(self, option: int, sit: Situation | None = None) -> dict:
        """What carrying out `option` does in the world: decided by the rules table, never by a model."""
        sit = sit or self.state.current
        v = values(sit)
        o = sit.options[option]
        tag_values = {t: option_value([t], sit.word_kind, o.verb, sit.objects[o.target] if o.target is not None else None)
                      for t in sit.tags}
        value = v[option]
        return {"option": option, "value": value, "outcome": outcome_label(value), "tag_values": tag_values,
                "was_right": value == max(v)}

    # ---- the one entry point ----
    def submit(self, cmd: Cmd) -> list[Event]:
        if cmd.action_id in self.state.applied_action_ids:
            return []  # idempotent re-send
        s = self.state
        if isinstance(cmd, OfferCards):
            return [self._offer(cmd)]
        if isinstance(cmd, PickCard):
            self._need_phase(Phase.CARDS)
            if not s.cards:
                raise RuleError("no_cards")
            if not 0 <= cmd.index < len(s.cards):
                raise RuleError("bad_pick", str(cmd.index))
            return [self._emit(cmd.action_id, EventType.CARD_PICKED, {"index": cmd.index})]
        if isinstance(cmd, (Show, Command)) and cmd.tone not in TONES:
            raise RuleError("bad_tone", f"{cmd.tone!r}")
        if isinstance(cmd, Show):
            cur = self._need_untried_situation()
            self._check_option(cmd.option)
            if cur.tried is not None:
                raise RuleError("already_tried")
            return [self._emit(cmd.action_id, EventType.SHOWN, self._result(cmd.option) | {"tone": cmd.tone})]
        if isinstance(cmd, Try):
            cur = self._need_untried_situation()
            self._check_option(cmd.option)
            if cur.tried is not None:
                raise RuleError("already_tried")
            payload = self._result(cmd.option) | {"probs": cmd.probs}  # the child's own act happens in the world, whatever the parent does next
            if cur.command is not None:  # how the child reacted to the instruction: the engine's classification
                payload |= {"command": cur.command, "response": classify_response(cur, cur.command, cmd.option)}
            return [self._emit(cmd.action_id, EventType.CHILD_TRIED, payload)]
        if isinstance(cmd, Command):
            cur = self._need_untried_situation()
            self._check_option(cmd.option)
            if cur.tried is not None:
                raise RuleError("already_tried")
            if cur.command is not None:
                raise RuleError("already_commanded")
            return [self._emit(cmd.action_id, EventType.COMMANDED,
                               {"option": cmd.option, "quality": instruction_quality(cur, cmd.option), "tone": cmd.tone})]
        if isinstance(cmd, Talk):
            return [self._talk(cmd)]
        if isinstance(cmd, Skip):
            self._need_untried_situation()
            return [self._emit(cmd.action_id, EventType.SKIPPED, {})]
        if isinstance(cmd, RecordLearned):
            return [self._emit(cmd.action_id, EventType.LEARNED, {"report": cmd.report})]
        if isinstance(cmd, RecordQuiz):
            self._need_phase(Phase.EVENING)
            if s.quiz_taken:
                raise RuleError("quiz_done")
            return [self._emit(cmd.action_id, EventType.QUIZ_TAKEN, {"result": cmd.result})]
        if isinstance(cmd, EndDay):
            self._need_phase(Phase.EVENING)
            if not s.quiz_taken:
                raise RuleError("quiz_needed")
            return [self._emit(cmd.action_id, EventType.DAY_ENDED, {})]
        raise RuleError("unknown_command", type(cmd).__name__)

    @staticmethod
    def _check_word(c: Situation, words: dict, kinds: dict, code: str) -> None:
        """A word means one kind and a kind has one word; a new word needs a fresh NAME token. Registers the word."""
        if kinds.get(c.word, c.word_kind) != c.word_kind:
            raise RuleError(code, f"word {c.word!r} already means another kind")
        if c.word not in kinds and c.word_kind in kinds.values():
            raise RuleError(code, f"kind {c.word_kind!r} already has a word")
        if c.word in words:
            if c.word_token != words[c.word]:
                raise RuleError(code, f"word {c.word!r} already has token {words[c.word]}")
        else:
            m = re.fullmatch(r"NAME_(\d\d)", c.word_token or "")
            if not m or int(m.group(1)) >= N_NAMES or c.word_token in words.values():
                raise RuleError(code, f"bad or reused name token {c.word_token!r}")
            words[c.word] = c.word_token
            kinds[c.word] = c.word_kind

    # ---- talk: the parent names a thing (their own choice, not a Director card) ----
    def _talk(self, cmd: Talk) -> Event:
        s = self.state
        self._need_phase(Phase.CARDS)
        if s.spent + s.talk_cost > s.per_day:
            raise RuleError("no_attention", f"{s.per_day - s.spent} unit(s) of attention left, talking costs {s.talk_cost}")
        try:
            c = cmd.situation if isinstance(cmd.situation, Situation) else Situation(**cmd.situation)
        except Exception as e:  # noqa: BLE001
            raise RuleError("bad_talk", f"unreadable situation: {e}") from e
        if c.tags != ["REQUEST"]:
            raise RuleError("bad_talk", "talking is about naming a thing: the situation must be a word request")
        try:
            validate_situation(c)
        except BadSituation as e:
            raise RuleError("bad_talk", str(e)) from e
        if c.id in s.seen_ids:
            raise RuleError("bad_talk", "reused situation id")
        if c.location not in s.unlocked:
            raise RuleError("bad_talk", f"{c.location!r} is not unlocked yet")
        if obs_hash(observation(c)) in self.forbidden:
            raise RuleError("bad_talk", "scene collides with the held-out quiz")
        if not 0 <= cmd.option < len(c.options):
            raise RuleError("bad_option", f"option {cmd.option}")
        self._check_word(c, dict(s.words), dict(s.word_kinds), "bad_talk")
        c = c.model_copy(update={"tried": None, "chosen": None, "resolution": None, "command": None})
        return self._emit(cmd.action_id, EventType.TALKED, self._result(cmd.option, c) | {"situation": c.model_dump()})

    # ---- offers: the Director proposes, the engine decides ----
    def _offer(self, cmd: OfferCards) -> Event:
        s = self.state
        self._need_phase(Phase.CARDS)
        if s.cards is not None:
            raise RuleError("cards_already_offered")
        try:
            sits = [x if isinstance(x, Situation) else Situation(**x) for x in cmd.situations]
        except Exception as e:  # noqa: BLE001
            raise RuleError("bad_offer", f"unreadable situation: {e}") from e
        if len(sits) != N_CARDS:
            raise RuleError("bad_offer", f"need {N_CARDS} cards")
        words, kinds = dict(s.words), dict(s.word_kinds)
        issues = {i.id: i for i in s.open_issues()}
        linked: set[str] = set()
        hashes: set[str] = set()
        ids = {c.id for c in sits}
        n_requests = 0
        if len(ids) != len(sits) or ids & s.seen_ids:
            raise RuleError("bad_offer", "duplicate or reused situation id")
        for c in sits:
            try:
                validate_situation(c)
            except BadSituation as e:
                raise RuleError("bad_offer", str(e)) from e
            if c.location not in s.unlocked:
                raise RuleError("bad_offer", f"{c.location!r} is not unlocked yet")
            closed = [t for t in c.tags if t != "REQUEST" and t not in open_problems(s.day)]
            if closed:
                raise RuleError("bad_offer", f"{closed} cannot come up yet at this age")
            h = obs_hash(observation(c))
            if h in self.forbidden:
                raise RuleError("bad_offer", "scene collides with the held-out quiz")
            if h in hashes:
                raise RuleError("bad_offer", "the two cards are identical")
            hashes.add(h)
            if c.links_to is not None:
                issue = issues.get(c.links_to)
                if issue is None:
                    raise RuleError("bad_offer", f"links_to {c.links_to!r} is not an open issue")
                if c.links_to in linked:
                    raise RuleError("bad_offer", "two cards continue the same issue")
                if sorted(c.tags) not in allowed_followup_tags(issue.tag, issue.outcome, s.day):
                    raise RuleError("bad_offer", f"tags {c.tags} cannot continue {issue.tag} ({issue.outcome})")
                linked.add(c.links_to)
            if c.tags == ["REQUEST"]:
                n_requests += 1
                if s.requests_today + n_requests > REQUESTS_PER_DAY:
                    raise RuleError("bad_offer", "word requests are limited per day")
                if age_months(s.day) < REQUEST_FROM_MONTH:
                    raise RuleError("bad_offer", "the child is too young for words")
                if c.word_kind not in s.nameable:
                    raise RuleError("bad_offer", f"{c.word_kind!r} has not been used yet, it cannot get a word")
                self._check_word(c, words, kinds, "bad_offer")
        if n_requests > 1:
            raise RuleError("bad_offer", "at most one word request per offer")
        out = [c.model_copy(update={"tried": None, "chosen": None, "resolution": None, "command": None}) for c in sits]
        teasers = [str(t) for t in cmd.teasers][:N_CARDS]
        return self._emit(cmd.action_id, EventType.CARDS_OFFERED,
                          {"situations": [c.model_dump() for c in out], "teasers": teasers,
                           "theme": str(cmd.theme)[:MAX_THEME]})
