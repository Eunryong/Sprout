"""Game loop. CLI and (later) FastAPI only call this class."""
import random
import shutil
import uuid
from dataclasses import asdict, replace

from pydantic import BaseModel

from server.agents import narrator
from server.agents.factory import make_client
from server.agents.director import DirectorAgent
from server.agents.narrator import NarratorAgent
from server.agents.schemas import DirectorContext
from server.config import GameCfg, load_configs
from server.learning import checkpoints
from server.learning.codec import Example, Fact, option_probs, seen_counts
from server.learning.evaluator import (
    by_topic, command_panel, evaluate, load_quiz, make_quiz, pick_daily, quiz_topics, save_quiz, score, tag_reliance,
)
from server.learning.learner import LearnReport, Learner, OnlineCfg
from server.learning.model import ModelCfg, build_model
from server.learning.policy import act
from server.world.engine import Engine
from server.world.events import (
    Command, EndDay, OfferCards, PickCard, RecordLearned, RecordQuiz, RuleError, Show, Skip, Talk, Try,
)
from server.world.generator import make_situation
from server.world.ontology import (
    ESCALATION, GAME, KINDS, ONTOLOGY_HASH, ONTOLOGY_PATH, PROBLEMS, REQUEST_KINDS, REQUESTS_PER_DAY, RULES, WAIT_VALUE, age_months, voice_of,
)
from server.world.worldgen import game_overrides, line_for, read_sheet, sheet_hash
from server.world.parenting import journal
from server.world.situation import Situation, observation, optimal, player_options
from server.world.ontology import outcome_label
from server.world.state import Phase

from .saves import SaveDir

DEFAULT = object()


class ActionResult(BaseModel):
    duplicate: bool = False
    narration: str = ""
    option: int | None = None
    was_right: bool | None = None
    value: int | None = None
    outcome: str | None = None
    probs: list[float] = []
    followed: bool | None = None  # a try after an instruction: did the child take the option the parent told it to?
    notes: str = ""  # what changed in the world because of the choice
    learn: LearnReport | None = None


class OptionCard(BaseModel):
    index: int
    verb: str
    text: str  # player-facing description (kind labels are fine here, never for the child)
    kind: str | None = None
    color: str | None = None
    size: str | None = None


class SceneView(BaseModel):
    day: int
    age_months: int = 0  # the child's age today (0 = no age schedule)
    done: int
    per_day: int
    phase: str
    location: str = ""
    narration: str = ""
    hint: str | None = None
    options: list[OptionCard] = []
    previews: list[dict] = []
    theme: str = ""
    tried: int | None = None
    command: int | None = None  # the option the parent told the child to take in this situation
    spent: int = 0  # attention used today (of per_day)


def default_client(word_map: dict):
    return make_client(word_map)


class Session:
    def __init__(self, save: SaveDir, meta: dict, engine: Engine, model, model0, learner: Learner,
                 buffer: list[Example], quiz: list, client):
        self.save, self.meta, self.engine = save, meta, engine
        self.model, self.model0, self.learner = model, model0, learner
        self.buffer, self.quiz = buffer, quiz
        self.seed = meta["seed"]
        self.game = GameCfg(**meta["game"])
        self.word_map = {k: tuple(v) for k, v in meta["word_map"].items()}
        self.forbidden = {q.hash for q in quiz}
        engine.forbidden = set(self.forbidden)
        client = default_client(self.word_map) if client is DEFAULT else client
        self.client = client
        self.director = DirectorAgent(client, self.word_map, on_call=save.append_llm)
        self.narrator = NarratorAgent(client, on_call=save.append_llm)
        self.warnings: list[str] = []

    # ---- construction ----
    @classmethod
    def new(cls, save_dir, seed: int = 0, game: GameCfg | None = None, mcfg: ModelCfg | None = None,
            ocfg: OnlineCfg | None = None, client=DEFAULT) -> "Session":
        g0, m0, o0 = load_configs()
        game, mcfg, ocfg = game or replace(g0, **{**GAME, **game_overrides(ONTOLOGY_PATH)}), mcfg or m0, ocfg or o0  # a generated world sets what it teaches
        save = SaveDir(save_dir)
        save.create()
        for src, dst in ((ONTOLOGY_PATH, save.root / "ontology.yaml"), (ONTOLOGY_PATH.with_name("world.json"), save.root / "world.json")):
            if src.exists() and not (dst.exists() and src.resolve() == dst.resolve()):
                shutil.copy(src, dst)  # the world this child grows up in, kept with the save
        rng = random.Random(seed)
        words = list(game.words)
        rng.shuffle(words)
        word_map = {k: (words[i], f"NAME_{i:02d}") for i, k in enumerate(REQUEST_KINDS)}
        mcfg.seed = seed
        model = build_model(mcfg)
        checkpoints.save(model, save.ckpt0, meta={"role": "initial"}, overwrite=False)
        topics = quiz_topics(word_map, [[t, e] for t, e in ESCALATION.items()])
        quiz = make_quiz(random.Random(seed + 10_000), topics, game.quiz_per_topic)
        save_quiz(quiz, save.quiz)
        meta = {"seed": seed, "game": asdict(game), "model": mcfg.to_dict(), "online": asdict(ocfg),
                "word_map": {k: list(v) for k, v in word_map.items()}, "ontology": ONTOLOGY_HASH}
        engine = Engine(per_day=game.per_day, talk_cost=game.talk_cost)
        for ev in engine.log:
            save.append_event(ev)
        save.write_meta(meta)
        model0 = checkpoints.load(save.ckpt0)[0]
        return cls(save, meta, engine, model, model0, Learner(model, ocfg), [], quiz, client)

    @classmethod
    def load(cls, save_dir, client=DEFAULT) -> "Session":
        save = SaveDir(save_dir)
        if not save.exists():
            raise FileNotFoundError(f"no save at {save_dir}")
        meta = save.read_meta()
        if meta.get("ontology") != ONTOLOGY_HASH:
            raise ValueError(f"this save was made with ontology {meta.get('ontology')}, the game has {ONTOLOGY_HASH}")
        ocfg = OnlineCfg(**meta["online"])
        engine = Engine(log=save.read_events())
        buffer = save.read_examples()
        src = save.latest if save.latest.exists() else save.ckpt0
        model, ck_meta = checkpoints.load(src)
        learner = Learner(model, ocfg)
        learner.n_calls = len(buffer)
        checkpoints.load_optimizer_state(src, learner.opt)
        s = cls(save, meta, engine, model, checkpoints.load(save.ckpt0)[0], learner, buffer,
                load_quiz(save.quiz), client)
        n_events = sum(e.type.value == "learned" for e in engine.log)
        if not (len(buffer) == n_events == ck_meta.get("n_learned", 0)):
            s.warnings.append(f"learning log mismatch: examples={len(buffer)} learned_events={n_events} "
                              f"checkpoint={ck_meta.get('n_learned', 0)} (interrupted save?)")
        return s

    def quiz_topic_names(self) -> set[str]:
        return {q.topic for q in self.quiz}

    def llm_stats(self) -> dict:
        return {"client": getattr(self.client, "name", "none"), "director": self.director.stats.summary(), "narrator": self.narrator.stats.summary(),
                "teasers_dropped": self.director.teasers_dropped}

    # ---- helpers ----
    @property
    def state(self):
        return self.engine.state

    @property
    def rng(self) -> random.Random:
        return self._rng("session")

    def _rng(self, salt: str) -> random.Random:
        return random.Random(f"{self.seed}:{len(self.engine.log)}:{salt}")  # same history -> same randomness

    def _submit(self, cmd):
        evs = self.engine.submit(cmd)  # raises RuleError before anything is persisted
        for ev in evs:
            self.save.append_event(ev)
        return evs

    @staticmethod
    def _aid(action_id: str | None) -> str:
        return action_id or uuid.uuid4().hex

    def debug_best(self) -> list[int]:
        """Engine truth (the best options). For synthetic players and tests only; the child never gets this."""
        cur = self.state.current
        return optimal(cur) if cur else []

    # ---- view ----
    def _still(self, sit: Situation) -> tuple[str, ...]:
        return tuple(i.tag for i in self.state.issues if i.id == sit.links_to)

    def _voice(self) -> str:
        """Whose scenes these are at the child's age ('baby': the baby is the one cared for)."""
        return voice_of(age_months(self.state.day))

    def view(self) -> SceneView:
        st = self.state
        v = SceneView(day=st.day, age_months=age_months(st.day), done=st.situations_done, per_day=st.per_day, phase=st.phase.value, spent=st.spent)
        if st.phase == Phase.CARDS and st.cards:
            v.theme = st.day_theme
            for i, c in enumerate(st.cards):
                teaser = st.card_teasers[i] if i < len(st.card_teasers) else ""
                text, _ = self.narrator.intro(c, False, self._still(c), self._voice())
                v.previews.append({"index": i, "location": c.location, "tags": c.tags, "teaser": teaser or text,
                                   "followup": c.links_to is not None, "word": c.word})
        elif st.phase == Phase.SITUATION:
            c = st.current
            v.location = c.location
            v.narration, v.hint = self.narrator.intro(c, st.current_first, self._still(c), self._voice())
            v.tried = c.tried
            v.command = c.command
            for po in player_options(c):
                t = po.target
                v.options.append(OptionCard(
                    index=po.index, verb=po.verb, text=narrator.option_text(po.verb, t, self._voice(), tags=list(c.tags)),
                    kind=t.kind if t else None, color=t.color if t else None, size=t.size if t else None))
        return v

    # ---- player actions ----
    def draw_cards(self, action_id: str | None = None) -> None:
        if self.state.cards is not None:
            return
        aid = self._aid(action_id)
        rng = self._rng("cards")
        ctx = DirectorContext.from_state(self.state, REQUESTS_PER_DAY)
        sits = self.director.offer(ctx, rng, self.forbidden)
        try:
            self._submit(OfferCards(action_id=aid, situations=sits, teasers=self.director.last_teasers,
                                    theme=self.director.last_theme))
        except RuleError:  # engine disagrees with the Director: use the validated template instead
            template = DirectorAgent(None, self.word_map, on_call=self.save.append_llm)
            sits = template.offer(ctx, rng, self.forbidden)
            self.director.last_source, self.director.last_error = "template", "engine rejected the proposal"
            self._submit(OfferCards(action_id=aid, situations=sits, teasers=template.last_teasers))

    def pick_card(self, index: int, action_id: str | None = None) -> None:
        self._submit(PickCard(action_id=self._aid(action_id), index=index))

    def _apply_world(self) -> None:
        """Ensure active world rules are applied."""
        pass

    def _context(self):
        cur = self.state.current
        if cur is None:
            return None, None
        return cur.model_copy(deep=True), observation(cur)

    @staticmethod
    def _mode(obs, tone: str) -> str:
        return ("command_" if obs.cmd is not None else "show_") + (obs.tone or tone)

    @staticmethod
    def _said(obs, tone: str):
        """A demonstration is said in a tone too; an instruction already carries the tone it was given in."""
        return obs if obs.tone is not None else obs.model_copy(update={"tone": tone})

    def _notes(self, before) -> str:
        after = self.state
        unlocked = [p for p in after.unlocked if p not in before.unlocked]
        opened = [i.tag for i in after.issues if i.open and i.id not in {x.id for x in before.issues}]
        named = [k for k in after.nameable if k not in before.nameable]
        return narrator.world_note(unlocked, opened, named)

    def _result(self, cur: Situation, ev, who: str, before) -> ActionResult:
        p = ev.payload
        after_unlocked = self.state.unlocked
        narration = self.narrator.outcome(cur, p["option"], p["outcome"], who, p["tag_values"],
                                          tuple(after_unlocked), self._voice())
        if isinstance(narration, tuple):
            narration = narration[0]
        return ActionResult(narration=narration,
                            option=p["option"], was_right=p["was_right"], value=p["value"], outcome=p["outcome"],
                            notes=self._notes(before))

    def _emphasis(self, tone: str) -> dict:
        """How a firm tone changes the example: more repeats now, or a heavier weight for good (game.firm_mode)."""
        if tone != "firm":
            return {}
        return {"weight": float(self.game.firm_repeat)} if self.game.firm_mode == "weight" else {"repeat": self.game.firm_repeat}

    @staticmethod
    def _fact(obs, option: int, outcome: str) -> Fact:
        """What happened when this option was done in this scene (docs/WORLD.md 22.14); `said` = the parent had told the child to."""
        o = obs.options[option]
        return Fact(option=o, result=outcome, said=obs.cmd is not None and o == obs.cmd)

    def _sting(self, rounds: int, fact: Fact) -> float:
        """Doing as the parent said and having it go badly stays with the child longer (the world sets how much, GameCfg.burn_*)."""
        return float(max(rounds, self.game.burn_rounds(fact.result) if fact.said else 0))

    def show(self, option: int, action_id: str | None = None, tone: str = "gentle",
             override_outcome: str | None = None, action_text: str = "", outcome_text: str = "",
             takeaway: str = "") -> ActionResult:
        """The parent does it and the child watches what comes of it, whatever that is."""
        if action_id in ("gentle", "firm", "shouting") and tone == "gentle":
            tone = action_id
            action_id = None
        cur, obs = self._context()
        before = self.state
        evs = self._submit(Show(action_id=self._aid(action_id), option=option, tone=tone))
        if not evs:
            return ActionResult(duplicate=True)
        res = self._result(cur, evs[0], "player", before)
        res_outcome = override_outcome if override_outcome in ("solved", "partial", "nothing", "worse") else res.outcome
        fact = self._fact(obs, option, res_outcome)
        if action_text:
            fact.action_text = action_text
            try:
                fact.option.option_text = action_text
            except Exception:
                object.__setattr__(fact.option, "option_text", action_text)
        if outcome_text:
            fact.outcome_text = outcome_text
        if hasattr(fact, "takeaway") and takeaway:
            fact.takeaway = takeaway
        elif takeaway:
            object.__setattr__(fact, "takeaway", takeaway)
        return self._learn(Example(id=f"ex{len(self.buffer):04d}", day=self.state.day, source="show", obs=self._said(obs, tone), mode=self._mode(obs, tone),
                                   strength=self._sting(self.game.firm_rounds if self._said(obs, tone).tone == "firm" else 1, fact), chosen=option,
                                   was_right=res.was_right, outcome=res_outcome, facts=[fact], **self._emphasis(tone)), res)

    def try_child(self, action_id: str | None = None, override_outcome: str | None = None,
                  action_text: str = "", outcome_text: str = "", takeaway: str = "") -> ActionResult:
        """The child chooses and does it. The world answers at once, and that is what it learns from."""
        cur, obs = self._context()
        if cur is None:
            raise RuleError("wrong_phase", "no situation in progress")
        before = self.state
        a = act(self.model, obs, self._rng("try"), seen=seen_counts(self.buffer), beta=self.game.curiosity)
        evs = self._submit(Try(action_id=self._aid(action_id), option=a.option, probs=a.probs))
        if not evs:
            return ActionResult(duplicate=True)
        res = self._result(cur, evs[0], "child", before)
        res.probs, res.followed = a.probs, None if cur.command is None else a.option == cur.command
        res_outcome = override_outcome if override_outcome in ("solved", "partial", "nothing", "worse") else res.outcome
        fact = self._fact(obs, a.option, res_outcome)
        if action_text:
            fact.action_text = action_text
            try:
                fact.option.option_text = action_text
            except Exception:
                object.__setattr__(fact.option, "option_text", action_text)
        if outcome_text:
            fact.outcome_text = outcome_text
        if hasattr(fact, "takeaway") and takeaway:
            fact.takeaway = takeaway
        elif takeaway:
            object.__setattr__(fact, "takeaway", takeaway)
        own_boost = float(self.game.own_rounds) if cur.command is None else 1.0
        strength = max(own_boost, self._sting(self.game.correction_rounds(res_outcome), fact))
        return self._learn(Example(id=f"ex{len(self.buffer):04d}", day=self.state.day, source="try", obs=obs, chosen=a.option, was_right=res.was_right,
                                   outcome=res_outcome, mode="own_try", strength=strength,
                                   facts=[fact]), res)

    def apply_api_outcome(self, outcome: str, action_text: str, outcome_text: str, takeaway: str = "") -> None:
        """When the API subsequently provides the real outcome and text for the chosen action, TinyLM learns it."""
        if not self.buffer or not self.buffer[-1].facts:
            return
        last_ex = self.buffer[-1]
        fact = last_ex.facts[0]
        if outcome in ("solved", "partial", "nothing", "worse"):
            fact.result = outcome
            last_ex.outcome = outcome
        if action_text:
            fact.action_text = action_text
            try:
                fact.option.option_text = action_text
            except Exception:
                object.__setattr__(fact.option, "option_text", action_text)
        if outcome_text:
            fact.outcome_text = outcome_text
        if hasattr(fact, "takeaway") and takeaway:
            fact.takeaway = takeaway
        elif takeaway:
            object.__setattr__(fact, "takeaway", takeaway)
        rng = getattr(self, "rng", self._rng("learn"))
        self.learner.learn(last_ex, self.buffer, rng)
        checkpoints.save(self.model, self.save.latest, opt=self.learner.opt,
                         meta={"n_learned": len(self.buffer), "events": len(self.engine.log)})

    def command(self, option: int, action_id: str | None = None, tone: str = "gentle") -> ActionResult:
        """Tell the child which option to take. Any option can be told (parents err). The engine records how good the
        instruction was; the player is not told, and the child only ever sees the option's verb and attributes."""
        cur, _ = self._context()
        evs = self._submit(Command(action_id=self._aid(action_id), option=option, tone=tone))
        if not evs:
            return ActionResult(duplicate=True)
        po = player_options(cur)[option]
        return ActionResult(option=option, narration=f"You tell the child: {narrator.option_text(po.verb, po.target, self._voice(), tags=list(cur.tags))}")

    def talk(self, kind: str | None = None, action_id: str | None = None) -> ActionResult:
        """Name a thing while handing it over (a word demonstration of the parent's own choosing, outside the Director's
        cards). The parent holds the thing they name, so the option shown is the one that fetches it."""
        if self.state.phase != Phase.CARDS:
            raise RuleError("wrong_phase", "talk between situations")
        rng = self._rng("talk")
        if kind is None:  # the least-named thing so far
            n = {k: self.state.topic_counts.get(f"REQUEST:{k}", 0) for k in REQUEST_KINDS}
            kind = rng.choice([k for k in REQUEST_KINDS if n[k] == min(n.values())])
        if kind not in self.word_map:
            raise RuleError("bad_talk", f"{kind!r} cannot be named")
        word, token = self.word_map[kind]
        sit = make_situation(rng, ["REQUEST"], f"talk{len(self.engine.log):04d}", "house", word, token, kind,
                             forbidden=self.forbidden)
        option = rng.choice(optimal(sit))
        obs = observation(sit)
        evs = self._submit(Talk(action_id=self._aid(action_id), situation=sit, option=option))
        if not evs:
            return ActionResult(duplicate=True)
        p = evs[0].payload
        res = ActionResult(narration=f"You hold up the {kind} and say \"{word}\".", option=option, was_right=p["was_right"],
                           value=p["value"], outcome=p["outcome"])
        return self._learn(Example(id=f"ex{len(self.buffer):04d}", day=self.state.day, source="talk", obs=obs,
                                   chosen=option, weight=1.0, was_right=p["was_right"], mode="talk",
                                   facts=[self._fact(obs, option, p["outcome"])]), res)

    def skip(self, action_id: str | None = None) -> ActionResult:
        evs = self._submit(Skip(action_id=self._aid(action_id)))
        return ActionResult(duplicate=not evs, narration="You let it pass. Nothing is learned.")

    def _learn(self, ex: Example, res: ActionResult) -> ActionResult:
        """Record the example, learn immediately (`ex.strength` rounds), checkpoint, then log the LEARNED event."""
        self.save.append_example(ex)
        self.buffer.append(ex)
        rep = self.learner.learn(ex, self.buffer, self._rng("learn"))
        for r in range(1, max(1, round(ex.strength))):  # a stronger lesson is learned again, in a further round
            more = self.learner.learn(ex, self.buffer, self._rng(f"learn{r}"))
            rep = rep.model_copy(update={"steps": rep.steps + more.steps, "loss_after": more.loss_after,
                                         "seconds": rep.seconds + more.seconds, "batch_size": more.batch_size})
        checkpoints.save(self.model, self.save.latest, opt=self.learner.opt,
                         meta={"n_learned": len(self.buffer), "events": len(self.engine.log)})
        self._submit(RecordLearned(action_id=f"learned-{ex.id}", report=rep.model_dump()))
        res.learn = rep
        return res

    # ---- evening ----
    def met_topics(self) -> set[str]:
        return {t for t, n in self.state.topic_counts.items() if n > 0}

    def end_day(self) -> dict:
        st = self.state
        if st.phase != Phase.EVENING:
            raise RuleError("wrong_phase", "the day is not over yet")
        met = self.met_topics()
        items = pick_daily(self.quiz, met, self._rng("quiz"), self.game.quiz_per_day)
        res = evaluate(self.model, self.model0, items)
        pool_items = [q for q in self.quiz if q.topic in met]
        pool = evaluate(self.model, self.model0, pool_items)
        res.update(panel=self.panel(), journal=self.journal().get(st.day, {}))
        res.update(day=st.day, pool_optimal=pool["optimal"], pool_p_opt=pool["p_opt"],
                   pool_untrained_optimal=pool["untrained_optimal"], pool_random=pool["random"], pool_n=pool["n"],
                   by_topic=by_topic(self.model, pool_items))
        self._submit(RecordQuiz(action_id=f"quiz-{st.day}", result=res))
        checkpoints.save(self.model, self.save.day_ckpt(st.day), meta={"day": st.day})
        self._submit(EndDay(action_id=f"endday-{st.day}"))
        return res

    def panel(self) -> dict:
        """What the way of raising left in the child, on held-out scenes of the problems it has met (docs/WORLD.md 19)."""
        return command_panel(self.model, self.quiz, self.met_topics() & set(PROBLEMS))

    def reveal(self) -> dict:
        """The rules of this world and how well the child learned each one (docs/WORLD.md 22.13). Only after the last day:
        during the game the parent finds the rules out through what happens. `p_pick` = how likely the child is to take the
        option of that relation on held-out scenes that have it, `chance` = what picking at random would give."""
        if self.state.day <= self.game.days:
            raise RuleError("not_over", f"the rules are shown after day {self.game.days}")
        attrs = {k: (spec.shape, spec.material) for k, spec in KINDS.items()}
        probs: dict[str, list[float]] = {}
        rules = []
        sheet = self.world_sheet()
        for tag, rows in RULES.items():
            items = [q for q in self.quiz if q.obs.tags == [tag]]
            for (verb, kind), val in sorted(rows.items(), key=lambda kv: (-kv[1], kv[0])):
                ps, chance = [], []
                for q in items:
                    idx = [i for i, o in enumerate(q.obs.options)
                           if o.verb == verb and o.target is not None and (o.target.shape, o.target.material) == attrs[kind]]
                    if idx:
                        pr = probs.setdefault(q.hash, option_probs(self.model, q.obs))
                        ps.append(pr[idx[0]])
                        chance.append(1 / len(q.obs.options))
                rules.append({"problem": tag, "verb": verb, "kind": kind, "value": val, "outcome": outcome_label(val),
                              "met": self.state.topic_counts.get(tag, 0), "scenes": len(ps), "line": line_for(sheet, [tag], verb, kind),
                              "p_pick": sum(ps) / len(ps) if ps else None, "chance": sum(chance) / len(chance) if chance else None})
        return {"world": sheet_hash(sheet) if sheet else ONTOLOGY_HASH, "premise": sheet.premise if sheet else "",
                "rules": rules, "wait": dict(WAIT_VALUE)}

    def world_sheet(self):
        """The Director's (or template's) sheet of this child's world; None for the hand-written one."""
        return read_sheet(self.save.root / "ontology.yaml")

    def journal(self) -> dict[int, dict]:
        """Parenting journal per day (talk / show / command / offer / praise / correct counts and the child's reactions),
        computed from the event log."""
        return journal(self.engine.log)

    # ---- "inside the child's head" (P2: per-topic skill + tag reliance; the attribute probe comes later) ----
    def mind(self, compare_day: int | None = None) -> dict:
        """Per-topic accuracy on the held-out pool for every topic the child has met, and whether it reads the
        situation tag. With compare_day, also that day's snapshot. Deterministic; never touches the model."""
        met = self.met_topics()
        items = [q for q in self.quiz if q.topic in met]
        before = None
        if compare_day is not None:
            before = checkpoints.load(self.save.day_ckpt(compare_day))[0] if compare_day > 0 else self.model0
        topics = {}
        for t in sorted(met):
            sub = [q for q in items if q.topic == t]
            e = {"accuracy": score(self.model, sub)["optimal"], "random": score(self.model, sub)["random"],
                 "n": len(sub), "practised": self.state.topic_counts[t]}
            if before is not None:
                e["before"] = score(before, sub)["optimal"]
            topics[t] = e
        out = {"topics": topics, "tag_reliance": tag_reliance(self.model, items)}
        if before is not None:
            out["tag_reliance_before"] = tag_reliance(before, items)
        return out
