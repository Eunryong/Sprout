"""The simple-choice game (docs/DIRECTION.md): one scene = options, the parent takes part once, the child acts, the world
answers, the child learns what happened. Days, cards, attention and quizzes still run underneath (Session), but this layer
never shows them: the player sees scenes only.

  scene()          the situation now and what the child expects of each option (its own guess, never the engine's rules)
  play("say", n)   tell the child to do option n, then the child chooses
  play("show", n)  do option n yourself; the child watches
  play("leave")    say nothing; the child chooses alone
"""
import random
from typing import Literal

from pydantic import BaseModel, Field

from server.learning.codec import Example, Fact, RESULTS, encode_try, novelty, novelty_key, option_probs, result_logits, seen_counts
from server.learning.knowledge import (
    EpisodeNode,
    KnowledgeGraph,
    cognitive_status,
    expand_knowledge,
    load_knowledge,
    principle_to_example,
    save_knowledge,
)
from server.learning.policy import act
from server.world.generator import make_situation
from server.world.ontology import ACTS_FROM_MONTH, MONTHS_PER_DAY, PROBLEMS, STAGES, VERBS, age_months, open_problems, outcome_label, stage_of, voice_of
from server.world.situation import Obj, OptionSpec, Situation, observation, option_value, player_options, values
from server.world.state import EventType, Phase

from server.world.worldgen import line_for

from server import i18n
from .explain import why_facts
from server.agents import narrator
from server.agents.narrator import after_words

from .memory import action_words, leanings, remember, will_remember
from .session import ActionResult, Session

Part = Literal["say", "show", "leave", "solo"]


def current_lang() -> str:
    return i18n.lang()


def _set_scene_text(obs, text: str) -> None:
    try:
        obs.scene_text = text
    except Exception:
        object.__setattr__(obs, "scene_text", text)


def _set_option_text(opt, text: str) -> None:
    try:
        opt.option_text = text
    except Exception:
        object.__setattr__(opt, "option_text", text)


def expectations(s: Session, obs) -> list[dict[str, float]]:
    """For every option: how likely the child thinks each result is (solved / partial / nothing / worse)."""
    import torch
    s.model.eval()
    with torch.no_grad():
        p = torch.softmax(result_logits(s.model, [encode_try(obs, o, obs.cmd is not None and o == obs.cmd) for o in obs.options]), dim=-1)
    return [dict(zip(RESULTS, row)) for row in p.tolist()]


class SimpleOption(BaseModel):
    index: int
    text: str
    speech: str = ""  # what the speaking child (age 3+) says or asks for this option
    expects: dict[str, float]  # result -> probability, the child's own guess
    seen: int = 0  # how many times the child has lived this kind of act on this kind of thing
    pull: float = 0.0  # curiosity: what its newness adds to what the child hopes for (fades as it is lived)
    memory: str = ""  # memory from the child's own history; '' if never lived
    lean: str = ""  # "remembers" | "curious": the child seems drawn to this option (its own odds of choosing it), '' otherwise


class SimpleScene(BaseModel):
    number: int  # scenes lived so far + 1
    step: int = 1  # 1 = with parent (request + guidance), 2 = child's solo moment
    step_total: int = 1  # 2 from age 3+ when two_step is enabled, 1 otherwise
    narration: str
    options: list[SimpleOption]
    age_months: int = 0
    age: str = ""  # "1 year 6 months"
    stage: dict | None = None  # {from, name, blurb} of the life stage
    stage_first: bool = False  # the first scene of this stage: the page shows the stage's introduction
    can_act: bool = True  # False: the child is too young to choose, the parent can only show
    ended: bool = False  # the story is over (the child is grown): see SimpleGame.summary()
    cognitive: dict = Field(default_factory=dict)


SimpleView = SimpleScene


def age_text(months: int) -> str:
    return i18n.age_text(months)


class Summary(BaseModel):
    age: str
    scenes: int
    stages: list[dict]  # [{name, scenes}]
    took_part: dict  # {showed, told, left}: how the parent took part
    moments: list[str]  # plain lines of what the child lived, latest of each trouble
    cognitive: dict = Field(default_factory=dict)


class SimpleTurn(BaseModel):
    part: Part  # what the parent or child did
    told: int | None  # option the parent said or showed
    acted: int  # option that happened (the parent's when shown, the child's otherwise)
    by: Literal["parent", "child"]
    followed: bool | None  # the child did what the parent said (None: the parent did not say anything)
    child_said: str = ""  # what the child said when making or acting on this choice
    narration: str  # what happened
    result: str  # solved / partial / nothing / worse
    learned: str  # "<option text> -> <result>"
    loss_before: float
    loss_after: float
    why: dict | None = None  # when the result surprised the child: the engine's facts (docs: explain.py)
    explanation: str = ""  # those facts in words (Narrator, or the fixed wording)
    remembers: str = ""  # what the child learned from this experience and its thought for next time
    world_says: str = ""  # the Director's line on why the world works so, once the result is in ('' when it has none)
    step: int = 1
    step_total: int = 1
    has_next_step: bool = False  # True after Step 1 when Step 2 (child's solo moment) follows immediately
    custom_action: str | None = None
    cognitive: dict = Field(default_factory=dict)
    synthesized_principle: str = ""
    expanded_concepts: list[str] = Field(default_factory=list)
    what_learned: str = ""


TurnOutcome = SimpleTurn


def _match_custom_action_to_option(
    custom_text: str, sit: Situation, objects: list[Obj]
) -> tuple[int, str, str | None]:
    text = custom_text.strip().lower()

    # 1. Match against existing options
    for idx, o in enumerate(sit.options):
        v = o.verb.lower()
        k = (objects[o.target].kind.lower() if o.target is not None and o.target < len(objects) else None)
        if v in text or (k and k in text):
            return idx, o.verb, k

    # 2. Match dynamically against VERBS and sit.objects
    inferred_verb: str | None = None
    for v in VERBS:
        if v.lower() in text:
            inferred_verb = v
            break
    if inferred_verb is None:
        inferred_verb = "WAIT" if "WAIT" in VERBS else (VERBS[0] if VERBS else "WAIT")

    target_idx: int | None = None
    inferred_kind: str | None = None
    for idx, obj in enumerate(objects):
        if obj.kind.lower() in text:
            target_idx = idx
            inferred_kind = obj.kind
            break

    # Check if (inferred_verb, inferred_kind) matches existing option
    for idx, o in enumerate(sit.options):
        k = (objects[o.target].kind if o.target is not None and o.target < len(objects) else None)
        if o.verb == inferred_verb and (target_idx is None or o.target == target_idx):
            return idx, inferred_verb, k

    new_opt = OptionSpec(verb=inferred_verb, target=target_idx)
    if new_opt in sit.options:
        return sit.options.index(new_opt), inferred_verb, inferred_kind

    sit.options.append(new_opt)
    return len(sit.options) - 1, inferred_verb, inferred_kind


def _map_bundle_options_to_situation(sit: Situation, b_opts: list[str]) -> Situation:
    """Map bundle options index-for-index to sit.options or match dynamically against sit.objects / VERBS."""
    objs = list(sit.objects)
    mapped_opts: list[OptionSpec] = []
    used_indices: set[int] = set()

    for i, opt_text in enumerate(b_opts):
        text = opt_text.strip().lower()
        matched_idx: int | None = None

        # 1. Match against existing options on sit if text contains verb or kind
        for idx, o in enumerate(sit.options):
            if idx in used_indices:
                continue
            v = o.verb.lower()
            k = (objs[o.target].kind.lower() if o.target is not None and 0 <= o.target < len(objs) else None)
            if v in text or (k and k in text):
                matched_idx = idx
                break

        # 2. Index-for-index mapping to sit.options
        if matched_idx is None and i < len(sit.options) and i not in used_indices:
            matched_idx = i

        # 3. Fallback to any remaining unused option in sit.options
        if matched_idx is None:
            for idx in range(len(sit.options)):
                if idx not in used_indices:
                    matched_idx = idx
                    break

        if matched_idx is not None:
            used_indices.add(matched_idx)
            mapped_opts.append(sit.options[matched_idx])
        else:
            verb = next((v for v in VERBS if v.lower() in text), "WAIT")
            target = next((j for j, obj in enumerate(objs) if obj.kind.lower() in text), None)
            mapped_opts.append(OptionSpec(verb=verb, target=target))

    return sit.model_copy(update={"objects": objs, "options": mapped_opts})


def _format_memory(m) -> str:
    if not m:
        return ""
    if isinstance(m, str):
        return m
    when = getattr(m, "when", "")
    action = getattr(m, "action", "")
    happened = getattr(m, "happened", "")
    if when:
        return f"{when}: {action} → {happened}"
    return f"{action} → {happened}"


class SimpleGame:
    def __init__(self, session: Session, two_step: bool = False):
        self.s = session
        self.two_step = two_step
        self.solo_sit: Situation | None = None
        self._solo_narration: str | None = None
        self._step1_trace: tuple[str, str | None, str] | None = None
        self.lived = sum(e.source in ("show", "try") and e.mode != "solo_try" for e in session.buffer)
        self._recent_narrations: list[str] = []
        self._current_bundle: dict | None = None
        self._scene_seq: int = 0
        self._current_scene_narration: str = ""
        self._current_option_texts: list[str] = []
        self._current_solo_narration: str = ""
        self._current_solo_option_texts: list[str] = []
        self._current_bundle_outcomes: list[str] = []
        self._current_solo_outcomes: list[str] = []
        self._lived_records: list[dict] = []
        self._last_ready_sit_id: str | None = None
        self._current_bundle_results: list[str] = []
        self._current_bundle_world_says: list[str] = []
        self._current_solo_results: list[str] = []
        self._current_solo_world_says: list[str] = []

        save_root = getattr(self.s, "save", None) and getattr(self.s.save, "root", None)
        k_path = (save_root / "knowledge.json") if save_root else None
        if k_path and k_path.exists():
            self.knowledge = load_knowledge(k_path)
        else:
            self.knowledge = KnowledgeGraph()

        if not self.knowledge.episodes and getattr(self.s, "buffer", None):
            for ex in self.s.buffer:
                for f in getattr(ex, "facts", []):
                    ep = EpisodeNode(
                        turn=len(self.knowledge.episodes) + 1,
                        scene_text=getattr(ex.obs, "scene_text", "") or "",
                        action_text=f.action_text or getattr(f.option, "option_text", "") or "",
                        outcome_text=f.outcome_text or "",
                        takeaway=f.takeaway or f.action_text or "",
                        outcome=f.result or "solved",
                        parent_mode=ex.source or "leave",
                    )
                    self.knowledge.episodes.append(ep)
            if self.knowledge.episodes and k_path:
                save_knowledge(self.knowledge, k_path)

        self.s.knowledge = self.knowledge

    @classmethod
    def new(cls, save_dir, seed: int = 0, two_step: bool = False, start_at_three: bool = False, persona: str = "warm", **kw) -> "SimpleGame":
        g = cls(Session.new(save_dir, seed=seed, **kw), two_step=two_step)
        if start_at_three:
            g.bootstrap_age_three(persona=persona)
        return g

    @classmethod
    def load(cls, save_dir, two_step: bool = False, **kw) -> "SimpleGame":
        g = cls(Session.load(save_dir, **kw), two_step=two_step)
        return g

    def bootstrap_age_three(self, persona: str = "warm") -> None:
        """Start directly at 36 months (3 years old) so the Director generates 3-year-old+ scenes from Scene 1."""
        curiosities = {"playful": 0.55, "warm": 0.4, "busy": 0.35, "strict": 0.25}
        if persona in curiosities and hasattr(self.s, "game"):
            self.s.game.curiosity = curiosities[persona]
        if MONTHS_PER_DAY > 0 and any(s.get("from", 0) >= 36 for s in STAGES):
            if self.s.state.current is not None or self.s.state.cards is not None:
                self.s.engine.state.current = None
                self.s.engine.state.cards = None
                self.s.engine.state.phase = Phase.CARDS
            while self.age_now() < 36 and self.s.state.day < self.s.game.days:
                ev = self.s.engine._emit(f"bootstrap:day{self.s.state.day}", EventType.DAY_ENDED, {})
                self.s.save.append_event(ev)
        self.lived = sum(e.source in ("show", "try") and e.mode != "solo_try" for e in self.s.buffer)
        if not self.knowledge.episodes and getattr(self.s, "buffer", None):
            for ex in self.s.buffer:
                for f in getattr(ex, "facts", []):
                    ep = EpisodeNode(
                        turn=len(self.knowledge.episodes) + 1,
                        scene_text=getattr(ex.obs, "scene_text", "") or "",
                        action_text=f.action_text or getattr(f.option, "option_text", "") or "",
                        outcome_text=f.outcome_text or "",
                        takeaway=f.takeaway or f.action_text or "",
                        outcome=f.result or "solved",
                        parent_mode=ex.source or "leave",
                    )
                    self.knowledge.episodes.append(ep)
            save_root = getattr(self.s, "save", None) and getattr(self.s.save, "root", None)
            if self.knowledge.episodes and save_root:
                save_knowledge(self.knowledge, save_root / "knowledge.json")

    def _ready(self) -> None:
        """Make sure a scene is under way: end the day when it is over, offer cards and take the first. Nothing after the last day."""
        if self.solo_sit is not None:
            return
        st = self.s.state
        if st.phase == Phase.EVENING:
            self.s.end_day()
        if self.ended:
            return
        if self.s.state.phase == Phase.CARDS:
            self.s.draw_cards()
            self.s.pick_card(0)
        if self.s.state.current is not None:
            if self.s.state.current.id != self._last_ready_sit_id:
                self._current_bundle = None
                self._current_bundle_outcomes = []
                self._current_bundle_results = []
                self._current_bundle_world_says = []
                self._scene_seq += 1
                step = getattr(self.s.state, "step", 1)
                rng = random.Random(f"{self.s.state.current.id}:{step}:{self._scene_seq}")
                deduped = self._dedup_situation(self.s.state.current, age=self.age_now(), rng=rng)
                if deduped.options != self.s.state.current.options or deduped.objects != self.s.state.current.objects:
                    self.s.engine.state.current = deduped
                self._last_ready_sit_id = self.s.state.current.id

    @property
    def ended(self) -> bool:
        return self.solo_sit is None and self.s.state.day > self.s.game.days

    def age_now(self) -> int:
        return age_months(self.s.state.day)

    def _voice(self) -> str:
        return self.s._voice()

    def can_act(self) -> bool:
        return self.age_now() >= ACTS_FROM_MONTH

    def has_two_steps(self) -> bool:
        """When two_step is enabled and the child can act, each scene has Step 1 (with parent) and Step 2 (child solo)."""
        return self.two_step and self.can_act()

    def _make_solo_situation(self, step1_tags: list[str], location: str = "house", step1_sit: Situation | None = None) -> Situation:
        rng = self.s._rng(f"solo-{self.lived}-{len(self.s.buffer)}")
        avail = open_problems(self.s.state.day) or list(PROBLEMS)
        diff_avail = [t for t in avail if t not in step1_tags and t != "REQUEST"]
        pool = diff_avail or [t for t in avail if t != "REQUEST"] or list(PROBLEMS)
        tag_exp = {t: sum(1 for e in self.s.buffer if t in e.obs.tags) for t in pool}
        min_exp = min(tag_exp.values())
        least_known = [t for t in pool if tag_exp[t] == min_exp]
        tag = rng.choice(least_known)
        sit = make_situation(rng, [tag], sid=f"solo{self.lived:04d}", location=location, forbidden=self.s.forbidden)
        self._solo_narration = None
        self._scene_seq += 1
        step = getattr(self.s.state, "step", 2)
        rng_dedup = random.Random(f"{sit.id}:{step}:{self._scene_seq}")
        return self._dedup_situation(sit, age=self.age_now(), rng=rng_dedup)

    @staticmethod
    def _dedup_situation(
        sit: Situation,
        objects: list[Obj] | None = None,
        world: dict | None = None,
        age: int = 12,
        rng: random.Random | None = None,
    ) -> Situation:
        seen = set()
        unique_opts = []
        for o in sit.options:
            key = (o.verb, o.target)
            if key not in seen:
                seen.add(key)
                unique_opts.append(o)
        objs = objects if objects is not None else sit.objects
        if unique_opts == sit.options and objs == sit.objects:
            return sit
        return sit.model_copy(update={"objects": objs, "options": unique_opts})

    _fresh_situation = _dedup_situation

    def scene(self) -> SimpleScene:
        self._ready()
        age, stg = self.age_now(), stage_of(self.age_now())
        if self.ended:
            return SimpleScene(number=self.lived + 1, narration="", options=[], age_months=age, age=age_text(age), stage=i18n.stage(stg), ended=True, cognitive=cognitive_status(self.knowledge, lang=current_lang()))
        first = self.solo_sit is None and not any(stage_of(age_months(e.day)) == stg for e in self.s.buffer)
        sc = self._solo_scene(age, stg) if self.solo_sit is not None else self._scene(age, stg)
        if sc.narration and (not self._recent_narrations or self._recent_narrations[-1] != sc.narration):
            self._recent_narrations.append(sc.narration)
            if len(self._recent_narrations) > 8:
                self._recent_narrations.pop(0)
        return sc.model_copy(update={"stage_first": first and stg is not None, "can_act": self.can_act()})

    def _sync_view(self, age: int | None = None, stg: dict | None = None) -> SimpleScene:
        if age is None:
            age = self.age_now()
        if stg is None:
            stg = stage_of(age)
        cur = self.s.state.current
        voice = self._voice()
        step_total = 2 if self.has_two_steps() else 1
        bundle = self._current_bundle

        if bundle and bundle.get("options"):
            b_opts = list(bundle.get("options") or [])
            cur = _map_bundle_options_to_situation(cur, b_opts)
            self.s.engine.state.current = cur
            self.s.obs = observation(cur)
            intro = bundle.get("narration", "")
            b_speeches = bundle.get("speeches", [])

            self._current_bundle_outcomes = list(bundle.get("outcomes") or [])
            self._current_bundle_results = list(bundle.get("results") or [])
            self._current_bundle_world_says = list(bundle.get("world_says") or [])

            options_out = []
            for i, opt_text in enumerate(b_opts):
                speech = b_speeches[i] if i < len(b_speeches) else (self.s.narrator.child_speech(cur, i, voice) if hasattr(self.s.narrator, "child_speech") else "")
                options_out.append(SimpleOption(
                    index=i, text=opt_text,
                    speech=speech,
                    expects={}, seen=0, pull=0.0, memory="", lean=""
                ))
        else:
            self._current_bundle_outcomes = []
            self._current_bundle_results = []
            self._current_bundle_world_says = []
            v = self.s.view()
            cur = self.s.state.current
            self.s.obs = observation(cur)
            intro = v.narration
            options_out = [
                SimpleOption(index=o.index, text=o.text, speech="", expects={}, seen=0, pull=0.0, memory="", lean="")
                for o in v.options
            ]

        # 1. Set self.s.obs.scene_text = intro
        _set_scene_text(self.s.obs, intro)

        # 2. For each i, opt in enumerate(self.s.obs.options): if i < len(options_out), set opt.option_text = options_out[i].text
        for i, opt in enumerate(self.s.obs.options):
            if i < len(options_out):
                _set_option_text(opt, options_out[i].text)

        # 3. Recompute probs and memories AFTER scene_text and option_text are set on self.s.obs
        seen = seen_counts(self.s.buffer)
        beta, tags = self.s.game.curiosity, list(cur.tags)
        ex = expectations(self.s, self.s.obs)
        if self.has_two_steps():
            mems = [None for _ in self.s.obs.options]
            lean = ["" for _ in self.s.obs.options]
        else:
            try:
                memories = remember(self.s.buffer, self.s.obs)
                if isinstance(memories, list):
                    mems = memories
                else:
                    mems = [remember(self.s.buffer, tags, o) for o in self.s.obs.options]
            except TypeError:
                mems = [remember(self.s.buffer, tags, o) for o in self.s.obs.options]
            probs = option_probs(self.s.model, self.s.obs, seen, beta=beta)
            lean = leanings(option_probs(self.s.model, self.s.obs), probs, [m is not None for m in mems])

        for i, opt in enumerate(options_out):
            if i < len(self.s.obs.options):
                opt.expects = ex[i]
                opt.seen = seen.get(novelty_key(self.s.obs.options[i]), 0)
                opt.pull = novelty(self.s.obs.options[i], seen, beta)
                opt.memory = _format_memory(mems[i])
                opt.lean = lean[i]

        self._current_scene_narration = intro
        self._current_option_texts = [opt.text for opt in options_out]

        return SimpleScene(
            number=self.lived + 1, step=1, step_total=step_total, narration=intro,
            age_months=age, age=age_text(age), stage=i18n.stage(stg),
            options=options_out,
            cognitive=cognitive_status(self.knowledge, lang=current_lang()))

    def _scene(self, age: int, stg: dict | None) -> SimpleScene:
        cur = self.s.state.current
        voice = self._voice()

        bundle = self._current_bundle if getattr(self, "_current_bundle_lang", None) == i18n.lang() and getattr(self, "_current_bundle_sit_id", None) == (cur.id if cur else None) else None
        if bundle is None:
            still = tuple(self.s._still(cur)) if (cur is not None and hasattr(self.s, "_still")) else ()
            narrator_override = "generate_scene_bundle" in getattr(self.s.narrator, "__dict__", {})
            if cur is not None and not narrator_override and hasattr(self.s, "director") and callable(getattr(self.s.director, "direct_scene", None)):
                try:
                    bundle = self.s.director.direct_scene(
                        cur, getattr(self.s.state, "age_months", age), voice, solo=False,
                        recent_scenes=self._recent_narrations, still=still
                    )
                except TypeError:
                    bundle = self.s.director.direct_scene(cur)
            if bundle is None and cur is not None and callable(getattr(self.s.narrator, "generate_scene_bundle", None)):
                bundle = self.s.narrator.generate_scene_bundle(
                    cur, getattr(self.s.state, "age_months", age), voice, solo=False,
                    recent_scenes=self._recent_narrations, still=still
                )
            if bundle is None and cur is not None and narrator_override and hasattr(self.s, "director") and callable(getattr(self.s.director, "direct_scene", None)):
                try:
                    bundle = self.s.director.direct_scene(
                        cur, getattr(self.s.state, "age_months", age), voice, solo=False,
                        recent_scenes=self._recent_narrations, still=still
                    )
                except TypeError:
                    bundle = self.s.director.direct_scene(cur)
            self._current_bundle = bundle
            self._current_bundle_lang = i18n.lang()
            self._current_bundle_sit_id = cur.id if cur else None

        return self._sync_view(age, stg)

    def _solo_scene(self, age: int, stg: dict | None) -> SimpleScene:
        sit = self.solo_sit
        voice = self._voice()

        bundle = self._current_bundle if getattr(self, "_current_bundle_lang", None) == i18n.lang() and getattr(self, "_current_bundle_sit_id", None) == (sit.id if sit else None) else None
        if bundle is None:
            narrator_override = "generate_scene_bundle" in getattr(self.s.narrator, "__dict__", {})
            if sit is not None and not narrator_override and hasattr(self.s, "director") and callable(getattr(self.s.director, "direct_scene", None)):
                try:
                    bundle = self.s.director.direct_scene(
                        sit, getattr(self.s.state, "age_months", age), voice, solo=True,
                        recent_scenes=self._recent_narrations
                    )
                except TypeError:
                    bundle = self.s.director.direct_scene(sit, solo=True)
            if bundle is None and callable(getattr(self.s.narrator, "generate_scene_bundle", None)):
                bundle = self.s.narrator.generate_scene_bundle(
                    sit, getattr(self.s.state, "age_months", age), voice, solo=True,
                    recent_scenes=self._recent_narrations
                )
            self._current_bundle = bundle
            self._current_bundle_lang = i18n.lang()
            self._current_bundle_sit_id = sit.id if sit else None

        if bundle and bundle.get("options"):
            b_opts = list(bundle.get("options") or [])
            if len(b_opts) != len(sit.options):
                sit = _map_bundle_options_to_situation(sit, b_opts)
                self.solo_sit = sit
            self.s.obs = observation(sit)
            intro = bundle.get("narration", self._solo_narration or "")
            self._solo_narration = intro
            b_speeches = bundle.get("speeches", [])
            pos = player_options(sit)

            self._current_solo_outcomes = list(bundle.get("outcomes") or [])
            self._current_solo_results = list(bundle.get("results") or [])
            self._current_solo_world_says = list(bundle.get("world_says") or [])

            options_out = [
                SimpleOption(
                    index=po.index, text=b_opts[i],
                    speech=b_speeches[i] if i < len(b_speeches) else narrator.child_speech(po.verb, po.target, voice, step=2, tags=list(sit.tags)),
                    expects={}, seen=0, pull=0.0, memory="", lean=""
                ) for i, po in enumerate(pos)
            ]
        else:
            self._current_solo_outcomes = []
            self._current_solo_results = []
            self._current_solo_world_says = []
            self.s.obs = observation(sit)
            pos = player_options(sit)

            if not getattr(self, "_solo_narration", None):
                if hasattr(self.s.narrator, "solo_intro"):
                    self._solo_narration = self.s.narrator.solo_intro(sit, voice)
                else:
                    self._solo_narration = narrator.solo_intro(sit, voice)
            intro = self._solo_narration
            tags = list(sit.tags)
            options_out = [
                SimpleOption(
                    index=po.index, text=getattr(narrator, "child_option_text", narrator.option_text)(po.verb, po.target, voice, tags=tags),
                    speech=narrator.child_speech(po.verb, po.target, voice, step=2, tags=tags),
                    expects={}, seen=0, pull=0.0, memory="", lean=""
                ) for po in pos
            ]

        _set_scene_text(self.s.obs, intro)
        for i, opt in enumerate(self.s.obs.options):
            if i < len(options_out):
                _set_option_text(opt, options_out[i].text)

        step1_buf = self.s.buffer[-1:] if self.s.buffer else []
        seen = seen_counts(step1_buf)
        beta = self.s.game.curiosity
        ex = expectations(self.s, self.s.obs)

        for i, opt in enumerate(options_out):
            if i < len(self.s.obs.options):
                opt.expects = ex[i]
                opt.seen = seen.get(novelty_key(self.s.obs.options[i]), 0)
                opt.pull = novelty(self.s.obs.options[i], seen, beta)

        self._current_solo_narration = intro
        self._current_solo_option_texts = [opt.text for opt in options_out]
        return SimpleScene(
            number=self.lived + 1, step=2, step_total=2, narration=intro,
            age_months=age, age=age_text(age), stage=i18n.stage(stg),
            options=options_out,
            cognitive=cognitive_status(self.knowledge, lang=current_lang()))

    def _play_solo(self, part: Part, option: int | None) -> SimpleTurn:
        self.s._apply_world()
        sit = self.solo_sit
        obs = observation(sit)
        self.s.obs = obs
        voice = self._voice()
        pos = player_options(sit)

        scene_narration = getattr(self, "_current_solo_narration", "")
        _set_scene_text(self.s.obs, scene_narration)
        if hasattr(self, "_current_solo_option_texts") and self._current_solo_option_texts:
            texts = {i: t for i, t in enumerate(self._current_solo_option_texts)}
        else:
            texts = {po.index: getattr(narrator, "child_option_text", narrator.option_text)(po.verb, po.target, voice, tags=list(sit.tags)) for po in pos}
        for i, opt in enumerate(self.s.obs.options):
            if i in texts:
                _set_option_text(opt, texts[i])

        bundle_speeches = (self._current_bundle.get("speeches") or []) if self._current_bundle else []
        speeches = {}
        for po in pos:
            if po.index < len(bundle_speeches) and bundle_speeches[po.index]:
                speeches[po.index] = bundle_speeches[po.index]
            else:
                speeches[po.index] = narrator.child_speech(po.verb, po.target, voice, step=2, tags=list(sit.tags))

        if part not in ("leave", "solo"):
            part = "solo" if option is not None else "leave"
        if part == "solo" and option not in texts:
            raise ValueError(f"solo needs an option number 0..{len(texts) - 1}")
        step1_buf = self.s.buffer[-1:] if self.s.buffer else []
        seen = seen_counts(step1_buf)
        pre = {"ex": expectations(self.s, self.s.obs), "plain": option_probs(self.s.model, self.s.obs), "curious": option_probs(self.s.model, self.s.obs, seen, self.s.game.curiosity)}
        if part == "leave" or option is None:
            tag_set = set(sit.tags)
            opt_counts: list[int] = []
            for o in self.s.obs.options:
                nk = novelty_key(o)
                c = sum(
                    1 for e in self.s.buffer
                    if (set(e.obs.tags) & tag_set)
                    for f in e.facts
                    if novelty_key(f.option) == nk
                )
                opt_counts.append(c)
            non_wait = [i for i, o in enumerate(sit.options) if o.verb != "WAIT"] or list(range(len(sit.options)))
            min_c = min(opt_counts[i] for i in non_wait)
            unexplored = [i for i in non_wait if opt_counts[i] == min_c]
            acted = self.s._rng("solo-try").choice(unexplored)
        else:
            acted = option

        vals = values(sit)
        val = vals[acted]
        out_label = outcome_label(val)
        api_outcome = (
            self._current_solo_outcomes[acted]
            if (self._current_solo_outcomes and acted < len(self._current_solo_outcomes) and self._current_solo_outcomes[acted])
            else out_label
        )
        bundle_res = (
            self._current_solo_results[acted]
            if (self._current_solo_results and acted < len(self._current_solo_results) and self._current_solo_results[acted])
            else ""
        )
        was_right = (api_outcome == "solved")

        so = sit.options[acted]
        tv = {t: option_value([t], sit.word_kind, so.verb, sit.objects[so.target] if so.target is not None else None) for t in sit.tags}
        chosen_text = (self._current_solo_option_texts[acted] if hasattr(self, "_current_solo_option_texts") and 0 <= acted < len(self._current_solo_option_texts) else texts.get(acted, ""))

        narration = ""
        takeaway = None
        outcome_calls = [
            lambda: self.s.narrator.outcome(sit, acted, api_outcome, "child", tv, tuple(self.s.state.unlocked), voice,
                                            scene_narration=scene_narration, chosen_text=chosen_text,
                                            outcome=api_outcome, bundle_result=bundle_res),
            lambda: self.s.narrator.outcome(sit, acted, api_outcome, "child", tv, tuple(self.s.state.unlocked), voice,
                                            scene_narration=scene_narration, chosen_text=chosen_text,
                                            bundle_result=bundle_res),
            lambda: self.s.narrator.outcome(sit, acted, api_outcome, "child", tv, tuple(self.s.state.unlocked), voice,
                                            scene_narration=scene_narration, chosen_text=chosen_text),
            lambda: self.s.narrator.outcome(sit, acted, api_outcome, "child", tv, tuple(self.s.state.unlocked), voice),
            lambda: self.s.narrator.outcome(sit, acted, api_outcome, "child"),
        ]
        out_res = None
        for call_fn in outcome_calls:
            try:
                out_res = call_fn()
                break
            except TypeError:
                continue
            except Exception:
                break

        if isinstance(out_res, tuple) and len(out_res) == 2:
            narration, takeaway = out_res
        elif isinstance(out_res, str):
            narration = out_res

        if not narration and bundle_res:
            narration = bundle_res
        if not narration:
            narration = outcome_label(val)

        if takeaway is None and hasattr(self.s.narrator, "get_takeaway"):
            try:
                takeaway = self.s.narrator.get_takeaway(sit, acted, api_outcome, "child", voice, chosen_text=chosen_text)
            except Exception:
                pass

        fact = Fact(option=self.s.obs.options[acted], result=api_outcome, said=False, action_text=chosen_text, outcome_text=narration)
        strength = max(2.0, self.s._sting(self.s.game.correction_rounds(api_outcome), fact))
        ex = Example(id=f"ex{len(self.s.buffer):04d}", day=self.s.state.day, source="try", obs=self.s.obs,
                     chosen=acted, was_right=was_right, outcome=api_outcome, mode="solo_try",
                     strength=strength, repeat=2, facts=[fact])
        res = self.s._learn(ex, ActionResult(narration=narration, option=acted, was_right=was_right, value=val, outcome=api_outcome))

        # Update self.s.buffer[-1].facts[0] immediately (BEFORE remember() and will_remember()!)
        if self.s.buffer and self.s.buffer[-1].facts:
            last_ex = self.s.buffer[-1]
            last_ex.outcome = api_outcome
            last_ex.was_right = was_right
            fact = last_ex.facts[0]
            fact.result = api_outcome
            fact.action_text = chosen_text
            _set_option_text(fact.option, chosen_text)
            fact.outcome_text = narration
            if hasattr(fact, "takeaway"):
                fact.takeaway = takeaway
            self.s.learner.learn(self.s.buffer[-1], self.s.buffer, self.s.rng)

        # Only AFTER updating fact and training TinyLM, call mem = remember(...) and learned_line = takeaway or will_remember(...)!
        try:
            mem = remember(self.s.buffer[-1:], self.s.obs, self.s.obs.options[acted])
        except TypeError:
            mem = remember(self.s.buffer[-1:], list(sit.tags), self.s.obs.options[acted], strict_tags=True)

        learned_line = takeaway or will_remember(mem, part=part, result=api_outcome, followed=None, voice=voice)

        how = str(part)
        ep = EpisodeNode(
            turn=len(self.knowledge.episodes) + 1,
            scene_text=getattr(self.s.obs, "scene_text", "") or "",
            action_text=chosen_text,
            outcome_text=narration,
            takeaway=takeaway or learned_line,
            outcome=api_outcome,
            parent_mode=how,
        )
        exp_res = expand_knowledge(
            self.knowledge,
            ep,
            client=getattr(self.s.narrator, "client", None),
            world_premise=getattr(self.s, "world_premise", ""),
            lang=current_lang(),
        )
        prin_node = exp_res.get("principle_node") or exp_res.get("new_principle")
        if prin_node is not None:
            prin_ex = principle_to_example(prin_node, day=self.s.state.day)
            learn_rng = self.s.rng if isinstance(getattr(self.s, "rng", None), random.Random) else (self.s.rng() if callable(getattr(self.s, "rng", None)) else self.s._rng("learn"))
            self.s.learner.learn(prin_ex, self.s.buffer, learn_rng, steps=5)

        if getattr(self.s, "save", None) and getattr(self.s.save, "root", None):
            save_knowledge(self.knowledge, self.s.save.root / "knowledge.json")

        cog_status = cognitive_status(self.knowledge, lang=current_lang())
        synth_prin = exp_res.get("synthesized_principle") or ""
        exp_concepts = [
            f"{c['name']}: {c['observation']}"
            for c in exp_res.get("concepts", [])
            if isinstance(c, dict) and c.get("name")
        ]
        if not exp_concepts and exp_res.get("new_concepts"):
            exp_concepts = [str(c) for c in exp_res.get("new_concepts", [])]

        what_learned = learned_line
        if synth_prin:
            what_learned = f"{what_learned} 💡 {synth_prin}" if what_learned else f"💡 {synth_prin}"

        self.solo_sit = None
        self._solo_narration = None
        self._step1_trace = None
        self.lived += 1

        says = line_for(self.s.world_sheet(), list(sit.tags), so.verb, sit.objects[so.target].kind if so.target is not None else None)
        if 0 <= acted < len(self._current_solo_world_says) and self._current_solo_world_says[acted]:
            says = self._current_solo_world_says[acted]

        why = why_facts(obs=self.s.obs, sit=sit, pre=pre, acted=acted, result=api_outcome, by="child", followed=None,
                        after=expectations(self.s, self.s.obs)[acted], text=chosen_text)

        turn = SimpleTurn(
            part=part, told=option if part == "solo" else None, acted=acted,
            by="child", followed=None, child_said=speeches.get(acted, ""), narration=narration,
            result=api_outcome, learned=f"{chosen_text} -> {api_outcome}",
            loss_before=res.learn.loss_before, loss_after=res.learn.loss_after,
            why=why, explanation=self.s.narrator.explain(why) if why else "", world_says=says,
            remembers=learned_line,
            step=2, step_total=2, has_next_step=False,
            cognitive=cog_status,
            synthesized_principle=synth_prin,
            expanded_concepts=exp_concepts,
            what_learned=what_learned)
        self._lived_records.append({
            "step": 2,
            "part": part,
            "acted": acted,
            "result": api_outcome,
            "narration": narration,
        })
        self._current_solo_outcomes = []
        self._current_solo_results = []
        self._current_solo_world_says = []
        self._current_bundle = None
        return turn

    def play(self, part: Part | int = "show", option: int | str | None = None, custom_text: str | None = None) -> SimpleTurn:
        self._ready()
        if self.ended:
            raise ValueError("the story is over: the child is grown")
        if isinstance(part, int):
            option = part
            part = "show"
        if self.solo_sit is not None:
            return self._play_solo(part, option)
        if part != "show" and not self.can_act():
            raise ValueError("the baby cannot choose yet: you can only show it")

        custom_action_text = custom_text.strip() if custom_text else None
        sit, obs = self.s._context()
        if custom_action_text:
            opt_idx, verb, kind = _match_custom_action_to_option(custom_action_text, sit, sit.objects)
            self.s.engine.state.current = sit
            option = opt_idx
            obs = observation(sit)

        self.s.obs = obs

        texts = {i: t for i, t in enumerate(self._current_option_texts)} if self._current_option_texts else {o.index: o.text for o in self.s.view().options}
        if custom_action_text and option not in texts:
            texts[option] = custom_action_text

        # Ensure self.s.obs.scene_text and self.s.obs.options[i].option_text are populated from the current view before acting
        scene_narration = getattr(self, "_current_scene_narration", "")
        _set_scene_text(self.s.obs, scene_narration)
        for i, opt in enumerate(self.s.obs.options):
            if i in texts:
                _set_option_text(opt, texts[i])

        if part not in ("leave",) and option not in texts:
            raise ValueError(f"{part} needs an option number 0..{len(texts) - 1}")
        if part == "say":
            self.s.command(option)
        voice = self._voice()
        pos = player_options(sit)
        child_step = 2 if part == "leave" else 1

        bundle_speeches = (self._current_bundle.get("speeches") or []) if self._current_bundle else []
        speeches = {}
        for po in pos:
            if po.index < len(bundle_speeches) and bundle_speeches[po.index]:
                speeches[po.index] = bundle_speeches[po.index]
            else:
                speeches[po.index] = narrator.child_speech(po.verb, po.target, voice, step=child_step, tags=list(sit.tags))

        seen = seen_counts(self.s.buffer)
        pre = {"ex": expectations(self.s, self.s.obs), "plain": option_probs(self.s.model, self.s.obs), "curious": option_probs(self.s.model, self.s.obs, seen, self.s.game.curiosity)}
        two = self.has_two_steps()
        r = self.s.show(option) if part == "show" else self.s.try_child()

        # After acted is determined (acted = option for show, or res.option for say/leave):
        acted = option if part == "show" else r.option

        api_outcome = (
            self._current_bundle_outcomes[acted]
            if (self._current_bundle_outcomes and acted < len(self._current_bundle_outcomes) and self._current_bundle_outcomes[acted])
            else r.outcome
        )
        bundle_res = (
            self._current_bundle_results[acted]
            if (self._current_bundle_results and acted < len(self._current_bundle_results) and self._current_bundle_results[acted])
            else ""
        )

        by = "parent" if part == "show" else "child"
        so = sit.options[acted]
        p_tv = {t: option_value([t], sit.word_kind, so.verb, sit.objects[so.target] if so.target is not None else None) for t in sit.tags}
        chosen_text = custom_action_text or (self._current_option_texts[acted] if hasattr(self, "_current_option_texts") and 0 <= acted < len(self._current_option_texts) else texts.get(acted, ""))

        # Call self.s.narrator.outcome(..., outcome=api_outcome, bundle_result=bundle_res) (with try/except TypeError fallback)
        narration = ""
        takeaway = None
        outcome_calls = [
            lambda: self.s.narrator.outcome(sit, acted, api_outcome, by, p_tv, tuple(self.s.state.unlocked), voice,
                                            custom_action=custom_action_text, scene_narration=scene_narration,
                                            chosen_text=chosen_text, outcome=api_outcome, bundle_result=bundle_res),
            lambda: self.s.narrator.outcome(sit, acted, api_outcome, by, p_tv, tuple(self.s.state.unlocked), voice,
                                            custom_action=custom_action_text, scene_narration=scene_narration,
                                            chosen_text=chosen_text, bundle_result=bundle_res),
            lambda: self.s.narrator.outcome(sit, acted, api_outcome, by, p_tv, tuple(self.s.state.unlocked), voice,
                                            custom_action=custom_action_text, scene_narration=scene_narration,
                                            chosen_text=chosen_text),
            lambda: self.s.narrator.outcome(sit, acted, api_outcome, by, p_tv, tuple(self.s.state.unlocked), voice,
                                            custom_action=custom_action_text),
            lambda: self.s.narrator.outcome(sit, acted, api_outcome, by),
        ]
        out_res = None
        for call_fn in outcome_calls:
            try:
                out_res = call_fn()
                break
            except TypeError:
                continue
            except Exception:
                break

        if isinstance(out_res, tuple) and len(out_res) == 2:
            narration, takeaway = out_res
        elif isinstance(out_res, str):
            narration = out_res

        if not narration and bundle_res:
            narration = bundle_res
        if not narration:
            narration = r.narration

        if takeaway is None and hasattr(self.s.narrator, "get_takeaway"):
            try:
                takeaway = self.s.narrator.get_takeaway(sit, acted, api_outcome, by, voice, custom_action=custom_action_text, chosen_text=chosen_text)
            except Exception:
                pass

        r.outcome = api_outcome
        r.was_right = (api_outcome == "solved")
        r.narration = narration

        # Update self.s.buffer[-1].facts[0] immediately (BEFORE remember() and will_remember()!)
        if self.s.buffer and self.s.buffer[-1].facts:
            last_ex = self.s.buffer[-1]
            last_ex.outcome = api_outcome
            last_ex.was_right = (api_outcome == "solved")
            fact = last_ex.facts[0]
            fact.result = api_outcome
            fact.action_text = chosen_text
            _set_option_text(fact.option, chosen_text)
            fact.outcome_text = narration
            if hasattr(fact, "takeaway"):
                fact.takeaway = takeaway
            self.s.learner.learn(self.s.buffer[-1], self.s.buffer, self.s.rng)

        # Only AFTER updating fact and training TinyLM, call mem = remember(...) and learned_line = takeaway or will_remember(...)!
        try:
            mem = remember(self.s.buffer[-1:] if two else self.s.buffer, self.s.obs, self.s.obs.options[acted])
        except TypeError:
            mem = remember(self.s.buffer[-1:] if two else self.s.buffer, list(sit.tags), self.s.obs.options[acted], strict_tags=two)

        learned_line = takeaway or will_remember(mem, part=part, result=api_outcome, followed=r.followed, voice=voice)
        if custom_action_text and custom_action_text not in learned_line:
            learned_line = f"{learned_line} ({custom_action_text})" if learned_line else custom_action_text

        how = str(part)
        ep = EpisodeNode(
            turn=len(self.knowledge.episodes) + 1,
            scene_text=getattr(self.s.obs, "scene_text", "") or "",
            action_text=chosen_text,
            outcome_text=narration,
            takeaway=takeaway or learned_line,
            outcome=api_outcome,
            parent_mode=how,
        )
        exp_res = expand_knowledge(
            self.knowledge,
            ep,
            client=getattr(self.s.narrator, "client", None),
            world_premise=getattr(self.s, "world_premise", ""),
            lang=current_lang(),
        )
        prin_node = exp_res.get("principle_node") or exp_res.get("new_principle")
        if prin_node is not None:
            prin_ex = principle_to_example(prin_node, day=self.s.state.day)
            learn_rng = self.s.rng if isinstance(getattr(self.s, "rng", None), random.Random) else (self.s.rng() if callable(getattr(self.s, "rng", None)) else self.s._rng("learn"))
            self.s.learner.learn(prin_ex, self.s.buffer, learn_rng, steps=5)

        if getattr(self.s, "save", None) and getattr(self.s.save, "root", None):
            save_knowledge(self.knowledge, self.s.save.root / "knowledge.json")

        cog_status = cognitive_status(self.knowledge, lang=current_lang())
        synth_prin = exp_res.get("synthesized_principle") or ""
        exp_concepts = [
            f"{c['name']}: {c['observation']}"
            for c in exp_res.get("concepts", [])
            if isinstance(c, dict) and c.get("name")
        ]
        if not exp_concepts and exp_res.get("new_concepts"):
            exp_concepts = [str(c) for c in exp_res.get("new_concepts", [])]

        what_learned = learned_line
        if synth_prin:
            what_learned = f"{what_learned} 💡 {synth_prin}" if what_learned else f"💡 {synth_prin}"

        if two:
            self.solo_sit = self._make_solo_situation(list(sit.tags), sit.location, step1_sit=sit)
            s1_kind = sit.objects[so.target].kind if so.target is not None else None
            self._step1_trace = (so.verb, s1_kind, api_outcome)
        else:
            self.lived += 1

        says = line_for(self.s.world_sheet(), list(sit.tags), so.verb, sit.objects[so.target].kind if so.target is not None else None)
        if not custom_action_text and 0 <= acted < len(self._current_bundle_world_says) and self._current_bundle_world_says[acted]:
            says = self._current_bundle_world_says[acted]

        why = why_facts(obs=self.s.obs, sit=sit, pre=pre, acted=acted, result=api_outcome, by=by, followed=r.followed,
                        after=expectations(self.s, self.s.obs)[acted], text=chosen_text)

        turn = SimpleTurn(
            part=part, told=None if part == "leave" else option, acted=acted,
            by=by, followed=r.followed,
            child_said=speeches.get(acted, "") if (by == "child" and self.can_act()) else "",
            narration=narration,
            result=api_outcome, learned=f"{chosen_text} -> {api_outcome}",
            loss_before=r.learn.loss_before, loss_after=r.learn.loss_after,
            why=why, explanation=self.s.narrator.explain(why) if why else "", world_says=says,
            remembers=learned_line,
            step=1, step_total=2 if two else 1, has_next_step=two,
            custom_action=custom_action_text,
            cognitive=cog_status,
            synthesized_principle=synth_prin,
            expanded_concepts=exp_concepts,
            what_learned=what_learned)
        self._lived_records.append({
            "step": 1,
            "part": part,
            "acted": acted,
            "result": api_outcome,
            "narration": narration,
            "custom": custom_action_text,
        })
        self._current_bundle_outcomes = []
        self._current_bundle_results = []
        self._current_bundle_world_says = []
        self._current_bundle = None
        return turn

    def summary(self) -> Summary:
        """A plain account of the life so far: no grades, only what happened and how the parent took part."""
        by_stage: dict[str, int] = {}
        for e in self.s.buffer:
            if e.source in ("show", "try") and e.mode != "solo_try":
                name = (i18n.stage(stage_of(age_months(e.day))) or {"name": "삶" if i18n.is_ko() else "life"})["name"]
                by_stage[name] = by_stage.get(name, 0) + 1
        days = self.s.journal().values()
        showed, told, tried = (sum(d[k] for d in days) for k in ("show", "command", "offer"))
        last: dict[str, tuple] = {}
        for e in self.s.buffer:
            for f in e.facts:
                last[i18n.tags_phrase(list(e.obs.tags)) or ("부탁" if i18n.is_ko() else "request")] = (e, f)
        moments = []
        for when, (e, f) in list(last.items())[-12:]:
            v = voice_of(age_months(e.day))
            act_s = f.action_text or action_words(f.option, v)
            out_s = f.outcome_text or f"{after_words(e.obs.tags, f.result, v, f.option.verb)}."
            moments.append(f"{when}: {act_s} — {out_s}")
        age = self.age_now()
        return Summary(age=age_text(age), scenes=self.lived, stages=[{"name": k, "scenes": v} for k, v in by_stage.items()],
                       took_part={"showed": showed, "told": told, "left": max(0, tried - told)}, moments=moments,
                       cognitive=cognitive_status(self.knowledge, lang=current_lang()))


_dedup_situation = SimpleGame._dedup_situation
_fresh_situation = SimpleGame._fresh_situation
