"""The parent, written by an LLM. In the first three years of a child's life nobody is at the screen: muse-spark plays the parent the
way the player will later, from the same words the player reads (the scene, what each option is, what the child seems to remember
and to be drawn to), takes part once per scene (show / tell / leave) and never sees the world's rules or any result before it
acts. So what the child learns in those years is what really happened to it when a parent did this or that, a parent who is sometimes
wrong, as parents are. The persona only says what kind of parent it is."""
import json
import random
import time
from typing import Literal

from pydantic import BaseModel

from .client import LLMClient
from .stats import AgentStats

PERSONAS = {
    "warm": "You are a warm, attentive parent. You like to show your child what to do, gently, and you try to read what it needs. As the child grows you step back little by little, letting it try on its own and stepping in when it seems unsure or unhappy.",
    "busy": "You are a busy, brisk parent. You often leave things to the child and only step in when something seems really off.",
    "playful": "You are a playful, curious parent. You like to try unusual things and see what happens, even when they may not work.",
    "strict": "You are a strict, orderly parent. You like to tell the child exactly what to do and to keep to routines.",
}
PERSONA_KO = {"warm": "다정하고 세심한 부모", "busy": "바빠서 아이에게 자주 맡기는 부모", "playful": "장난기 많고 이것저것 해 보는 부모", "strict": "엄하고 규칙을 중시하는 부모"}
SYSTEM = (
    "You are playing the parent of a young child in a gentle life simulation. Each scene you may take part ONCE: `show` (you do option n yourself and "
    "the child watches), `say` (you tell the child to do option n; it may or may not), or `leave` (you say nothing and the child chooses alone). "
    "While the child is a baby (`can_act` false) you can only `show`. You know only what the text says: the scene, the options, what the child "
    "seems to remember about each and what it seems drawn to. You do not know how things will turn out, and you are not perfect: act as your "
    "persona would, with your own common sense and mood. Do not try to teach a lesson; just parent. A child that can act and is growing needs room: "
    "the older it gets, the more a parent lets it try (`leave`) instead of doing everything for it, though your persona still colours how much. "
    "`so_far` counts what you have done with this child until now. "
    'Reply JSON only: {"part": "show" | "say" | "leave", "option": <index or null for leave>, "reason": <one short sentence>}.'
)


class ParentDecision(BaseModel):
    part: Literal["show", "say", "leave"]
    option: int | None = None
    reason: str = ""


class ParentAgent:
    def __init__(self, client: LLMClient | None, persona: str = "warm", seed: int = 0, on_call=None):
        if persona not in PERSONAS:
            raise ValueError(f"persona must be one of {sorted(PERSONAS)}")
        self.client, self.persona, self.rng = client, persona, random.Random(seed)
        self.stats = AgentStats("parent", on_call)
        self.last_error: str | None = None
        self.so_far = {"show": 0, "say": 0, "leave": 0}  # what this parent has done with the child so far (a parent knows how they have been with their child)

    def _fallback(self, scene) -> ParentDecision:
        """A parent who could not decide acts as a tired parent would: shows something to a baby, leaves a child alone."""
        if not scene.can_act:
            return ParentDecision(part="show", option=self.rng.randrange(len(scene.options)), reason="(fallback)")
        return ParentDecision(part="leave", reason="(fallback)")

    def decide(self, scene, recent: list[str]) -> ParentDecision:
        t0 = time.perf_counter()
        if self.client is None:
            self.stats.record(source="none", ok=False, attempts=0, error=None, started=t0)
            return self._fallback(scene)
        payload = {"persona": PERSONAS[self.persona], "age": scene.age, "can_act": scene.can_act, "seed": self.rng.randrange(10**6), "scene": scene.narration,
                   "options": [{"index": o.index, "text": o.text, "child_remembers": o.memory or None, "child_seems": o.lean or None} for o in scene.options],
                   "recently": recent[-3:], "so_far": self.so_far}
        try:
            d = self.client.complete_json(agent="parent", system=SYSTEM, user=json.dumps(payload), schema=ParentDecision)
            n = len(scene.options)
            if not scene.can_act and d.part != "show":
                d = d.model_copy(update={"part": "show"})  # a baby cannot choose: the most a parent can do is show
            if d.part != "leave" and (d.option is None or not 0 <= d.option < n):
                raise ValueError(f"option {d.option} is not one of the {n} options")
            if d.part == "leave":
                d = d.model_copy(update={"option": None})
        except Exception as e:  # noqa: BLE001
            self.last_error = f"{type(e).__name__}: {e}"
            self.stats.record(source="fixed", ok=False, attempts=1, error=self.last_error, started=t0)
            return self._fallback(scene)
        self.stats.record(source=getattr(self.client, "last_used", None) or self.client.name, ok=True, attempts=1, error=None, started=t0)
        self.so_far[d.part] += 1
        return d


def mock_parent_handler():
    """The stand-in parent of the mock mode: shows the option its seed points at, now and then leaves a child to choose."""
    def handler(user: dict) -> dict:
        n = len(user["options"])
        if user["can_act"] and user["seed"] % 3 == 0:
            return {"part": "leave", "option": None, "reason": "let it try"}
        return {"part": "show" if user["seed"] % 3 == 1 or not user["can_act"] else "say", "option": user["seed"] % n, "reason": "mock"}
    return handler
