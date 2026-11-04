import random

from pydantic import BaseModel

from server.world.situation import Observation

from .codec import option_probs


class ChildAct(BaseModel):
    option: int
    probs: list[float]


def act(model, obs: Observation, rng: random.Random | None = None, greedy: bool = False, seen: dict | None = None, beta: float | None = None) -> ChildAct:
    probs = option_probs(model, obs, seen, beta)
    if greedy or rng is None:
        option = max(range(len(probs)), key=probs.__getitem__)
    else:
        option = rng.choices(range(len(probs)), weights=probs)[0]
    return ChildAct(option=option, probs=probs)
