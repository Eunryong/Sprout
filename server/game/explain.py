"""Why did the child do that, and what did it not see coming? The engine works out the facts; the Narrator only puts them in words.

Only when the result surprised the child (it gave the true result under SURPRISE) there is something to explain. The reasons are
not guessed: each is the child's own numbers, recomputed (with and without curiosity, with the parent's word). The facts go to
`NarratorAgent.explain`, which keeps a line only if every number and the result in it are the engine's. Never fed back to the child."""
from server.world.situation import Observation

SURPRISE = 0.35  # the child gave the result that came up less than this, and it was not its top guess


def reasons_for(by: str, followed: bool | None, acted: int, plain: list[float], curious: list[float]) -> list[str]:
    """Why this option, from the child's own numbers. `plain` = chance of each option without curiosity, `curious` = with it."""
    if by == "parent":
        return ["shown"]
    out = []
    if followed:
        out.append("told")
    if curious[acted] >= 1.25 * plain[acted]:
        out.append("curious")
    if plain[acted] >= max(plain) - 1e-9:
        out.append("promising")
    return out or ["chance"]


def why_facts(*, obs: Observation, sit, pre: dict, acted: int, result: str, by: str, followed: bool | None, after: dict[str, float],
              text: str) -> dict | None:
    """The facts behind a surprising result, or None when the child was not surprised."""
    guess = pre["ex"][acted]
    top = max(guess, key=guess.get)
    if top == result or guess[result] >= SURPRISE:
        return None
    o = sit.options[acted]
    return {
        "action": text, "kind": sit.objects[o.target].kind if o.target is not None else None, "by": by,
        "expected": top, "expected_pct": round(100 * guess[top]),
        "actual": result, "before_pct": round(100 * guess[result]), "after_pct": round(100 * after[result]),
        "chose_pct": round(100 * pre["curious"][acted]),
        "reasons": reasons_for(by, followed, acted, pre["plain"], pre["curious"]),
    }
