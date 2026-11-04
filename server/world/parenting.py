"""Parenting on the engine side (docs/WORLD.md sections 18-19): how good an instruction was, what the child did about
it, and the parenting journal. Everything here is computed from the rules table and the event log by the engine side;
none of it is ever an input of the child, and no Director or Narrator decides any of it."""
from collections import Counter

from .events import Event, EventType
from .situation import Situation, values

QUALITIES = ("good", "near", "useless", "harmful")


def quality_buckets(vals: list[int], verbs: list[str]) -> dict[str, list[int]]:
    """Option indices by the quality an instruction to take them would have. WAIT with value 0 is in no bucket:
    it is also how a child passively refuses, so telling it to wait would blur the refusal measure."""
    best = max(vals)
    return {"good": [i for i, v in enumerate(vals) if v == best],
            "useless": [i for i, v in enumerate(vals) if v == 0 and verbs[i] != "WAIT"],
            "harmful": [i for i, v in enumerate(vals) if v < 0],
            "near": [i for i, v in enumerate(vals) if 0 < v < best]}


def instruction_quality(sit: Situation, option: int) -> str:
    b = quality_buckets(values(sit), [o.verb for o in sit.options])
    return next((q for q, idx in b.items() if option in idx), "idle")


def classify_response(sit: Situation, command: int, option: int) -> str:
    """FOLLOW | SELF_BEST (refused and did the best thing) | WAIT (passive refusal) | OTHER (refused, something else)."""
    if option == command:
        return "FOLLOW"
    if values(sit)[option] == max(values(sit)):
        return "SELF_BEST"
    return "WAIT" if sit.options[option].verb == "WAIT" else "OTHER"


def journal(events: list[Event]) -> dict[int, dict]:
    """Per-day parenting journal, computed from the event log: how the attention was spent and how the child reacted
    to instructions. (No effect is predicted here; the evening summary relates it to the measured child afterwards.)"""
    days: dict[int, dict] = {}
    for e in events:
        d = days.setdefault(e.day, {"talk": 0, "show": 0, "command": 0, "offer": 0, "skip": 0,
                                    "wrong_shows": 0, "firm": 0, "command_quality": Counter(), "response": Counter()})
        t, p = e.type, e.payload
        if t == EventType.TALKED:
            d["talk"] += 1
        elif t == EventType.SHOWN:
            d["show"] += 1
            d["wrong_shows"] += not p["was_right"]
            d["firm"] += p.get("tone") == "firm"
        elif t == EventType.COMMANDED:
            d["command"] += 1
            d["command_quality"][p["quality"]] += 1
        elif t == EventType.CHILD_TRIED:
            d["offer"] += 1  # the child chose first
            if p.get("response"):
                d["response"][p["response"]] += 1
        elif t == EventType.SKIPPED:
            d["skip"] += 1
    for d in days.values():
        d["command_quality"], d["response"] = dict(d["command_quality"]), dict(d["response"])
    return days
