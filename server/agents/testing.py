"""Fault injection: a stand-in "LLM" that answers like the mock but sometimes goes wrong in the ways real models do.
Used to check the whole guard / repair / fallback path and to validate the pass-rate script before a real key exists."""
import json
import random

from .client import ClientError, MockClient
from .director import mock_director_handler
from .narrator import mock_narrator_handler

DIRECTOR_FAULTS = ("raise", "bad_json", "unknown_tag", "no_solver", "one_card", "twin_topics", "leaky_teaser")
WORLD_FAULTS = ("raise", "bad_json", "off_pool", "no_solver", "too_many_traps", "bad_learning", "leaky_premise")
NARRATOR_FAULTS = ("raise", "names_kind", "contradicts", "empty", "internal_token")


class SloppyClient:
    """`rates`: fault name -> probability per call. A repair call (the prompt carries `previous_error`) fails with
    probability scaled by `repair_factor`, like a model that mostly fixes what it is told is wrong."""
    name = "sloppy"

    def __init__(self, word_map: dict, rates: dict[str, float], seed: int = 0, repair_factor: float = 0.3):
        self.inner = MockClient({"director": mock_director_handler(word_map), "narrator": mock_narrator_handler()})
        self.rates, self.rng, self.repair_factor = rates, random.Random(seed), repair_factor
        self.injected: dict[str, int] = {}
        self.last_used = self.name

    def _fault(self, faults: tuple[str, ...], repair: bool) -> str | None:
        for f in faults:
            if self.rng.random() < self.rates.get(f, 0.0) * (self.repair_factor if repair else 1.0):
                self.injected[f] = self.injected.get(f, 0) + 1
                return f
        return None

    def complete_json(self, *, agent, system, user, schema):
        payload = json.loads(user)
        repair = "previous_error" in payload
        out = self.inner.complete_json(agent=agent, system=system, user=user, schema=schema).model_dump()
        if agent == "director" and payload.get("task") == "world":
            f = self._fault(WORLD_FAULTS, repair)
            if f in ("raise", "bad_json"):
                raise ClientError("sloppy: simulated timeout" if f == "raise" else "sloppy: reply is not valid JSON")
            rules = out["rules"]
            if f == "off_pool":
                rules["THIRSTY"].append(["WIPE", "ball", 1])
            elif f == "no_solver":
                rules["DARK"] = [["GIVE", "flashlight", 1]]
            elif f == "too_many_traps":
                for t, p in payload["pools"]["problems"].items():
                    rules[t] = [r for r in rules[t] if not any(r[:2] == [v, k] for v, k, vals in p["rules"] if -1 in vals)]
                    rules[t] += [[v, k, -1] for v, k, vals in p["rules"] if -1 in vals]
            elif f == "bad_learning":
                out["learning"]["burn_worse_rounds"] = 9
            elif f == "leaky_premise":
                out["premise"] = "Hint: the cup solves it."
        elif agent == "director":
            f = self._fault(DIRECTOR_FAULTS, repair)
            if f == "raise":
                raise ClientError("sloppy: simulated timeout")
            if f == "bad_json":
                raise ClientError("sloppy: reply is not valid JSON")
            c0 = out["cards"][0]
            if f == "unknown_tag":
                c0["tags"] = ["DRAGON"]
            elif f == "no_solver":
                c0["options"] = [o for o in c0["options"] if o["verb"] == "WAIT"] + [
                    {"verb": "DRINK", "target": 0}, {"verb": "DRINK", "target": 1}, {"verb": "DRINK", "target": 2}]
            elif f == "one_card":
                out["cards"] = out["cards"][:1]
            elif f == "twin_topics":
                out["cards"][1] = json.loads(json.dumps(c0))
            elif f == "leaky_teaser":
                c0["teaser"] = f"Quick, grab the {c0['objects'][0]['kind']}!"
        else:
            f = self._fault(NARRATOR_FAULTS, repair)
            if f == "raise":
                raise ClientError("sloppy: simulated timeout")
            if f == "names_kind":
                out["text"] += " Maybe the spoon or the flashlight."
            elif f == "contradicts":
                out["text"] = "That solves it perfectly!" if payload.get("label") in ("worse", "nothing") else "It gets worse."
            elif f == "empty":
                out["text"] = ""
            elif f == "internal_token":
                out["text"] = "NAME_00 happened."
        return schema.model_validate(out)
