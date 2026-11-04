"""Closed ontology + rules table, loaded from config/ontology.yaml and validated once at import."""
import hashlib
import os
from dataclasses import dataclass
from pathlib import Path

import yaml

ONTOLOGY_PATH = Path(os.environ.get("GROWING_ONTOLOGY") or Path(__file__).resolve().parents[2] / "config" / "ontology.yaml")  # env: experiments only


@dataclass(frozen=True)
class KindSpec:
    name: str
    shape: str
    material: str
    sizes: tuple[str, ...]


def _load(path: Path = ONTOLOGY_PATH) -> dict:
    return yaml.safe_load(path.read_text())


_RAW = _load()
ONTOLOGY_HASH = hashlib.sha1(ONTOLOGY_PATH.read_bytes() + (b"no-schedule" if os.environ.get("GROWING_NO_SCHEDULE") else b"") + (b"tone-input" if os.environ.get("GROWING_TONE_INPUT") else b"") + (b"consequence" if os.environ.get("GROWING_CONSEQUENCE", "1") != "0" else b"")).hexdigest()[:12]  # a save belongs to one ontology version

TAGS: list[str] = _RAW["tags"]
VERBS: list[str] = _RAW["verbs"]
COLORS: list[str] = _RAW["colors"]
SIZES: list[str] = _RAW["sizes"]
SHAPES: list[str] = _RAW["shapes"]
MATERIALS: list[str] = _RAW["materials"]
LOCATIONS: list[str] = _RAW["locations"]
START_LOCATIONS: list[str] = _RAW["start_locations"]
KINDS: dict[str, KindSpec] = {k: KindSpec(k, v["shape"], v["material"], tuple(v["sizes"])) for k, v in _RAW["kinds"].items()}
PROBLEMS: list[str] = [t for t in TAGS if t != "REQUEST"]
RULES: dict[str, dict[tuple[str, str], int]] = {
    t: {(verb, kind): val for verb, kind, val in rows} for t, rows in _RAW["rules"].items()}
WAIT_VALUE: dict[str, int] = _RAW["wait_value"]
REQUEST_VERB: str = _RAW["request"]["verb"]
REQUEST_VALUE: int = _RAW["request"]["value"]
REQUEST_KINDS: list[str] = _RAW["request"]["kinds"]
OUTCOMES: dict[str, int] = _RAW["outcomes"]
ESCALATION: dict[str, str] = _RAW["followups"]["escalation"]
UNLOCKS: dict[str, str] = _RAW["followups"]["unlocks"]
NAMEABLE_MIN_VALUE: int = _RAW["followups"]["nameable_min_value"]
# needs that do not go away on their own: while one is unmet it comes back in the next scene (a hungry baby stays hungry)
LINGERS: list[str] = [t for t in _RAW["followups"].get("lingers", []) if t in PROBLEMS]
REQUESTS_PER_DAY: int = _RAW["followups"]["requests_per_day"]

# what solves each problem (the generator puts it on the table): the best rule of the tag
# Age schedule (docs/WORLD.md 22.7), optional: the child starts at `start_month` and a day is `months_per_day` months long.
# `opens` = the month from which a problem can come up. Without a schedule every problem is open from day 1.
# GROWING_NO_SCHEDULE=1 ignores it (experiments: the all-open control arm).
_SCHED: dict = {} if os.environ.get("GROWING_NO_SCHEDULE") else dict(_RAW.get("schedule") or {})
if _SCHED.get("opens") and any(isinstance(k, int) or str(k).isdigit() for k in _SCHED["opens"]):
    _inv: dict[str, int] = {}
    for k, ts in _SCHED["opens"].items():
        for t in ts:
            _inv[t] = int(k)
    _SCHED["opens"] = _inv
if _SCHED.get("closes") and any(isinstance(k, int) or str(k).isdigit() for k in _SCHED["closes"]):
    _inv_c: dict[str, int] = {}
    for k, ts in _SCHED["closes"].items():
        for t in ts:
            _inv_c[t] = int(k)
    _SCHED["closes"] = _inv_c
START_MONTH: int = _SCHED.get("start_month", 0)
MONTHS_PER_DAY: int = _SCHED.get("months_per_day", 0)
OPEN_MONTH: dict[str, int] = {t: _SCHED.get("opens", {}).get(t, 0) for t in PROBLEMS}
CLOSE_MONTH: dict[str, int] = {t: _SCHED.get("closes", {}).get(t, 10**9) for t in PROBLEMS}  # a problem stops coming up from this month
RAMP_SKIP: float = _SCHED.get("ramp_skip", 0.0)  # on the day a new problem opens (not day 1) it is left out of a card slot this often


def age_months(day: int) -> int:
    return START_MONTH + (day - 1) * MONTHS_PER_DAY


def open_problems(day: int) -> list[str]:
    return [t for t in PROBLEMS if OPEN_MONTH[t] <= age_months(day) < CLOSE_MONTH[t]]


# What this ontology sets for the game (days, scenes a day), when the child can first choose, the words from when, and the life stages.
GAME: dict = _RAW.get("game", {})
ACTS_FROM_MONTH: int = _RAW.get("acts_from_month", 0)
REQUEST_FROM_MONTH: int = _RAW.get("request_from_month", 0)
STAGES: list[dict] = _RAW.get("stages", [])


def voice_of(month: int) -> str:
    """Whose scenes these are at this age: 'baby' (the baby is the one cared for, it cannot choose yet) or 'grandma' (the child helps others)."""
    return (stage_of(month) or {}).get("voice", "grandma")


def stage_of(month: int) -> dict | None:
    """The life stage a month falls in ({from, name, blurb}), or None when the ontology has no stages."""
    return max((s for s in STAGES if s["from"] <= month), key=lambda s: s["from"], default=None)


def opened_on(tag: str) -> int:
    """The first day on which the problem can come up."""
    return 1 if OPEN_MONTH[tag] <= START_MONTH else -(-(OPEN_MONTH[tag] - START_MONTH) // MONTHS_PER_DAY) + 1


# What can sensibly be done with what (optional): verb -> kinds. Without the table anything goes (the small worlds). WAIT takes no thing.
AFFORDANCES: dict[str, set[str]] = {v: set(ks) for v, ks in (_RAW.get("affordances") or {}).items()}


# What a person might really try for a problem (optional): problem -> {(verb, kind)}. The wrong options of a scene are drawn from it (and the
# rules' own pairs are always kept), so a baby is not offered an errand. Without an entry for a problem anything afforded goes.
PLAUSIBLE: dict[str, set[tuple[str, str]]] = {t: {(v, k) for v, k in rows} for t, rows in (_RAW.get("plausible") or {}).items()}


# What belongs to babies only (optional): a wrong option in the scene of an older child never uses these (nobody feeds a bottle to a wet floor).
BABY_ONLY_VERBS: set[str] = set((_RAW.get("baby_only") or {}).get("verbs", []))
BABY_ONLY_KINDS: set[str] = set((_RAW.get("baby_only") or {}).get("kinds", []))


def afforded(verb: str, kind: str | None) -> bool:
    return not AFFORDANCES or verb == "WAIT" or kind in AFFORDANCES.get(verb, ())


NEEDS: dict[str, tuple[str, str]] = {t: max(r.items(), key=lambda kv: kv[1])[0] for t, r in RULES.items()}
# what makes each problem worse, for the solving kind only (the generator may put it on the table as a trap); empty in the shipped table
TRAPS: dict[str, list[str]] = {t: sorted(verb for (verb, kind), v in r.items() if v < 0 and kind == NEEDS[t][1])
                               for t, r in RULES.items()}

MAX_OPTS = 5
MIN_OPTS = 3
MAX_OBJECTS = 6
MIN_OBJECTS = 3
N_NAMES = 8


def _validate() -> None:
    assert set(RULES) == set(PROBLEMS) == set(WAIT_VALUE) - {"REQUEST"}, "every problem needs rules and a WAIT value"
    assert "WAIT" in VERBS and REQUEST_VERB in VERBS
    for t, rows in RULES.items():
        assert t in TAGS
        assert max(rows.values()) == 2, f"{t} must have a solving rule"
        for (verb, kind), v in rows.items():
            assert verb in VERBS and kind in KINDS and v in (-1, 0, 1, 2)  # -1 = makes it worse (none in the shipped table)
    for k, spec in KINDS.items():
        assert spec.shape in SHAPES and spec.material in MATERIALS and set(spec.sizes) <= set(SIZES), k
    for t, rows in RULES.items():
        for (verb, kind) in rows:
            assert afforded(verb, kind), f"{t}: {verb} {kind} is a rule but the affordances say it cannot be done"
    assert set(AFFORDANCES) <= set(VERBS) and all(set(ks) <= set(KINDS) for ks in AFFORDANCES.values()), "affordances name unknown verbs or kinds"
    assert set(REQUEST_KINDS) <= set(KINDS) and set(ESCALATION) <= set(PROBLEMS) and set(ESCALATION.values()) <= set(PROBLEMS)
    assert set(UNLOCKS) <= set(PROBLEMS) and set(UNLOCKS.values()) <= set(LOCATIONS) and set(START_LOCATIONS) <= set(LOCATIONS)


def _validate_schedule() -> None:
    assert set(_SCHED.get("opens", {})) <= set(TAGS) and set(_SCHED.get("closes", {})) <= set(PROBLEMS), "the schedule names a problem that does not exist"
    if _SCHED:
        assert MONTHS_PER_DAY > 0 and open_problems(1), "a schedule needs months_per_day > 0 and a problem open on day 1"
        assert len(open_problems(1)) >= 2, "two cards on different topics need two problems open on day 1"


_validate()
_validate_schedule()


def outcome_label(value: int) -> str:
    if value >= OUTCOMES["solved"]:
        return "solved"
    if value >= OUTCOMES["partial"]:
        return "partial"
    return "nothing" if value >= OUTCOMES["nothing"] else "worse"
