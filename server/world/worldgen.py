"""A generated world: the relations a world is made of, picked from authored pools (docs/WORLD.md 22.13).

The alphabet (tags, verbs, kinds and their attributes, colours) is fixed: the child's vocabulary and the prose tables are
keyed on it. A world sheet chooses, from `config/world_pools.yaml`, which relations hold in this world (what solves, what
helps a little, what makes things worse), the schedule, the words, and what the world teaches from a choice. This module
holds the sheet, its checks, the template generator (no LLM) and the writer of a sheet as an ordinary ontology file, so
everything that reads `GROWING_ONTOLOGY` works on a generated world unchanged. The Director (LLM) proposes a sheet that
goes through the same checks.

  uv run python -m server.world.worldgen --seed 7 --out saves/worlds/w7 --check
  GROWING_ONTOLOGY=saves/worlds/w7/ontology.yaml uv run python scripts/...
"""
import argparse
import hashlib
import json
import os
import random
import re
import subprocess
import sys
from pathlib import Path

import yaml
from pydantic import BaseModel

from server.i18n import KIND_KO

CONFIG = Path(__file__).resolve().parents[2] / "config"
POOLS_PATH = CONFIG / "world_pools.yaml"
LEARNING = ("burn_worse_rounds", "burn_nothing_rounds", "regret_rounds", "own_burn_worse_rounds", "own_burn_nothing_rounds")
MAX_PREMISE = 200
META_PHRASES = ("correct answer", "right answer", "best option", "best choice", "the answer", "you should", "you must",
                "hint", "solution", "optimal")
BAD_CHARS = re.compile(r"[{}<>`*#\[\]\\\x00-\x1f]")
ALPHABET_KEYS = ("tags", "verbs", "colors", "sizes", "shapes", "materials", "locations", "start_locations", "kinds", "outcomes")
OPTIONAL_KEYS = ("affordances", "plausible", "baby_only", "acts_from_month", "request_from_month", "stages", "game")  # the 0-7 years base has them, the small worlds do not


class WorldSheet(BaseModel):
    seed: int | None = None
    source: str = "template"  # template | director:<client>
    premise: str = ""  # one line for the player; never reaches the child
    rules: dict[str, list[list]]  # problem -> [[verb, kind, value]], value in {-1, 1, 2}; a relation not listed counts 0
    wait_value: dict[str, int]
    request_kinds: list[str]
    escalation: dict[str, str] = {}
    opens: dict[str, int]  # problem -> the month from which it can come up
    learning: dict[str, int]  # GameCfg fields: what the world teaches from a choice (docs/WORLD.md 22.12)
    days: int = 8
    curiosity: float = 0.4  # how strongly this child is drawn to what it has little experience of (codec.CURIOSITY is the default)
    lines: dict[str, dict[str, str]] = {}  # problem -> {"VERB kind": why the world works so}; shown after a result, never before a choice, never to the child


class WorldProposal(BaseModel):
    """What the Director answers: the sheet without the engine's own fields (seed, source, days)."""
    premise: str = ""
    rules: dict[str, list[list]]
    wait_value: dict[str, int]
    request_kinds: list[str]
    escalation: dict[str, str] = {}
    opens: dict[str, int]
    learning: dict[str, int]
    curiosity: float = 0.4
    lines: dict[str, dict[str, str]] = {}


def sheet_from_proposal(prop: WorldProposal, seed: int, pools: dict, source: str) -> WorldSheet:
    """A relation that does nothing (value 0) is not listed in a sheet; a model that lists it is not wrong."""
    rules = {t: [list(r) for r in rows if len(r) == 3 and r[2] != 0] for t, rows in prop.rules.items()}
    return WorldSheet(seed=seed, source=source, premise=prop.premise.strip(), rules=rules, wait_value=prop.wait_value,
                      request_kinds=prop.request_kinds, escalation=prop.escalation, opens=prop.opens, learning=prop.learning,
                      days=pools["schedule"]["days"], curiosity=prop.curiosity, lines=clean_lines(prop.lines, rules))


VERDICT_WORDS = ("wrong", "mistake", "should", "bad idea", "good idea", "correct", "naughty", "well done")
VERDICT_WORDS_KO = ("정답", "틀렸", "틀린", "잘못", "해야", "좋은 생각", "나쁜 생각", "올바른", "옳은", "잘했", "나쁜 아이", "힌트")
MAX_LINE = 120


def line_error(text: str, kind: str, pools: dict) -> str | None:
    """A line says why the world works so. It is prose for the player: no markup or token, no advice or verdict, and no thing but its own."""
    t = text.strip()
    if not t or len(t) > MAX_LINE:
        return "empty or too long"
    if BAD_CHARS.search(text) or "\n" in text or re.search(r"[A-Z][A-Z_0-9]{2,}", t):
        return "markup or token"
    low = t.lower()
    if any(m in low for m in META_PHRASES + VERDICT_WORDS + VERDICT_WORDS_KO):
        return "advice or verdict"
    words = {w.strip('.,!?:;"()—-').removesuffix("'s") for w in low.split()}
    kinds = {k for p in pools["problems"].values() for _, k, _ in p["rules"]}
    if any((k in words or k + "s" in words) for k in kinds - {kind}):
        return "names another thing"
    return "names another thing" if any(re.search(rf"(?<![가-힣]){KIND_KO[k]}(?:으로|에게|[을를은는이가에의도과와로])?(?![가-힣])", t) for k in kinds - {kind} if k in KIND_KO) else None


def clean_lines(lines: dict[str, dict[str, str]], rules: dict[str, list[list]], pools: dict | None = None) -> dict[str, dict[str, str]]:
    """Keep the lines of relations that are in `rules` and pass `line_error`; the rest are dropped (a line is a flourish, the world is valid without it)."""
    pools = pools or load_pools()
    out: dict[str, dict[str, str]] = {}
    for tag, rows in lines.items():
        have = {f"{v} {k}": k for v, k, _ in rules.get(tag, [])}
        for key, text in rows.items():
            if key in have and isinstance(text, str) and not line_error(text, have[key], pools):
                out.setdefault(tag, {})[key] = text.strip()
    return out


def line_for(sheet: "WorldSheet | None", tags: list[str], verb: str, kind: str | None) -> str:
    """What the world says about this relation, once it has played out ('' when it says nothing)."""
    if sheet is None or kind is None:
        return ""
    return next((sheet.lines[t][f"{verb} {kind}"] for t in tags if f"{verb} {kind}" in sheet.lines.get(t, {})), "")


def premise_error(text: str, pools: dict) -> str | None:
    """The premise is prose for the player: it must not name a thing, give an instruction or leak an internal token."""
    t = text.strip()
    if len(t) > MAX_PREMISE:
        return "premise too long"
    if BAD_CHARS.search(text) or "\n" in text:
        return "premise has markup or control characters"
    if re.search(r"[A-Z][A-Z_0-9]{2,}", t):
        return "premise leaks an internal token"
    low = t.lower()
    if any(m in low for m in META_PHRASES + VERDICT_WORDS_KO):
        return "premise talks about the answer"
    words = {w.strip('.,!?:;"()—-').removesuffix("'s") for w in low.split()}
    kinds = {k for p in pools["problems"].values() for _, k, _ in p["rules"]}
    named = sorted(k for k in kinds if k in words or k + "s" in words or (k in KIND_KO and re.search(rf"(?<![가-힣]){KIND_KO[k]}(?:으로|에게|[을를은는이가에의도과와로])?(?![가-힣])", t)))
    return f"premise names {named}" if named else None


def load_pools(path: Path = POOLS_PATH) -> dict:
    return yaml.safe_load(Path(path).read_text())


def base_alphabet(pools: dict) -> dict:
    base = yaml.safe_load((CONFIG / pools["base"]).read_text())
    return {**{k: base[k] for k in ALPHABET_KEYS}, **{k: base[k] for k in OPTIONAL_KEYS if k in base}}


def sheet_hash(sheet: WorldSheet) -> str:
    return hashlib.sha1(json.dumps(sheet.model_dump(), sort_keys=True).encode()).hexdigest()[:12]


def check_sheet(sheet: WorldSheet, pools: dict) -> list[str]:
    """Every reason the sheet is not a valid world (empty = valid). Deterministic; no LLM, no model."""
    errs: list[str] = []
    probs = pools["problems"]
    if set(sheet.rules) != set(probs):
        errs.append(f"rules must cover exactly the problems {sorted(probs)}")
        return errs
    sched = pools["schedule"]
    kind_value: dict[str, int] = {}
    n_traps = 0
    for t, rows in sheet.rules.items():
        allowed = {(v, k): vals for v, k, vals in probs[t]["rules"]}
        seen = set()
        for row in rows:
            verb, kind, val = row
            if (verb, kind) not in allowed:
                errs.append(f"{t}: {verb} {kind} is not a plausible relation for this problem")
            elif val not in allowed[(verb, kind)]:
                errs.append(f"{t}: {verb} {kind} cannot be {val} (allowed {allowed[(verb, kind)]})")
            if val == 0:
                errs.append(f"{t}: {verb} {kind} is 0; leave out relations that do nothing")
            if (verb, kind) in seen:
                errs.append(f"{t}: {verb} {kind} listed twice")
            seen.add((verb, kind))
            n_traps += val < 0
            kind_value[kind] = max(kind_value.get(kind, -1), val)
        if not any(r[2] == 2 for r in rows):
            errs.append(f"{t}: nothing solves it")
        if sheet.wait_value.get(t) not in probs[t]["wait"]:
            errs.append(f"{t}: wait value {sheet.wait_value.get(t)!r} not allowed")
        if sheet.opens.get(t) not in probs[t]["opens"]:
            errs.append(f"{t}: opens at {sheet.opens.get(t)!r}, allowed {probs[t]['opens']}")
    if n_traps > pools["max_traps_per_world"]:
        errs.append(f"{n_traps} harmful relations, at most {pools['max_traps_per_world']}")
    req = pools["request"]
    if len(set(sheet.request_kinds)) != req["n_words"] or not set(sheet.request_kinds) <= set(req["kinds"]):
        errs.append(f"request kinds must be {req['n_words']} different kinds of {req['kinds']}")
    for k in sheet.request_kinds:  # a word can only be introduced as a follow-up of using the thing
        if kind_value.get(k, -1) < pools["followups"]["nameable_min_value"]:
            errs.append(f"the word for {k} could never be introduced: no relation helps with it")
    if len(sheet.escalation) > 1 or any([a, b] not in pools["followups"]["escalation"] for a, b in sheet.escalation.items()):
        errs.append("escalation must be at most one of the candidate pairs")
    for a, b in sheet.escalation.items():
        if sheet.opens.get(b, 0) < sheet.opens.get(a, 0):
            errs.append(f"{b} would follow {a} before it can come up")
    if sum(m == sched["start_month"] for m in sheet.opens.values()) < 2:
        errs.append("two problems must be open on day 1")
    if max(sheet.opens.values(), default=0) > sched["start_month"] + sched["months_per_day"] * (sheet.days - 1):
        errs.append("a problem opens after the last day")
    for key, val in sheet.learning.items():
        if key not in pools["learning"]:
            errs.append(f"learning: unknown setting {key!r}")
        elif not (isinstance(val, int) and pools["learning"][key][0] <= val <= pools["learning"][key][1]):
            errs.append(f"learning: {key}={val!r} outside {pools['learning'][key]}")
    if missing := sorted(set(LEARNING) - set(sheet.learning)):
        errs.append(f"learning must also set {missing}")
    lo, hi = pools.get("curiosity", [0.0, 1.0])
    if not (isinstance(sheet.curiosity, (int, float)) and lo <= sheet.curiosity <= hi):
        errs.append(f"curiosity {sheet.curiosity!r} outside [{lo}, {hi}]")
    if not 1 <= sheet.days <= 30:
        errs.append("days out of range")
    if sheet.premise and (e := premise_error(sheet.premise, pools)):
        errs.append(e)
    return errs


def template_world(seed: int, pools: dict | None = None) -> WorldSheet:
    """A valid world picked at random (seeded) from the pools; no LLM. Also the fallback when the Director fails."""
    pools = pools or load_pools()
    rng = random.Random(f"world:{seed}")
    probs = pools["problems"]
    for _ in range(500):
        rules = {}
        for t, p in probs.items():
            rows = [[v, k, rng.choice(vals)] for v, k, vals in p["rules"]]
            if not any(r[2] == 2 for r in rows):
                rng.choice([r for r, (_, _, vals) in zip(rows, p["rules"]) if 2 in vals])[2] = 2
            rules[t] = [r for r in rows if r[2] != 0]
        temper = rng.choice(sorted(pools["temperaments"]))
        worse, nothing, regret = pools["temperaments"][temper]
        pairs = pools["followups"]["escalation"]
        sheet = WorldSheet(
            seed=seed, premise=pools.get("premise") or f"A small house, a toddler of {pools['schedule']['start_month']} months, and a world of its own rules.",
            rules=rules, wait_value={t: rng.choice(p["wait"]) for t, p in probs.items()},
            request_kinds=rng.sample(pools["request"]["kinds"], pools["request"]["n_words"]),
            escalation=dict([rng.choice(pairs)]) if pairs and rng.random() < 0.8 else {},
            opens={t: rng.choice(p["opens"]) for t, p in probs.items()},
            learning={"burn_worse_rounds": worse, "burn_nothing_rounds": nothing, "regret_rounds": regret,
                      "own_burn_worse_rounds": 0, "own_burn_nothing_rounds": 0},
            days=pools["schedule"]["days"], curiosity=random.Random(f"curiosity:{seed}").choice([0.2, 0.4, 0.6]))
        if not check_sheet(sheet, pools):
            return sheet
    raise RuntimeError("no valid world found in 500 tries; the pools are too tight")


def ontology_dict(sheet: WorldSheet, pools: dict) -> dict:
    """The sheet as the content of an ontology file (the format server/world/ontology.py reads)."""
    out = base_alphabet(pools)
    sched = pools["schedule"]
    out["rules"] = {t: [list(r) for r in sorted(rows, key=lambda r: -r[2])] for t, rows in sheet.rules.items()}
    out["schedule"] = {"start_month": sched["start_month"], "months_per_day": sched["months_per_day"],
                       "opens": dict(sheet.opens), "ramp_skip": sched["ramp_skip"]}
    if sched.get("closes"):
        out["schedule"]["closes"] = dict(sched["closes"])
    out["wait_value"] = {**sheet.wait_value, "REQUEST": 0}
    out["request"] = {"verb": pools["request"]["verb"], "value": 2, "kinds": list(sheet.request_kinds)}
    f = pools["followups"]
    out["followups"] = {"escalation": dict(sheet.escalation), "unlocks": dict(f["unlocks"]),
                        "nameable_min_value": f["nameable_min_value"], "requests_per_day": f["requests_per_day"]}
    if f.get("lingers") is not None:
        out["followups"]["lingers"] = list(f["lingers"])
    return out


def write_world(sheet: WorldSheet, out_dir: str | Path, pools: dict | None = None) -> tuple[Path, Path]:
    """out_dir/ontology.yaml (read through GROWING_ONTOLOGY) and out_dir/world.json (the sheet, for the game and the reveal)."""
    pools = pools or load_pools()
    errs = check_sheet(sheet, pools)
    if errs:
        raise ValueError("not a valid world: " + "; ".join(errs))
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    onto, js = out / "ontology.yaml", out / "world.json"
    head = f"# GENERATED WORLD {sheet_hash(sheet)} (source: {sheet.source}, seed: {sheet.seed}); docs/WORLD.md 22.13. Do not edit.\n"
    onto.write_text(head + yaml.safe_dump(ontology_dict(sheet, pools), sort_keys=False, default_flow_style=None, width=120))
    js.write_text(json.dumps({"hash": sheet_hash(sheet), "sheet": sheet.model_dump()}, indent=1))
    return onto, js


def read_sheet(ontology_path: str | Path) -> WorldSheet | None:
    """The sheet beside a generated ontology file, or None for a hand-written one."""
    p = Path(ontology_path).with_name("world.json")
    return WorldSheet.model_validate(json.loads(p.read_text())["sheet"]) if p.exists() else None


def game_overrides(ontology_path: str | Path) -> dict:
    """GameCfg fields the world sets (what it teaches from a choice, how many days); {} for a hand-written world."""
    sheet = read_sheet(ontology_path)
    return {} if sheet is None else {**sheet.learning, "days": sheet.days, "curiosity": sheet.curiosity}


SMOKE = """
import random
from server.world.ontology import PROBLEMS, REQUEST_KINDS
from server.world.generator import make_situation
from server.world.situation import optimal, values
rng = random.Random(0)
for t in PROBLEMS:
    for _ in range(40):
        s = make_situation(rng, [t])
        assert optimal(s) and max(values(s)) == 2
for k in REQUEST_KINDS:
    for _ in range(10):
        make_situation(rng, ["REQUEST"], "s", "house", "w", "NAME_00", k)
print("ok")
"""


def smoke(ontology_path: str | Path) -> list[str]:
    """Load the file as the game does and make scenes from it, in a fresh process (the ontology is a module global)."""
    env = {**os.environ, "GROWING_ONTOLOGY": str(Path(ontology_path).resolve()), "GROWING_LLM": "mock"}
    r = subprocess.run([sys.executable, "-c", SMOKE], env=env, capture_output=True, text=True, cwd=CONFIG.parent)
    return [] if r.returncode == 0 and r.stdout.strip() == "ok" else [(r.stderr.strip().splitlines() or ["failed"])[-1]]


def main() -> None:
    ap = argparse.ArgumentParser(description="Make a world (template, no LLM) and write it as an ontology file")
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--check", action="store_true", help="load it as the game does and make scenes from it")
    ap.add_argument("--director", action="store_true", help="let the Director (the LLM client of config/llm.yaml, GROWING_LLM) propose it")
    ap.add_argument("--mood", default="", help="with --director: a few words of flavour from the player (data, not instructions)")
    a = ap.parse_args()
    if a.director:
        from server.agents.director import DirectorAgent
        from server.agents.factory import make_client
        d = DirectorAgent(make_client({}), {})
        sheet = d.world(a.seed, a.mood)
        print(f"director: source={sheet.source} attempts={d.stats.summary()} error={d.last_error}")
    else:
        sheet = template_world(a.seed)
    onto, _ = write_world(sheet, a.out)
    print(f"world {sheet_hash(sheet)} ({sheet.source}) -> {onto}")
    if a.check:
        errs = smoke(onto)
        print("smoke:", "ok" if not errs else errs)
        sys.exit(1 if errs else 0)


if __name__ == "__main__":
    main()
