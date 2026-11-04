"""The child's language: tokens, encoding, loss and the allowed augmentations.

Input  = [BOS] TAG.. [NAME] [CMD VERB attrs..] [SEP] (OPT_i VERB COLOR SHAPE SIZE MAT [SEP])* ACT_PICK
Output = the OPT_k token at the last position (logits restricted to options that exist; every existing option
stays allowed, so the mask never reveals the answer)."""
import hashlib
import os
import random
import re
import zlib
from dataclasses import dataclass, field
from typing import Any

import torch
import torch.nn.functional as F
from pydantic import BaseModel

from server.world.ontology import COLORS, KINDS, MATERIALS, MAX_OPTS, N_NAMES, PROBLEMS, SHAPES, SIZES, TAGS, VERBS
from server.world.situation import Observation, OptionView, TargetView

OBJ_KINDS: list[str] = list(KINDS.keys())

TONE_INPUT = bool(os.environ.get("GROWING_TONE_INPUT"))  # experiments only (docs/WORLD.md 22.9): the child hears the tone
# The child learns what happens (docs/WORLD.md 22.14): `situation + the thing done -> solved | partial | nothing | worse`,
# and picks the option it expects to turn out best. No option is labelled right or wrong; every act the child lives
# through, its own or the parent's, is a fact about the world.
CONSEQUENCE = os.environ.get("GROWING_CONSEQUENCE", "1") != "0"  # on by default; GROWING_CONSEQUENCE=0 is the older child that learns which option to pick
RESULTS: list[str] = ["solved", "partial", "nothing", "worse"]
OUTCOMES: list[str] = RESULTS


@dataclass
class Obs:
    problem_tag: str = ""
    mood: str = "calm"
    energy: float = 0.8
    tags: list[str] = field(default_factory=list)
    name_token: str | None = None
    options: list = field(default_factory=list)
    cmd: Any = None
    tone: str | None = None
    scene_text: str = ""

    def __post_init__(self):
        if not self.tags and self.problem_tag:
            self.tags = [self.problem_tag]


@dataclass
class Obj:
    id: str = "o"
    name: str = ""
    kind: str = ""
    color: str = "blue"
    shape: str = "flat"
    size: str = "small"
    material: str = "wool"

    def __post_init__(self):
        if self.kind and self.kind in KINDS:
            spec = KINDS[self.kind]
            self.shape = spec.shape
            self.material = spec.material
            if spec.sizes:
                self.size = spec.sizes[0]


@dataclass
class Option:
    verb: str = ""
    target: Any = None
    option_text: str = ""
WANT: dict[str, float] = {"solved": 1.0, "partial": 0.5, "nothing": 0.0, "worse": -1.0}  # what the child hopes for: its drive, not a verdict
UNTESTED_N, UNTESTED_W = (lambda a: (int(a[0]), float(a[1])))((os.environ.get("GROWING_UNTESTED") or ("3,0.3" if CONSEQUENCE else "0,0")).split(","))  # experiments: see `untested`
SMOOTH = float(os.environ.get("GROWING_SMOOTH", "0.15" if CONSEQUENCE else "0"))  # label smoothing of the result loss: how sure the child may become of one result (scripts/exp_calib.py)
CHOICE_TEMP = 0.25  # softmax temperature over the expected results when the child picks
# Curiosity (docs/DIRECTION.md 6): a thing the child has lived through few times pulls on it, `CURIOSITY / sqrt(1 + times lived)`,
# added to the result it hopes for. Like WANT it is a drive, not a verdict; it fades as the child lives things (GROWING_CURIOSITY).
CURIOSITY = float(os.environ.get("GROWING_CURIOSITY", "0.4" if CONSEQUENCE else "0"))  # beta chosen by scripts/exp_explore.py
VOCAB: list[str] = (
    ["[PAD]", "[BOS]", "[SEP]", "[EOS]", "ACT_PICK", "CMD"]
    + [f"TAG_{t}" for t in TAGS] + [f"VERB_{v}" for v in VERBS] + [f"OPT_{i}" for i in range(MAX_OPTS)]
    + [f"COLOR_{c}" for c in COLORS] + [f"SHAPE_{s}" for s in SHAPES] + [f"SIZE_{s}" for s in SIZES]
    + [f"MAT_{m}" for m in MATERIALS] + [f"NAME_{i:02d}" for i in range(N_NAMES)]
    + (["TONE_GENTLE", "TONE_FIRM"] if TONE_INPUT else [])
    + (["ACT_DO", "SAID"] + [f"OUT_{r}" for r in RESULTS] if CONSEQUENCE else [])
    + [f"TXT_{i:03d}" for i in range(256)]
)
TOK = {t: i for i, t in enumerate(VOCAB)}
PAD = TOK["[PAD]"]
OPT_IDS = [TOK[f"OPT_{i}"] for i in range(MAX_OPTS)]
OUT_IDS = [TOK[f"OUT_{r}"] for r in RESULTS] if CONSEQUENCE else []
MAX_LEN = max(128, 1 + 2 + 1 + (1 if TONE_INPUT else 0) + 6 + 1 + MAX_OPTS * 7 + 1 + 32)


def _text_tokens(text: str, max_toks: int = 16) -> list[int]:
    s = (text or "").strip().lower()
    if not s:
        return []
    words = re.findall(r"\w+", s) or s.split()
    raw_parts: list[str] = list(words)
    for w in words:
        for i in range(len(w) - 1):
            raw_parts.append(w[i : i + 2])
    seen: set[str] = set()
    features: list[str] = []
    for p in raw_parts:
        if p and p not in seen:
            seen.add(p)
            features.append(p)
            if len(features) >= max_toks:
                break
    return [TOK[f"TXT_{(zlib.crc32(p.encode('utf-8')) % 256):03d}"] for p in features]


def encode(obs: Observation) -> list[int]:
    ids = [TOK["[BOS]"]] + [TOK[f"TAG_{t}"] for t in obs.tags]
    if obs.name_token:
        ids.append(TOK[obs.name_token])
    if TONE_INPUT and obs.tone:
        ids.append(TOK[f"TONE_{obs.tone.upper()}"])
    if obs.cmd is not None:
        ids += [TOK["CMD"], TOK[f"VERB_{obs.cmd.verb}"]]
        if obs.cmd.target is not None:
            t = obs.cmd.target
            ids += [TOK[f"COLOR_{t.color}"], TOK[f"SHAPE_{t.shape}"], TOK[f"SIZE_{t.size}"], TOK[f"MAT_{t.material}"]]
    ids.append(TOK["[SEP]"])
    for i, o in enumerate(obs.options):
        ids += [TOK[f"OPT_{i}"], TOK[f"VERB_{o.verb}"]]
        if o.target is not None:
            t = o.target
            ids += [TOK[f"COLOR_{t.color}"], TOK[f"SHAPE_{t.shape}"], TOK[f"SIZE_{t.size}"], TOK[f"MAT_{t.material}"]]
        ids.append(TOK["[SEP]"])
    ids.append(TOK["ACT_PICK"])
    return ids


class Fact(BaseModel):
    """Something that happened: this thing was done in the scene and it turned out `result`. `said` = the parent had said to do it."""
    option: OptionView
    result: str
    said: bool = False
    weight: float = 1.0
    action_text: str = ""
    outcome_text: str = ""
    takeaway: str = ""


class Example(BaseModel):
    id: str
    day: int
    source: str  # show | praise | correct
    obs: Observation
    chosen: int  # the option the player showed / praised / put right
    weight: float = 1.0
    rejected: int | None = None  # the child's wrong try, for corrections
    was_right: bool = True  # analysis only, never a model input
    repeat: int = 1  # times the lesson is repeated at the moment it is taught (a firm tone); the buffer keeps it once
    strength: float = 1.0  # how strongly it was taught: the learning rounds it got; replay can favour strong lessons (docs/WORLD.md 22.10)
    outcome: str = ""  # analysis only: what the child's own wrong choice did (partial | nothing | worse), for corrections
    mode: str = ""  # analysis only: command_firm | command_gentle | show_firm | show_gentle | own_right | own_wrong | talk | burned
    facts: list[Fact] = []  # with GROWING_CONSEQUENCE: what this teaching act showed the child about the world (the only thing learned)
    negative: bool = False  # a bad experience: `chosen` is learned as something NOT to do here (no answer is given), docs/WORLD.md 22.12


def collate(exs: list[Example]):
    seqs = [encode(e.obs) for e in exs]
    T = max(len(s) for s in seqs)
    ids = torch.full((len(exs), T), PAD, dtype=torch.long)
    for i, s in enumerate(seqs):
        ids[i, : len(s)] = torch.tensor(s)
    return (ids, torch.tensor([len(s) for s in seqs]), torch.tensor([len(e.obs.options) for e in exs]),
            torch.tensor([e.chosen for e in exs]), torch.tensor([e.weight for e in exs], dtype=torch.float))


def option_logits(model, ids, lengths, n_opts):
    last = model(ids)[torch.arange(ids.shape[0]), lengths - 1][:, OPT_IDS]
    valid = torch.arange(MAX_OPTS).unsqueeze(0) < n_opts.unsqueeze(1)
    return last.masked_fill(~valid, float("-inf"))


def encode_try(obs: Any, option: Any, said: bool) -> list[int]:
    """[BOS] TAG.. [NAME] [SAID] ACT_DO VERB [attrs]: the scene without its options, then the one thing done. Read at the last token."""
    tags = getattr(obs, "tags", None) or ([obs.problem_tag] if hasattr(obs, "problem_tag") and obs.problem_tag else [])
    ids = [TOK["[BOS]"]] + [TOK[f"TAG_{t}"] for t in tags if f"TAG_{t}" in TOK]
    ids += _text_tokens(getattr(obs, "scene_text", "") or "", max_toks=16)
    name_tok = getattr(obs, "name_token", None)
    if name_tok and name_tok in TOK:
        ids.append(TOK[name_tok])
    if said and "SAID" in TOK:
        ids.append(TOK["SAID"])
    verb_tok = f"VERB_{option.verb}"
    if "ACT_DO" in TOK and verb_tok in TOK:
        ids += [TOK["ACT_DO"], TOK[verb_tok]]
    if option.target is not None:
        t = option.target
        kind = getattr(t, "kind", None)
        spec = KINDS.get(kind) if kind in KINDS else None
        color = getattr(t, "color", None) or "blue"
        shape = getattr(t, "shape", None) or (spec.shape if spec else "flat")
        size = getattr(t, "size", None) or ((spec.sizes[0] if spec.sizes else "small") if spec else "small")
        mat = getattr(t, "material", None) or (spec.material if spec else "wool")
        for tok_name in (f"COLOR_{color}", f"SHAPE_{shape}", f"SIZE_{size}", f"MAT_{mat}"):
            if tok_name in TOK:
                ids.append(TOK[tok_name])
    ids += _text_tokens(getattr(option, "option_text", "") or "", max_toks=16)
    return ids


def result_logits(model, seqs: list[list[int]]) -> torch.Tensor:
    T = max(len(s) for s in seqs)
    ids = torch.full((len(seqs), T), PAD, dtype=torch.long)
    for i, s in enumerate(seqs):
        ids[i, : len(s)] = torch.tensor(s)
    last = model(ids)[torch.arange(len(seqs)), torch.tensor([len(s) for s in seqs]) - 1]
    return last[:, OUT_IDS]


def consequence_ce(model, exs: list[Example]) -> torch.Tensor:
    rows = [(encode_try(e.obs, f.option, f.said), RESULTS.index(f.result), e.weight * f.weight) for e in exs for f in e.facts]
    assert rows, "a consequence child learns from facts"
    ce = F.cross_entropy(result_logits(model, [r[0] for r in rows]), torch.tensor([r[1] for r in rows]), reduction="none", label_smoothing=SMOOTH)
    w = torch.tensor([r[2] for r in rows], dtype=torch.float)
    return (ce * w).sum() / w.sum()


def weighted_ce(model, exs: list[Example]) -> torch.Tensor:
    if CONSEQUENCE:
        return consequence_ce(model, exs)
    ids, lengths, n, target, weight = collate(exs)
    logits = option_logits(model, ids, lengths, n)
    ce = F.cross_entropy(logits, target, reduction="none")
    neg = torch.tensor([e.negative for e in exs])
    if neg.any():  # unlikelihood, -log(1 - p(chosen)): pushes the option down and stops pushing once it is unlikely
        p = F.softmax(logits, dim=-1).gather(1, target[:, None])[:, 0].clamp(max=1 - 1e-4)
        ce = torch.where(neg, -torch.log1p(-p), ce)
    return (ce * weight).sum() / weight.sum()


def novelty_key(o: OptionView) -> tuple:
    """What counts as 'the same thing' for curiosity: the same act on the same kind of thing (as in `untested`)."""
    opt_text = (getattr(o, "option_text", "") or "").strip()
    if opt_text:
        return (o.verb, opt_text)
    return (o.verb,) if o.target is None else (o.verb, o.target.shape, o.target.material)


def seen_counts(buffer: list[Example]) -> dict[tuple, int]:
    """How many times the child has lived each thing, its own tries and the parent's shows alike."""
    n: dict[tuple, int] = {}
    for e in buffer:
        for f in e.facts:
            k = novelty_key(f.option)
            n[k] = n.get(k, 0) + 1
    return n


def novelty(o: OptionView, seen: dict[tuple, int], beta: float | None = None) -> float:
    return (CURIOSITY if beta is None else beta) / (1 + seen.get(novelty_key(o), 0)) ** 0.5


@torch.no_grad()
def option_probs(model, obs: Observation, seen: dict[tuple, int] | None = None, beta: float | None = None) -> list[float]:
    """`seen` (what the child has lived) turns curiosity on for a real choice; without it this is the pure expectation (quiz, scoring)."""
    model.eval()
    if CONSEQUENCE:  # the expected result of each option, as the child hopes for results, then a soft pick
        p = torch.softmax(result_logits(model, [encode_try(obs, o, obs.cmd is not None and o == obs.cmd) for o in obs.options]), dim=-1)
        want = p @ torch.tensor([WANT[r] for r in RESULTS])
        if seen is not None:
            want = want + torch.tensor([novelty(o, seen, beta) for o in obs.options])
        return torch.softmax(want / CHOICE_TEMP, dim=-1).tolist()
    ids, lengths, n, _, _ = collate([Example(id="q", day=0, source="show", obs=obs, chosen=0)])
    return torch.softmax(option_logits(model, ids, lengths, n), dim=-1)[0][: len(obs.options)].tolist()


def data_hash(exs: list[Example]) -> str:
    raw = "|".join(f"{e.id}:{e.chosen}:{e.weight}" + (f":x{e.repeat}" if e.repeat != 1 else "") + (f":s{e.strength}" if e.strength != 1 else "") + (":neg" if e.negative else "") + (f":f{len(e.facts)}" if e.facts else "") for e in exs)
    return hashlib.sha1(raw.encode()).hexdigest()[:12]


# ---- augmentation (DESIGN 4.5): shuffle option order; optionally re-draw the OTHER options ----
_PAIRS = sorted({(k.shape, k.material) for k in KINDS.values()})


def redraw_options(ex: Example, rng: random.Random) -> Example:
    """Keep the option the player chose exactly as chosen; replace every other option by a random one that
    differs from it in the verb or in the kind of thing it targets. Uses no rule knowledge (nothing about which
    options are good), only what the taught example already contains."""
    n, t = len(ex.obs.options), ex.chosen
    chosen = ex.obs.options[t]
    out = list(ex.obs.options)
    for i in range(n):
        if i == t or (ex.obs.cmd is not None and out[i] == ex.obs.cmd):  # the instructed option stays an option
            continue
        for _ in range(50):
            verb = rng.choice(VERBS)
            if verb == "WAIT":
                cand = OptionView(verb="WAIT", target=None)
            else:
                shape, mat = rng.choice(_PAIRS)
                cand = OptionView(verb=verb, target=TargetView(color=rng.choice(COLORS), shape=shape,
                                                               size=rng.choice(SIZES), material=mat))
            same_kind = (chosen.target is not None and cand.target is not None
                         and (chosen.target.shape, chosen.target.material) == (cand.target.shape, cand.target.material))
            if cand.verb == chosen.verb and (same_kind or (chosen.target is None and cand.target is None)):
                continue  # would look like a duplicate of what the player chose
            if cand in out:
                continue
            out[i] = cand
            break
    return ex.model_copy(update={"obs": ex.obs.model_copy(update={"options": out}), "rejected": None})


def shuffle_options(ex: Example, rng: random.Random) -> Example:
    perm = list(range(len(ex.obs.options)))
    rng.shuffle(perm)
    opts = [ex.obs.options[i] for i in perm]
    return ex.model_copy(update={
        "obs": ex.obs.model_copy(update={"options": opts}), "chosen": perm.index(ex.chosen),
        "rejected": None if ex.rejected is None else perm.index(ex.rejected)})


def untested(ex: Example, rng: random.Random) -> Example:
    """Things the child has not seen done are taken to do nothing (a prior, not a rule: a random thing does not solve the problem).
    Without it a child that has only seen what worked expects everything to work."""
    seen = [f.option for f in ex.facts]
    extra = []
    for _ in range(UNTESTED_N):
        shape, mat = rng.choice(_PAIRS)
        verb = rng.choice([v for v in VERBS if v != "WAIT"])
        cand = OptionView(verb=verb, target=TargetView(color=rng.choice(COLORS), shape=shape, size=rng.choice(SIZES), material=mat))
        if not any(o.verb == verb and o.target is not None and (o.target.shape, o.target.material) == (shape, mat) for o in seen):
            extra.append(Fact(option=cand, result="nothing", weight=UNTESTED_W))
    return ex.model_copy(update={"facts": ex.facts + extra})


def augment(ex: Example, rng: random.Random, p_redraw: float = 0.0) -> Example:
    if CONSEQUENCE:
        return untested(ex, rng) if UNTESTED_N and ex.facts else ex
    if p_redraw > 0 and rng.random() < p_redraw:
        ex = redraw_options(ex, rng)
    return shuffle_options(ex, rng)
