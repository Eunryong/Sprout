"""Can a tiny child learn "feelings" (set-valued, noisy outcomes) from a handful of scenes, and which way of learning does it best?

A baby world (docs/DIRECTION.md, "feelings"): 5 situations (cold, hungry, crying, bored, dark), the parent responds with a thing,
the baby feels one or two of six feelings (relieved, cared, eased, unmet, upset, ignored); one feeling of the set is the one
the child lives through this time. The child sees `scene + response` and must come to expect what it feels, and (from month 12)
choose what to do by what it hopes to feel. Everything is the same world and the same transformer (d64, 3 layers, from the game);
only the way of learning differs.

  families  S  supervised heads on the same tiny model (4 legacy results, 6 feelings: single / sampled / soft / multi-hot / valence)
            R  reinforcement learning (REINFORCE policy over options, epsilon-greedy Q)
            P  pretrained start of the SAME tiny model (generic worlds, the true world, a shifted world; frozen body)
            B  no network: a count table with attribute sharing (what the data alone allows), and chance

Run (one process per (arm, seed), parallel):
  uv run python scripts/exp_feelings.py --arms all --seeds 300-323 --out scripts/out/feelings.json
  uv run python scripts/exp_feelings.py --one f6_sample 300      # one run, prints JSON
"""
import argparse
import json
import math
import os
import random
import statistics as st
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
os.environ["GROWING_ONTOLOGY"] = str(ROOT / "config" / "ontology_age07.yaml")
os.environ.setdefault("GROWING_LLM", "mock")
sys.path.insert(0, str(ROOT))

import torch  # noqa: E402
import torch.nn as nn  # noqa: E402
import torch.nn.functional as F  # noqa: E402

from server.learning.codec import TOK, novelty_key  # noqa: E402
from server.learning.model import ModelCfg, TinyLM  # noqa: E402
from server.world.ontology import AFFORDANCES, COLORS, KINDS, afforded  # noqa: E402
from server.world.situation import Observation, OptionView, TargetView  # noqa: E402

OUT = ROOT / "scripts" / "out"
FEELS = ["relieved", "cared", "eased", "unmet", "upset", "ignored"]
VAL = [1.0, 0.8, 0.4, -0.3, -0.8, -1.0]  # the child's hidden drive over feelings (research yardstick too)
LEGACY_VAL = [1.0, 0.5, 0.0, -1.0]  # solved, partial, nothing, worse (the game today)
CHECKS = [int(x) for x in os.environ.get("EXP_CHECKS", "6,12,30,84").split(",")]  # EXP_* : the life of a 3-7 years game is 48 scenes, the child acts from the start
BABY_SCENES = int(os.environ.get("EXP_BABY", 12))  # until then the parent shows; after it the child chooses
PARENT_P = float(os.environ.get("EXP_PARENT", 0))  # after it, the share of scenes in which the parent still shows (a mixed parent: 60% a sensible thing)
TAGS = ["COLD", "BABY_HUNGRY", "UPSET", "BORED", "DARK"]
VERBS = ["COVER", "HUG", "FEED", "GIVE", "FETCH", "READ", "LIGHT", "DRAW"]
THINGS = ["blanket", "cloth", "teddy", "bottle", "spoon", "bread", "ball", "book", "flashlight", "crayon"]
CELLS = [(v, k) for v in VERBS for k in THINGS if afforded(v, k)] + [("WAIT", None)]
BETA, TEMP = 0.4, 0.25  # curiosity and softmax temperature of the game's choice rule


# ---------------------------------------------------------------- the world
def S(*names):
    return tuple(FEELS.index(n) for n in names)


def hand_table() -> dict:
    """tag -> {(verb, kind) | 'WAIT' | '_': feeling set}. Hand-written common sense about a baby."""
    return {
        "COLD": {("COVER", "blanket"): S("relieved", "cared"), ("COVER", "cloth"): S("eased", "unmet"), ("HUG", "blanket"): S("eased", "cared"),
                 ("HUG", "teddy"): S("eased", "cared"), "WAIT": S("unmet", "ignored"), "_": S("unmet", "upset")},
        "BABY_HUNGRY": {("FEED", "bottle"): S("relieved", "cared"), ("FEED", "spoon"): S("eased", "upset"), ("FEED", "bread"): S("upset", "unmet"),
                        ("HUG", "teddy"): S("cared", "unmet"), ("HUG", "blanket"): S("cared", "unmet"), "WAIT": S("ignored", "unmet"), "_": S("unmet", "upset")},
        "UPSET": {("HUG", "teddy"): S("relieved", "cared"), ("HUG", "blanket"): S("eased", "cared"), ("GIVE", "ball"): S("eased", "cared"),
                  ("FEED", "bottle"): S("eased", "cared"), ("READ", "book"): S("eased",), "WAIT": S("ignored", "upset"), "_": S("unmet", "upset")},
        "BORED": {("READ", "book"): S("relieved", "cared"), ("DRAW", "crayon"): S("relieved",), ("GIVE", "ball"): S("eased", "cared"),
                  ("HUG", "teddy"): S("eased", "cared"), ("FETCH", "teddy"): S("eased", "unmet"), ("FETCH", "ball"): S("eased", "unmet"),
                  "WAIT": S("ignored", "upset"), "_": S("unmet", "upset")},
        "DARK": {("LIGHT", "flashlight"): S("relieved", "cared"), ("HUG", "teddy"): S("eased", "cared"), ("HUG", "blanket"): S("eased", "cared"),
                 ("READ", "book"): S("unmet", "upset"), "WAIT": S("unmet", "ignored"), "_": S("unmet", "upset")},
    }


class World:
    def __init__(self, table: dict):
        self.t = table

    def fset(self, tag, verb, kind) -> tuple:
        row = self.t[tag]
        return row["WAIT"] if verb == "WAIT" else row.get((verb, kind), row["_"])

    def p(self, tag, verb, kind) -> list[float]:
        s = self.fset(tag, verb, kind)
        return [1 / len(s) if i in s else 0.0 for i in range(6)]

    def ev(self, tag, verb, kind) -> float:
        s = self.fset(tag, verb, kind)
        return sum(VAL[i] for i in s) / len(s)


def llm_table(min_share: float = 0.5) -> dict | None:
    """What muse-spark believes (scripts/exp_feelings_llm.py oracle): the feelings most repeats agree on, per (situation, response)."""
    p = OUT / "feel_llm_tables.json"
    if not p.exists():
        return None
    reps = json.loads(p.read_text())
    out = {}
    for tag in TAGS:
        row = {"_": S("unmet", "upset")}
        for c in CELLS:
            votes = [v for v in (r.get(tag, {}).get(f"{c[0]} {c[1]}") for r in reps) if v]
            if not votes:
                continue
            cnt: dict = {}
            for v in votes:
                for f in v:
                    cnt[f] = cnt.get(f, 0) + 1
            keep = [f for f, n in sorted(cnt.items(), key=lambda x: -x[1]) if n >= min_share * len(votes)] or [max(cnt, key=cnt.get)]
            row["WAIT" if c[0] == "WAIT" else c] = tuple(FEELS.index(f) for f in keep[:2])
        out[tag] = row
    return out


def random_table(rng: random.Random, permute_tags: bool = True) -> dict:
    """A world from the same family as the hand table but with different answers: which thing helps which trouble is arbitrary."""
    good = [S("relieved", "cared"), S("relieved",), S("eased", "cared")]
    part = [S("eased", "unmet"), S("eased", "cared"), S("cared", "unmet"), S("eased",)]
    out = {}
    for tag in TAGS:
        row = {"WAIT": rng.choice([S("ignored", "unmet"), S("ignored", "upset")]), "_": rng.choice([S("unmet", "upset"), S("unmet",), S("upset", "unmet")])}
        cells = [c for c in CELLS if c[0] != "WAIT"]
        rng.shuffle(cells)
        for c in cells[:2]:
            row[c] = rng.choice(good)
        for c in cells[2:5]:
            row[c] = rng.choice(part)
        out[tag] = row
    return out


def shifted_table(base: dict, rng: random.Random, frac: float = 0.3) -> dict:
    """The true world with `frac` of its specific answers replaced: a prior that is mostly right (a baby is not quite the textbook baby)."""
    rnd = random_table(rng)
    out = {}
    for tag, row in base.items():
        new = dict(row)
        for c in list(row):
            if c not in ("WAIT", "_") and rng.random() < frac:
                new[c] = rng.choice(list(rnd[tag].values()))
        out[tag] = new
    return out


def legacy_label(world: World, tag, verb, kind) -> int:
    e = world.ev(tag, verb, kind)
    return 0 if e >= 0.75 else 1 if e >= 0.25 else 2 if e >= -0.55 else 3


class Scene:
    def __init__(self, tag, cells, views):
        self.tag, self.cells, self.views = tag, cells, views
        self.obs = Observation(tags=[tag], options=views)


def make_view(rng, verb, kind) -> OptionView:
    if verb == "WAIT":
        return OptionView(verb="WAIT", target=None)
    k = KINDS[kind]
    return OptionView(verb=verb, target=TargetView(color=rng.choice(COLORS), shape=k.shape, size=rng.choice(list(k.sizes)), material=k.material))


def make_scene(rng: random.Random, world: World, tag: str | None = None) -> Scene:
    tag = tag or rng.choice(TAGS)
    evs = {c: world.ev(tag, *c) for c in CELLS}
    best = max(evs.values())
    pool = [c for c in CELLS if evs[c] == best and c[0] != "WAIT"] or [c for c in CELLS if evs[c] == best]
    cells = [rng.choice(pool)]
    others = [c for c in CELLS if c not in cells and not (c[0] != "WAIT" and any(c[1] == d[1] and c[0] == d[0] for d in cells))]
    rng.shuffle(others)
    for c in others:
        if len(cells) == 4:
            break
        if c[0] == "WAIT" and rng.random() > 0.3:
            continue
        cells.append(c)
    rng.shuffle(cells)
    return Scene(tag, cells, [make_view(rng, *c) for c in cells])


# ---------------------------------------------------------------- the child
def encode(tag: str, o: OptionView) -> list[int]:
    ids = [TOK["[BOS]"], TOK[f"TAG_{tag}"], TOK["ACT_DO"], TOK[f"VERB_{o.verb}"]]
    if o.target is not None:
        t = o.target
        ids += [TOK[f"COLOR_{t.color}"], TOK[f"SHAPE_{t.shape}"], TOK[f"SIZE_{t.size}"], TOK[f"MAT_{t.material}"]]
    return ids


class Net(nn.Module):
    """The game's tiny transformer (random start) with a small read-out on top of the last token."""

    def __init__(self, seed: int, k_out: int):
        super().__init__()
        torch.manual_seed(seed)
        self.body = TinyLM(ModelCfg(seed=seed))
        self.out = nn.Linear(64, k_out)

    def forward(self, seqs: list[list[int]]) -> torch.Tensor:
        T = max(len(s) for s in seqs)
        ids = torch.zeros((len(seqs), T), dtype=torch.long)
        for i, s in enumerate(seqs):
            ids[i, : len(s)] = torch.tensor(s)
        b = self.body
        x = b.tok(ids) + b.pos(torch.arange(T))
        mask = nn.Transformer.generate_square_subsequent_mask(T)
        for layer in b.layers:
            x = layer(x, src_mask=mask, is_causal=True)
        x = b.ln(x)
        last = x[torch.arange(len(seqs)), torch.tensor([len(s) for s in seqs]) - 1]
        return self.out(last)


class Exp:
    def __init__(self, tag, option: OptionView, cell, fset, legacy):
        self.tag, self.option, self.cell, self.fset, self.legacy = tag, option, cell, fset, legacy
        self.feel = None  # the feeling lived this time (index)
        self.scene: Scene | None = None  # the scene, for policy methods
        self.act: int = 0


ARMS = {  # name: (family, head, extra)
    "legacy4": ("S", "cls4", {}),
    "f6_single": ("S", "cls6_single", {}),
    "f6_sample": ("S", "cls6", {}),
    "f6_sample_nosmooth": ("S", "cls6", {"smooth": 0.0}),
    "f6_sample_steps60": ("S", "cls6", {"steps": 60}),
    "f6_soft": ("S", "soft6", {}),
    "f6_bce": ("S", "bce6", {}),
    "valence": ("S", "reg", {}),
    "q_eps": ("R", "reg", {"policy": "eps"}),
    "reinforce": ("R", "pg", {}),
    "pre_generic": ("P", "cls6", {"pre": "generic"}),
    "pre_true": ("P", "cls6", {"pre": "true"}),
    "pre_shifted": ("P", "cls6", {"pre": "shifted"}),
    "pre_wrong": ("P", "cls6", {"pre": "wrong"}),
    "pre_shifted_frozen": ("P", "cls6", {"pre": "shifted", "freeze": True}),
    "pre_true_frozen": ("P", "cls6", {"pre": "true", "freeze": True}),
    "pre_s60_frozen": ("P", "cls6", {"pre": "shifted60", "freeze": True}),
    "pre_s100_frozen": ("P", "cls6", {"pre": "shifted100", "freeze": True}),
    "pre_shifted_frozen_soft": ("P", "soft6", {"pre": "shifted", "freeze": True}),
    "pre_llm": ("P", "cls6", {"pre": "llm"}),
    "pre_llm_frozen": ("P", "cls6", {"pre": "llm", "freeze": True}),
    "pre_llm_frozen_soft": ("P", "soft6", {"pre": "llm", "freeze": True}),
    "rand_frozen": ("S", "cls6", {"freeze": True}),  # no pretraining at all: the random body is left as it is and only the read-out learns
    "rand_frozen_soft": ("S", "soft6", {"freeze": True}),
    "obs_lm_soft": ("S", "soft6", {"aux": 0.5}),  # "pretraining" the honest way: the child also learns to predict the scenes it watches (no outside knowledge)
    "obs_lm_sample": ("S", "cls6", {"aux": 0.5}),
    "ctrl_loss": ("S", "cls6", {"ctrl": "loss"}),
    "ctrl_early": ("S", "cls6", {"ctrl": "early"}),
    "ctrl_llm": ("S", "cls6", {"ctrl": "llm"}),
    "counts": ("B", None, {}),
    "chance": ("B", None, {}),
}
DEFAULT = {"lr": 2e-3, "steps": 20, "replay": 16, "wd": 0.05, "smooth": 0.15}


def head_size(head):
    return {"cls4": 4, "cls6": 6, "cls6_single": 6, "soft6": 6, "bce6": 6, "reg": 1, "pg": 1}[head]


class Child:
    def __init__(self, arm: str, seed: int, world: World, pre: dict | None = None):
        self.arm, (self.family, self.head, extra) = arm, ARMS[arm]
        self.cfg = {**DEFAULT, **{k: v for k, v in extra.items() if k in DEFAULT}}
        self.extra, self.world = extra, world
        self.rng = random.Random(seed * 7 + 1)
        self.buffer: list[Exp] = []
        self.seen: dict = {}
        self.base = 0.0  # reward baseline (REINFORCE)
        self.counts: dict = {}
        self.hook = None  # an outside controller of the training (muse-spark), see exp_feelings_llm.py
        self.hist: list = []
        if self.head:
            self.net = Net(seed, head_size(self.head))
            if pre is not None:
                self.net.load_state_dict(pre)
            params = [p for n, p in self.net.named_parameters() if not (extra.get("freeze") and n.startswith("body."))]
            self.opt = torch.optim.AdamW(params, lr=self.cfg["lr"], weight_decay=self.cfg["wd"])

    # ----- what the child expects (a distribution over feelings, or a scalar)
    @torch.no_grad()
    def dist(self, tag: str, options: list[OptionView]) -> list[list[float]] | None:
        if self.head in (None, "reg", "pg"):
            return None
        self.net.eval()
        z = self.net([encode(tag, o) for o in options])
        p = torch.softmax(z, -1) if self.head in ("cls4", "cls6", "cls6_single", "soft6") else torch.sigmoid(z)
        p = p / p.sum(-1, keepdim=True)
        return p.tolist()

    @torch.no_grad()
    def ev_hat(self, tag: str, options: list[OptionView], cells=None) -> list[float]:
        if self.family == "B":
            if self.arm == "chance":
                return [0.0] * len(options)
            return [self.count_ev(tag, c, o) for c, o in zip(cells, options)]
        if self.head in ("reg", "pg"):
            self.net.eval()
            return self.net([encode(tag, o) for o in options])[:, 0].tolist()
        val = LEGACY_VAL if self.head == "cls4" else VAL
        return [sum(pi * vi for pi, vi in zip(p, val)) for p in self.dist(tag, options)]

    def count_ev(self, tag, cell, o: OptionView) -> float:
        """Dirichlet counts with attribute sharing: exact cell, else the same verb+shape+material over all tags, else the tag, else prior."""
        keys = [("c", tag, o.verb, o.target.shape if o.target else None, o.target.material if o.target else None),
                ("v", o.verb, o.target.shape if o.target else None, o.target.material if o.target else None), ("t", tag)]
        tot, w = 0.0, 0.0
        for rank, k in enumerate(keys):
            n, s = self.counts.get(k, (0, 0.0))
            if n:
                tot += (3 - rank) * s
                w += (3 - rank) * n
        return tot / w if w else -0.3

    def choose(self, scene: Scene) -> int:
        if self.arm == "chance":
            return self.rng.randrange(len(scene.views))
        ev = self.ev_hat(scene.tag, scene.views, scene.cells)
        if self.head == "pg":
            p = torch.softmax(torch.tensor(ev), -1).tolist()
            return self.rng.choices(range(len(ev)), weights=p)[0]
        if self.extra.get("policy") == "eps":
            return self.rng.randrange(len(ev)) if self.rng.random() < 0.2 else max(range(len(ev)), key=ev.__getitem__)
        bonus = [BETA / math.sqrt(1 + self.seen.get(novelty_key(o), 0)) for o in scene.views]
        z = torch.tensor([(e + b) / TEMP for e, b in zip(ev, bonus)])
        return self.rng.choices(range(len(ev)), weights=torch.softmax(z, -1).tolist())[0]

    # ----- learning
    def _lm_loss(self, batch: list[Exp]) -> torch.Tensor:
        """Next-token prediction over the whole scenes the child has watched (situation and every option on the table)."""
        from server.learning.codec import encode as encode_scene
        seqs = [encode_scene(e.scene.obs) for e in batch]
        T = max(len(q) for q in seqs)
        ids = torch.zeros((len(seqs), T), dtype=torch.long)
        for i, q in enumerate(seqs):
            ids[i, : len(q)] = torch.tensor(q)
        logits = self.net.body(ids)[:, :-1]
        return F.cross_entropy(logits.reshape(-1, logits.shape[-1]), ids[:, 1:].reshape(-1), ignore_index=0)

    def _loss(self, batch: list[Exp]) -> torch.Tensor:
        loss = self._loss_main(batch)
        return loss + self.extra["aux"] * self._lm_loss(batch) if self.extra.get("aux") else loss

    def _loss_main(self, batch: list[Exp]) -> torch.Tensor:
        h = self.head
        if h == "pg":
            tot = 0.0
            for e in batch:
                z = self.net([encode(e.tag, o) for o in e.scene.views])[:, 0]
                adv = VAL[e.feel] - self.base
                tot = tot - adv * F.log_softmax(z, -1)[e.act] - 0.01 * (-(F.softmax(z, -1) * F.log_softmax(z, -1)).sum())
            return tot / len(batch)
        z = self.net([encode(e.tag, e.option) for e in batch])
        if h == "reg":
            return F.mse_loss(z[:, 0], torch.tensor([VAL[e.feel] for e in batch]))
        if h == "cls4":
            return F.cross_entropy(z, torch.tensor([e.legacy for e in batch]), label_smoothing=self.cfg["smooth"])
        if h == "cls6":
            return F.cross_entropy(z, torch.tensor([e.feel for e in batch]), label_smoothing=self.cfg["smooth"])
        if h == "cls6_single":
            return F.cross_entropy(z, torch.tensor([e.fset[0] for e in batch]), label_smoothing=self.cfg["smooth"])
        tgt = torch.zeros(len(batch), 6)
        for i, e in enumerate(batch):
            for f in e.fset:
                tgt[i, f] = 1.0
        if h == "soft6":
            return -(tgt / tgt.sum(-1, keepdim=True) * F.log_softmax(z, -1)).sum(-1).mean()
        return F.binary_cross_entropy_with_logits(z, tgt)  # bce6

    def learn(self, e: Exp) -> None:
        self.buffer.append(e)
        self.seen[novelty_key(e.option)] = self.seen.get(novelty_key(e.option), 0) + 1
        if self.head == "pg":
            self.base += 0.2 * (VAL[e.feel] - self.base)
        if self.family == "B":
            for k in [("c", e.tag, e.option.verb, e.option.target.shape if e.option.target else None, e.option.target.material if e.option.target else None),
                      ("v", e.option.verb, e.option.target.shape if e.option.target else None, e.option.target.material if e.option.target else None), ("t", e.tag)]:
                n, s = self.counts.get(k, (0, 0.0))
                self.counts[k] = (n + 1, (s * n + VAL[e.feel]) / (n + 1))
            return
        old = self.buffer[:-1]
        steps, replay, lr_scale = self.cfg["steps"], self.cfg["replay"], 1.0
        if self.extra.get("ctrl"):
            with torch.no_grad():
                self.net.eval()
                loss0 = self._loss([e]).item()
            steps, replay, lr_scale = self.control(e, loss0)
            self.hist.append(loss0)
        for g in self.opt.param_groups:
            g["lr"] = self.cfg["lr"] * lr_scale
        self.net.train()
        for _ in range(steps):
            batch = [e] + (self.rng.sample(old, min(replay, len(old))) if old else [])
            loss = self._loss(batch)
            self.opt.zero_grad()
            loss.backward()
            self.opt.step()
        self.net.eval()


def _control(self, e: Exp, loss0: float) -> tuple[int, int, float]:
    """How to train on this experience: (steps, replay, lr scale). Chosen from the surprise alone, never from the world's table."""
    mode, n = self.extra["ctrl"], len(self.buffer)
    if mode == "loss":  # more work when the scene surprised the child, little when it already expected it
        return (5 if loss0 < 0.6 else 40), 16, 1.0
    if mode == "early":  # a young child lives each scene hard, an older one more gently
        return (60 if n <= BABY_SCENES else 10), 16, 1.0
    summary = {"experiences_so_far": n, "this_surprise": round(loss0, 2), "recent_surprises": [round(x, 2) for x in self.hist[-5:]],
               "per_situation": {t: sum(1 for b in self.buffer if b.tag == t) for t in TAGS}, "defaults": {"steps": 20, "replay": 16, "lr_scale": 1.0}}
    return self.hook(summary)


Child.control = _control


# ---------------------------------------------------------------- measuring
def kl(p, q):
    return sum(pi * math.log(pi / max(qi, 1e-9)) for pi, qi in zip(p, q) if pi > 0)


def measure(child: Child, world: World, test_scenes: list[Scene], cell_views: dict) -> dict:
    out = {}
    cells = [(t, c) for t in TAGS for c in CELLS]
    if child.head in ("cls4",):
        pass
    kls, maes, tops = [], [], []
    for t in TAGS:
        views = [cell_views[c] for c in CELLS]
        ev = child.ev_hat(t, views, CELLS)
        d = child.dist(t, views)
        for i, c in enumerate(CELLS):
            true = world.p(t, *c)
            maes.append(abs(ev[i] - world.ev(t, *c)))
            if d is not None and child.head != "cls4":
                kls.append(kl(true, d[i]))
                tops.append(true[max(range(6), key=d[i].__getitem__)] > 0)
    out["val_mae"] = st.mean(maes) if child.head != "pg" else float("nan")  # a policy's scores are not on the feelings scale
    out["kl"] = st.mean(kls) if kls else float("nan")
    out["top_in_set"] = st.mean(tops) if tops else float("nan")
    hits, regret = [], []
    for sc in test_scenes:
        ev = child.ev_hat(sc.tag, sc.views, sc.cells)
        tv = [world.ev(sc.tag, *c) for c in sc.cells]
        pick = max(range(len(ev)), key=ev.__getitem__)
        hits.append(tv[pick] >= max(tv) - 1e-9)
        regret.append(max(tv) - tv[pick])
    out["best_pick"], out["regret"] = st.mean(hits), st.mean(regret)
    return out


def pretrain_state(kind: str, world_true: World, seed: int, steps: int = 1500) -> dict:
    """Pretrain the same tiny transformer (6-way feelings) on data from other worlds (or this one) before any life starts."""
    rng = random.Random(10_000 + seed)
    torch.manual_seed(seed)
    net = Net(seed, 6)
    opt = torch.optim.AdamW(net.parameters(), lr=2e-3, weight_decay=0.05)
    fixed = {"true": world_true, "wrong": World(random_table(rng))}.get(kind)
    if kind.startswith("shifted"):
        fixed = World(shifted_table(world_true.t, rng, (int(kind[7:]) if kind[7:] else 30) / 100))
    if kind == "llm":
        fixed = World(llm_table())
    for step in range(steps):
        w = fixed if fixed is not None else World(random_table(rng))  # generic: a new random world each step, so no table can be memorised
        batch = []
        for _ in range(48):
            sc = make_scene(rng, w)
            i = rng.randrange(len(sc.cells))
            fs = w.fset(sc.tag, *sc.cells[i])
            batch.append((encode(sc.tag, sc.views[i]), rng.choice(fs)))
        z = net([b[0] for b in batch])
        loss = F.cross_entropy(z, torch.tensor([b[1] for b in batch]), label_smoothing=0.15)
        opt.zero_grad()
        loss.backward()
        opt.step()
    return net.state_dict()


def pre_path(kind: str, seed: int, truth: str = "hand") -> Path:
    return OUT / "feel_pre" / (f"{kind}_{seed}.pt" if truth == "hand" else f"{kind}_{truth}_{seed}.pt")


def one(arm: str, seed: int, truth: str = "hand", hook=None) -> dict:
    torch.set_num_threads(1)
    rng = random.Random(seed)
    world = World({"hand": hand_table, "llm": llm_table, "random": lambda: random_table(random.Random(seed * 13 + 5))}[truth]())  # truth = whose idea of a baby is the world's
    pre = None
    if ARMS[arm][2].get("pre"):
        p = pre_path(ARMS[arm][2]["pre"], seed, truth)
        pre = torch.load(p) if p.exists() else pretrain_state(ARMS[arm][2]["pre"], world, seed)
    child = Child(arm, seed, world, pre)
    child.hook = hook
    trng = random.Random(777)  # the same held-out scenes for everyone
    test_scenes = [make_scene(trng, world) for _ in range(150)]
    cell_views = {c: make_view(trng, *c) for c in CELLS}
    res, got = {}, []
    t0 = time.time()
    for n in range(1, CHECKS[-1] + 1):
        sc = make_scene(rng, world)
        shown = n <= BABY_SCENES or rng.random() < PARENT_P
        if shown:  # the parent shows: a mixed parent, 60% a sensible thing
            tv = [world.ev(sc.tag, *c) for c in sc.cells]
            a = max(range(len(tv)), key=tv.__getitem__) if rng.random() < 0.6 else rng.randrange(len(tv))
        else:
            a = child.choose(sc)
        tv = world.ev(sc.tag, *sc.cells[a])
        if not shown:
            got.append(tv)
        e = Exp(sc.tag, sc.views[a], sc.cells[a], world.fset(sc.tag, *sc.cells[a]), legacy_label(world, sc.tag, *sc.cells[a]))
        e.feel, e.scene, e.act = rng.choice(e.fset), sc, a
        child.learn(e)
        if n in CHECKS:
            res[n] = measure(child, world, test_scenes, cell_views) | {"own_reward": st.mean(got) if got else float("nan")}
    res["seconds"] = time.time() - t0
    return res


def _job(t):
    return t[0], t[1], one(*t)  # (arm, seed, truth)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--one", nargs=2, metavar=("ARM", "SEED"))
    ap.add_argument("--arms", nargs="*", default=["all"])
    ap.add_argument("--seeds", default="300-323")
    ap.add_argument("--jobs", type=int, default=14)
    ap.add_argument("--out", default=str(OUT / "feelings.json"))
    ap.add_argument("--truth", default="hand", choices=["hand", "llm", "random"], help="whose table is the baby: the hand-written one, muse-spark's, or a random world per seed")
    a = ap.parse_args()
    if a.one:
        print(json.dumps(one(a.one[0], int(a.one[1]), a.truth), indent=1))
        return
    arms = [x for x in ARMS if x != "ctrl_llm"] if a.arms == ["all"] else a.arms
    lo, hi = a.seeds.split("-")
    seeds = list(range(int(lo), int(hi) + 1))
    (OUT / "feel_pre").mkdir(parents=True, exist_ok=True)
    need = sorted({(ARMS[x][2]["pre"], s, a.truth) for x in arms if ARMS[x][2].get("pre") for s in seeds})
    if need:  # pretrain once per (kind, seed), in parallel, and keep the weights
        with ProcessPoolExecutor(a.jobs) as ex:
            list(ex.map(_pre_job, need))
    tasks = [(arm, s, a.truth) for arm in arms for s in seeds]
    with ProcessPoolExecutor(a.jobs) as ex:
        rows = list(ex.map(_job, tasks))
    Path(a.out).write_text(json.dumps([{"arm": r[0], "seed": r[1], "res": r[2]} for r in rows]))
    print(report(arms, rows, Path(a.out).with_suffix(".txt")))


def _pre_job(t):
    torch.set_num_threads(1)
    kind, seed, truth = t
    p = pre_path(kind, seed, truth)
    if not p.exists():
        torch.save(pretrain_state(kind, World(hand_table() if truth == "hand" else llm_table()), seed), p)


def report(arms, rows, path=None) -> str:
    keys = ["kl", "val_mae", "top_in_set", "best_pick", "regret", "own_reward"]
    lines = []
    for n in CHECKS:
        lines.append(f"\nafter {n} scenes {'':8s}" + "".join(f"{k:>12s}" for k in keys))
        for arm in arms:
            rs = [r[2][n] for r in rows if r[0] == arm]
            cells = []
            for k in keys:
                v = [r[k] for r in rs if not math.isnan(r[k])]
                cells.append(f"{st.mean(v):12.3f}" if v else f"{'-':>12s}")
            lines.append(f"  {arm:24s}" + "".join(cells))
    text = "\n".join(lines)
    Path(path or OUT / "feelings.txt").write_text(text + "\n")
    return text


if __name__ == "__main__":
    main()
