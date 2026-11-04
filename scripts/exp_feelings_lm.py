"""The same baby world as exp_feelings.py, learned by an already-trained small language model (SmolLM2-135M) instead of a random start.

The child is the language model: the scene and the parent's response are written as a short sentence, the model's next word is read
as the feeling (restricted to the six feeling words), and it learns online from each lived scene (LoRA or full fine-tuning), exactly
like the tiny transformer. Two ways of telling it the thing: `kind` (it is told "a big red blanket", so its world knowledge helps)
and `attrs` (only what the game's child sees: "a big red flat wool thing").

  uv run --with transformers --with peft python scripts/exp_feelings_lm.py --arms all --seeds 300-307 --jobs 4
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
sys.path.insert(0, str(ROOT / "scripts"))
import exp_feelings as X  # noqa: E402  (sets GROWING_ONTOLOGY and the world)

import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402

from server.world.ontology import KINDS  # noqa: E402

MODEL = "HuggingFaceTB/SmolLM2-135M"
TAG_TEXT = {"COLD": "The baby is cold and shivering.", "BABY_HUNGRY": "The baby is hungry and fussing for milk.", "UPSET": "The baby is crying and cannot settle.",
            "BORED": "The baby is bored and fretful.", "DARK": "The baby is frightened in the dark."}
VERB_TEXT = {"COVER": "wraps the baby in {k}", "HUG": "cuddles the baby with {k}", "FEED": "feeds the baby with {k}", "GIVE": "gives the baby {k} to hold",
             "FETCH": "brings the baby {k}", "READ": "shows the baby {k}", "LIGHT": "switches on {k} near the baby", "DRAW": "waves {k} in front of the baby",
             "WAIT": "waits and does nothing"}
FEEL_WORD = [" relieved", " cared", " eased", " unmet", " upset", " ignored"]
KIND_OF = {(k.shape, k.material): n for n, k in KINDS.items()}
ARMS = {  # name: (how the thing is told, how it learns)
    "lm_zero_kind": ("kind", "none"), "lm_zero_attrs": ("attrs", "none"),
    "lm_lora_kind": ("kind", "lora"), "lm_lora_attrs": ("attrs", "lora"),
    "lm_full_kind": ("kind", "full"), "lm_full_attrs": ("attrs", "full"),
}
SNAP = None


def snapshot() -> str:
    snaps = Path.home() / ".cache/huggingface/hub/models--HuggingFaceTB--SmolLM2-135M/snapshots"  # downloaded once, files only (no network here)
    return str(next(snaps.iterdir()))


def text(tag: str, o, told: str) -> str:
    if o.target is None:
        thing = ""
    elif told == "kind":
        thing = f"the {o.target.size} {o.target.color} {KIND_OF[(o.target.shape, o.target.material)]}"
    else:
        thing = f"a {o.target.size} {o.target.color} {o.target.shape} {o.target.material} thing"
    return f"{TAG_TEXT[tag]} The parent {VERB_TEXT[o.verb].format(k=thing)}. The baby feels"


class LMChild(X.Child):
    def __init__(self, arm: str, seed: int, world):
        from transformers import AutoModelForCausalLM, AutoTokenizer
        self.arm, self.family, self.head, self.extra, self.world = arm, "S", "cls6", {}, world
        self.told, self.mode = ARMS[arm]
        self.cfg = {**X.DEFAULT, "lr": 2e-4 if self.mode == "lora" else 2e-5}
        self.rng, self.buffer, self.seen, self.base, self.counts = random.Random(seed * 7 + 1), [], {}, 0.0, {}
        torch.manual_seed(seed)
        self.tok = AutoTokenizer.from_pretrained(SNAP)
        self.lm = AutoModelForCausalLM.from_pretrained(SNAP, torch_dtype=torch.float32)
        self.tok.pad_token = self.tok.pad_token or self.tok.eos_token
        self.tok.padding_side = "right"
        self.ids = [self.tok.encode(w)[0] for w in FEEL_WORD]
        assert len(set(self.ids)) == 6, "the six feeling words need distinct first tokens"
        if self.mode == "lora":
            from peft import LoraConfig, get_peft_model
            self.lm = get_peft_model(self.lm, LoraConfig(r=8, lora_alpha=16, target_modules=["q_proj", "k_proj", "v_proj", "o_proj"], lora_dropout=0.0))
        if self.mode != "none":
            self.opt = torch.optim.AdamW([p for p in self.lm.parameters() if p.requires_grad], lr=self.cfg["lr"], weight_decay=self.cfg["wd"])
        self.lm.eval()

    def _logits(self, texts: list[str]) -> torch.Tensor:
        enc = self.tok(texts, return_tensors="pt", padding=True)
        out = self.lm(**enc).logits
        last = enc["attention_mask"].sum(1) - 1 if self.tok.padding_side == "right" else torch.full((len(texts),), out.shape[1] - 1)
        return out[torch.arange(len(texts)), last][:, self.ids]

    @torch.no_grad()
    def dist(self, tag, options):
        self.lm.eval()
        return torch.softmax(self._logits([text(tag, o, self.told) for o in options]), -1).tolist()

    def learn(self, e) -> None:
        self.buffer.append(e)
        self.seen[X.novelty_key(e.option)] = self.seen.get(X.novelty_key(e.option), 0) + 1
        if self.mode == "none":
            return
        old = self.buffer[:-1]
        self.lm.train()
        for _ in range(self.cfg["steps"]):
            batch = [e] + (self.rng.sample(old, min(self.cfg["replay"], len(old))) if old else [])
            z = self._logits([text(b.tag, b.option, self.told) for b in batch])
            loss = F.cross_entropy(z, torch.tensor([b.feel for b in batch]), label_smoothing=self.cfg["smooth"])
            self.opt.zero_grad()
            loss.backward()
            self.opt.step()
        self.lm.eval()


def one(arm: str, seed: int) -> dict:
    global SNAP
    torch.set_num_threads(4)
    SNAP = SNAP or snapshot()
    rng = random.Random(seed)
    world = X.World(X.hand_table())
    child = LMChild(arm, seed, world)
    trng = random.Random(777)
    test_scenes = [X.make_scene(trng, world) for _ in range(150)]
    cell_views = {c: X.make_view(trng, *c) for c in X.CELLS}
    res, got, t0 = {}, [], time.time()
    for n in range(1, X.CHECKS[-1] + 1):
        sc = X.make_scene(rng, world)
        if n <= X.BABY_SCENES:
            tv = [world.ev(sc.tag, *c) for c in sc.cells]
            a = max(range(len(tv)), key=tv.__getitem__) if rng.random() < 0.6 else rng.randrange(len(tv))
        else:
            a = child.choose(sc)
        if n > X.BABY_SCENES:
            got.append(world.ev(sc.tag, *sc.cells[a]))
        e = X.Exp(sc.tag, sc.views[a], sc.cells[a], world.fset(sc.tag, *sc.cells[a]), 0)
        e.feel, e.scene, e.act = rng.choice(e.fset), sc, a
        child.learn(e)
        if n in X.CHECKS:
            res[n] = X.measure(child, world, test_scenes, cell_views) | {"own_reward": st.mean(got) if got else float("nan")}
    res["seconds"] = time.time() - t0
    return res


def _job(t):
    return t[0], t[1], one(*t)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--one", nargs=2, metavar=("ARM", "SEED"))
    ap.add_argument("--arms", nargs="*", default=["all"])
    ap.add_argument("--seeds", default="300-307")
    ap.add_argument("--jobs", type=int, default=4)
    ap.add_argument("--out", default=str(X.OUT / "feelings_lm.json"))
    a = ap.parse_args()
    if a.one:
        print(json.dumps(one(a.one[0], int(a.one[1])), indent=1))
        return
    arms = list(ARMS) if a.arms == ["all"] else a.arms
    lo, hi = a.seeds.split("-")
    tasks = [(arm, s) for arm in arms for s in range(int(lo), int(hi) + 1)]
    with ProcessPoolExecutor(a.jobs) as ex:
        rows = list(ex.map(_job, tasks))
    Path(a.out).write_text(json.dumps([{"arm": r[0], "seed": r[1], "res": r[2]} for r in rows]))
    text_ = X.report(arms, rows, X.OUT / "feelings_lm.txt")
    print(text_)


if __name__ == "__main__":
    main()
