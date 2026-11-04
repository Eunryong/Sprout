import math
import random
import time
from dataclasses import dataclass

import torch
from pydantic import BaseModel

from .codec import Example, augment, data_hash, weighted_ce


@dataclass
class OnlineCfg:
    lr: float = 2e-3
    steps: int = 20
    replay: int = 16
    oversample: int = 1
    weight_decay: float = 0.05
    batch_seed: int = 0
    # optional stabilizers (off by default; the sweep found them unnecessary once redraw + low lr + weight decay are on)
    lr_decay_tau: float = 0.0  # >0: lr_call = lr / (1 + n_calls / tau), i.e. smaller steps as the child matures
    cosine: bool = False  # anneal the lr inside one call down to 10%
    stop_loss: float = 0.0  # >0: stop a call early once the new example's loss is below this
    grad_clip: float = 0.0
    check_every: int = 5
    redraw: float = 1.0  # probability of re-drawing the other options of a sample (the chosen option is never changed)
    # Replay that remembers the way people do (docs/WORLD.md 22.10), off by default (uniform replay): an old lesson is replayed
    # in proportion to its strength x replay_decay^(days since it was taught). 1.0 = a lesson never fades.
    replay_by_strength: bool = False
    replay_decay: float = 1.0


class LearnReport(BaseModel):
    steps: int
    loss_before: float
    loss_after: float
    seconds: float
    batch_size: int
    data_hash: str


class Learner:
    """Immediate learning after a choice: the new example (oversampled) plus replayed older examples."""

    def __init__(self, model, cfg: OnlineCfg, ce_fn=None, aug_fn=None):  # injectable for experiments
        self.model, self.cfg = model, cfg
        self.ce = ce_fn or weighted_ce
        self.aug = aug_fn or augment
        self.n_calls = 0  # number of learn() calls so far (restored from the dataset size on load)
        self.opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)

    def _replay(self, old: list[Example], new: Example, rng: random.Random) -> list[Example]:
        c = self.cfg
        k = min(c.replay, len(old))
        if not old or (not c.replay_by_strength and c.replay_decay == 1.0):
            return rng.sample(old, k) if old else []
        # weighted sampling without replacement (Efraimidis-Spirakis): the larger the weight, the likelier it is drawn
        w = [(e.strength if c.replay_by_strength else 1.0) * c.replay_decay ** max(0, new.day - e.day) for e in old]
        keys = [rng.random() ** (1.0 / max(x, 1e-9)) for x in w]
        return [old[i] for i in sorted(range(len(old)), key=keys.__getitem__, reverse=True)[:k]]

    def learn(self, new: Example, buffer: list[Example], rng: random.Random, steps: int | None = None) -> LearnReport:
        t0 = time.perf_counter()
        steps = steps or self.cfg.steps
        old = [e for e in buffer if e.id != new.id]
        self.model.eval()
        with torch.no_grad():
            before = self.ce(self.model, [new]).item()
        self.model.train()
        c = self.cfg
        lr_call = c.lr / (1 + self.n_calls / c.lr_decay_tau) if c.lr_decay_tau > 0 else c.lr
        done = 0
        for step in range(steps):
            lr = lr_call * (0.1 + 0.45 * (1 + math.cos(math.pi * step / steps))) if c.cosine else lr_call
            for g in self.opt.param_groups:
                g["lr"] = lr
            replay = self._replay(old, new, rng)
            batch = [self.aug(new, rng, c.redraw) for _ in range(c.oversample * new.repeat)] + [self.aug(e, rng, c.redraw) for e in replay]
            loss = self.ce(self.model, batch)
            self.opt.zero_grad()
            loss.backward()
            if c.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), c.grad_clip)
            self.opt.step()
            done = step + 1
            if c.stop_loss > 0 and done % c.check_every == 0:
                with torch.no_grad():
                    if self.ce(self.model, [new]).item() < c.stop_loss:
                        break
        self.n_calls += 1
        self.model.eval()
        with torch.no_grad():
            after = self.ce(self.model, [new]).item()
        return LearnReport(steps=done, loss_before=before, loss_after=after,
                           seconds=time.perf_counter() - t0, batch_size=self.cfg.oversample * new.repeat + len(replay),
                           data_hash=data_hash(buffer))

    def learn_batch(self, buffer: list[Example], rng: random.Random, steps: int, batch_size: int = 20) -> LearnReport:
        """End-of-day batch alternative, used as the comparison condition in the D1 gate."""
        t0 = time.perf_counter()
        self.model.eval()
        with torch.no_grad():
            before = self.ce(self.model, buffer).item()
        self.model.train()
        for _ in range(steps):
            batch = [self.aug(e, rng, 0.0) for e in rng.sample(buffer, min(batch_size, len(buffer)))]
            loss = self.ce(self.model, batch)
            self.opt.zero_grad()
            loss.backward()
            self.opt.step()
        self.model.eval()
        with torch.no_grad():
            after = self.ce(self.model, buffer).item()
        return LearnReport(steps=steps, loss_before=before, loss_after=after,
                           seconds=time.perf_counter() - t0, batch_size=batch_size, data_hash=data_hash(buffer))
