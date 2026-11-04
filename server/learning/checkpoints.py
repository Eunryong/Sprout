from pathlib import Path

import torch

from .model import ModelCfg, TinyLM, build_model


def save(model: TinyLM, path: str | Path, opt=None, meta: dict | None = None, overwrite: bool = True) -> None:
    path = Path(path)
    if path.exists() and not overwrite:
        raise FileExistsError(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    torch.save({"state": model.state_dict(), "cfg": model.cfg.to_dict(),
                "opt": opt.state_dict() if opt else None, "meta": meta or {}}, tmp)
    tmp.replace(path)  # atomic: a crash never leaves a half-written checkpoint


def load(path: str | Path):
    blob = torch.load(path, weights_only=True)
    model = build_model(ModelCfg(**blob["cfg"]))
    model.load_state_dict(blob["state"])
    return model, blob["meta"]


def load_optimizer_state(path: str | Path, opt) -> bool:
    blob = torch.load(path, weights_only=True)
    if blob.get("opt"):
        opt.load_state_dict(blob["opt"])
        return True
    return False
