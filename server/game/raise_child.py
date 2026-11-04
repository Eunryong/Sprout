"""Raise a child from birth to three with an LLM parent, so that a player takes over a child that has already lived and learned.

The child is the same tiny randomly initialised transformer. For every scene of the first years muse-spark plays the parent (agents/parent.py:
from the player's words only, once per scene, sometimes wrong), the world answers by its own rules, and the child really trains after
each scene, exactly as it will with the player. The Director and the Narrator are left out (the template scenes and the fixed phrases are
enough: nobody reads these years), so only the parent's decisions cost a model call.

  uv run python -m server.game.raise_child --seed 11 --persona warm --out saves/children/mina
  uv run python -m server.game.raise_child --seed 11 --persona busy --out saves/children/dori --mock      # no network
"""
import argparse
import json
import os
import sys
import time
from pathlib import Path

if __name__ == "__main__":  # the world is chosen before anything reads the ontology (it is read once, at import)
    _args = sys.argv
    if "--world-seed" in _args and "--world" not in _args:  # a household of its own: its rules are drawn at random (config/world_pools_age07.yaml)
        from server.world.worldgen import CONFIG, load_pools, template_world, write_world
        _seed = int(_args[_args.index("--world-seed") + 1])
        _pools = load_pools(CONFIG / "world_pools_age07.yaml")
        _args += ["--world", str(write_world(template_world(_seed, _pools), Path("saves/worlds") / f"h{_seed}", _pools)[0])]
    os.environ["GROWING_ONTOLOGY"] = str(Path(_args[_args.index("--world") + 1]).resolve()) if "--world" in _args else \
        os.environ.get("GROWING_ONTOLOGY") or str(Path(__file__).resolve().parents[2] / "config" / "ontology_age07.yaml")

from server import i18n
from server.agents.parent import PERSONA_KO, PERSONAS, ParentAgent
from server.world.ontology import age_months

from .simple import SimpleGame

RAISED = "raised.json"


def raise_child(save_dir, seed: int, persona: str = "warm", until_month: int = 36, client=None, progress=None, max_scenes: int = 200) -> dict:
    """Raise a new child to `until_month` months (or `max_scenes`, for a world without ages). `client` is the parent's mind (an LLM client, or None for a tired fallback parent)."""
    save_dir = Path(save_dir)
    lang = i18n.lang()
    i18n.set_lang("en")  # the parent reads English; the album is told later in whatever language the player reads
    try:
        g = SimpleGame.new(save_dir, seed=seed, client=None)
        parent = ParentAgent(client, persona, seed, on_call=g.s.save.append_llm)
        log, recent, t0 = [], [], time.time()
        while True:
            sc = g.scene()
            if g.ended or g.age_now() >= until_month or g.lived >= max_scenes:
                break
            d = parent.decide(sc, recent)
            turn = g.play(d.part, d.option)
            recent.append(f"{d.part}: {sc.options[turn.acted].text} -> {turn.narration.split('. ', 1)[-1]}")
            log.append({"scene": sc.number, "age_months": sc.age_months, "part": d.part, "told_option": d.option, "acted": turn.acted, "by": turn.by,
                        "action": sc.options[turn.acted].text, "result": turn.result, "reason": d.reason, "scene_text": sc.narration})
            if progress:
                progress(sc.number, sc.age_months, d)
        (save_dir / "parent_log.jsonl").write_text("".join(json.dumps(x, ensure_ascii=False) + "\n" for x in log))
        info = {"name": save_dir.name, "persona": persona, "persona_ko": PERSONA_KO[persona], "seed": seed, "months": g.age_now(), "scenes": g.lived, "world": world_info(), "parent_model": getattr(client, "name", None),
                "parent_calls": parent.stats.calls, "parent_fallbacks": parent.stats.fallback + parent.stats.no_client, "seconds": round(time.time() - t0),
                "showed": sum(x["part"] == "show" for x in log), "told": sum(x["part"] == "say" for x in log), "left": sum(x["part"] == "leave" for x in log)}
        (save_dir / RAISED).write_text(json.dumps(info, ensure_ascii=False, indent=1))
        return info
    finally:
        i18n.set_lang(lang)


def main() -> None:
    ap = argparse.ArgumentParser(prog="server.game.raise_child")
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--persona", default="warm", choices=sorted(PERSONAS))
    ap.add_argument("--out", required=True, help="the save directory of the raised child (must not exist yet)")
    ap.add_argument("--until", type=int, default=36, help="months to raise the child to")
    ap.add_argument("--mock", action="store_true", help="a stand-in parent instead of muse-spark (no network)")
    ap.add_argument("--world", default="", help="an ontology.yaml (default: the 0-7 years world)")
    ap.add_argument("--world-seed", type=int, default=None, help="draw a world at random for this child (saved in saves/worlds/h<N>); the same number gives the same world")
    a = ap.parse_args()
    out = Path(a.out)
    if (out / "meta.json").exists():
        sys.exit(f"{out} already holds a child; choose another --out")
    from server.agents.factory import make_client
    client = make_client({}, mode="mock" if a.mock else "live")
    info = raise_child(out, a.seed, a.persona, a.until, client, lambda n, m, d: print(f"  scene {n:2d} · {age_months_text(m):>9s} · {d.part:5s} {'' if d.option is None else d.option}  {d.reason[:70]}", flush=True))
    print(json.dumps(info, ensure_ascii=False))


def world_info() -> dict | None:
    """Which world this child grew up in (the web page shows it on the chooser)."""
    from server.world.ontology import ONTOLOGY_HASH, ONTOLOGY_PATH
    from server.world.worldgen import read_sheet
    sheet = read_sheet(ONTOLOGY_PATH)
    return {"hash": ONTOLOGY_HASH, "seed": sheet.seed if sheet else None, "random": sheet is not None}


def age_months_text(m: int) -> str:
    y, mo = divmod(m, 12)
    return f"{y}y{mo}m"


if __name__ == "__main__":
    main()
