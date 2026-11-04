"""uv run python -m server.web [--lang ko] [--fresh] [--seed 0] [--save saves/web/current] [--port 8000] [--new-world | --world PATH] [--mood TEXT]

The game starts at three years old (36 months) with baseline 3-year-old intelligence pre-trained.
The world (the hidden rules) is fixed for the life of the server: by default the 0-7 years world (config/ontology_age07.yaml), or `--small`
for the five-problem one, or an ontology file (`--world`), or a Director-made one (`--new-world`: the 0-3 years pools only), or, when `--save`
already exists, the world that child grew up in."""
import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

ap = argparse.ArgumentParser(prog="server.web")
ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--save", default="saves/web/current", help="resumes this game if it exists")
ap.add_argument("--port", type=int, default=8000)
ap.add_argument("--host", default="127.0.0.1")
ap.add_argument("--new-world", action="store_true", help="ask the Director for a new world (the existing save is kept aside)")
ap.add_argument("--world", default="", help="an ontology.yaml (the existing save is kept aside). Default: the 0-7 years world, config/ontology_age07.yaml")
ap.add_argument("--small", action="store_true", help="the small five-problem world instead of the 0-7 years one")
ap.add_argument("--mood", default="", help="flavour text for --new-world")
ap.add_argument("--fresh", action="store_true", help="start a new child in the same world (the earlier child is kept aside). Without it a restart resumes the saved child")
ap.add_argument("--child", default="", help="take over a child raised to three (e.g. saves/children/mina); the earlier child is kept aside")
ap.add_argument("--children", default="saves/children", help="where the raised children wait (the page offers them on New child)")
ap.add_argument("--lang", choices=("en", "ko"), default=os.environ.get("GROWING_LANG", "en"), help="the language of the words the player reads (the page has a button to switch)")
a = ap.parse_args()
os.environ["GROWING_LANG"] = a.lang

import socket  # noqa: E402

with socket.socket() as s:  # before anything is moved or copied: a second copy must not touch the save the first one is playing
    if s.connect_ex((a.host, a.port)) == 0:
        sys.exit(f"Port {a.port} is already in use (another copy of the game still running?). Stop it, or start this one with --port {a.port + 1}.")

save = Path(a.save)
DEFAULT_WORLD = Path(__file__).resolve().parents[2] / "config" / "ontology_age07.yaml"
if a.child:
    src = Path(a.child)
    if not (src / "raised.json").exists():
        sys.exit(f"{src} is not a raised child (raised.json missing): make one with `uv run python -m server.game.raise_child`")
    if save.exists():
        save.rename(save.with_name(f"{save.name}-old-{time.strftime('%Y%m%d-%H%M%S')}"))
    shutil.copytree(src, save)
    print(f"taking over {src.name}; the earlier child, if any, is kept aside")
elif a.fresh and (save / "meta.json").exists():
    aside = save.with_name(f"{save.name}-old-{time.strftime('%Y%m%d-%H%M%S')}")
    save.rename(aside)
    save.mkdir(parents=True)
    for name in ("ontology.yaml", "world.json"):  # the world this child grew up in is the next child's world
        if (aside / name).exists():
            shutil.copy(aside / name, save / name)
    print(f"the earlier child is kept in {aside}; a new child starts in the same world")
fresh =a.new_world or a.world
if fresh and (save / "meta.json").exists():
    aside = save.with_name(f"{save.name}-old-{time.strftime('%Y%m%d-%H%M%S')}")
    save.rename(aside)
    print(f"the earlier child is kept in {aside}")
if a.new_world:
    print("asking the Director for a new world (it can take a minute or two)...", flush=True)
    r = subprocess.run([sys.executable, "-m", "server.web.world", "--seed", str(a.seed), "--mood", a.mood], capture_output=True, text=True)
    if r.returncode:
        sys.exit("the Director could not make a world:\n" + r.stderr[-600:])
    made = json.loads(r.stdout.strip().splitlines()[-1])
    print(f"world by {made['source']} in {made['seconds']}s, {made['lines']} lines, curiosity {made['curiosity']}: {made['premise']}")
    os.environ["GROWING_ONTOLOGY"] = made["path"]
elif a.world:
    os.environ["GROWING_ONTOLOGY"] = str(Path(a.world).resolve())
elif (save / "ontology.yaml").exists():
    os.environ["GROWING_ONTOLOGY"] = str((save / "ontology.yaml").resolve())  # the world this child grew up in
elif not a.small:
    os.environ["GROWING_ONTOLOGY"] = str(DEFAULT_WORLD)

try:
    import uvicorn  # noqa: E402
    from .app import create_app  # noqa: E402  (after the world is chosen: the ontology is read at import)
except ImportError as e:
    sys.exit(f"{e}\nThe web screen needs fastapi and uvicorn: run `uv sync`, then start it with `uv run python -m server.web`.")

app = create_app(a.save, a.seed, children_dir=a.children)
mode = os.environ.get("GROWING_LLM") or "see config/llm.yaml"
print(f"Sprout is ready: open http://{a.host}:{a.port}  (stop with Ctrl+C)\n"
      f"  save {a.save}   words and scenes: {mode}   world: {os.environ.get('GROWING_ONTOLOGY', 'hand-written')}\n"
      f"  with live words the first scene takes 10-20 seconds to appear", flush=True)
uvicorn.run(app, host=a.host, port=a.port, log_level="warning")
