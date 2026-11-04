"""Ask the Director for a new world and write it as an ontology file (a separate process: the ontology is a module global that is
read once at import, so the game that lives in the world has to start after this one has finished).

  uv run python -m server.web.world --seed 5 --out saves/worlds       prints one JSON line {"path", "source", "attempts", "premise"}
"""
import argparse
import json
import sys
import time

from server.agents.director import DirectorAgent
from server.agents.factory import make_client
from server.world.worldgen import check_sheet, load_pools, sheet_hash, write_world

ap = argparse.ArgumentParser(prog="server.web.world")
ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--mood", default="")
ap.add_argument("--out", default="saves/worlds")
a = ap.parse_args()
rows: list[dict] = []
t0 = time.perf_counter()
sheet = DirectorAgent(make_client({}), {}, on_call=rows.append).world(a.seed, a.mood)
pools = load_pools()
assert check_sheet(sheet, pools) == []
onto, _ = write_world(sheet, f"{a.out}/{sheet_hash(sheet)}", pools)
print(json.dumps({"path": str(onto), "source": sheet.source, "attempts": rows[0]["attempts"] if rows else 0, "premise": sheet.premise,
                  "seconds": round(time.perf_counter() - t0, 1), "lines": sum(len(v) for v in sheet.lines.values()), "curiosity": sheet.curiosity}))
sys.exit(0)
