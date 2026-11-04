"""How often does the real Director make a valid world? (docs/WORLD.md 22.13)

Calls Director.world() for several seeds with the LLM client of config/llm.yaml (GROWING_LLM, the key from .env / .env.local;
nothing about a key is printed or stored). Per seed: valid on the first try / after one repair / fell back to the template world,
the reasons of the rejected attempts, seconds. Also how different the worlds are from each other and from the template world.

  uv run python scripts/exp_world_live.py --seeds 0-7 --out scripts/out/world_live.json
"""
import argparse
import json
import statistics as st
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def one(seed: int, mood: str) -> dict:
    from server.agents.director import DirectorAgent
    from server.agents.factory import make_client
    from server.world.worldgen import check_sheet, load_pools, sheet_hash, template_world
    rows = []
    d = DirectorAgent(make_client({}), {}, on_call=rows.append)
    t0 = time.perf_counter()
    sheet = d.world(seed, mood)
    pools = load_pools()
    tpl = template_world(seed, pools)
    rel = lambda s: {(t, v, k, x) for t, rs in s.rules.items() for v, k, x in rs}  # noqa: E731
    return {"seed": seed, "source": sheet.source, "attempts": rows[0]["attempts"], "ok": rows[0]["ok"], "seconds": round(time.perf_counter() - t0, 1),
            "attempt_errors": rows[0]["attempt_errors"], "error": rows[0]["error"], "valid": check_sheet(sheet, pools) == [],
            "hash": sheet_hash(sheet), "traps": sum(x < 0 for _, _, _, x in rel(sheet)),
            "same_relations_as_template": round(len(rel(sheet) & rel(tpl)) / max(1, len(rel(sheet) | rel(tpl))), 2),
            "curiosity": sheet.curiosity, "n_rules": sum(len(v) for v in sheet.rules.values()),
            "n_lines": sum(len(v) for v in sheet.lines.values()), "learning": sheet.learning, "premise": sheet.premise, "sheet": sheet.model_dump()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", default="0-7")
    ap.add_argument("--mood", default="a rainy week at home")
    ap.add_argument("--jobs", type=int, default=3)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    lo, hi = (int(x) for x in (a.seeds.split("-") * 2)[:2])
    seeds = list(range(lo, hi + 1))
    t0 = time.time()
    with ThreadPoolExecutor(a.jobs) as ex:
        res = list(ex.map(lambda s: one(s, a.mood), seeds))
    first = [r for r in res if r["ok"] and r["attempts"] == 1]
    repaired = [r for r in res if r["ok"] and r["attempts"] > 1]
    fell = [r for r in res if not r["ok"]]
    lines = [f"seeds {a.seeds}, mood {a.mood!r}: {len(res)} calls",
             f"valid on the first try {len(first)}, after one repair {len(repaired)}, fell back to the template world {len(fell)}",
             f"seconds per world: median {st.median(r['seconds'] for r in res):.1f}, max {max(r['seconds'] for r in res):.1f}",
             f"distinct worlds {len({r['hash'] for r in res})} of {len(res)}; relations shared with the template world of the same seed (Jaccard) mean "
             f"{st.mean(r['same_relations_as_template'] for r in res):.2f}; harmful relations mean {st.mean(r['traps'] for r in res):.1f}",
             "rejected attempts (reasons):"]
    for r in res:
        for e in r["attempt_errors"]:
            lines.append(f"  seed {r['seed']}: {e[:230]}")
    lines.append(f"curiosity per world {[r['curiosity'] for r in res]}; lines kept per world {[r['n_lines'] for r in res]} of rules {[r['n_rules'] for r in res]}")
    lines += [f"  seed {r['seed']:2d} {r['source']:40s} attempts {r['attempts']} {r['seconds']:5.1f}s learning {r['learning']['burn_worse_rounds']}/{r['learning']['burn_nothing_rounds']}/{r['learning']['regret_rounds']}  {r['premise'][:70]}" for r in res]
    text = "\n".join(lines)
    print(text)
    print(f"\nwall={time.time() - t0:.0f}s")
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(res, indent=1))
    Path(a.out).with_suffix(".txt").write_text(text + "\n")


if __name__ == "__main__":
    main()
