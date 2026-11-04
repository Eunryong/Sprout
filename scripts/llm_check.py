"""One-minute check of a live setup, before spending a real run: one Director request and one Narrator request
through the configured models. Prints what came back, how long it took and whether the engine / guards accept it.

  GROWING_LLM=live uv run python scripts/llm_check.py          # key in .env.local, base URL in llm.yaml or MUSE_BASE_URL
"""
import json
import random
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from server.agents.director import DirectorAgent  # noqa: E402
from server.agents.factory import LLMConfigError, load_llm_cfg, make_client  # noqa: E402
from server.agents.narrator import NarratorAgent  # noqa: E402
from server.agents.schemas import DirectorContext  # noqa: E402
from server.world.situation import Obj, OptionSpec, Situation  # noqa: E402

WM = {"cup": ("bobo", "NAME_00"), "cloth": ("mimi", "NAME_01"), "paper": ("tata", "NAME_02"), "ball": ("lulu", "NAME_03")}


def main() -> int:
    cfg = load_llm_cfg()
    try:
        client = make_client(WM, cfg, mode="live")
    except LLMConfigError as e:
        print(f"setup problem: {e}")
        return 2
    ctx = DirectorContext(day=1, open_issues=[], nameable=[], requests_left=0, unlocked=["house"], recent_topics=[], start_index=0)
    d = DirectorAgent(client, WM)
    t0 = time.time()
    sits = d.offer(ctx, random.Random(0), set())
    print(f"Director: source={d.last_source} usable={d.last_error is None and d.last_source != 'template'} "
          f"{time.time() - t0:.1f}s attempts={d.stats.summary()}")
    if d.last_error:
        print(f"  last rejection: {d.last_error}")
    for s, teaser in zip(sits, d.last_teasers + [""] * 2):
        print(f"  card {s.tags} at {s.location}: {len(s.objects)} objects, {len(s.options)} options; teaser={teaser!r}")
    print(f"  theme={d.last_theme!r}  teasers dropped={d.teasers_dropped}")
    sit = Situation(id="chk", tags=["FLOOR_WET"], location="house",
                    objects=[Obj(kind="cloth", color="blue", shape="flat", size="big", material="fabric"),
                             Obj(kind="cup", color="red", shape="cylinder", size="small", material="ceramic"),
                             Obj(kind="ball", color="green", shape="sphere", size="small", material="rubber")],
                    options=[OptionSpec(verb="WIPE", target=0), OptionSpec(verb="DRINK", target=1), OptionSpec(verb="WAIT")])
    n = NarratorAgent(client)
    t0 = time.time()
    print(f"Narrator intro  ({time.time() - t0:.1f}s): {n.intro(sit, False)[0]}")
    print(f"Narrator result ({time.time() - t0:.1f}s): {n.outcome(sit, 0, 'solved', 'player', {'FLOOR_WET': 2})}")
    print("narrator:", n.stats.summary(), "| last rejection:", n.last_error)
    print("done; answers were recorded in", cfg.cache if cfg.record else "(recording off)")
    return 0 if d.last_source != "template" else 1


if __name__ == "__main__":
    sys.exit(main())
