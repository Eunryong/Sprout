"""P4 measurement: how usable are the Director's and Narrator's answers, played through the real Session stack.

  uv run python scripts/exp_llm.py --client mock --seeds 0-9 --days 3
  uv run python scripts/exp_llm.py --client sloppy --rates "no_solver=.2,unknown_tag=.1,leaky_teaser=.1,names_kind=.1"
  GROWING_LLM=live uv run python scripts/exp_llm.py --client live --seeds 0-2 --days 2 --max-calls 200   # needs the key
  uv run python scripts/exp_llm.py --client replay --seeds 0-2 --days 2        # offline, from the recorded cache

`live` is serial and capped by --max-calls (director + narrator calls). Everything else runs in parallel.
Pass/fail line = proposed P4 acceptance (edit the thresholds below once real numbers exist)."""
import argparse
import collections
import json
import random
import re
import shutil
import statistics as st
import sys
import tempfile
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

ACCEPT = {"director_ok": 0.90, "director_first_try": 0.60, "director_fallback_max": 0.10, "narrator_ok": 0.85,
          "director_p95_s": 8.0, "narrator_p95_s": 4.0, "teaser_drop_max": 0.15}


def parse_rates(s: str) -> dict[str, float]:
    return {k: float(v) for k, v in (kv.split("=") for kv in s.split(",") if kv)} if s else {}


def error_class(e: str | None) -> str:
    if not e:
        return "?"
    e = re.sub(r"\d+", "#", e.replace("\n", " "))
    return e[:70]


def tokens_of(client) -> tuple[int, int, int]:
    seen, stack, tin, tout, req = set(), [client], 0, 0, 0
    while stack:
        c = stack.pop()
        if c is None or id(c) in seen:
            continue
        seen.add(id(c))
        tin += getattr(c, "tokens_in", 0) or 0
        tout += getattr(c, "tokens_out", 0) or 0
        req += getattr(c, "requests", 0) or 0
        for attr in ("inner", "default"):
            stack.append(getattr(c, attr, None))
        stack += list(getattr(c, "chain", []) or []) + list((getattr(c, "by_agent", None) or {}).values())
    return tin, tout, req


def run(seed: int, kind: str, days: int, rates: dict, max_calls: int, cache: str | None):
    import torch
    torch.set_num_threads(1)
    from server.agents.factory import load_llm_cfg, make_client, mock_client
    from server.agents.testing import SloppyClient
    from server.game.autoplay import play_day
    from server.game.session import Session
    from server.world.events import EventType

    tmp = Path(tempfile.mkdtemp(prefix=f"llm{seed}-"))
    try:
        # the word map depends on the seed, so build the client from a throw-away session's map
        probe = Session.new(tmp / "probe", seed=seed, client=None)
        wm = probe.word_map
        if kind == "mock":
            client = mock_client(wm)
        elif kind == "sloppy":
            client = SloppyClient(wm, rates, seed=seed)
        else:
            cfg = load_llm_cfg()
            if cache:
                cfg.cache = cache
            client = make_client(wm, cfg, mode=kind)
        s = Session.new(tmp / "s", seed=seed, client=client)
        rng = random.Random(seed + 1)
        quiz = []
        for _ in range(days):
            quiz.append(play_day(s, rng)["optimal"])
            n = s.director.stats.calls + s.narrator.stats.calls
            if n >= max_calls:
                break
        rows = [json.loads(x) for x in s.save.llm_log.read_text().splitlines()] if s.save.llm_log.exists() else []
        sits = [c for e in s.engine.log if e.type == EventType.CARDS_OFFERED for c in e.payload["situations"]]
        tin, tout, req = tokens_of(client)
        return {"seed": seed, "rows": rows, "stats": s.llm_stats(), "quiz": quiz, "situations": sits,
                "tokens": [tin, tout, req], "injected": getattr(client, "injected", {}),
                "themes": [e.payload.get("theme", "") for e in s.engine.log if e.type == EventType.CARDS_OFFERED]}
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def pct(xs, q):
    xs = sorted(xs)
    return xs[min(len(xs) - 1, int(q * len(xs)))] if xs else float("nan")


def report(results: list[dict], args) -> tuple[str, dict, bool]:
    rows = [r for res in results for r in res["rows"]]
    D = [r for r in rows if r["agent"] == "director" and r["source"] != "none"]
    N = [r for r in rows if r["agent"] == "narrator" and r["source"] != "none"]
    out = {}
    lines = [f"client={args.client} seeds={len(results)} days={args.days}"]
    d_ok = sum(r["ok"] for r in D) / max(1, len(D))
    d_first = sum(r["ok"] and r["attempts"] == 1 for r in D) / max(1, len(D))
    d_fb = sum(not r["ok"] for r in D) / max(1, len(D))
    n_ok = sum(r["ok"] for r in N) / max(1, len(N))
    d_lat, n_lat = [r["seconds"] for r in D], [r["seconds"] for r in N]
    drop = sum(res["stats"]["teasers_dropped"] for res in results) / max(1, 2 * len(D))
    out.update(director_calls=len(D), director_ok=d_ok, director_first_try=d_first, director_fallback=d_fb,
               narrator_calls=len(N), narrator_ok=n_ok, director_p50_s=pct(d_lat, .5), director_p95_s=pct(d_lat, .95),
               narrator_p50_s=pct(n_lat, .5), narrator_p95_s=pct(n_lat, .95), teaser_drop=drop)
    lines.append(f"Director  calls={len(D):4d}  usable={d_ok:.1%}  first-try={d_first:.1%}  "
                 f"repaired={d_ok - d_first:.1%}  fallback={d_fb:.1%}  latency p50/p95={out['director_p50_s']:.2f}/{out['director_p95_s']:.2f}s")
    lines.append(f"Narrator  calls={len(N):4d}  usable={n_ok:.1%}  fallback={1 - n_ok:.1%}  "
                 f"latency p50/p95={out['narrator_p50_s']:.2f}/{out['narrator_p95_s']:.2f}s")
    lines.append(f"teasers dropped by the guard: {drop:.1%} of cards")
    for agent, rs in (("director", D), ("narrator", N)):
        errs = collections.Counter(error_class(e) for r in rs for e in (r.get("attempt_errors") or ([r["error"]] if r["error"] else [])))
        for e, n in errs.most_common(6):
            lines.append(f"   {agent} rejected x{n}: {e}")
    inj = collections.Counter()
    for res in results:
        inj.update(res["injected"])
    if inj:
        lines.append(f"injected faults: {dict(inj)}")
    sits = [s for res in results for s in res["situations"]]
    if sits:
        topics = collections.Counter("+".join(sorted(s["tags"])) if s["tags"] != ["REQUEST"] else "REQUEST" for s in sits)
        dup = sum(len({o["kind"] for o in s["objects"]}) < len(s["objects"]) for s in sits) / len(sits)
        colours = len({o["color"] for s in sits for o in s["objects"]})
        kinds = collections.Counter(o["kind"] for s in sits for o in s["objects"])
        wait = sum(any(o["verb"] == "WAIT" for o in s["options"]) for s in sits) / len(sits)
        out.update(cards=len(sits), same_kind_twice=dup, mean_objects=st.mean(len(s["objects"]) for s in sits),
                   mean_options=st.mean(len(s["options"]) for s in sits), with_wait=wait)
        lines.append(f"cards={len(sits)}  topics={dict(topics.most_common())}")
        lines.append(f"   objects/card={out['mean_objects']:.1f}  options/card={out['mean_options']:.1f}  same kind twice={dup:.0%}  "
                     f"WAIT offered={wait:.0%}  colours used={colours}/4  kinds={dict(kinds.most_common())}")
    themes = {t for res in results for t in res["themes"] if t}
    lines.append(f"distinct day themes: {len(themes)}")
    last = [res["quiz"][-1] for res in results if res["quiz"]]
    if last:
        out["last_quiz"] = st.mean(last)
        lines.append(f"child's evening quiz on the last day played: mean {st.mean(last):.2f} (min {min(last):.2f})")
    tin = sum(res["tokens"][0] for res in results)
    tout = sum(res["tokens"][1] for res in results)
    req = sum(res["tokens"][2] for res in results)
    if req:
        lines.append(f"HTTP requests={req}  tokens in/out={tin}/{tout}  (per day: {req / max(1, len(results) * args.days):.1f} requests)")
        out.update(requests=req, tokens_in=tin, tokens_out=tout)
    checks = [("Director usable >= %.0f%%" % (100 * ACCEPT["director_ok"]), d_ok >= ACCEPT["director_ok"]),
              ("Director first-try >= %.0f%%" % (100 * ACCEPT["director_first_try"]), d_first >= ACCEPT["director_first_try"]),
              ("Director fallback <= %.0f%%" % (100 * ACCEPT["director_fallback_max"]), d_fb <= ACCEPT["director_fallback_max"]),
              ("Narrator usable >= %.0f%%" % (100 * ACCEPT["narrator_ok"]), n_ok >= ACCEPT["narrator_ok"]),
              ("Director p95 <= %.0fs" % ACCEPT["director_p95_s"], out["director_p95_s"] <= ACCEPT["director_p95_s"]),
              ("Narrator p95 <= %.0fs" % ACCEPT["narrator_p95_s"], out["narrator_p95_s"] <= ACCEPT["narrator_p95_s"]),
              ("teaser drop <= %.0f%%" % (100 * ACCEPT["teaser_drop_max"]), drop <= ACCEPT["teaser_drop_max"])]
    lines.append("acceptance (proposed): " + "; ".join(f"{'PASS' if ok else 'FAIL'} {name}" for name, ok in checks))
    return "\n".join(lines), out, all(ok for _, ok in checks)


def parse_seeds(s: str) -> list[int]:
    a, _, b = s.partition("-")
    return list(range(int(a), int(b or a) + 1))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--client", choices=["mock", "sloppy", "live", "replay"], default="mock")
    ap.add_argument("--rates", default="", help='sloppy only: "no_solver=.2,unknown_tag=.1,..."')
    ap.add_argument("--seeds", default="0-4")
    ap.add_argument("--days", type=int, default=3)
    ap.add_argument("--jobs", type=int, default=8)
    ap.add_argument("--max-calls", type=int, default=10**9, help="stop a run after this many LLM calls (spend cap)")
    ap.add_argument("--cache", default=None, help="cache file (default: llm.yaml)")
    ap.add_argument("--out", default="scripts/out/p4_llm.json")
    a = ap.parse_args()
    seeds, rates, t0 = parse_seeds(a.seeds), parse_rates(a.rates), time.time()
    args = [(s, a.client, a.days, rates, a.max_calls, a.cache) for s in seeds]
    if a.client == "live":
        results = [run(*x) for x in args]
    else:
        with ProcessPoolExecutor(a.jobs) as ex:
            results = list(ex.map(run, *zip(*args)))
    text, data, ok = report(results, a)
    print(text + f"\nwall={time.time() - t0:.0f}s")
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps({"args": vars(a), "summary": data, "accepted": ok}, indent=1))
    Path(a.out).with_suffix(".txt").write_text(text + "\n")


if __name__ == "__main__":
    main()
