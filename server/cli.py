import argparse
import random
import sys
import time
from pathlib import Path

from server.game.autoplay import play_day
from server.game.session import Session
from server.game.simple import SimpleGame
from server.world.events import RuleError
from server.world.state import Phase

HELP = """commands:  pick N | show N | tell N | talk [kind] | try | skip | end | mind [DAY] | status | help | quit
  pick N     choose situation card N      show N     you carry out option N (the child learns from your choice)
  tell N     tell the child to take option N (before it chooses; you may be wrong)   talk [kind]  name a thing (between situations, costs attention)
  try        let the child choose and do it (it learns from what comes of it)
  skip       let it pass (no learning)    end        evening quiz, next day (after all situations)
  mind [DAY] what the child can do now, measured on the real model (DAY = compare with that day's snapshot)"""


def render(s: Session) -> None:
    v = s.view()
    print(f"\n=== Day {v.day} | attention {v.spent}/{v.per_day} | {v.phase} ===")
    if v.phase == "cards":
        for p in v.previews:
            tag = "  (continues an earlier problem)" if p["followup"] else ""
            print(f"  [{p['index']}] {p['location']:5s} {' + '.join(p['tags'])}{tag}\n       {p['teaser']}")
    elif v.phase == "situation":
        print(f"[{v.location}] {v.narration}")
        if v.hint:
            print(f"  ({v.hint})")
        for o in v.options:
            print(f"   [{o.index}] {o.text}")
        if v.command is not None:
            print(f"  you told the child to take [{v.command}]")
    else:
        print("The day is over. Type `end` for the evening quiz.")


def show_result(r) -> None:
    if r.duplicate:
        print("(already applied)")
        return
    if r.narration:
        print(r.narration)
    if r.notes:
        print(f"  * {r.notes}")
    if r.learn:
        l = r.learn
        print(f"  learned: {l.steps} steps | loss {l.loss_before:.2f} -> {l.loss_after:.2f} | {l.seconds:.2f}s")


def bar(x: float, w: int = 12) -> str:
    k = round(x * w)
    return "█" * k + "░" * (w - k)


def show_mind(s: Session, compare_day: int | None) -> None:
    m = s.mind(compare_day)
    if not m["topics"]:
        print("The child has not met any situation yet.")
        return
    print("topic                 practised  best option chosen (held-out situations)")
    for t, e in m["topics"].items():
        was = f"   was {e['before']:.0%}" if "before" in e else ""
        print(f"  {t:20s} {e['practised']:4d}      {bar(e['accuracy'])} {e['accuracy']:4.0%}  (chance {e['random']:.0%}){was}")
    r = m["tag_reliance"]
    print(f"reads the situation tag: {r['intact']:.0%} with it, {r['swapped']:.0%} when it is swapped (chance ~{m['topics'][next(iter(m['topics']))]['random']:.0%})")


def run_command(s: Session, line: str) -> bool:
    parts = line.split()
    if not parts:
        return True
    cmd, arg = parts[0], (int(parts[1]) if len(parts) > 1 and parts[1].lstrip("-").isdigit() else None)
    need = {"pick", "show", "tell"}
    if cmd in need and arg is None:
        print(f"usage: {cmd} N")
        return True
    if cmd in ("quit", "q", "exit"):
        return False
    if cmd == "help":
        print(HELP)
    elif cmd == "status":
        print(f"events={len(s.engine.log)} examples={len(s.buffer)} save={s.save.root}")
        for w in s.warnings:
            print("  warning:", w)
    elif cmd == "mind":
        show_mind(s, arg)
    elif cmd == "pick":
        s.pick_card(arg)
    elif cmd == "show":
        show_result(s.show(arg))
    elif cmd == "tell":
        show_result(s.command(arg))
    elif cmd == "talk":
        show_result(s.talk(parts[1] if len(parts) > 1 else None))
    elif cmd == "try":
        r = s.try_child()
        show_result(r)
        if r.followed is not None:
            print("  (the child did as you said)" if r.followed else "  (the child did something else)")
    elif cmd == "skip":
        show_result(s.skip())
    elif cmd == "end":
        r = s.end_day()
        print(f"Evening quiz ({r['n']} questions): child {r['optimal']:.2f}  | random {r['random']:.2f} | "
              f"untrained {r['untrained_optimal']:.2f}   (all met topics: {r['pool_optimal']:.2f}, n={r['pool_n']})")
    else:
        print("unknown command; type help")
    return True


def repl(s: Session) -> None:
    print(HELP)
    for w in s.warnings:
        print("warning:", w)
    while True:
        if s.state.phase == Phase.CARDS and s.state.cards is None:
            s.draw_cards()
        render(s)
        try:
            line = input("> ")
        except EOFError:
            print("\n(saved)")
            return
        try:
            if not run_command(s, line):
                print("(saved)")
                return
        except RuleError as e:
            print(f"! {e}")


PLAY_HELP = "say N = tell the child to do option N | show N = you do option N, the child watches | leave = say nothing | quit"


def play(g: SimpleGame, research: bool = False) -> None:
    """The simple-choice game: scenes only (docs/DIRECTION.md)."""
    print(PLAY_HELP)
    while True:
        sc = g.scene()
        print(f"\n--- scene {sc.number} ---\n{sc.narration}")
        for o in sc.options:
            if research:  # the child's raw numbers, for the researcher only
                best = max(o.expects, key=o.expects.get)
                print(f"  [{o.index}] {o.text:40s} research: guess {best} {o.expects[best]:.0%}, lived {o.seen}x, curiosity +{o.pull:.2f}")
            else:
                lean = {"remembers": "  <- the child seems drawn to this", "curious": "  <- the child seems curious about this"}.get(o.lean, "")
                print(f"  [{o.index}] {o.text:40s}{lean}" + (f"\n        {o.memory}" if o.memory else "  (new to the child)" if not lean else ""))
        try:
            parts = input("> ").split()
        except EOFError:
            print("\n(saved)")
            return
        if not parts:
            continue
        if parts[0] in ("quit", "q", "exit"):
            print("(saved)")
            return
        if parts[0] == "help":
            print(PLAY_HELP)
            continue
        if parts[0] not in ("say", "show", "leave"):
            print("unknown command; type help")
            continue
        try:
            n = int(parts[1]) if len(parts) > 1 else None
            t = g.play(parts[0], n)
        except (ValueError, RuleError) as e:
            print(f"! {e}")
            continue
        print(t.narration)
        if t.followed is not None:
            print("  (the child did as you said)" if t.followed else "  (the child did something else)")
        if research:
            print(f"  research: learned {t.learned}   loss {t.loss_before:.2f} -> {t.loss_after:.2f}")
        if t.world_says:
            print(f"  the world: {t.world_says}")
        if t.explanation:
            print(f"  {t.explanation}")
        print(f"  {t.remembers}")


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(prog="server.cli")
    sub = ap.add_subparsers(dest="mode", required=True)
    pl = sub.add_parser("play", help="simple choice: scenes only"); pl.add_argument("--seed", type=int, default=0); pl.add_argument("--save", default="")
    pl.add_argument("--load", default=""); pl.add_argument("--research", action="store_true", help="also show the child's raw numbers")
    n = sub.add_parser("new"); n.add_argument("--seed", type=int, default=0); n.add_argument("--save", default="")
    l = sub.add_parser("load"); l.add_argument("save")
    a = sub.add_parser("autoplay"); a.add_argument("--seed", type=int, default=0); a.add_argument("--days", type=int, default=3)
    a.add_argument("--save", default=""); a.add_argument("--p-show", type=float, default=0.5)
    args = ap.parse_args(argv)
    if args.mode == "play":
        if args.load:
            play(SimpleGame.load(args.load), args.research)
        else:
            path = args.save or f"saves/{time.strftime('%Y%m%d-%H%M%S')}-seed{args.seed}"
            print(f"new save: {path}")
            play(SimpleGame.new(path, seed=args.seed), args.research)
        return
    if args.mode == "load":
        repl(Session.load(args.save))
        return
    path = args.save or f"saves/{time.strftime('%Y%m%d-%H%M%S')}-seed{args.seed}"
    s = Session.new(path, seed=args.seed)
    print(f"new save: {path}")
    if args.mode == "new":
        repl(s)
        return
    rng = random.Random(args.seed + 1)
    for _ in range(args.days):
        r = play_day(s, rng, args.p_show)
        print(f"day {r['day']}: quiz {r['optimal']:.2f} (random {r['random']:.2f}, untrained {r['untrained_optimal']:.2f}) "
              f"pool {r['pool_optimal']:.2f}  examples={len(s.buffer)}")
    print("llm:", s.llm_stats())


if __name__ == "__main__":
    main()
