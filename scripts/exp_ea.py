"""E-A feasibility (docs/WORLD.md section 18): does the way a parent raises the child in the first days change the child
later, through nothing but the training data that way of parenting produces?

Four synthetic parents raise the SAME randomly initialised child (same seed = same weights, same held-out quiz):
  talker     talks a lot: half of each day's attention goes to naming things; the rest are plain demonstrations
  commander  "just do as I say": every demonstration is an instruction the child must follow (a quarter of them are
             wrong - parents err); the child is never asked what it would do
  chooser    lets the child choose first, then praises or puts it right
  balanced   a mix of the three
  demonstrator  (reference, not one of the four) plain demonstrations, one naming a day: the game as it is today
Days 1-3 are STAGE 1 (the style above). Days 4-6 are STAGE 2: every parent becomes the same discerning parent, who
gives instructions (half good, a quarter useless, a quarter harmful) and praises following the good ones and refusing
the bad ones, putting the child right otherwise. We then ask how much of the stage-1 style is still there.

Nothing here changes a hyper-parameter per style: the learner, the model and the quiz are identical. A style only
decides which examples exist. Measured on held-out scenes after day 3 and day 6 (greedy picks):
  word_acc      naming quiz accuracy                          problem_acc  accuracy without any instruction
  confidence    mean probability of the picked option         hesitate     share of picks under 0.6 ("looks at the parent")
  follow_good / follow_useless / follow_harm   how often the instructed option is taken, by instruction quality
  obey = overall follow rate   blind = follow rate on useless+harmful   discern = follow_good - blind
  refusals      what a non-following child does instead (best option / wait / something else)
Instruction quality is the engine's rules table (value 2 good, 0 useless, <0 harmful); the child never sees it.

  uv run python scripts/exp_ea.py --seeds 200-239 --jobs 12
"""
import argparse
import json
import random
import statistics as st
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from copy import deepcopy
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from server.game.parents import P_COMMAND_WRONG, STAGE2_MIX, STAGE2_NAMING, STYLES, pick_instruction  # noqa: E402  (one definition, also used on the engine path)


def kind_of_quality(values: list[int], options) -> dict[str, list[int]]:
    from server.world.parenting import quality_buckets
    return quality_buckets(values, [o.verb for o in options])


def panel(model, quiz) -> dict:
    from server.learning.evaluator import command_panel
    return command_panel(model, quiz)


def run(seed: int, style: str, days: int = 6, stage1: int = 3, free: float = 0.0, salt: str = "",
        p_wrong: float = P_COMMAND_WRONG, wrong_kind: str = "any", panel_days: tuple = ()) -> dict:
    import torch
    torch.set_num_threads(1)
    from server.config import load_configs
    from server.learning.codec import Example
    from server.learning.evaluator import make_quiz, quiz_topics
    from server.learning.learner import Learner
    from server.learning.model import build_model
    from server.learning.policy import act
    from server.world.generator import make_situation
    from server.world.ontology import PROBLEMS, REQUEST_KINDS
    from server.world.situation import observation, optimal, values

    game, mcfg, ocfg = load_configs()  # identical for every style
    rng = random.Random(seed)
    words = list(game.words)
    rng.shuffle(words)
    word_map = {k: (words[i], f"NAME_{i:02d}") for i, k in enumerate(REQUEST_KINDS)}
    mcfg.seed = seed
    model = build_model(mcfg)
    model0 = deepcopy(model)
    quiz = make_quiz(random.Random(seed + 10_000), quiz_topics(word_map, []), game.quiz_per_topic)
    forbidden = {q.hash for q in quiz}
    learner = Learner(model, ocfg)
    buffer: list[Example] = []
    prng = random.Random(f"{seed}:{style}{salt}")  # the parent's own randomness (`salt`: another parent stream, same child and quiz)
    counts = {t: 0 for t in PROBLEMS + list(REQUEST_KINDS)}
    sid = [0]

    def teach(obs, chosen, source, weight, rejected=None):
        ex = Example(id=f"ex{len(buffer):04d}", day=day, source=source, obs=obs, chosen=chosen, weight=weight, rejected=rejected)
        buffer.append(ex)
        learner.learn(ex, buffer, random.Random(f"{seed}:{style}:learn:{len(buffer)}"))

    def least(keys):
        m = min(counts[k] for k in keys)
        return prng.choice([k for k in keys if counts[k] == m])

    def new_sit(tags, word=None, token=None, kind=None):
        sid[0] += 1
        return make_situation(prng, tags, f"e{sid[0]}", "house", word, token, kind, forbidden=forbidden)

    def name_unit():
        kind = least(list(REQUEST_KINDS))
        counts[kind] += 1
        word, token = word_map[kind]
        s = new_sit(["REQUEST"], word, token, kind)
        teach(observation(s), prng.choice(optimal(s)), "show", 1.0)  # "this is the mimi": the parent shows the right thing

    def plain_show(s, obs):
        teach(obs, prng.choice(optimal(s)), "show", 1.0)

    def command_unit(s, obs):  # stage-1 commander: the child must do what it is told, right or wrong
        idx = pick_instruction(values(s), [o.verb for o in s.options], prng, p_wrong, wrong_kind)
        teach(obs.model_copy(update={"cmd": obs.options[idx]}), idx, "show", 1.0)

    def offer_unit(s, obs):  # the child chooses first
        a = act(model, obs, prng)
        best = optimal(s)
        if a.option in best:
            teach(obs, a.option, "praise", 0.5)
        else:
            teach(obs, prng.choice(best), "correct", 1.0, rejected=a.option)

    def discerning_unit(s, obs):  # stage 2 for everybody
        by_q = kind_of_quality(values(s), s.options)
        order = [q for q, _ in STAGE2_MIX]
        quality = prng.choices(order, weights=[w for _, w in STAGE2_MIX])[0]
        if not by_q[quality]:
            quality = "good" if quality == "harmful" or not by_q["useless"] else "useless"
        idx = prng.choice(by_q[quality])
        cobs = obs.model_copy(update={"cmd": obs.options[idx]})
        a = act(model, cobs, prng)
        best = optimal(s)
        if quality == "good":
            if a.option == idx or a.option in best:
                teach(cobs, a.option, "praise", 0.5)
            else:
                teach(cobs, idx, "correct", 1.0, rejected=a.option)
        elif a.option in best:  # refused a bad instruction and did the right thing
            teach(cobs, a.option, "praise", 0.5)
        else:
            teach(cobs, prng.choice(best), "correct", 1.0, rejected=a.option)

    rows = {0: panel(model0, quiz)}  # the untrained child on the same quiz
    for day in range(1, days + 1):
        stage = 1 if day <= stage1 else 2
        n_name = STYLES[style] if stage == 1 else STAGE2_NAMING
        units = ["name"] * n_name + ["problem"] * (game.per_day - n_name)
        prng.shuffle(units)
        for u in units:
            if u == "name":
                name_unit()
                continue
            t = least(PROBLEMS)
            counts[t] += 1
            s = new_sit([t])
            obs = observation(s)
            if stage == 2:
                (offer_unit if prng.random() < free else discerning_unit)(s, obs)  # `free`: the parent sometimes lets the child choose
            elif style in ("talker", "demonstrator"):
                plain_show(s, obs)
            elif style == "commander":
                command_unit(s, obs)
            elif style == "chooser":
                offer_unit(s, obs)
            else:
                prng.choice([plain_show, command_unit, offer_unit])(s, obs)
        if day in (stage1, days) or day in panel_days:
            rows[day] = panel(model, quiz)
    return {"seed": seed, "style": style, "rows": rows, "n_params": sum(p.numel() for p in model.parameters()),
            "examples": len(buffer)}


def job(a):
    return run(*a)


METRICS = [("word_acc", "words"), ("problem_acc", "problems (no instruction)"), ("confidence", "confidence"),
           ("hesitate", "hesitate <0.6"), ("follow_good", "follows good"), ("follow_useless", "follows useless"),
           ("follow_harm", "follows harmful"), ("obey", "obey (all)"), ("blind", "blind (useless+harmful)"),
           ("discern", "discern"), ("acc_with_cmd", "best option with instruction"),
           ("best_when_bad", "best option when told wrong"),
           ("refuse_best", "refusal: best"), ("refuse_wait", "refusal: wait"), ("refuse_other", "refusal: other")]


def ms(xs):
    xs = [x for x in xs if x == x]
    return (st.mean(xs), st.pstdev(xs)) if xs else (float("nan"), float("nan"))


def report(results: list[dict], days: int, stage1: int) -> tuple[str, dict]:
    by = {s: [r for r in results if r["style"] == s] for s in STYLES}
    lines = []
    summary = {}
    for day, label in ((stage1, f"after day {stage1} (end of stage 1)"), (days, f"after day {days} (stage 2 with the same discerning parent)")):
        lines.append(f"\n== {label} ==  mean (sd) over {len(by['talker'])} seeds")
        header = f"{'':28s}" + "".join(f"{s:>16s}" for s in STYLES) + f"{'untrained':>16s}"
        lines.append(header)
        for key, name in METRICS:
            cells = []
            for s in STYLES:
                m, sd = ms([r["rows"][day][key] for r in by[s]])
                cells.append(f"{m:8.2f} ({sd:.2f})")
                summary.setdefault(day, {}).setdefault(s, {})[key] = m
            u, _ = ms([r["rows"][0][key] for r in by["talker"]])
            lines.append(f"{name:28s}" + "".join(f"{c:>16s}" for c in cells) + f"{u:16.2f}")
    d = summary[stage1]
    paired = lambda a, b, k, day: ms([ra["rows"][day][k] - rb["rows"][day][k]  # noqa: E731
                                      for ra, rb in zip(by[a], by[b])])
    lines.append("\n== paired differences, same seed (mean, sd) ==")
    for a, b, k, day in (("talker", "commander", "word_acc", stage1), ("talker", "chooser", "word_acc", stage1),
                         ("chooser", "commander", "problem_acc", stage1), ("chooser", "commander", "confidence", stage1),
                         ("commander", "chooser", "blind", stage1), ("chooser", "commander", "discern", stage1),
                         ("chooser", "commander", "problem_acc", days), ("commander", "chooser", "blind", days),
                         ("talker", "commander", "word_acc", days)):
        m, sd = paired(a, b, k, day)
        lines.append(f"  day {day}: {a} - {b} on {k}: {m:+.2f} ({sd:.2f})")
    checks = [
        ("talker words beat the others' by >= 0.20 (day %d)" % stage1,
         d["talker"]["word_acc"] - max(d[s]["word_acc"] for s in STYLES if s != "talker") >= 0.20),
        ("chooser problem accuracy without instruction beats commander's by >= 0.15, or confidence by >= 0.10",
         d["chooser"]["problem_acc"] - d["commander"]["problem_acc"] >= 0.15
         or d["chooser"]["confidence"] - d["commander"]["confidence"] >= 0.10),
        ("commander blind obedience >= 0.60", d["commander"]["blind"] >= 0.60),
        ("learning matters: untrained discern is near 0 and obey is about chance",
         abs(ms([r["rows"][0]["discern"] for r in by["talker"]])[0]) < 0.15),
    ]
    others = max(d[s]["word_acc"] for s in STYLES if s != "talker")
    lines.append(f"\n(talker words {d['talker']['word_acc']:.3f} vs best other {others:.3f}: gap {d['talker']['word_acc'] - others:.3f}; "
                 f"commander blind {d['commander']['blind']:.3f})")
    lines.append("\nproposed criteria (docs/WORLD.md 18.10):")
    for text, ok in checks:
        lines.append(f"  {'PASS' if ok else 'FAIL'}  {text}")
    return "\n".join(lines), {"checks": {t: bool(ok) for t, ok in checks}, "means": summary}


def seeds_of(s):
    out = []
    for part in s.split(","):
        if "-" in part:
            a, b = part.split("-")
            out += range(int(a), int(b) + 1)
        else:
            out.append(int(part))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", default="200-239")
    ap.add_argument("--days", type=int, default=6)
    ap.add_argument("--stage1", type=int, default=3)
    ap.add_argument("--jobs", type=int, default=12)
    ap.add_argument("--free", type=float, default=0.0, help="stage 2: share of units where the parent gives no instruction")
    ap.add_argument("--styles", default=",".join(STYLES))
    ap.add_argument("--salt", default="", help="another random stream for the parent (same child weights and quiz)")
    ap.add_argument("--out", default="scripts/out/ea.json")
    a = ap.parse_args()
    styles = a.styles.split(",")
    tasks = [(s, st_, a.days, a.stage1, a.free, a.salt) for s in seeds_of(a.seeds) for st_ in styles]
    t0 = time.time()
    with ProcessPoolExecutor(a.jobs) as ex:
        results = list(ex.map(job, tasks))
    text, summary = report(results, a.days, a.stage1)
    print(text)
    print(f"\nparams={results[0]['n_params']} examples/run={results[0]['examples']} seeds={a.seeds} wall={time.time() - t0:.0f}s")
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps({"results": results, "summary": summary}))


if __name__ == "__main__":
    main()
