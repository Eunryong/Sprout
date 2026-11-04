"""Template situation generator: the Director's validated fallback (and the Mock Director's brain).

It composes situations inside the ontology and puts the solving thing on the table, plus traps. It never talks to
the child. Whether an option is good is decided later by the rules table, not here."""
import os
import random

from .ontology import (
    BABY_ONLY_KINDS, BABY_ONLY_VERBS, OPEN_MONTH, COLORS, KINDS, MIN_OBJECTS, MIN_OPTS,
    NEEDS, PLAUSIBLE, REQUEST_VERB, RULES, TRAPS, VERBS, afforded,
)
from .situation import Obj, OptionSpec, Situation, obs_hash, observation, values

MAX_OPTS = 4


# How wrong options are chosen (experiment knob, docs/WORLD.md section 16): "random" = any other thing / verb;
# "hard" = near misses: things that look like the solving one (same shape or material) and the solving pair of ANOTHER
# problem offered here as a decoy. Whether an option is good is still decided by the rules table, never here.
DEFAULT_DECOYS = os.environ.get("GROWING_DECOYS", "random")
P_TRAP = 0.5  # with a negative rule in the ontology: how often the "wrong verb, right thing" decoy is the harmful one

NATURAL_OPTIONS_BY_TAG: dict[str, list[tuple[str, str]]] = {}

SAME_CATEGORY_KINDS: tuple[set[str], ...] = (
    {"cup", "bottle"},
)


def sample_obj(rng: random.Random, kind: str, avoid: set | None = None) -> Obj:
    spec = KINDS[kind]
    for _ in range(20):
        o = Obj(kind=kind, color=rng.choice(COLORS), shape=spec.shape, size=rng.choice(spec.sizes), material=spec.material)
        if not avoid or (o.color, o.size) not in avoid:
            return o
    return o


def _build(rng: random.Random, tags: list[str], word_kind: str | None, decoys: str = "random"):
    request = tags == ["REQUEST"]
    pairs = [(REQUEST_VERB, word_kind)] if request else [NEEDS[t] for t in tags]
    good_kinds = [k for _, k in pairs]
    objs: list[Obj] = []
    good: list[tuple[str, int]] = []
    for verb, kind in pairs:
        copies = 1
        seen: set = set()
        for _ in range(copies):
            o = sample_obj(rng, kind, seen)
            seen.add((o.color, o.size))
            objs.append(o)
            good.append((verb, len(objs) - 1))

    older = not request and all(OPEN_MONTH.get(t, 0) >= 12 for t in tags)  # a problem of an older child: no baby things among the wrong options
    rule_pairs = {vk for t in tags if t in RULES for vk in RULES[t]}

    if request:
        sensible = None
    else:
        tag_cands: list[tuple[str, str]] = []
        for t in tags:
            tag_cands.extend(PLAUSIBLE.get(t, ()))
            if t in RULES:
                tag_cands.extend(RULES[t].keys())
        valid = {
            (v, k) for v, k in tag_cands
            if v in VERBS and k in KINDS and afforded(v, k)
            and not any(k in cat and any(gk in cat and gk != k for gk in good_kinds) for cat in SAME_CATEGORY_KINDS)
        }
        if older:
            valid = {
                (v, k) for v, k in valid
                if (v, k) in rule_pairs or (v not in BABY_ONLY_VERBS and k not in BABY_ONLY_KINDS)
            }
        sensible = valid if valid else None

    def fine(v: str, kind: str) -> bool:
        if older and (v in BABY_ONLY_VERBS or kind in BABY_ONLY_KINDS) and (v, kind) not in rule_pairs:
            return False
        return afforded(v, kind) and (sensible is None or (v, kind) in sensible)

    def conflicts_category(kind: str, current_kinds: list[str]) -> bool:
        return any(kind in cat and any(ck in cat for ck in current_kinds) for cat in SAME_CATEGORY_KINDS)

    sensible_others = [k for k in KINDS if k not in good_kinds and (sensible is None or any(kk == k for _, kk in sensible)) and not (older and k in BABY_ONLY_KINDS)]
    fallback_others = [k for k in KINDS if k not in good_kinds and not (older and k in BABY_ONLY_KINDS)]
    n_wrong = rng.choice([2, 3]) if request else (1 if len(pairs) > 1 and rng.random() < 0.5 else 2)

    wrong_idx = []
    cross = None  # another problem's solving pair, offered as a decoy
    if decoys == "hard" and not request:
        picks = []
        look_alike = [
            k for k in sensible_others
            if not conflicts_category(k, good_kinds + picks)
            and any(KINDS[k].shape == KINDS[g].shape or KINDS[k].material == KINDS[g].material for g in good_kinds)
        ]
        if look_alike:
            picks += rng.sample(look_alike, 1)
        other_tags = [
            t for t in NEEDS
            if t not in tags
            and NEEDS[t][1] not in good_kinds
            and NEEDS[t][1] not in picks
            and NEEDS[t][1] in KINDS
            and not conflicts_category(NEEDS[t][1], good_kinds + picks)
        ]
        if other_tags:
            t2 = rng.choice(other_tags)
            cross = NEEDS[t2]
            if cross[1] in KINDS:
                picks.append(cross[1])
        while len(picks) < n_wrong:
            rest = [k for k in sensible_others if k not in picks and not conflicts_category(k, good_kinds + picks)]
            if not rest:
                break
            picks.append(rng.choice(rest))
        while len(picks) < n_wrong:
            rest2 = [k for k in fallback_others if k not in picks and not conflicts_category(k, good_kinds + picks)]
            if not rest2:
                break
            picks.append(rng.choice(rest2))
    else:
        picks = []
        while len(picks) < n_wrong:
            cands = [k for k in sensible_others if k not in picks and not conflicts_category(k, good_kinds + picks)]
            if not cands:
                break
            picks.append(rng.choice(cands))
        while len(picks) < n_wrong:
            cands = [k for k in fallback_others if k not in picks and not conflicts_category(k, good_kinds + picks)]
            if not cands:
                break
            picks.append(rng.choice(cands))

    while len(objs) + len(picks) < MIN_OBJECTS:
        rest = [k for k in KINDS if k not in good_kinds and k not in picks and not conflicts_category(k, good_kinds + picks)]
        if not rest:
            rest = [k for k in KINDS if k not in good_kinds and k not in picks]
            if not rest:
                break
        picks.append(rng.choice(rest))

    for k in picks:
        objs.append(sample_obj(rng, k))
        wrong_idx.append(len(objs) - 1)

    opts: list[OptionSpec] = []
    if request:
        opts.append(OptionSpec(verb=REQUEST_VERB, target=good[0][1]))
        for i in wrong_idx:
            if len(opts) >= MAX_OPTS:
                break
            opts.append(OptionSpec(verb=REQUEST_VERB, target=i))
    else:
        for v, i in good:
            opts.append(OptionSpec(verb=v, target=i))

        used_kinds = {objs[o.target].kind for o in opts if o.target is not None}
        shuffled_wrong = list(wrong_idx)
        rng.shuffle(shuffled_wrong)

        if cross is not None and cross[1] in [objs[i].kind for i in wrong_idx]:
            c_idx = next(i for i in wrong_idx if objs[i].kind == cross[1])
            if fine(cross[0], cross[1]):
                opts.append(OptionSpec(verb=cross[0], target=c_idx))
                used_kinds.add(cross[1])

        # 1. Prioritize distinct wrong objects with sensible affordances
        for i in shuffled_wrong:
            if len(opts) >= MAX_OPTS:
                break
            k = objs[i].kind
            if k in used_kinds:
                continue
            cand_verbs = [v for v in VERBS if v != "WAIT" and fine(v, k)]
            if cand_verbs:
                right_verbs = [v for v in cand_verbs if any(v == gv for gv, _ in pairs)]
                chosen_v = rng.choice(right_verbs) if right_verbs and rng.random() < 0.6 else rng.choice(cand_verbs)
                opts.append(OptionSpec(verb=chosen_v, target=i))
                used_kinds.add(k)

        # 1b. If sensible distinct-object actions are fewer than MAX_OPTS, try other sensible verbs on the primary (good) object
        if len(opts) < MAX_OPTS:
            for _, gi in good:
                if len(opts) >= MAX_OPTS:
                    break
                gk = objs[gi].kind
                cand_verbs = [v for v in VERBS if v != "WAIT" and fine(v, gk) and OptionSpec(verb=v, target=gi) not in opts]
                if cand_verbs:
                    opts.append(OptionSpec(verb=rng.choice(cand_verbs), target=gi))

        # 1c. If tag explicitly includes WAIT in PLAUSIBLE, include WAIT before falling back to unrelated objects
        if len(opts) < MAX_OPTS and any(any(v == "WAIT" for v, _ in PLAUSIBLE.get(t, ())) for t in tags):
            if OptionSpec(verb="WAIT") not in opts and rng.random() < 0.5:
                opts.append(OptionSpec(verb="WAIT"))

        # 2. If sensible actions are still fewer than MAX_OPTS, try other afforded actions on unused wrong objects
        if len(opts) < MAX_OPTS:
            for i in shuffled_wrong:
                if len(opts) >= MAX_OPTS:
                    break
                k = objs[i].kind
                if k in used_kinds:
                    continue
                cand_verbs = [v for v in VERBS if v != "WAIT" and afforded(v, k) and not (older and (v in BABY_ONLY_VERBS or k in BABY_ONLY_KINDS))]
                if cand_verbs:
                    right_verbs = [v for v in cand_verbs if any(v == gv for gv, _ in pairs)]
                    chosen_v = rng.choice(right_verbs) if right_verbs and rng.random() < 0.6 else rng.choice(cand_verbs)
                    opts.append(OptionSpec(verb=chosen_v, target=i))
                    used_kinds.add(k)

        # 3. If distinct-object actions are still fewer than MAX_OPTS, check remaining wrong objects
        if len(opts) < MAX_OPTS:
            for i in shuffled_wrong:
                if len(opts) >= MAX_OPTS:
                    break
                k = objs[i].kind
                cand_verbs = [v for v in VERBS if v != "WAIT" and fine(v, k) and OptionSpec(verb=v, target=i) not in opts]
                if cand_verbs:
                    opts.append(OptionSpec(verb=rng.choice(cand_verbs), target=i))

        # 4. Only add WAIT when sensible object actions are not enough to reach MIN_OPTS (3)
        if len(opts) < 3 and OptionSpec(verb="WAIT") not in opts:
            opts.append(OptionSpec(verb="WAIT"))

        # 5. Engine fallback: ensure at least MIN_OPTS
        if len(opts) < 3:
            for i in range(len(objs)):
                if len(opts) >= 3:
                    break
                k = objs[i].kind
                aff_verbs = [v for v in VERBS if v != "WAIT" and afforded(v, k) and not (older and (v in BABY_ONLY_VERBS or k in BABY_ONLY_KINDS))]
                rng.shuffle(aff_verbs)
                for v in aff_verbs:
                    cand = OptionSpec(verb=v, target=i)
                    if cand not in opts:
                        opts.append(cand)
                        break

    uniq: list[OptionSpec] = []
    for o in opts:
        if o not in uniq:
            uniq.append(o)

    uniq = uniq[:MAX_OPTS]
    rng.shuffle(uniq)
    return objs, uniq


def _repair_options(sit: Situation, objects: list[Obj]) -> list[OptionSpec]:
    tag = sit.tags[0] if sit.tags else "REQUEST"
    repaired: list[OptionSpec] = []
    seen_verbs: set[str] = set()
    seen_pairs: set[tuple[str, int | None]] = set()
    seen_kinds: set[str] = set()

    for o in sit.options:
        k = objects[o.target].kind if o.target is not None else None
        pair = (o.verb, o.target)
        if pair in seen_pairs:
            continue
        if tag != "REQUEST":
            if o.verb != "WAIT" and o.verb in seen_verbs:
                continue
            if k is not None and k in seen_kinds:
                continue
        repaired.append(o)
        seen_pairs.add(pair)
        if o.verb != "WAIT":
            seen_verbs.add(o.verb)
        if k is not None:
            seen_kinds.add(k)
        if len(repaired) >= MAX_OPTS:
            break

    if len(repaired) < MAX_OPTS and tag != "REQUEST":
        candidates: list[tuple[str, str]] = []
        if tag in RULES:
            for (v, k), val in sorted(RULES[tag].items(), key=lambda x: -x[1]):
                candidates.append((v, k))
        for v, k in PLAUSIBLE.get(tag, ()):
            if (v, k) not in candidates:
                candidates.append((v, k))

        # Pass 1: candidates with unseen verb and unseen kind
        for v, k in candidates:
            if len(repaired) >= MAX_OPTS:
                break
            if v != "WAIT" and v in seen_verbs:
                continue
            if k in seen_kinds:
                continue
            idx = next((i for i, ob in enumerate(objects) if ob.kind == k), None)
            if idx is not None:
                repaired.append(OptionSpec(verb=v, target=idx))
                seen_verbs.add(v)
                seen_kinds.add(k)
                seen_pairs.add((v, idx))

        # Pass 2: candidates with unseen verb even if kind was seen (e.g. GIVE cup when DRINK cup exists)
        if len(repaired) < MAX_OPTS:
            for v, k in candidates:
                if len(repaired) >= MAX_OPTS:
                    break
                if v != "WAIT" and v in seen_verbs:
                    continue
                idx = next((i for i, ob in enumerate(objects) if ob.kind == k), None)
                if idx is not None and (v, idx) not in seen_pairs:
                    repaired.append(OptionSpec(verb=v, target=idx))
                    seen_verbs.add(v)
                    seen_pairs.add((v, idx))

        # Pass 3: if still < 3, afforded actions on existing objects with unseen verbs
        if len(repaired) < 3:
            for idx, ob in enumerate(objects):
                if len(repaired) >= 3:
                    break
                for v in VERBS:
                    if v == "WAIT" or v in seen_verbs or not afforded(v, ob.kind):
                        continue
                    if (v, idx) not in seen_pairs:
                        repaired.append(OptionSpec(verb=v, target=idx))
                        seen_verbs.add(v)
                        seen_pairs.add((v, idx))
                        break

        # Pass 4: if still < 3, add WAIT
        if len(repaired) < 3 and "WAIT" not in seen_verbs and OptionSpec(verb="WAIT") not in repaired:
            repaired.append(OptionSpec(verb="WAIT"))
            seen_verbs.add("WAIT")

        # Pass 5: if still < 3, fallback to remaining options from sit.options
        if len(repaired) < 3:
            for o in sit.options:
                if len(repaired) >= 3:
                    break
                if o not in repaired:
                    repaired.append(o)

    return repaired[:MAX_OPTS]


def make_situation(rng: random.Random, tags: list[str], sid: str = "s", location: str = "house", word: str | None = None,
                   word_token: str | None = None, word_kind: str | None = None, links_to: str | None = None,
                   forbidden: set[str] = frozenset(), decoys: str | None = None) -> Situation:
    """A valid situation for `tags` that does not collide with `forbidden` (quiz) scene hashes."""
    for _ in range(300):
        objs, opts = _build(rng, tags, word_kind, decoys or DEFAULT_DECOYS)
        sit = Situation(id=sid, tags=list(tags), location=location, word=word, word_token=word_token,
                        word_kind=word_kind, links_to=links_to, objects=objs, options=opts)
        repaired_opts = _repair_options(sit, objs)
        if len(repaired_opts) >= 3:
            sit = sit.model_copy(update={"options": repaired_opts})
        v = values(sit)
        if max(v) >= 2 and min(v) < max(v) and obs_hash(observation(sit)) not in forbidden:
            return sit
    raise RuntimeError("could not generate a valid situation")
