"""Text guards for player-facing LLM prose. Prose never decides anything; these checks only keep it from lying about
facts the engine settled or from steering the player (naming things, giving the answer, leaking tokens)."""
import re

from server import i18n
from server.world.ontology import KINDS, LOCATIONS, TAGS, VERBS

META_PHRASES = ("correct answer", "right answer", "best option", "best choice", "the answer", "you should",
                "you must", "hint:", "solution", "optimal", "the right one", "the correct one")
BAD_CHARS = re.compile(r"[{}<>`*#\[\]\\\x00-\x1f]")
UP_TOKENS = set(TAGS) | set(VERBS)
SOLVED_WORDS = ("solved", "solves", "perfect", "fixed", "all better", "gone", "problem is over")
FAILED_WORDS = ("worse", "failed", "fails", "unfortunately", "wrong", "nothing changes", "no use", "useless")
# Korean: the same checks. No "정답"/"틀렸" (nothing in this game is right or wrong) and no steering.
META_PHRASES_KO = ("정답", "올바른", "옳은", "틀렸", "틀린", "힌트", "최선", "해야 해", "해야 합", "하는 게 좋", "추천")
SOLVED_WORDS_KO = ("해결", "완벽", "다 나았", "모두 나았", "싹 사라", "문제가 끝", "풀렸")
FAILED_WORDS_KO = ("더 나빠", "악화", "더 심해", "더 커졌", "실패", "소용없", "소용이 없", "아쉽게도", "변하지 않", "아무 일도 일어나지")
# a Korean noun is followed by a particle, never by other letters: "공이" is the ball, "공기" is not
_PARTICLE = r"(?:으로|에게|한테|이랑|[을를은는이가에의도만과와랑로])?"


def _words(text: str) -> list[str]:
    out = []
    for w in text.split():
        w = w.strip('.,!?:;"()—-').lower()
        if w.endswith("'s"):
            w = w[:-2]
        out.append(w)
    return out


def kinds_in(text: str) -> set[str]:
    ws = set(_words(text))
    found = {k for k in KINDS if k in ws or k + "s" in ws or k + "es" in ws}
    if i18n.is_ko():
        found |= {k for k, name in i18n.KIND_KO.items() if k in KINDS and re.search(rf"(?<![가-힣]){name}{_PARTICLE}(?![가-힣])", text)}
    return found


def text_ok(text: str, *, max_len: int, allowed_kinds: frozenset[str] | set[str] = frozenset(),
            allowed_places: set[str] | None = None, must_contain: str | None = None) -> str | None:
    """None if the text is acceptable, else the reason it is not."""
    t = text.strip()
    if not t:
        return "empty"
    if len(t) > max_len:
        return "too long"
    if BAD_CHARS.search(t) or "\n" in text:
        return "markup or control characters"
    if set(re.findall(r"[A-Z][A-Z_0-9]{2,}", t)) & (UP_TOKENS | {"REQUEST"}) or "NAME_" in t or "OPT_" in t:
        return "leaks an internal token"
    low = t.lower()
    if any(m in low for m in META_PHRASES) or (i18n.is_ko() and any(m in t for m in META_PHRASES_KO)):
        return "talks about the answer"
    extra = kinds_in(t) - set(allowed_kinds)
    if extra:
        return f"names {sorted(extra)}"
    if allowed_places is not None:
        ws = set(_words(t))
        bad = [p for p in LOCATIONS if p in ws and p not in allowed_places and p != "house"]
        if bad:
            return f"mentions a place that is not open: {bad}"
        if i18n.is_ko():
            bad = [p for p in LOCATIONS if p != "house" and p not in allowed_places and i18n.PLACE_KO.get(p) and re.search(rf"(?<![가-힣]){i18n.PLACE_KO[p]}(?![가-힣])|{i18n.PLACE_KO[p]}{_PARTICLE}(?![가-힣])", t)]
            if bad:
                return f"mentions a place that is not open: {bad}"
    if must_contain and must_contain.lower() not in low:
        return f"must mention {must_contain!r}"
    return None


def outcome_ok(text: str, label: str) -> str | None:
    """The prose must not contradict the outcome label the engine recorded."""
    low = text.lower()
    solved_words, failed_words = (SOLVED_WORDS + SOLVED_WORDS_KO, FAILED_WORDS + FAILED_WORDS_KO) if i18n.is_ko() else (SOLVED_WORDS, FAILED_WORDS)
    if label == "solved":
        hit = [w for w in failed_words if w in low]
    elif label == "worse":
        hit = [w for w in solved_words if w in low]
    else:  # nothing / partial: neither solved nor worse
        hit = [w for w in solved_words + ("worse",) + (FAILED_WORDS_KO[:3] if i18n.is_ko() else ()) if w in low]
    return f"contradicts the outcome ({hit[0]})" if hit else None
