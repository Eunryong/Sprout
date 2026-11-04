"""The language of the player-facing text: English (default) or Korean.

Only what the player reads changes. The child, the engine and the rules never see a language: they work on tags, verbs and
attributes. The language is one value per server (`GROWING_LANG`, `--lang`, or the page's language button through
`POST /api/lang`), so a Korean child and an English child are the same child.

Korean needs particles that follow the last sound of the word before them (컵으로 / 공으로, 컵을 / 공을), so templates mark
them: "{k:으로} 마셔요" and `fill` picks the right one.
"""
import os
import re

LANGS = ("en", "ko")
_lang = os.environ.get("GROWING_LANG", "en")
if _lang not in LANGS:
    _lang = "en"


def lang() -> str:
    return _lang


def is_ko() -> bool:
    return _lang == "ko"


def set_lang(x: str) -> None:
    global _lang
    if x not in LANGS:
        raise ValueError(f"unknown language {x!r} (use one of {LANGS})")
    _lang = x


# ---- particles ----
_PAIRS = {"을": ("을", "를"), "은": ("은", "는"), "이": ("이", "가"), "과": ("과", "와"), "으로": ("으로", "로"), "에": ("에", "에")}
_SLOT = re.compile(r"\{(\w+)(?::([^}]+))?\}")


def _final(word: str) -> int:
    """The final consonant of the last syllable (0 = none). A word in Latin letters (a made-up word like "bobo") reads as open."""
    for ch in reversed(word):
        if "가" <= ch <= "힣":
            return (ord(ch) - 0xAC00) % 28
        if ch.isalnum():
            return 0
    return 0


def josa(word: str, spec: str) -> str:
    with_final, without = _PAIRS[spec]
    f = _final(word)
    if spec == "으로":
        return without if f in (0, 8) else with_final  # 8 = ㄹ: 연필로, not 연필으로
    return with_final if f else without


def fill(template: str, **words: str) -> str:
    """'{k:으로} 마셔요' + k='컵' -> '컵으로 마셔요'."""
    return _SLOT.sub(lambda m: words[m[1]] + (josa(words[m[1]], m[2]) if m[2] else ""), template)


# ---- words ----
KIND_KO = {
    "cup": "컵", "bread": "빵", "teddy": "곰인형", "cloth": "헝겊", "paper": "종이", "box": "상자", "blanket": "담요", "flashlight": "손전등",
    "spoon": "숟가락", "ball": "공", "bottle": "젖병", "soap": "비누", "bandage": "반창고", "crayon": "크레용", "book": "책",
    "umbrella": "우산", "backpack": "가방", "fan": "부채",
}
COLOR_KO = {"red": "빨간", "blue": "파란", "green": "초록", "yellow": "노란"}
SIZE_KO = {"small": "작은", "big": "큰"}
PLACE_KO = {"house": "집", "yard": "마당"}
# a trouble as a short name: "지난번(목마름)", and the home of the "when" in a memory
TAG_KO = {
    "BABY_HUNGRY": "아기 배고픔", "UPSET": "울음", "COLD": "추위", "THIRSTY": "목마름", "HUNGRY": "배고픔", "FLOOR_WET": "젖은 바닥",
    "MESSY": "어질러진 장난감", "DARK": "어둠", "VISITOR": "손님", "DIRTY_HANDS": "더러운 손", "HURT": "다친 무릎", "BORED": "심심함",
    "RAIN": "비", "SCHOOL_MORNING": "등교 아침", "FRIEND_SAD": "슬픈 친구", "REQUEST": "부탁", "HOT": "더위",
}
STAGE_KO = {
    "Baby": ("아기", "모든 것이 처음이에요. 먹이고, 안아 주고, 달래 주세요. 아기는 아직 스스로 고를 수 없어요. 당신이 하는 일을 지켜봐요."),
    "Toddler": ("걸음마 아이", "첫걸음, 첫말. 아이가 손을 뻗어 이것저것 해 보기 시작해요."),
    "Little kid": ("꼬마", "질문과 어지르기, 그리고 상상 놀이. 아이는 점점 스스로 하는 일이 늘어요."),
    "Schoolchild": ("학생", "등굣길 아침과 친구들. 아이 스스로의 선택이 하루하루 더 중요해져요."),
}


def tag_name(tag: str) -> str:
    return TAG_KO.get(tag, tag.lower().replace("_", " ")) if is_ko() else tag.lower().replace("_", " ")


def tags_phrase(tags: list[str]) -> str:
    """The troubles of a scene in a few words ("thirsty and floor wet" / "목마름 · 젖은 바닥"); '' when there are none."""
    return (" · " if is_ko() else " and ").join(tag_name(t) for t in tags if t != "REQUEST")


def stage(stg: dict | None) -> dict | None:
    """A life stage as the player reads it (name and blurb in the language; an unknown stage stays as written)."""
    if stg is None or not is_ko() or stg["name"] not in STAGE_KO:
        return stg
    name, blurb = STAGE_KO[stg["name"]]
    return {**stg, "name": name, "blurb": blurb}


def age_text(months: int) -> str:
    y, m = divmod(months, 12)
    if is_ko():
        return " ".join(p for p in (f"{y}년" if y else "", f"{m}개월" if m or not y else "") if p)
    return " ".join(p for p in (f"{y} year{'s' if y != 1 else ''}" if y else "", f"{m} month{'s' if m != 1 else ''}" if m or not y else "") if p)
