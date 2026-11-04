"""The Narrator's compact Korean language helpers (해요체)."""
from server import i18n
from server.i18n import KIND_KO, COLOR_KO, SIZE_KO, fill

def object_label(kind: str, color: str | None = None, size: str | None = None, voice: str = "grandma", tags: list[str] | tuple[str, ...] = ()) -> str:
    return "무언가" if kind == "something" else KIND_KO.get(kind, kind or "")


def after_words(tag: str | None, label: str, voice: str = "grandma", verb: str | None = None, kind: str | None = None) -> str:
    p = i18n.tag_name(tag) if tag else ""
    m = {"solved": "기분이 좋아지고 편안해졌어요", "partial": "조금 나아졌어요", "nothing": "그대로예요"}.get(label, "더 힘들어졌어요")
    return f"{p} {m}".strip() if p else m


def action_phrase(verb: str, kind: str | None, voice: str = "grandma") -> str:
    k = object_label(kind) if kind else ""
    v = verb.lower().replace("_", " ")
    return fill(f"{{k:을}} {v}하기", k=k) if k else f"{v}하기"


def intro(sit, first_use: bool = False, still: tuple[str, ...] = (), voice: str = "grandma") -> tuple[str, str | None]:
    if getattr(sit, "word", None):
        k = object_label(getattr(sit, "word_kind", ""))
        return f'할머니: "{sit.word}, 부탁해!"', (f'"{sit.word}"라는 말은 {k}{i18n.josa(k, "을")} 뜻해요.' if first_use and k else None)
    lines = [f"{i18n.tag_name(t)} 상황이 아직 그대로예요." if t in still else f"{i18n.tag_name(t)} 상황이에요." for t in getattr(sit, "tags", ())]
    return " ".join(lines) or "주변을 둘러봐요.", None


def solo_intro(sit, voice: str = "self") -> str:
    tag = next((t for t in getattr(sit, "tags", ()) if t != "REQUEST"), None)
    return f"{i18n.tag_name(tag) if tag else '새로운 상황'} 앞에서 아이가 혼자 생각해요: '어떻게 해 볼까?'"


def option_text(verb: str, obj, voice: str = "grandma", tags: list[str] | tuple[str, ...] = ()) -> str:
    return child_option_text(verb, obj, voice=voice, tags=tags)


def option_text_ko(verb: str, kind: str | None = None, tag: str = "") -> str:
    return option_text(verb, kind, tags=[tag] if tag else ())


def child_option_text(verb: str, obj, voice: str = "self", tags: list[str] | tuple[str, ...] = ()) -> str:
    kind = getattr(obj, "kind", obj if isinstance(obj, str) else None)
    k = object_label(kind) if kind else ""
    v = verb.lower().replace("_", " ")
    return fill(f"{{k:을}} {v}하기", k=k) if k else f"{v}하기"


def child_speech(verb: str, target, voice: str = "grandma", step: int = 1, tags: list[str] | tuple[str, ...] = ()) -> str:
    if voice == "baby" or step == 1:
        return ""
    kind = getattr(target, "kind", target if isinstance(target, str) else None)
    k = object_label(kind) if kind else ""
    v = verb.lower().replace("_", " ")
    return f'"{k} {v}해 볼까?"'.strip() if k else f'"{v}해 볼까?"'


def doing(verb: str, obj, who: str, voice: str = "grandma", tags: list[str] | tuple[str, ...] = ()) -> str:
    act = child_option_text(verb, obj, voice, tags=tags)
    if who == "child":
        return f"아이는 {act}를 해요"
    return f"당신은 {act}를 해요"


def doing_words(verb: str, obj, who: str, voice: str = "grandma", tags: list[str] | tuple[str, ...] = ()) -> str:
    return doing(verb, obj, who, voice, tags=tags)


def flavor(sit, label: str, tag_values: dict[str, int] | None = None, voice: str = "grandma", verb: str | None = None, kind: str | None = None) -> str:
    return f"결과는 {label}입니다."


def world_note(unlocked: list[str], opened: list[str], nameable: list[str]) -> str:
    notes = [f"{i18n.PLACE_KO.get(p, p)}에 이제 갈 수 있어요." for p in unlocked]
    notes += [f"{i18n.tag_name(t)} 상황은 아직 끝나지 않았어요." for t in opened]
    notes += [f"할머니가 {object_label(k)}에 눈길을 줬어요. 이름을 붙여 줄지도 몰라요." for k in nameable]
    return " ".join(notes)


def explain_fixed(w: dict) -> str:
    head = f"아이는 {w.get('expected')} 결과를 기대했지만 실제로는 {w.get('actual')} 상태가 되었어요."
    if w.get("reasons") == ["shown"]:
        return f"{head} 당신이 직접 해 보였고 아이는 지켜보기만 했어요."
    reasons = ", ".join(w.get("reasons", []))
    return f"{head} 선택 이유: {reasons}." if reasons else head


def intro_error(text: str, sit=None, solo: bool = False) -> str | None:
    t = text.strip()
    if not t:
        return "empty"
    if len(t) > 170:
        return "too long"
    from server.agents.guards import BAD_CHARS
    if BAD_CHARS.search(t) or "\n" in text:
        return "markup or control characters"
    if sit and getattr(sit, "word", None) and sit.word not in t:
        return f"must mention {sit.word!r}"
    if solo:
        for phrase in ("당신", "같이 놀아요", "놀아주세요", "씻겨주세요", "해 주세요", "해주세요", "읽어주세요", "엄마", "아빠"):
            if phrase in t:
                return f"addresses parent or asks for help in solo mode: {phrase}"
    return None


def outcome_error(text: str, sit=None, kind: str | None = None, unlocked: set[str] | None = None) -> str | None:
    t = text.strip()
    if not t:
        return "empty"
    if len(t) > 150:
        return "too long"
    from server.agents.guards import BAD_CHARS
    if BAD_CHARS.search(t) or "\n" in text:
        return "markup or control characters"
    return None


def mock(user: dict) -> dict:
    task = user.get("task")
    if task == "ask_child":
        return {"answer": "음... 아직 잘 몰라요."}
    if task == "solo_intro":
        tag = user.get("problems", [{}])[0].get("tag") if user.get("problems") else None
        return {"text": f"{i18n.tag_name(tag) if tag else '새로운 상황'} 앞에서 아이가 혼자 생각해요: '어떻게 해 볼까?'"}
    if task == "intro":
        if user.get("word"):
            return {"text": f'할머니: "{user["word"]}, 부탁해!"'}
        lines = [f"{i18n.tag_name(p['tag'])} 상황이 아직 그대로예요." if p.get("continues") else f"{i18n.tag_name(p['tag'])} 상황이에요." for p in user.get("problems", [])]
        return {"text": " ".join(lines) or "주변을 둘러봐요."}
    if task == "explain":
        return {"text": explain_fixed({"expected": "nothing", "actual": "solved", "reasons": ["chance"]})}
    label = user.get("label", "solved")
    if user.get("word"):
        return {"text": f'할머니가 웃어요: "{user["word"]}!"' if label == "solved" else f'할머니가 고개를 저어요: "{user["word"]} 말고."'}
    return {"text": f"결과는 {label}입니다.", "takeaway": "경험을 통해 배웠어요."}
