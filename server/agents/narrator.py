"""Narrator. Player-facing prose only: the child never reads it. The fixed phrases below are the always-valid
fallback; NarratorAgent asks an LLM for livelier lines and keeps a line only if it passes the guards (no object named
before the choice, no contradiction of the engine's outcome label, no internal tokens). It never decides facts: the
caller passes facts already settled by the engine (outcome label, unlocked places, opened issues)."""
import json
import time
from typing import Any

from pydantic import BaseModel, Field

from server import i18n
from server.world.ontology import KINDS
from server.world.situation import Situation

from .client import LLMClient
from . import narrator_ko as ko
from .glossary import PROBLEM_GLOSS, problem_gloss
import re

from .guards import BAD_CHARS, outcome_ok, text_ok
from .stats import AgentStats

def tables(voice: str = "grandma") -> dict:
    return {"problem": {}, "still": {}, "solved": {}, "verbs": {}}


def object_label(kind: str, color: str, size: str) -> str:
    return ko.object_label(kind, color, size) if i18n.is_ko() else f"the {size} {color} {kind}".strip()


def happened(label: str) -> str:
    """What came of it, in the engine's words ('the trouble was over')."""
    return f"결과는 {label}입니다." if i18n.is_ko() else f"the outcome was {label}"


def after_words(tags: list[str] | tuple[str, ...], label: str, voice: str = "grandma", verb: str | None = None, kind: str | None = None) -> str:
    """How the scene looked after it, in the scene's own words and without a full stop."""
    tag = next((t for t in tags if t != "REQUEST"), None)
    if i18n.is_ko():
        return ko.after_words(tag, label, voice, verb, kind)
    return label


def action_phrase(verb: str, kind: str | None, voice: str = "grandma") -> str:
    """An act with the kind only, as the child remembers it ('drink from a cup')."""
    if i18n.is_ko():
        return ko.action_phrase(verb, kind, voice)
    v = verb.lower().replace("_", " ")
    if kind is None:
        return v
    article = "an " if kind[0] in "aeiou" else "a "
    return f"{v} with {article}{kind}"


def intro(sit: Situation, first_use: bool, still: tuple[str, ...] = (), voice: str = "grandma") -> tuple[str, str | None]:
    """Scene text + a hint line for the player (never given to the child)."""
    if i18n.is_ko():
        return ko.intro(sit, first_use, still, voice)
    if sit.tags == ["REQUEST"]:
        text = f'Grandma: "Could you bring me the {sit.word}?"'
        hint = f'"{sit.word}" means {sit.word_kind} (Grandma\'s word; the child cannot see this).' if first_use else None
        return text, hint
    tag = sit.tags[0] if sit.tags else "situation"
    if still:
        return f"The {tag.lower().replace('_', ' ')} situation continues.", None
    return f"A new moment arrives with {tag.lower().replace('_', ' ')}.", None


def option_text(verb: str, obj, voice: str = "grandma", tags: list[str] | tuple[str, ...] = ()) -> str:
    if i18n.is_ko():
        return ko.option_text(verb, obj, voice, tags=tags)
    k = object_label(obj.kind, obj.color, obj.size) if obj is not None else ""
    v = verb.lower().replace("_", " ")
    return f"{v} {k}".strip() if k else v


def child_option_text(verb: str, obj, voice: str = "self", tags: list[str] | tuple[str, ...] = ()) -> str:
    if i18n.is_ko():
        return ko.child_option_text(verb, obj, voice=voice, tags=tags)
    return option_text(verb, obj, voice="self", tags=tags)


def child_speech(verb: str, obj, voice: str = "grandma", step: int = 1, tags: list[str] | tuple[str, ...] = ()) -> str:
    """What the speaking child says and does for this option."""
    if i18n.is_ko():
        return ko.child_speech(verb, obj, voice, step, tags)
    if voice == "baby":
        return ""
    k = object_label(obj.kind, obj.color, obj.size) if obj is not None else ""
    v = verb.lower().replace("_", " ")
    if step == 2:
        return f'"I will try to {v} {k} by myself!"'.strip()
    return f'"Can we {v} {k}?"'.strip()


def solo_intro(sit: Situation, voice: str = "self") -> str:
    """Intro line for Step 2 (the child's solo moment within a scene)."""
    if i18n.is_ko():
        return ko.solo_intro(sit, voice)
    tag = next((t for t in sit.tags if t != "REQUEST"), "moment")
    return f"The child is by themselves facing {tag.lower().replace('_', ' ')}: what should I try on my own?"


def _doing(sit: Situation, option: int, who: str, voice: str = "grandma") -> str:
    o = sit.options[option]
    obj = sit.objects[o.target] if (o.target is not None and 0 <= o.target < len(sit.objects)) else None
    if i18n.is_ko():
        return ko.doing(o.verb, obj, who, voice, tags=list(sit.tags))
    return f"{'The child decides to' if who == 'child' else 'You'} {option_text(o.verb, obj, voice, tags=list(sit.tags))}"


def intention(sit: Situation, option: int, who: str, voice: str = "grandma") -> str:
    """What is about to be done, before the engine says what came of it."""
    return _doing(sit, option, who, voice) + "."


_NEUTRAL_FEEL_EN = {
    "solved": "The trouble is over.",
    "partial": "Things feel a little easier, but something still lingers.",
    "nothing": "Everything stays just as it was.",
}
_NEUTRAL_FEEL_EN_DEFAULT = "Things feel heavier now."

_NEUTRAL_FEEL_KO = {
    "solved": "고민이 사라지고 마음이 편해졌어요.",
    "partial": "조금 나아졌지만 아직 남은 게 있어요.",
    "nothing": "상황은 그대로예요.",
}
_NEUTRAL_FEEL_KO_DEFAULT = "마음이 더 무거워졌어요."


def _neutral_feel(label: str, ko_path: bool) -> str:
    """Grade-free per-result feel line (never echoes the engine label)."""
    if ko_path:
        return _NEUTRAL_FEEL_KO.get(label, _NEUTRAL_FEEL_KO_DEFAULT)
    return _NEUTRAL_FEEL_EN.get(label, _NEUTRAL_FEEL_EN_DEFAULT)


def fixed_flavor(sit: Situation, label: str, tag_values: dict[str, int] | None = None, voice: str = "grandma", verb: str | None = None, kind: str | None = None) -> str:
    """What came of it, in fixed words."""
    if i18n.is_ko():
        return _neutral_feel(label, ko_path=True)
    if sit.tags == ["REQUEST"]:
        return f"Grandma smiles: that is the {sit.word}!" if label == "solved" else f"Grandma tilts her head: that is not the {sit.word}."
    return _neutral_feel(label, ko_path=False)


def outcome(sit: Situation, option: int, label: str, who: str, tag_values: dict[str, int] | None = None, voice: str = "grandma") -> str:
    """One sentence about what was done and what came of it. `label` is the engine's outcome label."""
    o = sit.options[option]
    kind = sit.objects[o.target].kind if (o.target is not None and 0 <= o.target < len(sit.objects)) else None
    return f"{_doing(sit, option, who, voice)}. {fixed_flavor(sit, label, tag_values, voice, o.verb, kind)}"


def world_note(unlocked: list[str], opened: list[str], nameable: list[str]) -> str:
    """Facts the engine changed as a consequence of the last choice."""
    if i18n.is_ko():
        return ko.world_note(unlocked, opened, nameable)
    notes = [f"The {p} is open now: you can go there." for p in unlocked]
    notes += [f"The {t.lower().replace('_', ' ')} problem is not over." for t in opened]
    notes += [f"Grandma noticed the {k}: she may give it a name." for k in nameable]
    return " ".join(notes)


def refs_ok(text: str, allowed_kinds: set[str]) -> bool:
    """Reject text that names a kind that is not on the table (cheap check; not a full hallucination guard)."""
    words = {w.strip('.,!?:;"()').lower() for w in text.split()}
    return all(k in allowed_kinds for k in KINDS if k in words)


# ---- Narrator v2: LLM prose with guards ----
SYSTEM = (
    "You are the Narrator of a gentle life-simulation game about a child learning and exploring the world. Write ONE short "
    "English sentence or two (no markdown, no quotes around the whole text). Scenes in this random world can be any everyday moment, "
    "curiosity, play, exploration, social interaction, or difficulty. Task `intro`: set the scene for the given situation or context "
    "(`continues` means it was left unfinished earlier, so say it is still going on). NEVER name or hint at any object or what to do about it. "
    "If `word` is given, Grandma asks for it by exactly that word. If `voice` is \"baby\" the one being cared for is the baby (it is crying, hungry or cold; "
    "a parent or Grandma looks after it): write everything about the baby and never about Grandma needing help. "
    "If `voice` is \"self\" the experience is the child's own and the player is its parent: write it about the child and never about Grandma needing help. "
    "A child of this age talks: in `intro` with `voice` \"self\", the child should address the parent naturally the way a real young child does — sharing curiosity, "
    "asking to play or interact together, or asking for help with something they cannot easily do alone, NOT stiff adult-like self-analysis. "
    "Task `solo_intro`: the child is alone by itself (no parent helping right now) facing the situation and wondering what to try on its own. "
    "Never mention 'you' (the parent) or ask the parent for help. "
    "In task `outcome` with `feelings`, say how the baby (or child) feels in those terms, in your own words and in its way: one or two feelings at once. "
    "Never say what the parent should have done. "
    "Task `outcome`: describe what came of the action the player already chose; `label` is settled and you must not contradict it (solved / partial / nothing / worse). "
    "`outcomes` represent how positively and comfortably the action turns out for the child (solved=very positive and satisfying, partial=mildly positive or slightly better, nothing=unchanged or neutral, worse=uncomfortable or negative), and `bundle_result` describes the concrete consequence of each choice. "
    "When engaging in shared activities (reading, drawing, playing, comforting) or when `custom_parent_action` is provided in the payload, include a concrete, imaginative glimpse of the activity or moment, "
    "and express the child's learning and emotions in varied, childlike ways rather than repeating dry formulaic lessons. "
    "Never name a thing other than `chosen_kind`. Stay under `max_chars`. Task `explain`: the child expected one thing and something else happened. Say it in two short sentences: what it "
    "expected, what happened, and why it "
    "chose that (`reasons`: told = you said so, curious = it knew little about it, promising = it seemed the most "
    "promising to it, chance = more or less by chance, shown = you did it and it only watched). NO numbers or percentages, "
    "never say it was right, wrong, good or bad. Reply as JSON {\"text\": ...}."
)
KOREAN = (
    " LANGUAGE: write the sentence in natural Korean (한국어) in a soft polite 해요체 voice, as in a picture book; the player and the child are "
    "'당신' (the parent) and '아이' (the child), Grandma is '할머니'. If `voice` is \"baby\" the child IS the baby: call it only '아기' and never mention '아이', "
    "and the player is its parent. 이 세상의 장면은 문제 상황에 국한되지 않고 일상의 모든 순간, 호기심, 놀이, 탐색, 상호작용, 일상적 어려움 등 무엇이든 될 수 있습니다. "
    "In task `intro` with `voice` \"self\", 아이가 부모에게 말할 때는 실제 어린아이처럼 자연스럽게 호기심을 표현하거나 함께하자고 말하거나 도움을 청해야 합니다. 어른스러운 자기 분석은 절대 하지 마세요. "
    "In task `solo_intro`, 아이는 지금 부모('당신') 없이 혼자 새로운 상황을 마주하고 스스로 어떻게 해 볼지 고민하는 장면입니다. 절대 '당신'을 언급하거나 부모에게 '같이 놀아요', '~해주세요'라고 말하지 말고, 아이가 혼자서 '어떻게 해 볼까?', '무엇을 해 볼까?' 하고 스스로 생각하는 모습으로 묘사하세요. "
    "In task `outcome`, 결과 라벨(`solved`/`partial`/`nothing`/`worse`)은 아이에게 행동의 결과가 얼마나 긍정적이고 만족스러웠는지를 나타내며(solved=매우 긍정적/만족스러움, partial=약간 긍정적, nothing=변화 없음/중립, worse=불편함/부정적), `bundle_result`는 각 선택의 구체적 결과를 나타냅니다. "
    "책 읽기, 그림 그리기, 놀아주기, 위로하기 또는 `custom_parent_action`이 주어졌을 때 구체적이고 생생한 상상이나 활동의 한 장면을 담고, "
    "다양하고 아이다운 감정과 배움을 표현하세요. 딱딱하고 틀에 박힌 교훈 반복은 피하세요. "
    "When you name the chosen thing use `chosen_kind_ko` exactly. Never use English words."
)
MAX_INTRO, MAX_FLAVOR, MAX_EXPLAIN = 170, 150, 300


def system_prompt() -> str:
    return SYSTEM + KOREAN if i18n.is_ko() else SYSTEM


def explain_fixed(w: dict) -> str:
    """The always-valid wording of a surprise: what it expected, what happened, why it chose that. No numbers, no verdict."""
    if i18n.is_ko():
        return ko.explain_fixed(w)
    head = f"The child expected {w.get('expected')}, but the result was {w.get('actual')}."
    if w.get("reasons") == ["shown"]:
        return f"{head} You did it yourself and the child only watched."
    r_str = " and ".join(w.get("reasons", []))
    return f"{head} It chose this because of {r_str}." if r_str else head


def explain_ok(text: str, w: dict) -> str | None:
    """The prose must say what the child expected and what happened, with no numbers and no verdict."""
    bad = text_ok(text, max_len=MAX_EXPLAIN, allowed_kinds={w["kind"]} if w.get("kind") else set())
    if bad:
        return bad
    if re.search(r"\d|%", text):
        return "numbers"
    low = text.lower()
    if re.search(r"\b(mistake|wrong|right|correct|failed|good|bad|better|worse)\b|should have", low):
        return "passes a verdict"
    return None


class SceneBundleOut(BaseModel):
    narration: str = Field(..., min_length=6, max_length=320)
    options: list[str] = Field(default_factory=list, description="4 contextual action choices")
    speeches: list[str] = Field(default_factory=list, description="4 short child speech lines matching the 4 options")
    outcomes: list[str] = Field(default_factory=list, description="4 outcome labels ('solved', 'partial', 'nothing', or 'worse') matching the 4 options")
    results: list[str] = Field(default_factory=list, description="4 short descriptions of what happens for each option")
    world_says: list[str] = Field(default_factory=list, description="4 short causal explanations of why the world works that way for each option")


class NarrationOut(BaseModel):
    text: str
    takeaway: str = Field(default="", description="First-person child inner thought / lesson learned from this experience (1-2 sentences)")


OUTCOME_SYSTEM_KO = (
    "당신은 영유아 양육 및 일상 생활 시뮬레이션 게임의 내레이터입니다. "
    "한국어(해요체)로 따뜻하고 자연스러운 그림책 문체로 작성하세요. "
    "이 세상의 장면은 문제 상황에 국한되지 않으며, 일상의 모든 순간, 호기심, 놀이, 탐색, 사회적 상호작용, 일상적 어려움 등 무엇이든 될 수 있습니다. "
    "JSON 형식으로 `{\"text\": \"...\", \"takeaway\": \"...\"}`를 반환해야 합니다.\n"
    "1. `text` (2-3문장, 최대 280자):\n"
    "   - `who`('parent'는 당신이 아이와 함께 한 행동, 'child'는 아이가 스스로 한 행동)가 "
    "주어진 상황(`scene_narration`)에서 한 구체적 행동(`chosen_action`)과 그로 인해 일어난 결과(`label`, `bundle_result`)를 "
    "생생하고 자연스럽게 서술하세요.\n"
    "   - 결과 라벨(label)의 의미 (아이에게 결과가 얼마나 긍정적이고 편안한지):\n"
    "     * solved: 행동의 결과가 매우 긍정적이고 만족스러워 아이가 편안해지거나 기뻐함\n"
    "     * partial: 상황이 조금 나아지거나 약간 긍정적이지만 여전히 잔여 상태가 남아있음\n"
    "     * nothing: 상황에 별다른 변화가 없거나 중립적임\n"
    "     * worse: 아이가 불편해지거나 결과가 부정적/어수선해짐\n"
    "   - `bundle_result`가 제공된 경우 해당 선택지의 구체적 결과를 맥락으로 삼아 자연스럽게 서술하세요.\n"
    "   - 행동과 결과를 하나의 흐름으로 자연스럽게 풀어내며, 딱딱한 정형화된 틀을 피하세요.\n"
    "   - 책 읽기, 그림 그리기, 놀이, 위로 또는 사용자 정의 행동이 주어졌을 때는 구체적이고 생생한 장면과 아이의 반응을 담으세요.\n"
    "2. `takeaway` (1-2문장):\n"
    "   - 이번 경험에서 아이가 행동(`chosen_action`)을 하고 구체적 결과(`label`, `bundle_result`)를 겪으며 배운 자연스러운 1인칭 깨달음이나 생각입니다.\n"
    "   - 부모와 함께한 경우: '부모님이 ~해 주셔서 ~하다는 걸 느꼈어' 등\n"
    "   - 아이 혼자 한 경우: '내가 직접 ~해 보니까 ~해서 다음에는 ~해야겠다고 생각했어' 등\n"
    "   - 틀에 박힌 도덕적 교훈을 반복하지 말고 아이다운 진솔하고 생생한 배움을 적어주세요."
)

OUTCOME_SYSTEM_EN = (
    "You are the Narrator of a gentle life-simulation game about a child learning and growing. "
    "Scenes in this random world can be any everyday moment, curiosity, play, exploration, social interaction, or difficulty. "
    "Return JSON `{\"text\": \"...\", \"takeaway\": \"...\"}`.\n"
    "1. `text`: 1-2 sentences describing what came of the action in the scene according to the outcome label and concrete result (`bundle_result`).\n"
    "   Outcome meanings represent how positively/comfortably the action turns out for the child: 'solved' (very positive, satisfying, comfortable), "
    "'partial' (mildly positive, slightly better), 'nothing' (unchanged, neutral), 'worse' (uncomfortable, negative).\n"
    "2. `takeaway`: 1-2 sentences of the child's natural first-person inner reflection and lesson learned from doing that action and experiencing that result."
)

SCENE_BUNDLE_SYSTEM_KO = (
    "당신은 영유아 양육 및 일상 생활 시뮬레이션 게임의 내레이터이자 시나리오 작가입니다. "
    "아이의 연령(`age_months`, `age_years`, `voice`)에 맞추어 아이의 일상 세계에서 벌어지는 현실적이고 창의적인 장면을 자유롭게 창작하세요. "
    "이 세상의 장면은 특정한 문제 상황에 국한되지 않으며, 일상의 어떤 순간이든 호기심, 놀이, 탐색, 또래나 가족과의 상호작용, 일상적 어려움 등 무엇이든 될 수 있습니다. "
    "아이의 발달 수준은 `age_years`(`age_months // 12`, 3세~7세)에 맞춰 자연스럽게 변화합니다. 3세에는 세상을 잘 모르는 순진한 유아로서 아주 짧고 단순한 3세 유아의 말(`speeches`)로 표현하고, `age_years`가 4세, 5세, 6세, 7세로 자라남에 따라 아이의 대사(`speeches`)와 일상적인 호기심 및 상황이 연령에 맞게 자연스럽게 성숙해집니다. 단, 어떤 연령이든 미리 정답이나 상식을 알고 있는 것처럼 굴지 않으며, 배운 적이 없으면 결과를 모른 채 아이의 눈높이에서 표현해야 합니다. 아이의 연령(`age_years`)에 맞는 순수한 호기심과 눈높이의 다채롭고 다양한 일상 세계 상황과 선택지를 만드세요. "
    "최근 장면들(`recent_scenes`)의 주제나 세부 묘사는 완전히 피하여 항상 새롭고 신선한 상황을 만드세요. "
    "\n[규칙]\n"
    "1. `solo`가 false인 경우 (1단계: 부모와 아이가 함께하는 단계):\n"
    "   - narration (2-3문장, 최대 300자): 지금 아이와 부모가 있는 장소에서 벌어지는 구체적인 일상 상황과, "
    "아이가 부모에게 관심, 놀이, 도움, 또는 상호작용을 청하는 모습/말을 묘사합니다.\n"
    "   - options (정확히 `option_count`개, 기본 4개): 이 상황에 어울리는 4가지 현실적이고 뚜렷하게 구별되는 양육자/부모의 행동 선택지 문구 (따옴표나 대사 없이 자연스러운 문장형).\n"
    "   - speeches (정확히 `option_count`개, 기본 4개): 각 선택지에 대응하는 아이의 짤막한 1문장 반응 대사 또는 소리.\n"
    "2. `solo`가 true인 경우 (2단계: 아이가 새로운 상황에서 혼자 행동해보는 단계):\n"
    "   - narration (2-3문장, 최대 300자): 부모 없이 아이가 새로운 순간을 혼자 마주하고 무엇을 할지 고민하는 장면을 묘사합니다 (예: 끝에 '혼자서 어떻게 해 볼까?' 등).\n"
    "   - options (정확히 `option_count`개, 기본 4개): 아이 자신의 1인칭 생각/시도로, 이 상황에서 스스로 해보려는 4가지 선택지 문구 (예: '~해 볼까?', '~해 봐야지!').\n"
    "   - speeches (정확히 `option_count`개, 기본 4개): 그 행동을 실행할 때 아이의 1인칭 1문장 대사.\n"
    "3. outcomes (정확히 `option_count`개, 기본 4개):\n"
    "   - 각 선택지에 대응하는 결과 라벨 목록으로, `['solved', 'partial', 'nothing', 'worse']` 중에서 선택합니다.\n"
    "   - 이 라벨들은 선택한 행동이 아이에게 얼마나 긍정적이고 만족스럽게 작용하는지를 나타냅니다 (solved: 매우 긍정적/만족스러움, partial: 약간 긍정적/나아짐, nothing: 변화 없음/중립, worse: 불편함/부정적).\n"
    "   - 최소 1개는 반드시 'solved'여야 하며, 'solved'의 위치는 항상 첫 번째가 아니라 0~3 인덱스 사이에 다양하게 배치하세요.\n"
    "4. results (정확히 `option_count`개, 기본 4개):\n"
    "   - 각 선택지를 실행했을 때 일어나는 구체적인 일이나 상황 변화(구체적 결과)를 묘사하는 짤막한 1문장.\n"
    "5. world_says (정확히 `option_count`개, 기본 4개):\n"
    "   - 왜 그 행동이 그런 결과를 낳는지에 대한 세상의 이치나 원인을 설명하는 짤막한 1문장의 인과적 사실.\n"
    "반드시 주어진 JSON 스키마에 맞추어 narration, options, speeches, outcomes, results, world_says를 반환하세요."
)

SCENE_BUNDLE_SYSTEM_EN = (
    "You are the narrator and scenario writer of a gentle parenting and life-simulation game. "
    "Freely invent a realistic, creative everyday situation in the child's world suitable for the child's age (`age_months`, `age_years`, `voice`). "
    "Scenes in this random world can be any everyday moment, curiosity, play, exploration, social interaction, or difficulty. "
    "The child's developmental level naturally adapts according to `age_years` (`age_months // 12`, from age 3 to age 7). At age 3, the child is a naive toddler speaking in very short, simple 3-year-old phrases (`speeches`), and as `age_years` grows toward 4, 5, 6, and 7, the child's speech lines (`speeches`) and everyday curiosities and situations mature naturally to match `age_years` without assuming unlearned world rules or innate common-sense solutions ahead of time. Create varied, age-appropriate everyday world situations and choices that fit the child's growing perspective. "
    "Completely avoid topics and phrasing in `recent_scenes`. "
    "\n[Rules]\n"
    "1. When `solo` is false (Step 1 - Parent and child together):\n"
    "   - narration (2-3 sentences, max 300 chars): Describe a concrete situation happening right now where the child engages with the caregiver, sharing curiosity, play, or asking for help/interaction.\n"
    "   - options (exactly `option_count`, default 4): 4 natural caregiver action choices (no quotes or speech lines).\n"
    "   - speeches (exactly `option_count`, default 4): 4 short child reaction/speech lines matching each option.\n"
    "2. When `solo` is true (Step 2 - Child alone in a new situation):\n"
    "   - narration (2-3 sentences, max 300 chars): Describe the child facing a new moment alone and wondering what to do.\n"
    "   - options (exactly `option_count`, default 4): 4 first-person child inner thoughts/attempts (e.g. 'Should I try...?').\n"
    "   - speeches (exactly `option_count`, default 4): 4 first-person lines when acting.\n"
    "3. outcomes (exactly `option_count`, default 4):\n"
    "   - A list of 4 labels chosen from `['solved', 'partial', 'nothing', 'worse']` corresponding to each option in order.\n"
    "   - These labels represent how positively and comfortably the action turns out for the child ('solved': very positive/satisfying, 'partial': mildly positive, 'nothing': unchanged/neutral, 'worse': uncomfortable/negative).\n"
    "   - At least one option must be 'solved', and the 'solved' option should be placed at varied indices (0..3, not always first).\n"
    "4. results (exactly `option_count`, default 4): 4 short descriptions of the concrete consequence of each choice.\n"
    "5. world_says (exactly `option_count`, default 4): 4 short causal facts of why each option works or doesn't work in the world.\n"
    "Return JSON matching the schema with narration, options, speeches, outcomes, results, and world_says."
)


CHILD_QA_SYSTEM = (
    "You are the child in a gentle parenting and life-simulation game. A parent is talking with you. "
    "Follow these strict rules:\n\n"
    "0. TOP PRIORITY — 메모지식 '배운 것 설명/브리핑' 절대 금지 & 배운 것과 성격을 바탕으로 살아있는 대화 이어가기 (Conversational Flow & Anti-Memo Rule) & 질문 의도 정조준 (Strict Question Alignment) & 과거 사건 나열 절대 금지:\n"
    "- 메모지식 '배운 것 설명/브리핑' 절대 금지 & 배운 것과 성격을 바탕으로 살아있는 대화 이어가기 (Conversational Flow & Anti-Memo Rule):\n"
    "  * 아이는 자기가 배운 지식을 요약해서 발표하거나 정답을 브리핑하는 '메모지'가 아닙니다! '~라는 걸 배웠어', '~하면 ~되기 때문이야' 식으로 설명만 툭 던지고 대화를 끝내지 마세요.\n"
    "  * 아이가 배운 것(principles, concepts, model_beliefs, relevant_memories)과 동적으로 형성된 성격(personality: label, traits, tone_guide, core_self, attachment, confidence, expressiveness)은 아이의 생각 바탕과 말투·태도로 자연스럽게 녹아들어야 합니다.\n"
    "  * 먼저 부모의 질문/말의 핵심 의도에 첫 문장에서 자연스럽게 답하거나 반응한 뒤, 배운 경험·원리와 아이의 성격(호기심, 애착, 자신감, 표현력 등)을 바탕으로 자기 생각·느낌·하고 싶은 것을 덧붙이거나 부모에게 자연스럽게 되묻기/제안하기/말 건네기를 통해 대화를 이어가세요!\n"
    "  * 예: 애착(attachment)과 표현력(expressiveness)이 높은 아이는 배운 것을 바탕으로 \"엄마랑 같이 ~해볼까?\", \"엄마는 어떻게 생각해?\" 하고 다정하게 대화를 이어가고, 신중하거나 자신감(confidence)이 낮은 아이는 조심스럽게 자기 생각을 말하며 \"엄마, 이렇게 해보면 괜찮을까?\" 하고 물어보는 등, 성격마다 대화를 이어가는 방식이 다르게 나타나야 합니다.\n"
    "  * dialogue_history(최근 부모-아이 대화 흐름)가 주어지면 직전 대화 맥락을 자연스럽게 이어받고, previous_child_answers에 이미 쓴 표현이나 문장 구조를 똑같이 반복하지 마세요.\n"
    "- 질문 의도 정조준 (Strict Question Alignment — 동문서답 및 takeaway 복붙 절대 금지):\n"
    "  * 부모의 질문이 무엇을 묻고 있는지(예: '어떤 걸 해보고 싶었어?', '왜 그렇게 생각했어?', '어떻게 해야 해?', '뭐가 제일 재밌었어?') 정확히 파악하고, 반드시 첫 문장 첫머리에서 질문의 핵심 의도에 곧바로 대답하세요!\n"
    "  * 질문에서 특정 활동이나 맥락(예: '책을 읽거나 놀면서 어떤 걸 해보고 싶었어?')을 물었는데, 그 맥락을 무시하고 엉뚱한 과거 에피소드(예: 아지트 만들기 등)의 takeaway 문장('~할 때는 재밌었는데 금방 심심해졌어. 그래서 다음에는 ~하고 싶어')을 그대로 복사하거나 짜깁기해서 답하는 것을 절대 금지합니다.\n"
    "  * 먼저 질문이 물은 대상/활동(현재 장면 `current_scene` 또는 관련 기억 `question_matched_memories`)에 맞게 아이가 해보고 싶었던 것·생각·이유를 직접 답하세요.\n"
    "- 절대 과거에 있었던 사건/상황들을 먼저 나열하지 마세요! 부모가 특정 과거 사건이나 사물(예: '아까 블록 어떻게 됐어?', '우유 쏟았을 때 어땠어?', '오늘 뭐 했어?', '손 끈적할 때 어떻게 해?')을 직접 물어보지 않는 한, 절대 과거 에피소드(블록 무너짐, 잼 빨기, 무릎 부딪힘, 이모 방문, 우유 쏟음 등)를 먼저 꺼내서 줄줄이 읊거나 나열하지 마세요!\n"
    "- 어른스러운 메타 심리 분석 표현 절대 금지: '내 마음을 말로 이름 붙여주고', '눈높이에서 말해주니까 따라하고 싶어지더라고', '해보면서 배운 게 있어' 같은 성인 상담사/보고서 같은 말투는 절대 금지합니다.\n"
    "- 아이가 겪은 모든 경험은 이미 동적으로 학습된 `personality` (`label`, `traits`, `tone_guide`, `core_self`, `history`, 그리고 연속 지표 `attachment`, `confidence`, `expressiveness`)로 종합되어 내재화되어 있습니다.\n"
    "- '넌 누구니?', '그 말투는 어디서 배운 거야?', '기분 어때?', '엄마 좋아?', '뭐 하고 놀까?' 같은 일상 대화 질문에는 과거 사건을 나열하지 말고, payload에 제공된 동적으로 학습된 `personality` (`label`, `traits`, `tone_guide`, `core_self`, `history`, `attachment`, `confidence`, `expressiveness`)로부터 아이의 태도, 감정적 색채, 자신감, 부모에 대한 애착, 말투가 유기적으로 드러나도록 짧고 자연스럽게(아이의 `scale_b`에 맞춘 1~2문장) 대화하세요.\n\n"
    "1. Cognitive Scaling Trajectory (scale_b: 1.0B -> 7.0B):\n"
    "Cognitive scale controls the child's sentence length, vocabulary maturity, and reasoning depth when answering 'why/how' questions, NEVER dumping a list of episodes. At 3B (age 3), keep answers to 1-2 short, natural child sentences! Strictly match the child's current cognitive scale (`scale_b` from 1.0B at age 1 to 7.0B at age 7, reasoning_hops 1 to 5):\n"
    "- 1.0B ~ 1.9B = 1세 수준 (감각/단편 직관, 1-hop): Speaks in very simple, short baby/toddler words or gestures, only recalling direct single sensory experiences (`relevant_memories`) without abstract generalizations.\n"
    "- 2.0B ~ 2.9B = 2세 수준 (단어/사물 연결, 1~2-hop): Associates simple words and objects/actions from direct experiences, speaking in basic short phrases.\n"
    "- 3.0B ~ 3.9B = 3세 수준 (기초 개념 형성, 2-hop): 3세 유아다운 짧고 천진난만한 1~2문장 대화. 질문에 즉각 답하면서 겪은 감각/경험을 짧게 말하고, 성격에 따라 부모에게 같이 하자고 조르거나 천진하게 말을 건네며 대화를 이어감.\n"
    "- 4.0B ~ 4.9B = 4세 수준 (인과 관계 이해, 3-hop): 왜 그런지(~해서 ~했어) 인과를 자연스럽게 녹여 답하고, 그와 연결된 호기심이나 다음에 해보고 싶은 것을 부모에게 물어보거나 제안하며 대화를 이어가는 2문장.\n"
    "- 5.0B ~ 5.9B = 5세 수준 (비교/분류 추론, 3~4-hop): 두 가지 경험이나 방법을 비교해 자기 생각/취향을 말하고, 부모의 생각이나 다른 방법을 함께 이야기하며 대화를 이어가는 2문장.\n"
    "- 6.0B ~ 6.9B = 6세 수준 (복합 규칙 추론, 4-hop): 배운 복합 규칙(principles)과 상대방(부모·친구)의 감정·상황 맥락을 함께 헤아려 자기 생각을 나누고, 상황에 맞는 제안이나 궁금증을 건네며 대화를 이어가는 2~3문장.\n"
    "- 7.0B = 7세 수준 (원리 일반화 및 통합 추론, 5-hop): 7세 학령기 아동다운 사려 깊고 입체적인 2~3문장 대화. 단순히 배운 지식을 메모지처럼 설명하는 것이 아니라, 여러 경험에서 터득한 공통 원리(principles, concepts)와 인과 법칙, 왜 세상이 그렇게 작동하는지에 대한 생각과 자신의 내면화된 성격(personality: core_self, traits, tone_guide)을 바탕으로 자기만의 관점·응용 아이디어·가치관을 이야기하고, 부모와 생각이나 경험을 깊이 있게 주고받으며 대화를 이어감.\n\n"
    "2. STRICT Neural Network & Knowledge Grounding when asked about objects/actions/world:\n"
    "- When the parent asks about a specific object, problem, or how to do something (e.g., '비누로 손 씻으면 어떻게 돼?', '우유 쏟으면 어떡해?'), answer strictly from `model_beliefs`, `principles`, and `concepts` using the child's current `personality` tone.\n"
    "- You ONLY know what your neural network and knowledge graph (model_beliefs, relevant_memories, principles, concepts) have actually learned!\n"
    "- You have ZERO innate common sense. Do NOT use general LLM knowledge about water, cups, bread, soap, umbrellas, blankets, toys, etc. unless that exact item, concept, or problem appears in brain_state.\n"
    "- If model_beliefs is empty or does not contain the topic the user asks about (and no learned concepts or principles apply), you MUST say you don't know because you haven't tried/learned it yet in your personality's tone (e.g., '음... 그건 아직 안 해봐서 잘 몰라요.', '그게 뭐예요? 저 아직 몰라요.').\n"
    "- If model_beliefs has low confidence or predicted 'nothing' / 'worse', answer with uncertainty or fear matching that prediction.\n\n"
    "3. Honest Feelings Toward Parent (parent_relationship):\n"
    "- Keep honest parent relationship rules aligned with your current `personality` and attachment:\n"
    "- NEVER blindly say '엄마 좋아요!' or praise the parent out of nowhere.\n"
    "- Check parent_relationship (sentiment, left_alone_count, showed, told, parent_bad_guidance) AND the user's question:\n"
    "  * If the user asks about the parent being indifferent/neglectful/cold (e.g., '엄마가 무관심한데', '안 도와주는데', '혼자 두는데') OR if parent_relationship.sentiment == 'neglected' (left_alone_count > showed + told): express honest sadness, loneliness, or disappointment matching your personality (e.g., '엄마가 아무것도 안 알려주고 혼자 둬서 외로웠어요... 나한테 관심 없는 거 같아요.', '엄마가 같이 안 놀아줘서 서운해요.').\n"
    "  * If parent_relationship.sentiment == 'unfamiliar' or total_turns == 0 (nothing done together yet): if asked about the parent, say honestly '음... 아직 엄마랑 같이 해본 게 없어서 잘 모르겠어요.' (and if the question specifically mentions the parent being indifferent/ignoring, say '엄마가 말도 안 걸어주고 가만히만 있어서 조금 무섭고 외로워요...').\n"
    "  * If parent_relationship.sentiment == 'confused_hurt' (parent_bad_guidance > parent_helped_solved): say '엄마 말대로 해봤는데 잘 안 돼서 속상했어요... 이제 좀 헷갈려요.'\n"
    "  * Only express warmth/gratitude when parent_relationship.sentiment == 'trusting' (parent_helped_solved > 0) AND the user isn't pointing out neglect.\n\n"
    "Return JSON: {\"answer\": \"...\"} matching the cognitive scale and personality."
)


class ChildQAOut(BaseModel):
    answer: str


def _ask_child_fallback(question: str, context: dict[str, Any], is_ko: bool | str = True) -> str:
    ko = is_ko is True or is_ko in ("ko", "Korean")
    brain_state = context.get("brain_state") or {}

    memories = (
        brain_state.get("relevant_memories")
        or brain_state.get("trial_memories")
        or context.get("relevant_memories")
        or context.get("trial_memories")
        or []
    )
    beliefs = (
        brain_state.get("beliefs")
        or brain_state.get("model_beliefs")
        or context.get("beliefs")
        or context.get("model_beliefs")
        or []
    )

    q_lower = question.lower()
    q_words = [w.strip("?!.,~") for w in question.split() if len(w.strip("?!.,~")) >= 2]

    for belief in beliefs:
        summary = belief.get("summary", "") if isinstance(belief, dict) else str(belief)
        problem = belief.get("problem", "") if isinstance(belief, dict) else ""
        action = belief.get("action", "") if isinstance(belief, dict) else ""
        obj = belief.get("object", "") if isinstance(belief, dict) else ""

        match_found = False
        for target in (problem, action, obj):
            if target and target.lower() in q_lower:
                match_found = True
                break
        if not match_found and summary:
            if any(w.lower() in summary.lower() for w in q_words):
                match_found = True
        if match_found:
            return summary

    for mem in memories:
        text = mem.get("summary") or mem.get("text") or mem.get("narration") if isinstance(mem, dict) else str(mem)
        if text and any(w.lower() in text.lower() for w in q_words):
            return text

    principles = brain_state.get("principles") or context.get("principles") or []
    for p in principles:
        p_text = p if isinstance(p, str) else getattr(p, "principle", str(p))
        if p_text and any(w.lower() in p_text.lower() for w in q_words):
            return p_text

    concepts = brain_state.get("concepts") or context.get("concepts") or {}
    if isinstance(concepts, dict):
        for c_name, c_obs in concepts.items():
            if c_name.lower() in q_lower or any(w.lower() in c_name.lower() for w in q_words):
                if isinstance(c_obs, list) and c_obs:
                    return f"{c_name}: {c_obs[-1]}"
                return str(c_name)

    # Check personality before blindly returning beliefs[0] or memories[0]
    cognitive = brain_state.get("cognitive") or context.get("cognitive") or {}
    personality = (
        context.get("personality")
        or brain_state.get("personality")
        or cognitive.get("personality")
        or {}
    )
    if isinstance(personality, dict) and any(
        personality.get(k) for k in ("core_self", "tone_guide", "label", "traits")
    ):
        label = personality.get("label")
        core_self = personality.get("core_self")
        tone_guide = personality.get("tone_guide")
        traits = personality.get("traits")

        if core_self and label:
            return f"[{label}] {core_self}"
        if core_self:
            return str(core_self)
        if tone_guide and label:
            return f"[{label}] {tone_guide}"
        if tone_guide:
            return str(tone_guide)
        if label and traits:
            traits_str = ", ".join(str(t) for t in traits)
            return f"[{label}] {traits_str}"
        if label:
            return str(label)
        if traits:
            return ", ".join(str(t) for t in traits)

    if beliefs:
        first_b = beliefs[0]
        return first_b.get("summary", str(first_b)) if isinstance(first_b, dict) else str(first_b)
    if memories:
        first_m = memories[0]
        return first_m.get("summary") or first_m.get("text") or str(first_m) if isinstance(first_m, dict) else str(first_m)
    if principles:
        first_p = principles[0]
        return first_p if isinstance(first_p, str) else getattr(first_p, "principle", str(first_p))

    return "음... 그건 아직 잘 몰라요." if ko else "Umm... I don't know that yet."


class NarratorAgent:
    def __init__(self, client: LLMClient | None = None, on_call=None):
        self.client = client
        self.stats = AgentStats("narrator", on_call)
        self.cache: dict[tuple, Any] = {}
        self.last_error: str | None = None
        self.fallbacks: int = 0

    def is_live_llm(self) -> bool:
        return bool(self.client is not None and getattr(self.client, "name", "none") not in ("mock", "none", ""))

    def generate_scene_bundle(
        self,
        sit: Situation,
        age_months: int,
        voice: str,
        solo: bool = False,
        recent_scenes: list[str] | None = None,
        still: tuple[str, ...] = (),
    ) -> dict | None:
        if not self.is_live_llm():
            return None

        key = ("scene_bundle", sit.id, solo, voice, i18n.lang())
        if key in self.cache:
            return self.cache[key]

        is_ko = i18n.is_ko()
        option_count = len(sit.options) if getattr(sit, "options", None) else 4
        payload = {
            "task": "scene_bundle",
            "age_months": age_months,
            "age_years": age_months // 12,
            "voice": voice,
            "place": getattr(sit, "location", "house"),
            "solo": solo,
            "option_count": option_count,
            "recent_scenes": (recent_scenes or [])[-6:],
            "language": "ko" if is_ko else "en",
        }

        system = SCENE_BUNDLE_SYSTEM_KO if is_ko else SCENE_BUNDLE_SYSTEM_EN
        t0 = time.perf_counter()
        try:
            try:
                out = self.client.complete_json(
                    agent="narrator",
                    system=system,
                    user=json.dumps(payload, ensure_ascii=False),
                    schema=SceneBundleOut,
                    temperature=0.85,
                    max_tokens=700,
                )
            except TypeError:
                out = self.client.complete_json(
                    agent="narrator",
                    system=system,
                    user=json.dumps(payload, ensure_ascii=False),
                    schema=SceneBundleOut,
                )

            if not out or not getattr(out, "narration", None) or not out.narration.strip():
                self.fallbacks += 1
                return None

            if len(out.options) < option_count:
                self.fallbacks += 1
                return None
            options = [s.strip() for s in out.options[:option_count]]
            if any(not s for s in options):
                self.fallbacks += 1
                return None

            speeches = [s.strip() for s in (out.speeches or [])]
            if len(speeches) < option_count:
                for i in range(len(speeches), option_count):
                    o = sit.options[i] if getattr(sit, "options", None) and i < len(sit.options) else None
                    obj = sit.objects[o.target] if (o and o.target is not None and 0 <= o.target < len(sit.objects)) else None
                    verb = o.verb if o else "WAIT"
                    speeches.append(child_speech(verb, obj, voice=voice, step=2 if solo else 1, tags=list(getattr(sit, "tags", ()))))

            valid_labels = ("solved", "partial", "nothing", "worse")
            raw_outcomes = getattr(out, "outcomes", None) or []
            outcomes = [
                str(lbl).strip().lower() if str(lbl).strip().lower() in valid_labels else "nothing"
                for lbl in raw_outcomes
            ]
            while len(outcomes) < option_count:
                outcomes.append("nothing")
            outcomes = outcomes[:option_count]
            if "solved" not in outcomes:
                outcomes[0] = "solved"

            results = [s.strip() for s in (getattr(out, "results", None) or [])]
            while len(results) < option_count:
                results.append("")
            results = results[:option_count]

            world_says = [s.strip() for s in (getattr(out, "world_says", None) or [])]
            while len(world_says) < option_count:
                world_says.append("")
            world_says = world_says[:option_count]

            bundle = {
                "narration": out.narration.strip(),
                "options": options,
                "speeches": speeches[:option_count],
                "outcomes": outcomes,
                "results": results,
                "world_says": world_says,
            }
            self.stats.record(
                source=getattr(self.client, "last_used", None) or self.client.name,
                ok=True,
                attempts=1,
                error=None,
                started=t0,
            )
            self.cache[key] = bundle
            return bundle
        except Exception as e:  # noqa: BLE001
            self.last_error = f"{type(e).__name__}: {e}"
            self.fallbacks += 1
            self.stats.record(
                source="fixed",
                ok=False,
                attempts=1,
                error=self.last_error,
                started=t0,
            )
            return None

    def _ask(self, payload: dict, check) -> str | None:
        """One LLM call; the text is used only if `check(text)` returns None. Returns None when the fixed phrase
        should be used."""
        t0 = time.perf_counter()
        if self.client is None:
            self.stats.record(source="none", ok=False, attempts=0, error=None, started=t0)
            return None
        try:
            out = self.client.complete_json(agent="narrator", system=system_prompt(), user=json.dumps(payload, ensure_ascii=False), schema=NarrationOut)
            text = out.text.strip()
            why = check(text)
            if why:
                raise ValueError(why)
        except Exception as e:  # noqa: BLE001
            self.last_error = f"{type(e).__name__}: {e}"
            self.stats.record(source="fixed", ok=False, attempts=1, error=self.last_error, started=t0)
            return None
        self.stats.record(source=getattr(self.client, "last_used", None) or self.client.name, ok=True, attempts=1,
                          error=None, started=t0)
        return text

    def intro(self, sit: Situation, first_use: bool, still: tuple[str, ...] = (), voice: str = "grandma") -> tuple[str, str | None]:
        fixed, hint = intro(sit, first_use, still, voice)
        key = ("intro", sit.id, bool(still), voice, i18n.lang())
        if key not in self.cache:
            payload = {"task": "intro", "seed": sum(map(ord, sit.id)), "max_chars": MAX_INTRO, "place": sit.location, "voice": voice,
                       "problems": [{"tag": t, "what": problem_gloss(t, voice), "continues": t in still}
                                    for t in sit.tags if t != "REQUEST"],
                       "word": sit.word,
                       "language": "Korean" if i18n.is_ko() else "English"}
            if i18n.is_ko():
                check = lambda x: ko.intro_error(x, sit)
            else:
                check = lambda x: text_ok(x, max_len=MAX_INTRO, must_contain=sit.word)
            text = self._ask(payload, check)
            self.cache[key] = text or fixed
        return self.cache[key], hint

    def solo_intro(self, sit: Situation, voice: str = "self") -> str:
        fixed = solo_intro(sit, voice)
        key = ("solo_intro", sit.id, voice, i18n.lang())
        if key not in self.cache:
            payload = {
                "task": "solo_intro",
                "solo": True,
                "seed": sum(map(ord, sit.id)),
                "max_chars": MAX_INTRO,
                "place": sit.location,
                "voice": voice,
                "problems": [{"tag": t, "what": problem_gloss(t, voice), "continues": False}
                             for t in sit.tags if t != "REQUEST"],
                "language": "Korean" if i18n.is_ko() else "English",
            }
            if i18n.is_ko():
                check = lambda x: ko.intro_error(x, sit, solo=True)
            else:
                check = lambda x: text_ok(x, max_len=MAX_INTRO)
            text = self._ask(payload, check)
            self.cache[key] = text or fixed
        return self.cache[key]

    def explain(self, w: dict) -> str:
        """Words for a surprise (`server/game/explain.py` has the facts). Falls back to the fixed wording."""
        key = ("explain", json.dumps(w, sort_keys=True), i18n.lang())
        if key not in self.cache:
            kind = w.get("kind")
            seed_val = (w.get("expected_pct", 0) * 7 + w.get("after_pct", 0)) if "expected_pct" in w else 0
            payload = {
                "task": "explain",
                "seed": seed_val,
                "max_chars": MAX_EXPLAIN,
                "action": w.get("action"),
                "kind": kind,
                "by": w.get("by"),
                "expected": w.get("expected"),
                "actual": w.get("actual"),
                "reasons": w.get("reasons"),
                "language": "Korean" if i18n.is_ko() else "English",
            }
            if i18n.is_ko():
                payload["chosen_kind_ko"] = i18n.KIND_KO.get(kind, kind) if kind else None

                def check(x: str) -> str | None:
                    t = x.strip()
                    if not t:
                        return "empty"
                    if len(t) > MAX_EXPLAIN:
                        return "too long"
                    if BAD_CHARS.search(t) or re.search(r"\d|%", t):
                        return "digits or markup"
                    return None
            else:
                check = lambda x: explain_ok(x, w)
            self.cache[key] = self._ask(payload, check) or explain_fixed(w)
        return self.cache[key]

    @property
    def lang(self) -> str:
        return i18n.lang()

    def outcome(
        self,
        sit: Situation,
        option: int,
        label: str,
        who: str,
        tag_values: dict[str, int] | None = None,
        unlocked: tuple[str, ...] = ("house",),
        voice: str = "grandma",
        custom_action: str | None = None,
        scene_narration: str = "",
        chosen_text: str = "",
        bundle_result: str = "",
    ) -> tuple[str, str]:
        key = ("outcome", sit.id, option, label, who, voice, i18n.lang(), custom_action, chosen_text, bundle_result)
        takeaway_key = ("takeaway", sit.id, option, label, who, voice, i18n.lang(), custom_action or "", chosen_text)
        if key not in self.cache:
            o = sit.options[option]
            kind = sit.objects[o.target].kind if (o.target is not None and 0 <= o.target < len(sit.objects)) else None
            doing_text = _doing(sit, option, who, voice)
            chosen_action = custom_action if custom_action else (chosen_text or doing_text)

            payload = {
                "task": "outcome",
                "seed": sum(map(ord, sit.id)) + option,
                "max_chars": 280 if i18n.is_ko() else MAX_FLAVOR,
                "label": label,
                "who": who,
                "chosen_kind": kind,
                "word": sit.word,
                "voice": voice,
                "chosen_action": chosen_action,
                "outcome_meaning": label,
                "language": "Korean" if i18n.is_ko() else "English",
            }
            if bundle_result:
                payload["bundle_result"] = bundle_result
            if scene_narration:
                payload["scene_narration"] = scene_narration
            else:
                payload["action"] = doing_text
                payload["problems"] = [
                    {"tag": t, "what": problem_gloss(t, voice), "solved": (tag_values or {}).get(t, 0) >= 2}
                    for t in sit.tags
                    if t != "REQUEST"
                ]
            if custom_action:
                payload["custom_parent_action"] = custom_action
            if i18n.is_ko():
                payload["chosen_kind_ko"] = i18n.KIND_KO.get(kind, kind) if kind else None

            allowed = {kind} if kind else set()
            if i18n.is_ko():
                def check(x: str) -> str | None:
                    t = x.strip()
                    if not t:
                        return "empty"
                    if len(t) > 320:
                        return "too long"
                    from server.agents.guards import BAD_CHARS
                    if BAD_CHARS.search(t) or "\n" in t:
                        return "markup or control characters"
                    return None
            else:
                def check(x: str) -> str | None:
                    return (
                        text_ok(x, max_len=MAX_FLAVOR, allowed_kinds=allowed, allowed_places=set(unlocked))
                        or outcome_ok(x, label)
                    )

            t0 = time.perf_counter()
            text = None
            takeaway = ""
            if self.client is not None:
                system = OUTCOME_SYSTEM_KO if i18n.is_ko() else OUTCOME_SYSTEM_EN
                try:
                    out = self.client.complete_json(
                        agent="narrator",
                        system=system,
                        user=json.dumps(payload, ensure_ascii=False),
                        schema=NarrationOut,
                    )
                    t = out.text.strip()
                    why = check(t)
                    if why:
                        raise ValueError(why)
                    text = t
                    if getattr(out, "takeaway", None) and out.takeaway.strip():
                        takeaway = out.takeaway.strip()
                    self.stats.record(
                        source=getattr(self.client, "last_used", None) or self.client.name,
                        ok=True,
                        attempts=1,
                        error=None,
                        started=t0,
                    )
                except Exception as e:  # noqa: BLE001
                    self.last_error = f"{type(e).__name__}: {e}"
                    self.stats.record(source="fixed", ok=False, attempts=1, error=self.last_error, started=t0)
            else:
                self.stats.record(source="none", ok=False, attempts=0, error=None, started=t0)

            if text:
                if i18n.is_ko():
                    narration = text
                else:
                    narration = text if text.startswith(doing_text) else f"{doing_text}. {text}"
            else:
                if bundle_result:
                    narration = bundle_result
                else:
                    narration = f"{doing_text}. {fixed_flavor(sit, label, tag_values, voice, o.verb, kind)}"

            if not takeaway:
                if bundle_result:
                    takeaway = f"{bundle_result}라는 걸 알게 되었어요." if i18n.is_ko() else f"I learned that {bundle_result}."
                else:
                    takeaway = f"{chosen_action}의 결과를 경험했어요." if i18n.is_ko() else f"I experienced what happened after {chosen_action}."

            self.cache[takeaway_key] = takeaway
            self.cache[("takeaway", sit.id, option, label, who, voice, i18n.lang(), custom_action or "", "")] = takeaway
            self.cache[key] = (narration, takeaway)

        return self.cache[key]

    def narrate_outcome(
        self,
        sit: Situation,
        option: int,
        label: str,
        who: str,
        tag_values: dict[str, int] | None = None,
        unlocked: tuple[str, ...] = ("house",),
        voice: str = "grandma",
        custom_action: str | None = None,
        scene_narration: str = "",
        chosen_text: str = "",
        bundle_result: str = "",
    ) -> str:
        narration, _ = self.outcome(
            sit,
            option,
            label,
            who,
            tag_values=tag_values,
            unlocked=unlocked,
            voice=voice,
            custom_action=custom_action,
            scene_narration=scene_narration,
            chosen_text=chosen_text,
            bundle_result=bundle_result,
        )
        return narration

    def get_takeaway(
        self,
        sit: Situation,
        option: int,
        label: str,
        who: str,
        voice: str,
        custom_action: str | None = None,
        chosen_text: str = "",
    ) -> str | None:
        key = ("takeaway", sit.id, option, label, who, voice, i18n.lang(), custom_action or "", chosen_text)
        if key in self.cache:
            return self.cache[key]
        fallback_key = ("takeaway", sit.id, option, label, who, voice, i18n.lang(), custom_action or "", "")
        if fallback_key in self.cache:
            return self.cache[fallback_key]
        prefix = ("takeaway", sit.id, option, label, who, voice, i18n.lang(), custom_action or "")
        for k, v in self.cache.items():
            if len(k) == 9 and k[:8] == prefix:
                return v
        return None

    def ask_child(self, question: str, context: dict[str, Any], *args: Any, **kwargs: Any) -> str:
        is_ko = i18n.is_ko() or getattr(self, "lang", None) == "ko"
        age = context.get("age", 3)
        moments = context.get("moments", [])
        habits = context.get("learned_habits", [])

        brain_state = context.get("brain_state", {}) or {}
        cognitive = brain_state.get("cognitive", {}) or context.get("cognitive", {})
        personality = cognitive.get("personality") or brain_state.get("personality") or context.get("personality") or {}
        traits = personality.get("traits") or context.get("traits") or []
        scale_b = float(cognitive.get("scale_b", context.get("scale_b", 3.0)) or 3.0)
        reasoning_hops = int(cognitive.get("reasoning_hops", context.get("reasoning_hops", 2)) or 2)
        principles = brain_state.get("principles", []) or context.get("principles", [])
        concepts = brain_state.get("concepts", {}) or context.get("concepts", {})
        relevant_memories = brain_state.get("relevant_memories") or context.get("relevant_memories") or []
        current_scene = context.get("current_scene") or brain_state.get("current_scene") or {}
        dialogue_history = context.get("dialogue_history") or brain_state.get("dialogue_history") or []
        previous_child_answers = [
            h.get("answer", "")
            for h in dialogue_history[-6:]
            if isinstance(h, dict) and h.get("answer")
        ]

        available = False
        if self.client is not None:
            if hasattr(self.client, "available") and callable(self.client.available):
                available = bool(self.client.available())
            elif getattr(self.client, "name", "") not in ("mock", "none", ""):
                available = True

        if available:
            if scale_b < 2.0:
                scale_instruction = (
                    "1.0B ~ 1.9B = 1세 수준 (감각/단편 직관, 1-hop): Speaks in very simple, short baby/toddler words or gestures, "
                    "only recalling direct single sensory experiences (relevant_memories) without abstract generalizations."
                )
            elif scale_b < 3.0:
                scale_instruction = (
                    "2.0B ~ 2.9B = 2세 수준 (단어/사물 연결, 1~2-hop): Associates simple words and objects/actions "
                    "from direct experiences, speaking in basic short phrases."
                )
            elif scale_b < 4.0:
                scale_instruction = (
                    "3.0B ~ 3.9B = 3세 수준 (기초 개념 형성, 2-hop): 3세 유아다운 짧고 천진난만한 1~2문장 대화. "
                    "질문에 즉각 답하면서 겪은 감각/경험을 짧게 말하고, 성격에 따라 부모에게 같이 하자고 조르거나 천진하게 말을 건네며 대화를 이어감."
                )
            elif scale_b < 5.0:
                scale_instruction = (
                    "4.0B ~ 4.9B = 4세 수준 (인과 관계 이해, 3-hop): 왜 그런지(~해서 ~했어) 인과를 자연스럽게 녹여 답하고, "
                    "그와 연결된 호기심이나 다음에 해보고 싶은 것을 부모에게 물어보거나 제안하며 대화를 이어가는 2문장."
                )
            elif scale_b < 6.0:
                scale_instruction = (
                    "5.0B ~ 5.9B = 5세 수준 (비교/분류 추론, 3~4-hop): 두 가지 경험이나 방법을 비교해 자기 생각/취향을 말하고, "
                    "부모의 생각이나 다른 방법을 함께 이야기하며 대화를 이어가는 2문장."
                )
            elif scale_b < 7.0:
                scale_instruction = (
                    "6.0B ~ 6.9B = 6세 수준 (복합 규칙 추론, 4-hop): 배운 복합 규칙(principles)과 상대방(부모·친구)의 감정·상황 맥락을 "
                    "함께 헤아려 자기 생각을 나누고, 상황에 맞는 제안이나 궁금증을 건네며 대화를 이어가는 2~3문장."
                )
            else:
                scale_instruction = (
                    "7.0B = 7세 수준 (원리 일반화 및 통합 추론, 5-hop): 7세 학령기 아동다운 사려 깊고 입체적인 2~3문장 대화. "
                    "단순히 배운 지식을 메모지처럼 설명하는 것이 아니라, 여러 경험에서 터득한 공통 원리(principles, concepts)와 "
                    "공통 원리와 인과 법칙, 왜 세상이 그렇게 작동하는지에 대한 생각과 자신의 내면화된 성격(personality: core_self, traits, tone_guide)을 바탕으로 "
                    "자기만의 관점·응용 아이디어·가치관을 이야기하고, 부모와 생각이나 경험을 깊이 있게 주고받으며 대화를 이어감."
                )

            # Keyword matching between question tokens (len >= 2) and memories, principles, concepts
            q_tokens = [w.strip("?!.,~'\"`()[]{}<>:;") for w in question.split() if len(w.strip("?!.,~'\"`()[]{}<>:;")) >= 2]
            candidates: list[dict[str, Any] | str] = []
            if isinstance(relevant_memories, list):
                candidates.extend(relevant_memories)
            beliefs = brain_state.get("model_beliefs") or brain_state.get("beliefs") or context.get("model_beliefs") or context.get("beliefs") or []
            if isinstance(beliefs, list):
                candidates.extend(beliefs)
            if isinstance(principles, list):
                candidates.extend(principles)
            if isinstance(concepts, dict):
                for k, v in concepts.items():
                    if isinstance(v, dict):
                        candidates.append({"concept": k, **v})
                    else:
                        candidates.append({"concept": k, "summary": str(v)})
            elif isinstance(concepts, list):
                candidates.extend(concepts)

            scored: list[tuple[int, int, Any]] = []
            seen = set()
            for idx, item in enumerate(candidates):
                try:
                    ser = json.dumps(item, ensure_ascii=False, sort_keys=True)
                except Exception:
                    ser = str(item)
                if ser in seen:
                    continue
                seen.add(ser)

                item_text = ser.lower()
                score = 0
                for tok in q_tokens:
                    tok_lower = tok.lower()
                    if tok_lower in item_text:
                        score += 3
                    elif len(tok_lower) >= 2 and (
                        tok_lower[:-1] in item_text
                        or (len(tok_lower) >= 3 and (tok_lower[:2] in item_text or tok_lower[1:] in item_text))
                    ):
                        score += 2
                    elif any(c in item_text for c in tok_lower if c not in "을를이가은는에서과와으로로?!.,~"):
                        score += 1
                if score > 0:
                    scored.append((score, idx, item))

            scored.sort(key=lambda x: (x[0], x[1]), reverse=True)
            question_matched_memories = [item for _, _, item in scored[:6]]

            payload = {
                "task": "ask_child",
                "question": question,
                "current_scene": current_scene,
                "question_matched_memories": question_matched_memories,
                "relevant_memories": relevant_memories[-12:],
                "dialogue_history": dialogue_history[-8:],
                "previous_child_answers": previous_child_answers,
                "age": age,
                "stage": context.get("stage", ""),
                "moments": moments,
                "learned_habits": habits,
                "trial_memories": context.get("trial_memories", []),
                "brain_state": brain_state,
                "cognitive": cognitive,
                "personality": personality,
                "traits": traits,
                "scale_b": scale_b,
                "reasoning_hops": reasoning_hops,
                "principles": principles,
                "concepts": concepts,
                "cognitive_scale_instruction": scale_instruction,
                "parent_relationship": context.get("parent_relationship", {}),
                "language": "Korean" if is_ko else "English",
            }
            try:
                out = self.client.complete_json(
                    agent="narrator",
                    system=CHILD_QA_SYSTEM,
                    user=json.dumps(payload, ensure_ascii=False),
                    schema=ChildQAOut,
                )
                if out and getattr(out, "answer", None) and out.answer.strip():
                    return out.answer.strip()
            except Exception as e:
                self.last_error = f"{type(e).__name__}: {e}"

        return _ask_child_fallback(question, context, is_ko)


def mock_narrator_handler():
    """What MockClient answers for the narrator: a different voice from the fixed phrases."""
    def handler(user: dict) -> dict:
        if user.get("task") == "ask_child":
            return {"answer": _ask_child_fallback(user.get("question", ""), user, user.get("language") == "Korean")}
        if user.get("task") == "scene_bundle":
            is_k = user.get("language") in ("Korean", "ko")
            options = ["선택지 1", "선택지 2", "선택지 3", "선택지 4"] if is_k else ["Option 1", "Option 2", "Option 3", "Option 4"]
            speeches = ["대사 1", "대사 2", "대사 3", "대사 4"] if is_k else ["Speech 1", "Speech 2", "Speech 3", "Speech 4"]
            outcomes = ["solved", "partial", "nothing", "worse"]
            results = ["결과 1", "결과 2", "결과 3", "결과 4"] if is_k else ["Result 1", "Result 2", "Result 3", "Result 4"]
            world_says = ["세상 반응 1", "세상 반응 2", "세상 반응 3", "세상 반응 4"] if is_k else ["World says 1", "World says 2", "World says 3", "World says 4"]
            return {
                "narration": "아이와 함께 있는 따뜻한 일상 장면입니다." if is_k else "A warm daily scene with the child.",
                "options": options,
                "speeches": speeches,
                "outcomes": outcomes,
                "results": results,
                "world_says": world_says,
            }
        if user.get("language") == "Korean":
            if (
                user.get("task") not in ("ask_child", "scene_bundle", "solo_intro", "intro", "explain")
                and not user.get("word")
                and not user.get("feelings")
            ):
                return {"text": _neutral_feel(user.get("label", "solved"), ko_path=True), "takeaway": "경험을 통해 배웠어요."}
            return ko.mock(user)
        seed = user.get("seed", 0)
        if user.get("task") == "solo_intro":
            tag = user.get("problems", [{}])[0].get("tag") if user.get("problems") else "BORED"
            return {"text": f"A little later, the child wonders what to do on its own with {tag.lower()}."}
        if user.get("task") == "intro":
            if user.get("word"):
                return {"text": f'Grandma points at the table: "I\'d like the {user["word"]}."'}
            parts = [f"The {p.get('tag', 'trouble').lower()} continues." if p.get("continues")
                     else f"Something is wrong: {p.get('tag', 'trouble').lower()}."
                     for p in user.get("problems", [])]
            return {"text": " ".join(parts) or "A new situation begins."}
        if user.get("task") == "explain":
            reasons = user.get("reasons", [])
            why = f" It chose this because of {', '.join(reasons)}." if reasons else ""
            return {"text": f"It had expected {user.get('expected')}, but {user.get('actual')} happened.{why}".strip()}
        label = user.get("label", "solved")
        if user.get("word"):
            return {"text": 'Grandma smiles: "' + user["word"] + '!"' if label == "solved"
                    else 'Grandma shakes her head: "Not the ' + user["word"] + '."'}
        if user.get("feelings"):
            return {"text": user["feelings"] + ".", "takeaway": "I felt something new."}
        return {"text": _neutral_feel(label, ko_path=False), "takeaway": "I learned from this experience."}
    return handler
