import json
import random
import re
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from server import i18n
from server.agents.client import MockClient
from server.agents.narrator import NarrationOut, NarratorAgent, SceneBundleOut, explain_fixed, mock_narrator_handler
from server.agents import narrator_ko as ko
from server.world.generator import make_situation


@pytest.fixture(autouse=True)
def reset_lang():
    i18n.set_lang("en")
    yield
    i18n.set_lang("en")


def test_korean_intro_calls_llm_with_korean_payload():
    sit = make_situation(random.Random(1), ["FLOOR_WET"], sid="test_sit_ko_intro")
    mock_client = MagicMock()
    mock_client.name = "mock"
    mock_client.complete_json.return_value = NarrationOut(text="바닥에 물이 흥건해요!")
    n = NarratorAgent(mock_client)

    i18n.set_lang("ko")
    text_ko, hint = n.intro(sit, False)
    assert text_ko == "바닥에 물이 흥건해요!"
    mock_client.complete_json.assert_called_once()
    call_args = mock_client.complete_json.call_args[1]
    payload = json.loads(call_args["user"])
    assert payload["language"] == "Korean"
    assert payload["task"] == "intro"


def test_korean_intro_fallback_on_error():
    sit = make_situation(random.Random(1), ["FLOOR_WET"], sid="test_sit_ko_fallback")
    mock_client = MagicMock()
    mock_client.complete_json.side_effect = RuntimeError("network error")
    n = NarratorAgent(mock_client)

    i18n.set_lang("ko")
    text_ko, hint = n.intro(sit, False)
    assert re.search(r"[가-힣]", text_ko)
    assert n.last_error is not None


def test_korean_outcome_calls_llm_with_korean_payload():
    sit = make_situation(random.Random(1), ["FLOOR_WET"], sid="test_sit_ko_outcome")
    mock_client = MagicMock()
    mock_client.name = "mock"
    mock_client.complete_json.return_value = NarrationOut(text="물이 말끔히 닦였어요.")
    n = NarratorAgent(mock_client)

    i18n.set_lang("ko")
    text_ko = n.outcome(sit, 0, "solved", "player")
    assert "물이 말끔히 닦였어요." in text_ko
    mock_client.complete_json.assert_called_once()
    call_args = mock_client.complete_json.call_args[1]
    payload = json.loads(call_args["user"])
    assert payload["language"] == "Korean"
    assert payload["task"] == "outcome"
    assert "chosen_kind_ko" in payload


def test_korean_outcome_fallback_on_error():
    sit = make_situation(random.Random(1), ["FLOOR_WET"], sid="test_sit_ko_outcome_fallback")
    mock_client = MagicMock()
    mock_client.complete_json.side_effect = RuntimeError("error")
    n = NarratorAgent(mock_client)

    i18n.set_lang("ko")
    text_ko, takeaway = n.outcome(sit, 0, "solved", "player")
    assert re.search(r"[가-힣]", text_ko)
    assert n.last_error is not None


def test_korean_explain_calls_llm_with_korean_payload():
    w = {
        "kind": "cup",
        "expected": "nothing",
        "actual": "solved",
        "expected_pct": 10,
        "after_pct": 80,
        "reasons": ["chance"],
    }
    mock_client = MagicMock()
    mock_client.name = "mock"
    mock_client.complete_json.return_value = NarrationOut(text="아이는 아무 일도 일어나지 않을 거라 여겼지만 문제가 잘 풀렸어요.")
    n = NarratorAgent(mock_client)

    i18n.set_lang("ko")
    text_ko = n.explain(w)
    assert text_ko == "아이는 아무 일도 일어나지 않을 거라 여겼지만 문제가 잘 풀렸어요."
    mock_client.complete_json.assert_called_once()
    call_args = mock_client.complete_json.call_args[1]
    payload = json.loads(call_args["user"])
    assert payload["language"] == "Korean"
    assert payload["task"] == "explain"
    assert payload["chosen_kind_ko"] == i18n.KIND_KO.get("cup")


def test_korean_explain_guard_rejects_digits():
    w = {
        "kind": "cup",
        "expected": "nothing",
        "actual": "solved",
        "expected_pct": 10,
        "after_pct": 80,
        "reasons": ["chance"],
    }
    mock_client = MagicMock()
    mock_client.name = "mock"
    # LLM text containing digits should fail check and trigger fallback
    mock_client.complete_json.return_value = NarrationOut(text="80% 확률로 문제가 풀렸어요.")
    n = NarratorAgent(mock_client)

    i18n.set_lang("ko")
    text_ko = n.explain(w)
    assert text_ko == explain_fixed(w)
    assert n.last_error is not None


def test_ko_intro_and_outcome_error_guards():
    assert ko.intro_error("") == "empty"
    assert ko.intro_error("a" * 171) == "too long"
    assert ko.intro_error("bad <markup>") == "markup or control characters"
    assert ko.intro_error("정상적인 시작 문장입니다.") is None

    assert ko.outcome_error("") == "empty"
    assert ko.outcome_error("b" * 151) == "too long"
    assert ko.outcome_error("line1\nline2") == "markup or control characters"
    assert ko.outcome_error("정상적인 결과 문장입니다.") is None


def test_child_speech_step1_is_empty_in_korean():
    cup = SimpleNamespace(kind="cup", color="red", size="big")
    assert ko.child_speech("GIVE", cup, voice="self", step=1, tags=["THIRSTY"]) == ""
    assert ko.child_speech("DRINK", cup, voice="grandma", step=1, tags=["THIRSTY"]) == ""
    assert ko.child_speech("WIPE", None, voice="self", step=1, tags=["FLOOR_WET"]) == ""


def test_child_speech_step2_dynamically_composed_in_korean():
    cup = SimpleNamespace(kind="cup", color="red", size="big")
    s_give = ko.child_speech("GIVE", cup, voice="self", step=2, tags=["THIRSTY"])
    assert "컵" in s_give and "give" in s_give

    s_drink = ko.child_speech("DRINK", cup, voice="self", step=2, tags=["THIRSTY"])
    assert "컵" in s_drink and "drink" in s_drink

    s_wait = ko.child_speech("WAIT", None, voice="self", step=2, tags=["THIRSTY"])
    assert "wait" in s_wait


def test_child_option_text_and_speech_for_bored_options():
    teddy = SimpleNamespace(kind="teddy", color="brown", size="medium")
    ball = SimpleNamespace(kind="ball", color="red", size="small")
    crayon = SimpleNamespace(kind="crayon", color="blue", size="small")
    book = SimpleNamespace(kind="book", color="green", size="medium")

    assert "곰인형" in ko.child_option_text("HUG", teddy, voice="self", tags=["BORED"])
    assert "곰인형" in ko.child_speech("HUG", teddy, voice="self", step=2, tags=["BORED"])

    assert "공" in ko.child_option_text("GIVE", ball, voice="self", tags=["BORED"])
    assert "공" in ko.child_speech("GIVE", ball, voice="self", step=2, tags=["BORED"])

    assert "크레용" in ko.child_option_text("DRAW", crayon, voice="self", tags=["BORED"])
    assert "크레용" in ko.child_speech("DRAW", crayon, voice="self", step=2, tags=["BORED"])

    assert "책" in ko.child_option_text("READ", book, voice="self", tags=["BORED"])
    assert "책" in ko.child_speech("READ", book, voice="self", step=2, tags=["BORED"])


def test_narrator_solo_intro_and_error_guard():
    assert ko.intro_error("아이가 당신 곁에 와서 심심해요, 같이 놀아요!", solo=True) is not None
    assert ko.intro_error("아이가 당신 곁에 와서 심심해요, 같이 놀아요!", solo=False) is None
    assert ko.intro_error("혼자서 어떤 재미있는 놀이를 해 볼까?", solo=True) is None

    sit = make_situation(random.Random(1), ["FLOOR_WET"], sid="test_sit_solo_intro")
    mock_client = MagicMock()
    mock_client.name = "mock"
    mock_client.complete_json.return_value = NarrationOut(text="혼자서 어떤 재미있는 놀이를 해 볼까?")
    n = NarratorAgent(mock_client)

    i18n.set_lang("ko")
    solo_text = n.solo_intro(sit, voice="self")
    assert solo_text == "혼자서 어떤 재미있는 놀이를 해 볼까?"
    mock_client.complete_json.assert_called_once()
    payload = json.loads(mock_client.complete_json.call_args[1]["user"])
    assert payload["language"] == "Korean"
    assert payload["task"] == "solo_intro"
    assert payload["solo"] is True

    mock_client_err = MagicMock()
    mock_client_err.complete_json.side_effect = RuntimeError("error")
    n_fallback = NarratorAgent(mock_client_err)
    fallback_text = n_fallback.solo_intro(sit, voice="self")
    assert "혼자" in fallback_text or re.search(r"[가-힣]", fallback_text)


def test_child_speech_baby_is_empty():
    cup = SimpleNamespace(kind="cup", color="red", size="big")
    assert ko.child_speech("GIVE", cup, voice="baby", step=1, tags=["THIRSTY"]) == ""
    assert ko.child_speech("GIVE", cup, voice="baby", step=2, tags=["THIRSTY"]) == ""


def test_hardcoded_dialogue_tables_removed_from_narrator_ko():
    assert not hasattr(ko, "SPEECH_BY_TAG_VERB_KIND")
    assert not hasattr(ko, "STEP1_BY_TAG")


def test_mock_narrator_handler_handles_korean():
    handler = mock_narrator_handler()
    sit = make_situation(random.Random(1), ["FLOOR_WET"], sid="test_sit_mock")

    client = MockClient({"narrator": handler})
    n = NarratorAgent(client)

    i18n.set_lang("ko")
    intro_text, _ = n.intro(sit, False)
    assert re.search(r"[가-힣]", intro_text)

    outcome_text, _ = n.outcome(sit, 0, "solved", "player")
    assert re.search(r"[가-힣]", outcome_text)

    w = {
        "kind": "cup",
        "expected": "nothing",
        "actual": "solved",
        "expected_pct": 10,
        "after_pct": 80,
        "reasons": ["chance"],
    }
    explain_text = n.explain(w)
    assert re.search(r"[가-힣]", explain_text)


def test_english_narrator_flow():
    sit = make_situation(random.Random(1), ["FLOOR_WET"], sid="test_sit_en")
    handler = mock_narrator_handler()
    client = MockClient({"narrator": handler})
    n = NarratorAgent(client)

    i18n.set_lang("en")
    intro_text, _ = n.intro(sit, False)
    assert "floor" in intro_text.lower() or "water" in intro_text.lower() or "Oh no" in intro_text

    outcome_text, _ = n.outcome(sit, 0, "solved", "player")
    assert "That solves it!" in outcome_text or "You" in outcome_text


def test_pair_action_ko():
    assert "비누" in ko.option_text_ko("WASH", "soap") and "wash" in ko.option_text_ko("WASH", "soap")
    assert "책" in ko.option_text_ko("READ", "book") and "read" in ko.option_text_ko("READ", "book")
    assert "크레용" in ko.option_text_ko("DRAW", "crayon") and "draw" in ko.option_text_ko("DRAW", "crayon")
    assert "곰인형" in ko.option_text_ko("HUG", "teddy") and "hug" in ko.option_text_ko("HUG", "teddy")

    soap = SimpleNamespace(kind="soap", color=None, size=None)
    assert "비누" in ko.option_text("WASH", soap) and "wash" in ko.option_text("WASH", soap) 


def test_narrate_outcome_custom_action():
    sit = make_situation(random.Random(1), ["FLOOR_WET"], sid="test_sit_custom_action")
    mock_client = MagicMock()
    mock_client.name = "mock"
    mock_client.complete_json.return_value = NarrationOut(text="공룡 이야기를 들으며 즐거워했어요.")
    n = NarratorAgent(mock_client)

    i18n.set_lang("ko")
    out = n.narrate_outcome(sit, 0, "solved", "player", custom_action="동화책을 실감나게 읽어주기")
    assert "공룡 이야기를 들으며 즐거워했어요." in out
    mock_client.complete_json.assert_called_once()
    payload = json.loads(mock_client.complete_json.call_args[1]["user"])
    assert payload["custom_parent_action"] == "동화책을 실감나게 읽어주기"


def test_ask_child_fallback_and_live():
    from server.agents.narrator import ChildQAOut

    n_offline = NarratorAgent(None)
    i18n.set_lang("ko")
    ctx = {
        "age": 3,
        "stage": "toddler",
        "brain_state": {
            "model_beliefs": [{"problem": "DIRTY_HANDS", "action": "WASH", "summary": "비누로 손을 씻으면 깨끗해져요"}],
            "relevant_memories": [{"summary": "오늘 엄마와 함께 손을 씻었어요"}],
        },
        "traits": [],
    }
    ans_belief = n_offline.ask_child("손 더러울 때 비누로 손 씻기 어떻게 해?", ctx)
    assert "비누로 손을 씻으면 깨끗해져요" in ans_belief

    ans_mem = n_offline.ask_child("오늘 뭐 했어?", ctx)
    assert "손을 씻었어요" in ans_mem or "비누" in ans_mem

    i18n.set_lang("en")
    ans_en = n_offline.ask_child("What did you do?", {"brain_state": {}})
    assert "don't know" in ans_en or "haven't tried" in ans_en

    live_client = MagicMock()
    live_client.available.return_value = True
    live_client.complete_json.return_value = ChildQAOut(answer="엄마 아빠랑 거품 놀이한 게 제일 재미있었어요!")
    n_live = NarratorAgent(live_client)
    i18n.set_lang("ko")
    live_ans = n_live.ask_child("뭐가 제일 기억에 남아?", ctx)
    assert live_ans == "엄마 아빠랑 거품 놀이한 게 제일 재미있었어요!"


def test_ask_child_cognitive_scaling_instructions():
    from server.agents.narrator import CHILD_QA_SYSTEM, ChildQAOut

    expected_tiers = [
        "1.0B ~ 1.9B = 1세 수준 (감각/단편 직관, 1-hop)",
        "2.0B ~ 2.9B = 2세 수준 (단어/사물 연결, 1~2-hop)",
        "3.0B ~ 3.9B = 3세 수준 (기초 개념 형성, 2-hop)",
        "4.0B ~ 4.9B = 4세 수준 (인과 관계 이해, 3-hop)",
        "5.0B ~ 5.9B = 5세 수준 (비교/분류 추론, 3~4-hop)",
        "6.0B ~ 6.9B = 6세 수준 (복합 규칙 추론, 4-hop)",
        "7.0B = 7세 수준 (원리 일반화, 5-hop)",
    ]
    for tier in expected_tiers:
        assert tier in CHILD_QA_SYSTEM

    scales_and_labels = [
        (1.0, "1세 수준"),
        (1.5, "1세 수준"),
        (2.0, "2세 수준"),
        (2.8, "2세 수준"),
        (3.0, "3세 수준"),
        (3.7, "3세 수준"),
        (4.0, "4세 수준"),
        (4.5, "4세 수준"),
        (5.0, "5세 수준"),
        (5.5, "5세 수준"),
        (6.0, "6세 수준"),
        (6.8, "6세 수준"),
        (7.0, "7세 수준"),
    ]

    for scale, expected_label in scales_and_labels:
        client = MagicMock()
        client.available.return_value = True
        client.complete_json.return_value = ChildQAOut(answer="응!")
        n = NarratorAgent(client)
        ctx = {"cognitive": {"scale_b": scale}}
        n.ask_child("질문", ctx)
        call_args = client.complete_json.call_args[1]
        payload = json.loads(call_args["user"])
        assert expected_label in payload["cognitive_scale_instruction"]


def test_ask_child_personality_and_no_episode_listing_prompt():
    from server.agents.narrator import CHILD_QA_SYSTEM, ChildQAOut, NarratorAgent
    from server.game.memory import probe_child_brain

    # 1. CHILD_QA_SYSTEM explicitly forbids listing past episodes unsolicited and defines personality-specific styles
    assert "과거 사건 나열 절대 금지" in CHILD_QA_SYSTEM or "과거 사건" in CHILD_QA_SYSTEM
    assert "절대 과거에 있었던 사건/상황들을 먼저 나열하지 마세요" in CHILD_QA_SYSTEM
    archetypes = [
        "warm_affectionate",
        "bold_explorer",
        "timid_anxious",
        "lonely_sulky",
        "wary_sensitive",
    ]
    for arch in archetypes:
        assert arch in CHILD_QA_SYSTEM

    # 2. NarratorAgent.ask_child() includes personality and traits in the LLM payload
    client = MagicMock()
    client.available.return_value = True
    client.complete_json.return_value = ChildQAOut(answer="헤헤, 나는 엄마 귀여운 꼬마지!")
    n = NarratorAgent(client)

    test_personality = {
        "archetype": "warm_affectionate",
        "label": "다정하고 애교 많은 사랑둥이",
        "traits": ["다정함", "애교"],
    }
    ctx = {
        "cognitive": {"personality": test_personality},
    }
    n.ask_child("넌 누구니?", ctx)
    call_args = client.complete_json.call_args[1]
    payload = json.loads(call_args["user"])
    assert payload["personality"] == test_personality
    assert payload["traits"] == ["다정함", "애교"]

    # 3. Fallback uses personality instead of blindly dumping beliefs/episodes
    n_offline = NarratorAgent(None)
    i18n.set_lang("ko")
    fb_ctx = {
        "personality": test_personality,
        "brain_state": {
            "model_beliefs": [{"problem": "DIRTY_HANDS", "action": "WASH", "summary": "비누로 손을 씻으면 깨끗해져요"}],
        },
    }
    fb_ans = n_offline.ask_child("넌 누구니?", fb_ctx)
    assert "엄마 귀여운 꼬마" in fb_ans
    assert "비누" not in fb_ans

    # Different archetype gives different answer
    sulky_ctx = {
        "personality": {
            "archetype": "lonely_sulky",
            "traits": ["서운함"],
        },
    }
    sulky_ans = n_offline.ask_child("넌 누구니?", sulky_ctx)
    assert "흥" in sulky_ans or "심심" in sulky_ans

    # 4. probe_child_brain returns personality and traits
    fake_session = SimpleNamespace(model=None, model0=None, buffer=[], journal=[])
    brain = probe_child_brain(fake_session, lang="ko")
    assert "personality" in brain
    assert "traits" in brain
    assert isinstance(brain["personality"], dict)
    assert isinstance(brain["traits"], list)


def test_compose_lesson_ko():
    import server.game.memory as mem
    assert not hasattr(mem, "_compose_lesson_ko")


def test_solo_intro_lines_tags_and_phrasing():
    from server.agents.narrator import solo_intro

    sit = SimpleNamespace(tags=["PLAY_INVITE"], id="test_sit", location="house")
    i18n.set_lang("ko")
    res_ko = solo_intro(sit)
    assert "혼자" in res_ko or re.search(r"[가-힣]", res_ko)

    i18n.set_lang("en")
    res_en = solo_intro(sit)
    assert "themselves" in res_en or "alone" in res_en or "play" in res_en.lower()


def test_child_option_text_and_doing_child():
    from server.agents.narrator import child_option_text

    book = SimpleNamespace(kind="book", color="red", size="small")
    soap = SimpleNamespace(kind="soap", color="white", size="small")

    # In Korean
    i18n.set_lang("ko")
    assert "책" in child_option_text("READ", book, voice="self", tags=["BORED"])
    assert "비누" in child_option_text("WASH", soap, voice="self", tags=["DIRTY_HANDS"])
    assert "책" in ko.doing("READ", book, who="child", voice="self", tags=["BORED"])

    # In English
    i18n.set_lang("en")
    en_text = child_option_text("READ", book, voice="self", tags=["BORED"])
    assert "read" in en_text.lower()


def test_mock_solo_intro_handler():
    handler = mock_narrator_handler()

    # Korean
    ko_req = {
        "task": "solo_intro",
        "language": "Korean",
        "problems": [{"tag": "BORED"}],
        "seed": 1,
    }
    ko_res = handler(ko_req)
    assert "혼자" in ko_res["text"] and "어떻게" in ko_res["text"]

    # English
    en_req = {
        "task": "solo_intro",
        "language": "English",
        "problems": [{"tag": "BORED"}],
        "seed": 1,
    }
    en_res = handler(en_req)
    assert "bored" in en_res["text"].lower()


def test_is_live_llm():
    assert NarratorAgent(None).is_live_llm() is False

    mock_client = MagicMock()
    mock_client.name = "mock"
    assert NarratorAgent(mock_client).is_live_llm() is False

    none_client = MagicMock()
    none_client.name = "none"
    assert NarratorAgent(none_client).is_live_llm() is False

    empty_client = MagicMock()
    empty_client.name = ""
    assert NarratorAgent(empty_client).is_live_llm() is False

    live_client = MagicMock()
    live_client.name = "live_meta"
    assert NarratorAgent(live_client).is_live_llm() is True


def test_generate_scene_bundle_offline_and_mock_returns_none():
    sit = make_situation(random.Random(1), ["FLOOR_WET"], sid="test_bundle_offline")
    assert NarratorAgent(None).generate_scene_bundle(sit, 24, "self") is None

    mock_client = MagicMock()
    mock_client.name = "mock"
    assert NarratorAgent(mock_client).generate_scene_bundle(sit, 24, "self") is None


def test_generate_scene_bundle_live_client_success_and_caching():
    sit_base = make_situation(random.Random(1), ["FLOOR_WET"], sid="test_bundle_live")
    sit = sit_base.model_copy(update={"options": list(sit_base.options) + [sit_base.options[0]]})
    assert len(sit.options) == 4
    live_client = MagicMock()
    live_client.name = "live_meta"
    live_client.complete_json.return_value = SceneBundleOut(
        narration="더운 여름날 그늘 아래서 땀을 흘리고 있어요.",
        options=["물컵으로 시원하게 물 마시기", "손수건으로 땀 닦기", "선풍기 바람 쐬기", "가만히 쉬기"],
        speeches=["시원해요!", "좋아요!", "시원한 바람!", "쉴래요!"],
        outcomes=["solved", "partial", "nothing", "worse"],
        results=["물이 시원해요", "땀이 닦여요", "바람이 시원해요", "그대로 더워요"],
        world_says=["물은 갈증을 해소해 줍니다", "수건은 땀을 흡수합니다", "바람은 열을 식힙니다", "쉬는 것만으로는 덥습니다"],
    )
    n = NarratorAgent(live_client)

    i18n.set_lang("ko")
    bundle = n.generate_scene_bundle(
        sit,
        age_months=36,
        voice="self",
        solo=False,
        recent_scenes=["이전 장면 1", "이전 장면 2"],
    )
    assert bundle is not None
    assert bundle["narration"] == "더운 여름날 그늘 아래서 땀을 흘리고 있어요."
    assert len(bundle["options"]) == 4
    assert len(bundle["speeches"]) == 4
    assert bundle["outcomes"] == ["solved", "partial", "nothing", "worse"]
    assert len(bundle["results"]) == 4
    assert len(bundle["world_says"]) == 4
    assert live_client.complete_json.call_count == 1

    # Check payload structure: verify that tags and option_slots are NOT sent to LLM
    call_args = live_client.complete_json.call_args[1]
    payload = json.loads(call_args["user"])
    assert payload["task"] == "scene_bundle"
    assert payload["age_months"] == 36
    assert payload["age_years"] == 3
    assert payload["solo"] is False
    assert payload["recent_scenes"] == ["이전 장면 1", "이전 장면 2"]
    assert payload["option_count"] == 4
    assert "tags" not in payload
    assert "option_slots" not in payload
    assert "problems" not in payload
    assert "still" not in payload

    # Calling again should use cache
    bundle_cached = n.generate_scene_bundle(
        sit,
        age_months=36,
        voice="self",
        solo=False,
        recent_scenes=["이전 장면 1", "이전 장면 2"],
    )
    assert bundle_cached == bundle
    assert live_client.complete_json.call_count == 1


def test_generate_scene_bundle_padding_and_error_fallback():
    sit = make_situation(random.Random(1), ["FLOOR_WET"], sid="test_bundle_pad")
    live_client = MagicMock()
    live_client.name = "live_meta"
    # Provide only 2 speeches for 4 options
    live_client.complete_json.return_value = SceneBundleOut(
        narration="더운 여름날 그늘 아래서 땀을 흘리고 있어요.",
        options=["옵션 1", "옵션 2", "옵션 3", "옵션 4"],
        speeches=["대사 1", "대사 2"],
    )
    n = NarratorAgent(live_client)
    i18n.set_lang("ko")
    bundle = n.generate_scene_bundle(sit, age_months=36, voice="self", solo=True)
    assert bundle is not None
    assert len(bundle["speeches"]) == len(sit.options)
    assert bundle["speeches"][0] == "대사 1"
    assert bundle["speeches"][1] == "대사 2"
    assert len(bundle["speeches"][2]) > 0
    assert len(bundle["outcomes"]) == len(sit.options)
    assert "solved" in bundle["outcomes"]

    # Error triggers fallback and increments fallbacks counter
    live_client.complete_json.side_effect = RuntimeError("meta api error")
    bundle_err = n.generate_scene_bundle(
        sit, age_months=36, voice="grandma", solo=True
    )
    assert bundle_err is None
    assert n.fallbacks >= 1


def test_outcome_live_korean_stores_takeaway_and_omits_doing_prefix():
    from server.agents.narrator import _doing

    sit = make_situation(random.Random(1), ["FLOOR_WET"], sid="test_sit_takeaway")
    live_client = MagicMock()
    live_client.name = "live_meta"
    narration_text = "물이 쏟아진 바닥을 마른 걸레로 뽀송하게 닦아내자 방 안이 다시 깨끗해졌어요."
    takeaway_text = "물이 엎질러졌을 때는 걸레로 닦으면 다시 깨끗해진다는 걸 배웠어!"
    live_client.complete_json.return_value = NarrationOut(
        text=narration_text,
        takeaway=takeaway_text,
    )
    n = NarratorAgent(live_client)

    i18n.set_lang("ko")
    res, takeaway = n.outcome(
        sit,
        0,
        "solved",
        "player",
        scene_narration="바닥에 물이 흥건해요.",
        chosen_text="걸레로 바닥을 깨끗이 닦아주기",
    )
    # Natural LLM text returned directly without prepended _doing
    assert res == narration_text
    assert takeaway == takeaway_text
    prefix = _doing(sit, 0, "player", "grandma")
    assert not res.startswith(prefix)

    # Takeaway stored in cache and retrievable via get_takeaway
    retrieved = n.get_takeaway(
        sit,
        0,
        "solved",
        "player",
        "grandma",
        chosen_text="걸레로 바닥을 깨끗이 닦아주기",
    )
    assert retrieved == takeaway_text

    # Fallback lookup without chosen_text also retrieves it
    retrieved_fallback = n.get_takeaway(sit, 0, "solved", "player", "grandma")
    assert retrieved_fallback == takeaway_text

    # Verify user payload does not contain consequence or feelings, and contains chosen_action & outcome_meaning
    call_args = live_client.complete_json.call_args[1]
    payload = json.loads(call_args["user"])
    assert "consequence" not in payload
    assert "feelings" not in payload
    assert payload["chosen_action"] == "걸레로 바닥을 깨끗이 닦아주기"
    assert payload["scene_narration"] == "바닥에 물이 흥건해요."
    assert "outcome_meaning" in payload


def test_outcome_english_retains_doing_prefix():
    sit = make_situation(random.Random(1), ["FLOOR_WET"], sid="test_sit_en_live")
    live_client = MagicMock()
    live_client.name = "live_meta"
    live_client.complete_json.return_value = NarrationOut(
        text="The floor is completely dry again.",
        takeaway="Wiping water right away is important.",
    )
    n = NarratorAgent(live_client)

    i18n.set_lang("en")
    res, takeaway = n.outcome(sit, 0, "solved", "player")
    assert "You " in res
    assert "The floor is completely dry again." in res
    assert takeaway == "Wiping water right away is important."
    assert n.get_takeaway(sit, 0, "solved", "player", "grandma") == "Wiping water right away is important."


def test_outcome_bundle_result_context_and_fallback():
    sit = make_situation(random.Random(1), ["FLOOR_WET"], sid="test_bundle_res")
    live_client = MagicMock()
    live_client.name = "live_meta"
    live_client.complete_json.return_value = NarrationOut(
        text="깨끗하게 닦였어요.",
        takeaway="물기를 바로 닦으면 미끄럽지 않아요.",
    )
    n = NarratorAgent(live_client)

    i18n.set_lang("ko")
    # Live LLM call receives bundle_result in payload
    narration, takeaway = n.outcome(
        sit,
        0,
        "solved",
        "player",
        bundle_result="바닥의 물기가 말끔히 닦였어요.",
    )
    assert narration == "깨끗하게 닦였어요."
    call_args = live_client.complete_json.call_args[1]
    payload = json.loads(call_args["user"])
    assert payload["bundle_result"] == "바닥의 물기가 말끔히 닦였어요."

    # When fallback occurs (e.g. mock / error), bundle_result is used directly as narration
    f_client = MagicMock()
    f_client.name = "failing"
    f_client.complete_json.side_effect = RuntimeError("network error")
    fn = NarratorAgent(f_client)
    f_narration, f_takeaway = fn.outcome(
        sit,
        0,
        "solved",
        "player",
        bundle_result="바닥의 물기가 말끔히 닦였어요.",
    )
    assert f_narration == "바닥의 물기가 말끔히 닦였어요."


def test_static_dictionaries_removed():
    import server.agents.narrator as narrator

    for name in (
        "VERB_PHRASES",
        "RESULTS",
        "HAPPENED_WORDS",
        "EXPECTED_WORDS",
        "REASON_WORDS",
        "OUTCOME_MEANINGS_KO",
        "OUTCOME_MEANINGS_EN",
    ):
        assert not hasattr(narrator, name), f"{name} should not exist in narrator"
    for name in ("VERBS", "RESULTS", "HAPPENED", "EXPECTED", "REASON"):
        assert not hasattr(ko, name), f"{name} should not exist in narrator_ko"


def test_director_agent_live_llm_and_direct_scene():
    from server.agents.director import DirectorAgent

    assert DirectorAgent(None, {}).is_live_llm() is False

    mock_c = MagicMock()
    mock_c.name = "mock"
    d_mock = DirectorAgent(mock_c, {})
    assert d_mock.is_live_llm() is False
    sit = make_situation(random.Random(1), ["FLOOR_WET"], sid="test_dir_scene")
    assert d_mock.direct_scene(sit, 24, "grandma") is None

    live_c = MagicMock()
    live_c.name = "live_meta"
    live_c.complete_json.return_value = SceneBundleOut(
        narration="할머니와 아이가 방 안에 있어요.",
        options=["옵션 1", "옵션 2", "옵션 3", "옵션 4"],
        speeches=["대사 1", "대사 2", "대사 3", "대사 4"],
        outcomes=["solved", "partial", "nothing", "worse"],
        results=["결과 1", "결과 2", "결과 3", "결과 4"],
        world_says=["세상 1", "세상 2", "세상 3", "세상 4"],
    )
    d_live = DirectorAgent(live_c, {})
    assert d_live.is_live_llm() is True

    bundle = d_live.direct_scene(sit, 24, "grandma")
    assert bundle is not None
    assert bundle["narration"] == "할머니와 아이가 방 안에 있어요."
    assert len(bundle["options"]) == len(sit.options)
    assert len(bundle["results"]) == len(sit.options)
    assert len(bundle["world_says"]) == len(sit.options)
    assert live_c.complete_json.call_args[1]["agent"] == "director"

    # Cached call
    cached = d_live.direct_scene(sit, 24, "grandma")
    assert cached == bundle
    assert live_c.complete_json.call_count == 1
