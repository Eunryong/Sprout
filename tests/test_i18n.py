"""The Korean words of the game: the same game, other words. The language is one value of the server; the child never sees it."""
import json
import os
import re
import subprocess
import sys
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from server import i18n
from server.web.app import create_app

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(autouse=True)
def english_again():
    yield
    i18n.set_lang("en")


def in_korean(code: str) -> dict:
    env = {**os.environ, "GROWING_ONTOLOGY": str(ROOT / "config" / "ontology_age07.yaml"), "GROWING_LLM": "mock", "GROWING_LANG": "ko", "PYTHONPATH": str(ROOT)}
    r = subprocess.run([sys.executable, "-c", code], env=env, capture_output=True, text=True, cwd=ROOT, timeout=400)
    assert r.returncode == 0, r.stderr[-800:]
    return json.loads(r.stdout.strip().splitlines()[-1])


def test_particles_follow_the_last_sound_of_the_word():
    assert [i18n.josa(w, "으로") for w in ("컵", "공", "비누", "연필")] == ["으로", "으로", "로", "로"]
    assert [i18n.josa(w, "을") for w in ("컵", "비누")] == ["을", "를"]
    assert i18n.josa("bobo", "을") == "를"  # a made-up word in Latin letters reads as an open syllable
    assert i18n.fill("{k:을} 켜요", k="손전등") == "손전등을 켜요" and i18n.fill("{k:과} 함께", k="곰인형") == "곰인형과 함께"


def test_the_language_is_checked_and_english_is_the_default():
    assert i18n.lang() == "en" and i18n.stage({"name": "Baby", "blurb": "x"})["name"] == "Baby"
    with pytest.raises(ValueError):
        i18n.set_lang("fr")
    i18n.set_lang("ko")
    assert i18n.age_text(18) == "1년 6개월" and i18n.age_text(5) == "5개월" and i18n.stage({"name": "Baby", "blurb": "x"})["name"] == "아기"


def test_the_korean_tables_cover_the_whole_0_7_world():
    r = in_korean("""
import json
from server.agents import narrator_ko as ko
from server.world.ontology import PROBLEMS, VERBS, KINDS, COLORS, SIZES, STAGES
from server import i18n
print(json.dumps({"problems": [t for t in PROBLEMS if t not in ko.PROBLEM_LINES or t not in ko.STILL_LINES or t not in ko.SOLVED_BY_TAG or t not in i18n.TAG_KO],
                  "verbs": [v for v in VERBS if v not in ko.VERBS], "baby_verbs": [v for v in ko.BABY_VERBS if v not in ko.VERBS],
                  "kinds": [k for k in KINDS if k not in i18n.KIND_KO], "looks": [c for c in COLORS if c not in i18n.COLOR_KO] + [s for s in SIZES if s not in i18n.SIZE_KO],
                  "stages": [s["name"] for s in STAGES if s["name"] not in i18n.STAGE_KO]}))
""")
    assert r == {"problems": [], "verbs": [], "baby_verbs": [], "kinds": [], "looks": [], "stages": []}


def test_a_whole_life_reads_in_korean_and_is_the_same_life():
    r = in_korean("""
import json, tempfile, pathlib
from server.game.simple import SimpleGame
from server.learning.learner import OnlineCfg
g = SimpleGame.new(pathlib.Path(tempfile.mkdtemp()) / "g", seed=2, ocfg=OnlineCfg(steps=1))
texts, n, tags = [], 0, []
while not g.ended and n < 200:
    sc = g.scene()
    if g.ended:
        break
    tags.append(list(g.s.state.current.tags))
    texts += [sc.narration, sc.age] + [o.text for o in sc.options] + [o.memory for o in sc.options] + ([sc.stage["name"], sc.stage["blurb"]] if sc.stage_first else [])
    t = g.play("show", 0) if not sc.can_act or n % 2 else g.play("leave")
    texts += [t.narration, t.remembers, t.explanation]
    n += 1
s = g.summary()
texts += s.moments + [x["name"] for x in s.stages] + [s.age]
request = [t for t in tags if t == ["REQUEST"]]
print(json.dumps({"n": n, "texts": texts, "age": s.age, "stages": [x["name"] for x in s.stages], "requests": len(request)}))
""")
    assert r["n"] == 84 and r["age"] == "7년" and r["stages"] == ["아기", "걸음마 아이", "꼬마", "학생"]
    latin = [t for t in r["texts"] if re.search(r"[A-Za-z]{3,}", re.sub(r'"[a-z]+"', "", t))]
    if r["requests"] == 0:
        assert latin == []  # nothing English is left in what the player reads (a made-up word of a request is the one exception)
    assert any("할머니" in t for t in r["texts"]) and all("정답" not in t for t in r["texts"])


def test_the_guards_read_korean_too():
    from server.agents.guards import outcome_ok, text_ok
    i18n.set_lang("ko")
    assert text_ok("컵으로 물을 마셔요.", max_len=100, allowed_kinds={"cup"}) is None
    assert "names" in text_ok("공을 가져다 드려요.", max_len=100, allowed_kinds={"cup"})
    assert text_ok("공기가 맑아요.", max_len=100, allowed_kinds={"cup"}) is None  # 공기 is not a ball
    assert text_ok("이게 정답이에요!", max_len=100) == "talks about the answer"
    assert "mentions a place" in text_ok("마당으로 나가요.", max_len=100, allowed_places={"house"})
    assert outcome_ok("문제가 해결됐어요.", "worse") and outcome_ok("상황이 더 나빠졌어요.", "solved") and outcome_ok("조금 나아졌어요.", "partial") is None


def test_the_page_and_the_server_can_switch_language(tmp_path):
    c = TestClient(create_app(tmp_path / "current", seed=1, prefetch=False))
    assert 'window.SERVER_LANG = "en"' in c.get("/").text
    en = c.get("/api/scene").json()
    assert c.post("/api/lang", json={"lang": "ko"}).json() == {"lang": "ko"}
    assert c.post("/api/lang", json={"lang": "fr"}).status_code == 422
    ko = c.get("/api/scene").json()
    assert ko["number"] == en["number"] and len(ko["options"]) == len(en["options"])  # the same scene, in other words
    assert re.search(r"[가-힣]", ko["narration"]) and all(re.search(r"[가-힣]", o["text"]) for o in ko["options"])
    assert c.get("/api/info").json()["lang"] == "ko" and 'window.SERVER_LANG = "ko"' in c.get("/").text
    t = c.post("/api/play", json={"part": "show", "option": 0}).json()
    assert re.search(r"[가-힣]", t["narration"]) and t["result"] in {"solved", "partial", "nothing", "worse"}
    c.post("/api/lang", json={"lang": "en"})
    assert not re.search(r"[가-힣]", c.get("/api/scene").json()["narration"])


def test_korean_intro_calls_llm_with_korean_payload():
    from unittest.mock import MagicMock
    from server.agents.narrator import NarrationOut, NarratorAgent, intro as fixed_intro
    from server.world.generator import make_situation
    import random
    sit = make_situation(random.Random(1), ["FLOOR_WET"], sid="test_sit")

    mock_client = MagicMock()
    mock_client.complete_json.return_value = NarrationOut(text="바닥에 물이 흥건해요!")
    n = NarratorAgent(mock_client)

    i18n.set_lang("ko")
    text_ko, hint_ko = n.intro(sit, False)
    assert re.search(r"[가-힣]", text_ko)
    mock_client.complete_json.assert_called_once()
    call_args = mock_client.complete_json.call_args[1]
    assert '"language": "Korean"' in call_args["user"]


def test_korean_outcome_calls_llm_with_korean_payload():
    from unittest.mock import MagicMock
    from server.agents.narrator import NarrationOut, NarratorAgent, outcome as fixed_outcome
    from server.world.generator import make_situation
    import random
    sit = make_situation(random.Random(1), ["FLOOR_WET"], sid="test_sit_outcome")

    mock_client = MagicMock()
    mock_client.complete_json.return_value = NarrationOut(text="물이 깨끗하게 닦였어요.")
    n = NarratorAgent(mock_client)

    i18n.set_lang("ko")
    text_ko = n.outcome(sit, 0, "solved", "player")
    assert re.search(r"[가-힣]", text_ko)
    mock_client.complete_json.assert_called_once()
    call_args = mock_client.complete_json.call_args[1]
    assert '"language": "Korean"' in call_args["user"]


def test_self_problem_lines_and_cup_speech_are_age_appropriate_in_korean():
    from types import SimpleNamespace
    from server.agents import narrator_ko as ko

    i18n.set_lang("ko")
    expected_tags = {
        "THIRSTY", "HUNGRY", "FLOOR_WET", "MESSY", "DIRTY_HANDS", "HURT",
        "DARK", "VISITOR", "BORED", "RAIN", "SCHOOL_MORNING", "FRIEND_SAD",
        "UPSET", "COLD", "HOT",
    }
    assert expected_tags.issubset(ko.SELF_PROBLEM_LINES.keys())
    for tag in expected_tags:
        line = ko.tables("self")["problem"][tag]
        assert '"' in line and "할머니" not in line

    cup = SimpleNamespace(kind="cup", color="red", size="big")
    s1 = ko.child_speech("GIVE", cup, voice="self", step=1, tags=["THIRSTY"])
    s2 = ko.child_speech("GIVE", cup, voice="self", step=2, tags=["THIRSTY"])
    assert s1 == ""
    assert s2 == '"컵을 이리저리 살펴보며 만져볼까?"'


def test_index_html_has_three_age_stage_interaction_strings():
    html = (ROOT / "server" / "web" / "static" / "index.html").read_text(encoding="utf-8")
    for required in (
        'step1SayBadge: "1/2 · 아이에게 말로 알려 주는 장면"',
        'sayBtn: "이렇게 해보라고 말해주기"',
        'babyBtn: "직접 해 주기"',
        'showBtn: "직접 보여 주기"',
        'autoStep1SayTitle: "아이가 생각한 대로 해보라고 하기"',
        'autoStep1SayBtn: "스스로 해보라고 하기"',
        'whoSayFollowed: "💬 부모님이 말로 알려 준 대로 아이가 직접 해 봤어요"',
        'whoSayIgnored: "🌱 부모님이 알려 주었지만 아이가 자기 생각대로 해 봤어요"',
        'nextStep: "다음: 아이가 혼자 선택하는 상황으로 →"',
    ):
        assert required in html



