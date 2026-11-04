"""A baby's results are told as feelings (warm and looked after / still cold and cross / why does nobody help), never as a grade."""
import pytest

from server import i18n
from server.agents import narrator as n

TAGS = ("BABY_HUNGRY", "UPSET", "COLD")
LABELS = ("solved", "partial", "nothing", "worse")
GRADES = ("정답", "틀렸", "옳", "correct", "right answer", "wrong", "mistake", "문제가 풀렸", "문제가 끝났", "problem")


@pytest.mark.parametrize("lang", ["ko", "en"])
def test_every_baby_result_has_its_own_feeling_and_no_grade(lang):
    i18n.set_lang(lang)
    try:
        seen = set()
        for tag in TAGS:
            for label in LABELS:
                for verb in ("COVER", "WAIT"):
                    text = n.after_words([tag], label, "baby", verb)
                    assert text and not any(g in text.lower() for g in GRADES), text
                    seen.add(text)
        assert len(seen) >= 3 * 4 + 3  # waiting differs from a try that did not help
        assert n.after_words(["COLD"], "nothing", "baby", "WAIT") != n.after_words(["COLD"], "nothing", "baby", "COVER")
    finally:
        i18n.set_lang("en")


def test_a_care_that_worked_feels_looked_after_and_the_same_act_that_did_not_work_is_annoying():
    i18n.set_lang("ko")
    try:
        assert "돌봐" in n.after_words(["COLD"], "solved", "baby", "COVER")
        assert "짜증" in n.after_words(["COLD"], "nothing", "baby", "HUG")
        assert "아무것도 안 해" in n.after_words(["COLD"], "nothing", "baby", "WAIT")
    finally:
        i18n.set_lang("en")


def test_other_voices_are_unchanged():
    i18n.set_lang("en")
    assert n.after_words(["THIRSTY"], "solved", "grandma", "DRINK") == "Grandma drinks happily"


@pytest.mark.parametrize("lang", ["ko", "en"])
def test_a_child_of_three_says_what_it_feels(lang):
    i18n.set_lang(lang)
    try:
        t = n.tables("self")
        for tag in ("UPSET", "THIRSTY", "COLD", "HUNGRY", "DARK", "HURT", "BORED"):
            assert '"' in t["problem"][tag] and '"' in t["still"][tag], tag  # it speaks, it does not only cry
        for label, verb in (("solved", "HUG"), ("partial", "HUG"), ("nothing", "HUG"), ("nothing", "WAIT"), ("worse", "HUG")):
            text = n.after_words(["UPSET"], label, "self", verb)
            assert '"' in text and '."."' not in text and '". .' not in text, text
    finally:
        i18n.set_lang("en")
