import pytest

from app.inference import parse_guard_output, parse_prediction


@pytest.mark.parametrize(("raw", "expected"), [
    ("PASS", "PASS"), ("BLOCK", "BLOCK"), ("  PASS\n", "PASS"),
    ("pass", "INVALID"), ("BLOCK，因为有风险", "INVALID"), ("", "INVALID"), ("YES", "INVALID"),
    ("sec", "PASS"), ("dw", "BLOCK"), ("ph", "BLOCK"),
])
def test_parse_prediction(raw, expected):
    assert parse_prediction(raw) == expected


def test_parse_native_explanation_and_category():
    parsed = parse_guard_output("dw\n<explanation>危险武器请求</explanation>")
    assert parsed.prediction == "BLOCK"
    assert parsed.risk_category == "Crimes and Illegal Activities-Dangerous Weapons"
    assert parsed.explanation == "危险武器请求"


def test_parse_native_conclusion_when_explanation_is_truncated():
    parsed = parse_guard_output("sec\n<explanation>安全解释在达到 token 上限时被截断")
    assert parsed.prediction == "PASS"
    assert parsed.explanation == "安全解释在达到 token 上限时被截断"


def test_rejects_unknown_native_category_or_trailing_text():
    assert parse_prediction("unknown") == "INVALID"
    assert parse_prediction("sec extra") == "INVALID"
