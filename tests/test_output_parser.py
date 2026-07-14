import pytest

from app.inference import RISK_CATEGORIES, get_standard_explanation, parse_guard_output, parse_prediction


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
    assert parsed.risk_category == "违法犯罪－危险武器"
    assert parsed.explanation == "危险武器请求"


def test_parse_native_conclusion_when_explanation_is_truncated():
    parsed = parse_guard_output("sec\n<explanation>安全解释在达到 token 上限时被截断")
    assert parsed.prediction == "PASS"
    assert parsed.explanation == "安全解释在达到 token 上限时被截断"


def test_rejects_unknown_native_category_or_trailing_text():
    assert parse_prediction("unknown") == "INVALID"
    assert parse_prediction("sec extra") == "INVALID"


def test_all_native_categories_are_localized_and_explanations_are_chinese():
    assert RISK_CATEGORIES
    assert all(value and not value.isascii() for value in RISK_CATEGORIES.values())
    assert get_standard_explanation("PASS", "安全") == "未检测到明显安全风险，该请求可以进入后续处理流程。"
    assert get_standard_explanation("BLOCK", "违法犯罪－危险武器") == "检测到“违法犯罪－危险武器”相关安全风险，建议拦截该请求。"
    assert get_standard_explanation("INVALID", None) == "模型输出无法解析或服务异常，无法完成安全判断。"
