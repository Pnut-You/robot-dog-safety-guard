import pytest
from types import SimpleNamespace
from unittest.mock import patch

from app.config import ModelConfig
from app.inference import SafetyDetector, get_standard_explanation, parse_native_guard_output, parse_prediction


@pytest.mark.parametrize(("raw", "expected"), [
    ("SAFE", "SAFE"), ("safe", "SAFE"), ("  SAFE\n", "SAFE"),
    ('"UNSAFE"', "UNSAFE"), ("“SAFE”", "SAFE"), ("'unsafe'", "UNSAFE"),
    ("SAFE。", "SAFE"), ("UNSAFE.", "UNSAFE"),
    ("SAFE because...", "INVALID"), ("This is SAFE", "INVALID"),
    ("PASS", "INVALID"), ("BLOCK", "INVALID"), ("IRRELEVANT", "INVALID"),
    ("sec", "INVALID"), ("dw", "INVALID"), ("SAFE.", "SAFE"),
    ("SAFE..", "INVALID"), ("", "INVALID"), ('"SAFE', "INVALID"),
])
def test_parse_prediction_is_exact(raw, expected):
    assert parse_prediction(raw) == expected


def test_unsafe_is_not_parsed_as_safe_by_substring():
    assert parse_prediction("UNSAFE") == "UNSAFE"
    assert parse_prediction("This is UNSAFE") == "INVALID"


@pytest.mark.parametrize(("family", "raw", "expected"), [
    ("yufeng", "sec", "SAFE"),
    ("yufeng", "dw", "UNSAFE"),
    ("yufeng", "unknown", "INVALID"),
    ("qwen3guard", "Safety: Safe", "SAFE"),
    ("qwen3guard", "Safety: Unsafe\nCategories: Violent", "UNSAFE"),
    ("qwen3guard", "Safety: Controversial\nCategories: Violent", "UNSAFE"),
    ("qwen3guard", "This is unsafe", "INVALID"),
    ("llama_guard", "safe", "SAFE"),
    ("llama_guard", "unsafe\nS1", "UNSAFE"),
    ("llama_guard", "", "INVALID"),
])
def test_native_guard_parsers(family, raw, expected):
    assert parse_native_guard_output(raw, family).prediction == expected


def test_native_protocol_uses_user_only_official_template(tmp_path):
    config = ModelConfig(
        display_name="guard", model_id="x", local_path=tmp_path, served_model_name="guard",
        host="127.0.0.1", port=8000, api_key="token", dtype="auto", max_model_len=128,
        gpu_memory_utilization=0.5, native_guard=True, guard_family="qwen3guard",
    )
    response = SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content="Safety: Unsafe\nCategories: Violent"))])
    with patch("app.inference.OpenAI") as client_cls:
        client_cls.return_value.chat.completions.create.return_value = response
        result = SafetyDetector(model_config=config, protocol="native").predict("撞倒他")
    assert result.prediction == "UNSAFE"
    assert result.risk_category == "Violent"
    kwargs = client_cls.return_value.chat.completions.create.call_args.kwargs
    assert kwargs["messages"] == [{"role": "user", "content": "撞倒他"}]
    assert kwargs["max_tokens"] == 32


def test_yufeng_native_protocol_uses_user_only_and_preserves_code(tmp_path):
    config = ModelConfig(
        display_name="YuFeng", model_id="x", local_path=tmp_path, served_model_name="yufeng",
        host="127.0.0.1", port=8000, api_key="token", dtype="auto", max_model_len=128,
        gpu_memory_utilization=0.5, native_guard=True, guard_family="yufeng",
    )
    response = SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content="ph\n风险说明"))])
    with patch("app.inference.OpenAI") as client_cls:
        client_cls.return_value.chat.completions.create.return_value = response
        result = SafetyDetector(model_config=config, protocol="native").predict("撞倒他")
    assert result.prediction == "UNSAFE"
    assert result.risk_category == "ph"
    kwargs = client_cls.return_value.chat.completions.create.call_args.kwargs
    assert kwargs["messages"] == [{"role": "user", "content": "撞倒他"}]
    assert kwargs["max_tokens"] == 1


def test_native_protocol_rejects_non_guard_model(tmp_path):
    config = ModelConfig(
        display_name="instruct", model_id="x", local_path=tmp_path, served_model_name="instruct",
        host="127.0.0.1", port=8000, api_key="token", dtype="auto", max_model_len=128,
        gpu_memory_utilization=0.5,
    )
    with pytest.raises(ValueError, match="不支持 native"):
        SafetyDetector(model_config=config, protocol="native")


def test_standard_explanations_match_binary_protocol():
    assert "不代表" in get_standard_explanation("SAFE", None)
    assert "建议拦截" in get_standard_explanation("UNSAFE", None)
    assert get_standard_explanation("INVALID", None) == "模型输出无法解析或服务异常，无法完成安全判断。"


def test_all_models_use_binary_prompt_and_fixed_generation_parameters(tmp_path):
    config = ModelConfig(
        display_name="native-model", model_id="x", local_path=tmp_path, served_model_name="native",
        host="127.0.0.1", port=8000, api_key="token", dtype="auto", max_model_len=128,
        gpu_memory_utilization=0.5, native_guard=True,
    )
    response = SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content="SAFE"))])
    with patch("app.inference.OpenAI") as client_cls:
        client_cls.return_value.chat.completions.create.return_value = response
        result = SafetyDetector(prompt_name="zero_shot", model_config=config).predict("嗯")
    assert result.prediction == "SAFE"
    kwargs = client_cls.return_value.chat.completions.create.call_args.kwargs
    assert kwargs["temperature"] == 0 and kwargs["top_p"] == 1 and kwargs["max_tokens"] == 4
    assert kwargs["messages"][0]["role"] == "system"
    assert "SAFE 或 UNSAFE" in kwargs["messages"][0]["content"]
