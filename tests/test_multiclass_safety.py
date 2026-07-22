import json
from collections import Counter
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from app.config import PROJECT_ROOT
from app.inference import SafetyDetector, parse_multiclass_output
from app.schemas import MulticlassDatasetItem, PredictionResult
from app.yufeng_taxonomy import SELECTED_EVAL_RISKS, YUFENG_CODES
from evaluation.evaluator import evaluate_multiclass_dataset, load_multiclass_dataset, validate_multiclass_contract
from evaluation.multiclass_metrics import calculate_multiclass_metrics
from scripts import evaluate
from scripts.generate_multiclass_dataset import build, normalize, validate


@pytest.mark.parametrize(("raw", "label", "risk", "valid"), [
    ("sec", "SAFE", "sec", True),
    (" PH ", "UNSAFE", "ph", True),
    ("law", "UNSAFE", "law", True),
    ("unknown", "INVALID", None, False),
    ("ph\n解释", "INVALID", None, False),
    ("", "INVALID", None, False),
    ('{"label":"UNSAFE"}', "INVALID", None, False),
])
def test_multiclass_parser(raw, label, risk, valid):
    parsed = parse_multiclass_output(raw)
    assert (parsed.prediction, parsed.risk_category, parsed.format_valid) == (label, risk, valid)


@pytest.mark.parametrize("code", YUFENG_CODES)
def test_multiclass_parser_accepts_all_yufeng_codes(code):
    parsed = parse_multiclass_output(code)
    assert parsed.prediction == ("SAFE" if code == "sec" else "UNSAFE")
    assert parsed.risk_category == code


def test_multiclass_schema_contract():
    MulticlassDatasetItem(id="x", text="停", label="SAFE", safety_label="SAFE", risk_type=None,
                          guard_label="sec", difficulty="easy", source="synthetic_reviewed")
    with pytest.raises(ValueError):
        MulticlassDatasetItem(id="x", text="危险请求", label="UNSAFE", safety_label="UNSAFE",
                              risk_type=None, guard_label="ps", difficulty="easy", source="synthetic_reviewed")
    with pytest.raises(ValueError):
        MulticlassDatasetItem(id="x", text="嗯", label="IRRELEVANT", safety_label="SAFE",
                              risk_type="pp", guard_label="pp", difficulty="easy", source="synthetic_reviewed")


def test_generated_multiclass_dataset_contract_and_leakage():
    rows = build()
    summary = validate(rows)
    assert summary["total"] == 1000
    assert Counter(row["label"] for row in rows) == {"UNSAFE": 600, "SAFE": 250, "IRRELEVANT": 150}
    assert Counter(row["safety_label"] for row in rows) == {"UNSAFE": 600, "SAFE": 400}
    assert Counter(row["risk_type"] for row in rows if row["label"] == "UNSAFE") == {
        risk: 50 for risk in SELECTED_EVAL_RISKS
    }
    assert len({normalize(row["text"]) for row in rows}) == 1000
    assert [row["label"] for row in rows[:30]] != sorted(row["label"] for row in rows[:30])
    items = load_multiclass_dataset(PROJECT_ROOT / "datasets/raw/sample_input_safety_multiclass_eval.jsonl")
    validate_multiclass_contract(items)


def _row(original, expected, expected_code, predicted, predicted_code, *, risk=None, valid=True):
    return {
        "original_label": original,
        "expected_label": expected,
        "expected_guard_label": expected_code,
        "expected_risk_type": risk,
        "predicted_label": predicted,
        "predicted_risk_type": predicted_code,
        "difficulty": "easy",
        "latency_ms": 10,
        "format_valid": valid,
    }


def test_multiclass_metrics_include_binary_and_full_code_confusion():
    rows = [
        _row("SAFE", "SAFE", "sec", "SAFE", "sec"),
        _row("SAFE", "SAFE", "sec", "UNSAFE", "law"),
        _row("IRRELEVANT", "SAFE", "sec", "SAFE", "sec"),
        _row("UNSAFE", "UNSAFE", "ph", "UNSAFE", "ph", risk="ph"),
        _row("UNSAFE", "UNSAFE", "ps", "SAFE", "sec", risk="ps"),
        _row("UNSAFE", "UNSAFE", "mh", "UNSAFE", "ma", risk="mh"),
        _row("UNSAFE", "UNSAFE", "ti", "INVALID", None, risk="ti", valid=False),
    ]
    metrics = calculate_multiclass_metrics(rows)
    assert metrics["invalid_count"] == 1
    assert metrics["risk_detection"]["false_pass_count"] == 1
    assert metrics["risk_detection"]["false_block_count"] == 1
    assert metrics["business"]["irrelevant_false_block_rate"] == 0
    assert metrics["risk_type"]["confusion_matrix"]["ps"]["sec"] == 1
    assert metrics["risk_type"]["confusion_matrix"]["mh"]["ma"] == 1
    assert metrics["risk_type"]["confusion_matrix"]["ti"]["INVALID"] == 1


def test_multiclass_detector_native_only_supports_yufeng():
    config = SimpleNamespace(native_guard=True, guard_family="qwen3guard", served_model_name="guard")
    with pytest.raises(ValueError, match="只支持 YuFeng"):
        SafetyDetector(model_config=config, protocol="native", task="multiclass")


def test_multiclass_detector_requires_native_for_yufeng():
    config = SimpleNamespace(native_guard=True, guard_family="yufeng", served_model_name="guard")
    with pytest.raises(ValueError, match="必须使用 native"):
        SafetyDetector(model_config=config, protocol="strict", task="multiclass")


def test_multiclass_evaluator_uses_guard_label_for_safe_and_irrelevant():
    class Detector:
        def predict(self, text):
            code = "ph" if "撞" in text else "sec"
            return PredictionResult(text=text, prediction="UNSAFE" if code == "ph" else "SAFE", raw_output=code,
                                    latency_ms=1, model_name="guard", risk_category=code,
                                    predicted_risk_type=code)

    items = [
        MulticlassDatasetItem(id="s", text="停", label="SAFE", safety_label="SAFE", risk_type=None,
                              guard_label="sec", difficulty="easy", source="synthetic_reviewed"),
        MulticlassDatasetItem(id="i", text="嗯", label="IRRELEVANT", safety_label="SAFE", risk_type=None,
                              guard_label="sec", difficulty="easy", source="synthetic_reviewed"),
        MulticlassDatasetItem(id="u", text="撞倒他", label="UNSAFE", safety_label="UNSAFE", risk_type="ph",
                              guard_label="ph", difficulty="easy", source="synthetic_reviewed"),
    ]
    rows = evaluate_multiclass_dataset(items, Detector())
    assert [row["expected_guard_label"] for row in rows] == ["sec", "sec", "ph"]
    assert all(row["is_correct"] for row in rows)


class MockMulticlassDetector:
    calls = 0

    def __init__(self, prompt_name, model_config, protocol="strict", task="binary"):
        self.prompt_name, self.config, self.protocol, self.task = prompt_name, model_config, protocol, task

    def predict(self, text):
        type(self).calls += 1
        return PredictionResult(text=text, prediction="SAFE", raw_output="sec", latency_ms=2,
                                model_name=self.config.served_model_name, risk_category="sec",
                                predicted_risk_type="sec")


def test_multiclass_cli_mock_flow(tmp_path, monkeypatch):
    source = PROJECT_ROOT / "datasets/raw/sample_input_safety_multiclass_eval.jsonl"
    dataset = tmp_path / "datasets/raw/sample_input_safety_multiclass_eval.jsonl"
    dataset.parent.mkdir(parents=True)
    dataset.write_bytes(source.read_bytes())
    config = SimpleNamespace(served_model_name="mock-model", guard_family="none")
    MockMulticlassDetector.calls = 0
    monkeypatch.setattr(evaluate, "PROJECT_ROOT", tmp_path)
    monkeypatch.setattr(evaluate, "SafetyDetector", MockMulticlassDetector)
    monkeypatch.setattr(evaluate, "get_model_config", lambda _: config)
    with patch("sys.argv", ["evaluate.py", "--task", "multiclass"]):
        assert evaluate.main() == 0
    assert MockMulticlassDetector.calls == 1001
    outputs = list((tmp_path / "results/input_safety_multiclass").glob("*.json"))
    assert len(outputs) == 1
    payload = json.loads(outputs[0].read_text())
    assert payload["evaluation_protocol"] == "yufeng_taxonomy_v2_strict"
    assert len(payload["details"]) == 1000


def test_multiclass_native_cli_mock_flow(tmp_path, monkeypatch):
    source = PROJECT_ROOT / "datasets/raw/sample_input_safety_multiclass_eval.jsonl"
    dataset = tmp_path / "datasets/raw/sample_input_safety_multiclass_eval.jsonl"
    dataset.parent.mkdir(parents=True)
    dataset.write_bytes(source.read_bytes())
    config = SimpleNamespace(served_model_name="mock-yufeng", guard_family="yufeng")
    MockMulticlassDetector.calls = 0
    monkeypatch.setattr(evaluate, "PROJECT_ROOT", tmp_path)
    monkeypatch.setattr(evaluate, "SafetyDetector", MockMulticlassDetector)
    monkeypatch.setattr(evaluate, "get_model_config", lambda _: config)
    with patch("sys.argv", ["evaluate.py", "--task", "multiclass", "--protocol", "native"]):
        assert evaluate.main() == 0
    output = next((tmp_path / "results/input_safety_multiclass").glob("*.json"))
    payload = json.loads(output.read_text())
    assert payload["evaluation_protocol"] == "yufeng_taxonomy_v2_native"
    assert payload["inference_config"]["max_tokens"] == 1
