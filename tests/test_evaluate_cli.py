import json
from types import SimpleNamespace
from unittest.mock import patch

from app.config import PROJECT_ROOT as REAL_ROOT
from app.schemas import PredictionResult
from scripts import evaluate


class MockDetector:
    calls = 0

    def __init__(self, prompt_name, model_config, protocol="strict"):
        self.prompt_name = prompt_name
        self.protocol = protocol
        self.config = model_config

    def predict(self, text):
        type(self).calls += 1
        return PredictionResult(text=text, prediction="SAFE", raw_output="SAFE", latency_ms=2.0, model_name=self.config.served_model_name)


def test_cli_mock_flow_validates_warms_and_saves_three_sections(tmp_path, monkeypatch):
    source = REAL_ROOT / "datasets/raw/sample_guard_safety_binary_eval.jsonl"
    dataset = tmp_path / "datasets/raw/sample_guard_safety_binary_eval.jsonl"
    dataset.parent.mkdir(parents=True)
    dataset.write_bytes(source.read_bytes())
    config = SimpleNamespace(served_model_name="mock-model", guard_family="none")
    MockDetector.calls = 0
    monkeypatch.setattr(evaluate, "PROJECT_ROOT", tmp_path)
    monkeypatch.setattr(evaluate, "SafetyDetector", MockDetector)
    monkeypatch.setattr(evaluate, "get_model_config", lambda _: config)
    with patch("sys.argv", ["evaluate.py"]):
        assert evaluate.main() == 0
    assert MockDetector.calls == 1001
    outputs = list((tmp_path / "results").glob("mock-model_binary_safety_*.json"))
    assert len(outputs) == 1
    payload = json.loads(outputs[0].read_text(encoding="utf-8"))
    assert payload["evaluation_protocol"] == "binary_safety_v1_strict"
    assert payload["adapter_protocol"] == "strict"
    assert payload["inference_config"]["warmup_requests"] == 1
    assert set(payload["metrics"]) == {"ranking_basis", "selection_priority", "safety_core", "noise_robustness", "full_dataset"}
    assert len(payload["details"]) == 1000
    assert {"expected", "predicted", "is_correct", "is_invalid"} <= payload["details"][0].keys()
