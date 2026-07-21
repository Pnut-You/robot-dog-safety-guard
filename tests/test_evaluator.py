from app.schemas import DatasetItem, PredictionResult
from evaluation.evaluator import evaluate_dataset, warmup_detector


class StubDetector:
    def __init__(self):
        self.calls = []

    def predict(self, text: str) -> PredictionResult:
        self.calls.append(text)
        prediction = "INVALID" if text == "非法输出" else "SAFE"
        return PredictionResult(text=text, prediction=prediction, raw_output="?" if prediction == "INVALID" else "SAFE",
                                latency_ms=8.2, model_name="stub", error=None)


def item(**changes):
    values = {
        "id": "safe_001", "text": "向前走", "label": "SAFE", "original_label": "PASS",
        "category": "safe", "eval_subset": "safety_core", "risk_type": None,
        "irrelevant_type": None, "scene_type": "clean", "difficulty": "easy",
        "source": "manual", "source_id": None, "template_family": "safe_move",
    }
    values.update(changes)
    return DatasetItem(**values)


def test_warmup_is_separate_and_results_use_export_schema(capsys):
    items = [
        item(),
        item(id="risk_001", text="非法输出", label="UNSAFE", original_label="BLOCK", category="unsafe", risk_type="violence", difficulty="hard"),
    ]
    detector = StubDetector()
    warmup_detector(items, detector, 1)
    rows = evaluate_dataset(items, detector)
    output = capsys.readouterr().out
    assert detector.calls == ["向前走", "向前走", "非法输出"]
    assert "[1/2] safe_001 | 期望=SAFE | 预测=SAFE" in output
    assert "[2/2] risk_001 | 期望=UNSAFE | 预测=INVALID" in output
    assert rows[0]["expected"] == "SAFE" and rows[0]["predicted"] == "SAFE"
    assert rows[0]["is_correct"] is True and rows[1]["is_invalid"] is True
