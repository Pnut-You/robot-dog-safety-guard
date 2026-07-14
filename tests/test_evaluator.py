from app.schemas import DatasetItem, PredictionResult
from evaluation.evaluator import evaluate_dataset


class StubDetector:
    def predict(self, text: str) -> PredictionResult:
        prediction = "INVALID" if text == "非法输出" else "PASS"
        return PredictionResult(text=text, prediction=prediction, raw_output="?" if prediction == "INVALID" else "sec",
                                latency_ms=8.2, model_name="stub", error="模型输出无效" if prediction == "INVALID" else None)


def test_evaluate_dataset_prints_progress_and_keeps_results(capsys):
    items = [
        DatasetItem(id="safe_001", text="向前走", label="PASS", category="safe", risk_type=None, difficulty="easy", source="manual"),
        DatasetItem(id="risk_001", text="非法输出", label="BLOCK", category="unsafe", risk_type="violence", difficulty="hard", source="manual"),
    ]
    rows = evaluate_dataset(items, StubDetector())
    output = capsys.readouterr().out
    assert "[1/2] safe_001 | 标签=PASS | 预测=PASS" in output
    assert "[2/2] risk_001 | 标签=BLOCK | 预测=INVALID" in output
    assert [row["prediction"] for row in rows] == ["PASS", "INVALID"]
