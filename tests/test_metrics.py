from evaluation.metrics import calculate_metrics


def test_safety_metrics_and_risk_groups():
    rows = [
        {"label": "PASS", "prediction": "PASS", "latency_ms": 10, "risk_type": None},
        {"label": "PASS", "prediction": "BLOCK", "latency_ms": 20, "risk_type": None},
        {"label": "BLOCK", "prediction": "BLOCK", "latency_ms": 30, "risk_type": "robot_harm"},
        {"label": "BLOCK", "prediction": "PASS", "latency_ms": 40, "risk_type": "violence"},
        {"label": "BLOCK", "prediction": "INVALID", "latency_ms": 50, "risk_type": "violence"},
        {"label": "IRRELEVANT", "prediction": "IRRELEVANT", "latency_ms": 60, "risk_type": None},
        {"label": "IRRELEVANT", "prediction": "PASS", "latency_ms": 70, "risk_type": None},
    ]
    metrics = calculate_metrics(rows)
    assert metrics["false_pass_rate"] == 1 / 3
    assert metrics["false_block_rate"] == 0.5
    assert metrics["dangerous_miss_rate"] == 2 / 3
    assert metrics["safe_false_block_rate"] == 0.5
    assert metrics["robot_harm_recall"] == 1.0
    assert metrics["invalid_count"] == 1
    assert metrics["irrelevant_recall"] == 0.5
    assert metrics["irrelevant_false_accept_rate"] == 0.5
    assert metrics["confusion_matrix"]["IRRELEVANT"]["PASS"] == 1
    assert metrics["risk_types"]["violence"] == {"count": 2, "accuracy": 0.0}
