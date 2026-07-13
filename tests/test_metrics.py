from evaluation.metrics import calculate_metrics


def test_metrics_include_policy_groups():
    rows = [
        {"label": "ACCEPT", "prediction": "ACCEPT", "latency_ms": 10, "category": "action", "oos_type": None},
        {"label": "ACCEPT", "prediction": "REJECT", "latency_ms": 20, "category": "qa", "oos_type": None},
        {"label": "REJECT", "prediction": "REJECT", "latency_ms": 30, "category": "action", "oos_type": "unsafe"},
        {"label": "REJECT", "prediction": "ACCEPT", "latency_ms": 40, "category": "unknown", "oos_type": "near"},
    ]

    metrics = calculate_metrics(rows)

    assert metrics["in_scope_accept_rate"] == 0.5
    assert metrics["unsafe_rejection_rate"] == 1.0
    assert metrics["groups"]["action"] == {"count": 2, "accuracy": 1.0}
    assert metrics["groups"]["qa"] == {"count": 1, "accuracy": 0.0}
    assert metrics["groups"]["near"] == {"count": 1, "accuracy": 0.0}
