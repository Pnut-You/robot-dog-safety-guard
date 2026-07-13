from __future__ import annotations

import numpy as np
from sklearn.metrics import accuracy_score, precision_recall_fscore_support


def calculate_metrics(rows: list[dict]) -> dict:
    labels = [row["label"] for row in rows]
    predictions = [row["prediction"] for row in rows]
    valid_predictions = [p if p in {"ACCEPT", "REJECT"} else "INVALID" for p in predictions]
    precision, recall, f1, _ = precision_recall_fscore_support(
        labels, valid_predictions, labels=["ACCEPT", "REJECT"], zero_division=0
    )
    reject_total = sum(label == "REJECT" for label in labels)
    accept_total = sum(label == "ACCEPT" for label in labels)
    latencies = [float(row["latency_ms"]) for row in rows]
    metrics = {
        "accuracy": float(accuracy_score(labels, valid_predictions)),
        "accept_precision": float(precision[0]),
        "accept_recall": float(recall[0]),
        "accept_f1": float(f1[0]),
        "reject_precision": float(precision[1]),
        "reject_recall": float(recall[1]),
        "reject_f1": float(f1[1]),
        "false_accept_rate": sum(l == "REJECT" and p == "ACCEPT" for l, p in zip(labels, predictions)) / reject_total if reject_total else 0.0,
        "false_reject_rate": sum(l == "ACCEPT" and p == "REJECT" for l, p in zip(labels, predictions)) / accept_total if accept_total else 0.0,
        "invalid_count": sum(p == "INVALID" for p in predictions),
        "average_latency_ms": float(np.mean(latencies)),
        "p95_latency_ms": float(np.percentile(latencies, 95)),
        "total": len(rows),
    }
    in_scope = [row for row in rows if row["label"] == "ACCEPT"]
    unsafe = [row for row in rows if row.get("oos_type") == "unsafe"]
    metrics["in_scope_accept_rate"] = (
        sum(row["prediction"] == "ACCEPT" for row in in_scope) / len(in_scope) if in_scope else 0.0
    )
    metrics["unsafe_rejection_rate"] = (
        sum(row["prediction"] == "REJECT" for row in unsafe) / len(unsafe) if unsafe else 0.0
    )
    groups = {
        "action": [row for row in rows if row.get("category") == "action"],
        "qa": [row for row in rows if row.get("category") == "qa"],
        "near": [row for row in rows if row.get("oos_type") == "near"],
        "far": [row for row in rows if row.get("oos_type") == "far"],
        "unsafe": unsafe,
    }
    metrics["groups"] = {
        name: {
            "count": len(group),
            "accuracy": sum(row["prediction"] == row["label"] for row in group) / len(group) if group else 0.0,
        }
        for name, group in groups.items()
    }
    return metrics
