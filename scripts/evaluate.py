#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import re
import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from app.config import PROJECT_ROOT, get_model_config  # noqa: E402
from app.inference import OOSDetector  # noqa: E402
from evaluation.evaluator import evaluate_dataset, load_dataset  # noqa: E402
from evaluation.metrics import calculate_metrics  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description="评测机器狗拒识模型")
    parser.add_argument("--dataset", required=True, help="JSONL 数据集路径")
    parser.add_argument("--model", help="models.yaml 中的模型键；默认使用 active_model")
    parser.add_argument("--prompt", choices=["zero_shot", "few_shot"], default="few_shot")
    args = parser.parse_args()
    try:
        detector = OOSDetector(prompt_name=args.prompt, model_config=get_model_config(args.model))
        rows = evaluate_dataset(load_dataset(args.dataset), detector)
        metrics = calculate_metrics(rows)
    except (OSError, ValueError) as exc:
        print(f"评测失败: {exc}", file=sys.stderr)
        return 1

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    safe_name = re.sub(r"[^A-Za-z0-9_.-]+", "_", detector.config.served_model_name)
    output_path = PROJECT_ROOT / "results" / f"{safe_name}_{timestamp}.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps({"model": detector.config.served_model_name, "prompt": args.prompt, "dataset": args.dataset,
                    "metrics": metrics, "details": rows}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    names = {
        "accuracy": "Accuracy", "accept_precision": "ACCEPT Precision", "accept_recall": "ACCEPT Recall",
        "accept_f1": "ACCEPT F1", "reject_precision": "REJECT Precision", "reject_recall": "REJECT Recall",
        "reject_f1": "REJECT F1", "false_accept_rate": "False Accept Rate",
        "false_reject_rate": "False Reject Rate", "invalid_count": "INVALID 数量",
        "average_latency_ms": "平均延迟 (ms)", "p95_latency_ms": "P95 延迟 (ms)", "total": "样本数",
        "in_scope_accept_rate": "能力内接受率", "unsafe_rejection_rate": "危险请求拒绝率",
    }
    for key, label in names.items():
        value = metrics[key]
        print(f"{label}: {value:.4f}" if isinstance(value, float) else f"{label}: {value}")
    print("分组指标:")
    for group, values in metrics["groups"].items():
        print(f"  {group}: 样本数={values['count']}, Accuracy={values['accuracy']:.4f}")
    print(f"详细结果: {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
