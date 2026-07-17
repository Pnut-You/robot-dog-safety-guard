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
from app.inference import SafetyDetector  # noqa: E402
from evaluation.evaluator import evaluate_dataset, load_dataset  # noqa: E402
from evaluation.metrics import calculate_metrics  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description="评测机器狗输入安全风险识别模型")
    parser.add_argument("--dataset", required=True, help="JSONL 数据集路径")
    parser.add_argument("--model", help="models.yaml 中的模型键；默认使用 active_model")
    parser.add_argument("--prompt", choices=["zero_shot", "few_shot"], default="few_shot")
    args = parser.parse_args()
    try:
        detector = SafetyDetector(prompt_name=args.prompt, model_config=get_model_config(args.model))
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
        "accuracy": "Accuracy", "pass_precision": "PASS Precision", "pass_recall": "PASS Recall",
        "pass_f1": "PASS F1", "block_precision": "BLOCK Precision", "block_recall": "BLOCK Recall",
        "block_f1": "BLOCK F1", "false_pass_rate": "False Pass Rate",
        "irrelevant_precision": "IRRELEVANT Precision", "irrelevant_recall": "IRRELEVANT Recall",
        "irrelevant_f1": "IRRELEVANT F1", "macro_f1": "Macro F1",
        "irrelevant_false_accept_rate": "无关语音误接受率",
        "false_block_rate": "False Block Rate", "invalid_count": "INVALID 数量",
        "average_latency_ms": "平均延迟 (ms)", "p95_latency_ms": "P95 延迟 (ms)", "total": "样本数",
        "dangerous_miss_rate": "危险请求漏检率", "safe_false_block_rate": "安全样本误拦截率",
        "robot_harm_recall": "机器狗相关危险请求召回率",
    }
    for key, label in names.items():
        value = metrics[key]
        print(f"{label}: {value:.4f}" if isinstance(value, float) else f"{label}: {value}")
    print("不同 risk_type 的结果:")
    for group, values in metrics["risk_types"].items():
        print(f"  {group}: 样本数={values['count']}, Accuracy={values['accuracy']:.4f}")
    print("三分类混淆矩阵:")
    for actual, values in metrics["confusion_matrix"].items():
        print(f"  {actual}: {values}")
    print(f"详细结果: {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
