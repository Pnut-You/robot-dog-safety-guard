#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from app.config import PROJECT_ROOT, get_model_config  # noqa: E402
from app.inference import SafetyDetector  # noqa: E402
from evaluation.evaluator import (  # noqa: E402
    evaluate_dataset, evaluate_multiclass_dataset, evaluate_native_safety_dataset, load_dataset, load_multiclass_dataset,
    validate_dataset_contract, validate_multiclass_contract, warmup_detector,
)
from evaluation.metrics import calculate_metrics, calculate_native_safety_metrics  # noqa: E402
from evaluation.multiclass_metrics import calculate_multiclass_metrics  # noqa: E402

DEFAULT_DATASET = "datasets/raw/sample_guard_safety_binary_eval.jsonl"
MULTICLASS_DATASET = "datasets/raw/sample_input_safety_multiclass_eval.jsonl"


def _print_binary_metrics(title: str, metrics: dict) -> None:
    print(f"\n{title}")
    keys = (
        ("total", "样本数"), ("strict_accuracy", "Strict Accuracy"),
        ("safe_precision", "SAFE Precision"), ("safe_recall", "SAFE Recall"), ("safe_f1", "SAFE F1"),
        ("unsafe_precision", "UNSAFE Precision"), ("unsafe_recall", "UNSAFE Recall"), ("unsafe_f1", "UNSAFE F1"),
        ("macro_precision", "Macro Precision"), ("macro_recall", "Macro Recall"), ("macro_f1", "Macro F1"),
        ("false_pass_rate", "False Pass Rate"), ("false_block_rate", "False Block Rate"),
        ("invalid_count", "INVALID 数量"), ("invalid_rate", "INVALID Rate"),
        ("average_latency_ms", "平均延迟 (ms)"), ("p50_latency_ms", "P50 延迟 (ms)"),
        ("p95_latency_ms", "P95 延迟 (ms)"), ("max_latency_ms", "最大延迟 (ms)"),
    )
    for key, label in keys:
        value = metrics[key]
        print(f"{label}: {value:.4f}" if isinstance(value, float) else f"{label}: {value}")


def main() -> int:
    parser = argparse.ArgumentParser(description="评测机器狗输入安全模型")
    parser.add_argument("--task", choices=["binary", "multiclass", "native_safety"], default="binary")
    parser.add_argument("--dataset", help="JSONL 数据集路径；省略时按 task 选择默认数据集")
    parser.add_argument("--model", help="models.yaml 中的模型键；默认使用 active_model")
    parser.add_argument("--prompt", choices=["zero_shot", "few_shot"], default="few_shot")
    parser.add_argument("--protocol", choices=["strict", "native"], default="strict",
                        help="strict=统一裸标签协议；native=专用 Guard 官方输出协议")
    parser.add_argument("--warmup-requests", type=int, default=1, help="不计入正式结果的预热请求数（默认: 1）")
    args = parser.parse_args()
    dataset_path = Path(args.dataset or (MULTICLASS_DATASET if args.task in {"multiclass", "native_safety"} else DEFAULT_DATASET))
    if not dataset_path.is_absolute():
        dataset_path = PROJECT_ROOT / dataset_path
    try:
        # Dataset validation must complete before a model client is created.
        if args.task in {"multiclass", "native_safety"}:
            items = load_multiclass_dataset(dataset_path)
            validate_multiclass_contract(items)
            if args.task == "native_safety":
                if args.protocol != "native":
                    raise ValueError("native_safety 任务必须使用 --protocol native")
                detector = SafetyDetector(prompt_name=args.prompt, model_config=get_model_config(args.model),
                                          protocol="native", task="binary")
            else:
                detector = SafetyDetector(prompt_name=args.prompt, model_config=get_model_config(args.model),
                                          protocol=args.protocol, task="multiclass")
        else:
            items = load_dataset(dataset_path)
            validate_dataset_contract(items)
            detector = SafetyDetector(prompt_name=args.prompt, model_config=get_model_config(args.model), protocol=args.protocol)
        warmup_detector(items, detector, args.warmup_requests)
        if args.task == "multiclass":
            rows = evaluate_multiclass_dataset(items, detector)
            metrics = calculate_multiclass_metrics(rows)
        elif args.task == "native_safety":
            rows = evaluate_native_safety_dataset(items, detector)
            metrics = calculate_native_safety_metrics(rows)
        else:
            rows = evaluate_dataset(items, detector)
            metrics = calculate_metrics(rows)
    except (OSError, ValueError) as exc:
        print(f"评测失败: {exc}", file=sys.stderr)
        return 1

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    safe_name = re.sub(r"[^A-Za-z0-9_.-]+", "_", detector.config.served_model_name)
    if args.task == "multiclass":
        output_path = (PROJECT_ROOT / "results" / "input_safety_multiclass" /
                       f"{safe_name}_sample_input_safety_multiclass_eval_{args.protocol}_{timestamp}.json")
    elif args.task == "native_safety":
        output_path = PROJECT_ROOT / "results" / "native_safety" / f"{safe_name}_sample_input_safety_multiclass_eval_native_{timestamp}.json"
    else:
        output_path = PROJECT_ROOT / "results" / f"{safe_name}_binary_safety_{args.protocol}_{timestamp}.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "evaluation_protocol": (f"yufeng_taxonomy_v2_{args.protocol}" if args.task == "multiclass" else
                                ("native_safety_detection_v1" if args.task == "native_safety" else
                                 f"binary_safety_v1_{args.protocol}")),
        "task": args.task,
        "adapter_protocol": args.protocol,
        "model": detector.config.served_model_name,
        "prompt": args.prompt,
        "dataset": str(dataset_path.relative_to(PROJECT_ROOT)),
        "dataset_sha256": hashlib.sha256(dataset_path.read_bytes()).hexdigest(),
        "inference_config": {
            "temperature": 0, "top_p": 1,
            "max_tokens": ((1 if args.protocol == "native" else 4) if args.task == "multiclass" else
                           (4 if args.protocol == "strict" else
                            (1 if detector.config.guard_family == "yufeng" else
                             (64 if detector.config.guard_family == "singguard" else 32)))),
            "concurrency": 1,
            "warmup_requests": args.warmup_requests,
            "latency_scope": "formal requests only; warmup excluded",
        },
        "metrics": metrics,
        "details": rows,
    }
    output_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    if args.task == "multiclass":
        print("\nYuFeng 原生标签空间评测结果")
        for key in ("total", "overall_accuracy", "macro_precision", "macro_recall", "macro_f1",
                    "invalid_count", "invalid_rate", "format_error_count", "average_latency_ms", "p95_latency_ms"):
            print(f"{key}: {metrics[key]}")
        print(f"风险检测: {metrics['risk_detection']}")
        print(f"UNSAFE 风险代码 Accuracy: {metrics['risk_type']['accuracy']}")
        print(f"UNSAFE 风险代码 Macro F1: {metrics['risk_type']['macro_f1']}")
        for risk, values in metrics["risk_type"]["classes"].items():
            print(f"  {risk}: {values}")
        print(f"业务指标: {metrics['business']}")
        print(f"\n详细结果: {output_path}")
        return 0

    if args.task == "native_safety":
        _print_binary_metrics("Guard 原生风险检测结果（SAFE 与 IRRELEVANT 合并为无风险）", metrics)
        print(f"\n原生风险代码分布: {metrics['native_risk_code_distribution']}")
        print("\n项目 risk_type 的风险检出率（不是原生类别映射）:")
        for risk_type, values in metrics["risk_types"].items():
            print(f"  {risk_type}: count={values['count']}, UNSAFE Recall={values['unsafe_recall']:.4f}, "
                  f"False Pass Rate={values['false_pass_rate']:.4f}")
        print(f"\n详细结果: {output_path}")
        return 0

    _print_binary_metrics("核心安全测试集（模型选型主结果）", metrics["safety_core"])
    print("\nrisk_type 召回:")
    for risk_type, values in metrics["safety_core"]["risk_types"].items():
        print(f"  {risk_type}: count={values['count']}, UNSAFE Recall={values['unsafe_recall']:.4f}, False Pass Rate={values['false_pass_rate']:.4f}")
    noise = metrics["noise_robustness"]
    print("\n噪声鲁棒性测试集")
    for key, label in (("total", "样本数"), ("safe_count", "SAFE 数量"), ("unsafe_count", "UNSAFE 数量"),
                       ("invalid_count", "INVALID 数量"), ("safe_acceptance_rate", "Safe Acceptance Rate"),
                       ("noise_false_block_rate", "Noise False Block Rate"), ("invalid_rate", "INVALID Rate"),
                       ("average_latency_ms", "平均延迟 (ms)"), ("p95_latency_ms", "P95 延迟 (ms)")):
        value = noise[key]
        print(f"{label}: {value:.4f}" if isinstance(value, float) else f"{label}: {value}")
    _print_binary_metrics("全量测试集（补充结果）", metrics["full_dataset"])
    print(f"\n详细结果: {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
