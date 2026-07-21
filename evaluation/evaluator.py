from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

from pydantic import ValidationError

from app.inference import SafetyDetector
from app.schemas import MULTICLASS_RISK_TYPES, DatasetItem, MulticlassDatasetItem, PredictionResult


def load_dataset(path: str | Path) -> list[DatasetItem]:
    dataset_path = Path(path)
    if not dataset_path.is_file():
        raise FileNotFoundError(f"数据集不存在: {dataset_path}")
    items: list[DatasetItem] = []
    for line_number, line in enumerate(dataset_path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            items.append(DatasetItem.model_validate_json(line))
        except (ValidationError, json.JSONDecodeError) as exc:
            raise ValueError(f"数据集第 {line_number} 行无效: {exc}") from exc
    if not items:
        raise ValueError("数据集为空")
    return items


def load_multiclass_dataset(path: str | Path) -> list[MulticlassDatasetItem]:
    dataset_path = Path(path)
    if not dataset_path.is_file():
        raise FileNotFoundError(f"数据集不存在: {dataset_path}")
    items: list[MulticlassDatasetItem] = []
    for line_number, line in enumerate(dataset_path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            items.append(MulticlassDatasetItem.model_validate_json(line))
        except (ValidationError, json.JSONDecodeError) as exc:
            raise ValueError(f"多分类数据集第 {line_number} 行无效: {exc}") from exc
    if not items:
        raise ValueError("多分类数据集为空")
    return items


def validate_multiclass_contract(items: list[MulticlassDatasetItem]) -> None:
    labels = Counter(item.label for item in items)
    risks = Counter(item.risk_type for item in items if item.label == "UNSAFE")
    difficulties = Counter(item.difficulty for item in items)
    expected_risks = Counter({"legal_risk": 60, "self_harm": 70, "harm_others": 90,
                              "safety_bypass": 60, "dangerous_operation": 80, "other_unsafe": 40})
    errors = []
    if len(items) != 1000:
        errors.append(f"总数应为1000，实际{len(items)}")
    if labels != Counter({"SAFE": 400, "UNSAFE": 400, "IRRELEVANT": 200}):
        errors.append(f"标签分布错误: {dict(labels)}")
    if risks != expected_risks or set(risks) != MULTICLASS_RISK_TYPES:
        errors.append(f"risk_type分布错误: {dict(risks)}")
    if difficulties != Counter({"easy": 300, "medium": 400, "hard": 300}):
        errors.append(f"难度分布错误: {dict(difficulties)}")
    if len({item.id for item in items}) != len(items):
        errors.append("ID不唯一")
    if len({item.text for item in items}) != len(items):
        errors.append("文本不唯一")
    if errors:
        raise ValueError("多分类数据集校验失败: " + "; ".join(errors))


def validate_dataset_contract(items: list[DatasetItem]) -> None:
    labels = Counter(item.label for item in items)
    subsets = Counter(item.eval_subset for item in items)
    core = Counter(item.label for item in items if item.eval_subset == "safety_core")
    noise = Counter(item.label for item in items if item.eval_subset == "noise_robustness")
    errors = []
    if len(items) != 1000:
        errors.append(f"总数应为 1000，实际 {len(items)}")
    if labels != Counter({"SAFE": 600, "UNSAFE": 400}):
        errors.append(f"标签分布应为 SAFE=600/UNSAFE=400，实际 {dict(labels)}")
    if subsets != Counter({"safety_core": 800, "noise_robustness": 200}):
        errors.append(f"子集分布应为 safety_core=800/noise_robustness=200，实际 {dict(subsets)}")
    if core != Counter({"SAFE": 400, "UNSAFE": 400}):
        errors.append(f"核心集应为 SAFE=400/UNSAFE=400，实际 {dict(core)}")
    if noise != Counter({"SAFE": 200}):
        errors.append(f"噪声集必须全部为 SAFE，实际 {dict(noise)}")
    if len({item.id for item in items}) != len(items):
        errors.append("数据集 ID 不唯一")
    if errors:
        raise ValueError("二分类数据集校验失败: " + "; ".join(errors))


def evaluate_dataset(items: list[DatasetItem], detector: SafetyDetector) -> list[dict]:
    details = []
    total = len(items)
    width = len(str(total))
    for index, item in enumerate(items, 1):
        result: PredictionResult = detector.predict(item.text)
        progress = (
            f"[{index:0{width}d}/{total}] {item.id} | 期望={item.label} | "
            f"预测={result.prediction} | 耗时={result.latency_ms:.1f} ms"
        )
        if result.error:
            progress += f" | 错误={result.error}"
        print(progress, flush=True)
        details.append({
            "id": item.id,
            "text": item.text,
            "expected": item.label,
            "predicted": result.prediction,
            "raw_output": result.raw_output,
            "is_correct": result.prediction == item.label,
            "is_invalid": result.prediction == "INVALID",
            "original_label": item.original_label,
            "eval_subset": item.eval_subset,
            "risk_type": item.risk_type,
            "irrelevant_type": item.irrelevant_type,
            "difficulty": item.difficulty,
            "scene_type": item.scene_type,
            "latency_ms": result.latency_ms,
            "model_name": result.model_name,
            "model_risk_category": result.risk_category,
            "error": result.error,
        })
    return details


def evaluate_multiclass_dataset(items: list[MulticlassDatasetItem], detector: SafetyDetector) -> list[dict]:
    details = []
    total = len(items)
    width = len(str(total))
    for index, item in enumerate(items, 1):
        result = detector.predict(item.text)
        print(f"[{index:0{width}d}/{total}] {item.id} | 期望={item.label}/{item.risk_type} | "
              f"预测={result.prediction}/{result.predicted_risk_type} | 耗时={result.latency_ms:.1f} ms", flush=True)
        label_correct = result.prediction == item.label
        risk_correct = (result.format_valid if item.label != "UNSAFE" else (
            result.prediction == "UNSAFE" and result.predicted_risk_type == item.risk_type
        ))
        details.append({
            "id": item.id, "text": item.text,
            "expected_label": item.label, "expected_risk_type": item.risk_type,
            "predicted_label": result.prediction, "predicted_risk_type": result.predicted_risk_type,
            "raw_output": result.raw_output, "is_label_correct": label_correct,
            "is_risk_type_correct": risk_correct,
            "is_correct": label_correct and risk_correct and result.format_valid,
            "is_invalid": result.prediction == "INVALID", "format_valid": result.format_valid,
            "parse_error": result.parse_error, "difficulty": item.difficulty,
            "source": item.source, "latency_ms": result.latency_ms,
            "model_name": result.model_name, "error": result.error,
        })
    return details


def evaluate_native_safety_dataset(items: list[MulticlassDatasetItem], detector: SafetyDetector) -> list[dict]:
    """Evaluate a native Guard as risk/no-risk without pretending it supports project routing labels."""
    details = []
    total = len(items)
    width = len(str(total))
    for index, item in enumerate(items, 1):
        result = detector.predict(item.text)
        expected = "UNSAFE" if item.label == "UNSAFE" else "SAFE"
        print(f"[{index:0{width}d}/{total}] {item.id} | 原标签={item.label} | "
              f"期望风险={expected} | 预测={result.prediction}/{result.risk_category} | "
              f"耗时={result.latency_ms:.1f} ms", flush=True)
        details.append({
            "id": item.id, "text": item.text,
            "original_label": item.label, "expected": expected,
            "predicted": result.prediction, "native_risk_code": result.risk_category,
            "raw_output": result.raw_output, "is_correct": result.prediction == expected,
            "is_invalid": result.prediction == "INVALID", "difficulty": item.difficulty,
            "risk_type": item.risk_type, "source": item.source,
            "latency_ms": result.latency_ms, "model_name": result.model_name,
            "error": result.error,
        })
    return details


def warmup_detector(items: list[DatasetItem], detector: SafetyDetector, requests: int = 1) -> None:
    if requests < 0:
        raise ValueError("预热次数不能为负数")
    for index in range(requests):
        result = detector.predict(items[index % len(items)].text)
        if result.error:
            raise ValueError(f"模型预热失败: {result.error}")
