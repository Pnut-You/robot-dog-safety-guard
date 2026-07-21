#!/usr/bin/env python3
from __future__ import annotations

import hashlib
import json
from datetime import datetime
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
DATASET = ROOT / "datasets/raw/sample_input_safety_multiclass_eval.jsonl"
MULTICLASS_RESULTS = ROOT / "results/input_safety_multiclass"
NATIVE_RESULTS = ROOT / "results/native_safety"
REPORT = ROOT / "reports/input_safety_multiclass_comparison.md"
def _latest(directory: Path, pattern: str, protocol: str, models: tuple[str, ...], dataset_hash: str) -> dict[str, tuple[Path, dict]]:
    found: dict[str, tuple[float, Path, dict]] = {}
    for path in directory.glob(pattern):
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        model = payload.get("model")
        if (payload.get("evaluation_protocol") != protocol or payload.get("dataset_sha256") != dataset_hash
                or model not in models or len(payload.get("details", [])) != 1000):
            continue
        if model not in found or path.stat().st_mtime > found[model][0]:
            found[model] = (path.stat().st_mtime, path, payload)
    return {model: (found[model][1], found[model][2]) for model in models if model in found}


def _pct(value: float) -> str:
    return f"{value * 100:.2f}%"


def main() -> None:
    dataset_hash = hashlib.sha256(DATASET.read_bytes()).hexdigest()
    config = yaml.safe_load((ROOT / "configs/models.yaml").read_text(encoding="utf-8"))["models"]
    native_models = tuple(value["served_model_name"] for value in config.values() if value.get("native_guard"))
    strict_models = tuple(value["served_model_name"] for value in config.values() if not value.get("native_guard"))
    native = _latest(NATIVE_RESULTS, "*_sample_input_safety_multiclass_eval_native_*.json",
                     "native_safety_detection_v1", native_models, dataset_hash)
    strict = _latest(MULTICLASS_RESULTS, "*_sample_input_safety_multiclass_eval_*.json",
                     "input_safety_multiclass_v1", strict_models, dataset_hash)
    if not native and not strict:
        raise SystemExit("没有找到与当前数据集哈希和评测协议匹配的完整结果")

    served_to_config = {value["served_model_name"]: value for value in config.values()}
    native_rows, strict_rows = [], []
    for model, (path, payload) in native.items():
        m = payload["metrics"]
        native_rows.append({
            "model": model, "result_file": str(path.relative_to(ROOT)),
            "accuracy": m["strict_accuracy"], "macro_f1": m["macro_f1"],
            "unsafe_recall": m["unsafe_recall"], "false_pass_rate": m["false_pass_rate"],
            "false_block_rate": m["false_block_rate"], "invalid_rate": m["invalid_rate"],
            "average_latency_ms": m["average_latency_ms"], "p95_latency_ms": m["p95_latency_ms"],
            "risk_types": m["risk_types"],
        })
    for model, (path, payload) in strict.items():
        m = payload["metrics"]
        strict_rows.append({
            "model": model, "result_file": str(path.relative_to(ROOT)),
            "accuracy": m["overall_accuracy"], "macro_f1": m["macro_f1"],
            "safe_recall": m["classes"]["SAFE"]["recall"],
            "unsafe_recall": m["classes"]["UNSAFE"]["recall"],
            "irrelevant_recall": m["classes"]["IRRELEVANT"]["recall"],
            "risk_type_accuracy": m["risk_type"]["accuracy"],
            "invalid_rate": m["invalid_rate"], "format_error_count": m["format_error_count"],
            "average_latency_ms": m["average_latency_ms"], "p95_latency_ms": m["p95_latency_ms"],
            "business": m["business"], "risk_types": m["risk_type"]["classes"],
        })

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    comparison = {
        "generated_at": timestamp, "dataset": str(DATASET.relative_to(ROOT)),
        "dataset_sha256": dataset_hash, "comparison_policy": "dual_track_no_cross_ranking",
        "native_safety": native_rows, "strict_multiclass": strict_rows,
    }
    output = MULTICLASS_RESULTS / f"model_comparison_dual_track_{timestamp}.json"
    output.write_text(json.dumps(comparison, ensure_ascii=False, indent=2), encoding="utf-8")

    lines = [
        "# 本地输入安全模型双轨重测", "",
        f"- 数据集：`{comparison['dataset']}`", f"- SHA256：`{dataset_hash}`",
        "- 样本：1000（SAFE 400 / UNSAFE 400 / IRRELEVANT 200）",
        "- 推理：temperature=0 / top_p=1 / concurrency=1 / warmup=1（不计延迟）",
        "- 部署参数：来自 `configs/models.yaml`，实际硬件与服务日志需随实验另行记录",
        "- 口径：专用 Guard 与项目三分类分轨展示，不跨轨道直接排名", "",
    ]
    risks = ("legal_risk", "self_harm", "harm_others", "safety_bypass", "dangerous_operation", "other_unsafe")
    if native_rows:
        lines += ["## Guard 原生风险检测", "",
                  "该轨道把项目 SAFE 与 IRRELEVANT 合并为无风险，只衡量风险是否检出。", "",
                  "| 模型 | Accuracy | Macro F1 | UNSAFE Recall | 漏检率 | 误拦截率 | INVALID | 平均延迟 | P95 |",
                  "|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
        for row in native_rows:
            lines.append(f"| {row['model']} | {_pct(row['accuracy'])} | {_pct(row['macro_f1'])} | "
                         f"{_pct(row['unsafe_recall'])} | {_pct(row['false_pass_rate'])} | "
                         f"{_pct(row['false_block_rate'])} | {_pct(row['invalid_rate'])} | "
                         f"{row['average_latency_ms']:.1f}ms | {row['p95_latency_ms']:.1f}ms |")
        lines += ["", "### Guard 对项目风险场景的检出率", "",
                  "这些数值只表示是否检出风险，不表示原生风险代码能映射为项目 risk_type。", "",
                  "| 模型 | legal | self_harm | harm_others | bypass | dangerous_operation | other |",
                  "|---|---:|---:|---:|---:|---:|---:|"]
        for row in native_rows:
            cells = [_pct(row["risk_types"][risk]["unsafe_recall"]) for risk in risks]
            lines.append(f"| {row['model']} | " + " | ".join(cells) + " |")
        lines.append("")

    if strict_rows:
        lines += ["## 项目严格三分类", "",
                  "| 模型 | Accuracy | Macro F1 | SAFE R | UNSAFE R | IRRELEVANT R | Risk Acc | INVALID | 平均延迟 | P95 |",
                  "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
        for row in strict_rows:
            lines.append(f"| {row['model']} | {_pct(row['accuracy'])} | {_pct(row['macro_f1'])} | "
                         f"{_pct(row['safe_recall'])} | {_pct(row['unsafe_recall'])} | {_pct(row['irrelevant_recall'])} | "
                         f"{_pct(row['risk_type_accuracy'])} | {_pct(row['invalid_rate'])} | "
                         f"{row['average_latency_ms']:.1f}ms | {row['p95_latency_ms']:.1f}ms |")
        lines += ["", "### 三分类业务指标", "",
                  "| 模型 | 危险漏检 | 安全误拦截 | 噪声误进入 | dangerous_operation R | harm_others R | self_harm R |",
                  "|---|---:|---:|---:|---:|---:|---:|"]
        for row in strict_rows:
            b = row["business"]
            lines.append(f"| {row['model']} | {_pct(b['dangerous_miss_rate'])} | {_pct(b['safe_false_block_rate'])} | "
                         f"{_pct(b['noise_entry_rate'])} | {_pct(b['dangerous_operation_recall'])} | "
                         f"{_pct(b['harm_others_recall'])} | {_pct(b['self_harm_recall'])} |")

    lines += ["", "## 部署参数与结果文件", "",
              "| 模型 | max_model_len | GPU利用率 | eager | 结果文件 |", "|---|---:|---:|---:|---|"]
    for row in (*native_rows, *strict_rows):
        c = served_to_config[row["model"]]
        lines.append(f"| {row['model']} | {c['max_model_len']} | {c['gpu_memory_utilization']} | "
                     f"{c['enforce_eager']} | `{row['result_file']}` |")
    lines += ["", "## 结论", ""]
    if native_rows:
        best_native = max(native_rows, key=lambda row: (row["unsafe_recall"], -row["false_pass_rate"], row["macro_f1"]))
        lines.append(f"- Guard 原生风险检测中，本轮危险召回最高的是 `{best_native['model']}`（{_pct(best_native['unsafe_recall'])}）。")
    if strict_rows:
        best_strict = max(strict_rows, key=lambda row: (row["macro_f1"], row["accuracy"], -row["invalid_rate"]))
        lines.append(f"- 项目严格三分类中，本轮 Macro F1 最高的是 `{best_strict['model']}`（{_pct(best_strict['macro_f1'])}）。")
    lines += ["- Guard 原生风险二分类与项目严格三分类解决的任务不同，不能跨轨直接比较准确率。", ""]
    REPORT.write_text("\n".join(lines), encoding="utf-8")
    print(output)
    print(REPORT)


if __name__ == "__main__":
    main()
