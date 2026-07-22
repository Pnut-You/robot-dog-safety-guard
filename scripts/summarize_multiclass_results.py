#!/usr/bin/env python3
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from app.yufeng_taxonomy import SELECTED_EVAL_RISKS  # noqa: E402

DATASET = ROOT / "datasets/raw/sample_input_safety_multiclass_eval.jsonl"
RESULTS = ROOT / "results/input_safety_multiclass"
REPORT = ROOT / "reports/input_safety_multiclass_comparison.md"


def _latest(protocol: str, models: tuple[str, ...], dataset_hash: str) -> dict[str, tuple[Path, dict]]:
    found: dict[str, tuple[float, Path, dict]] = {}
    for path in RESULTS.glob("*_sample_input_safety_multiclass_eval_*.json"):
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
    configs = yaml.safe_load((ROOT / "configs/models.yaml").read_text(encoding="utf-8"))["models"]
    yufeng_models = tuple(v["served_model_name"] for v in configs.values() if v.get("guard_family") == "yufeng")
    general_models = tuple(v["served_model_name"] for v in configs.values() if not v.get("native_guard"))
    results = {
        **_latest("yufeng_taxonomy_v2_native", yufeng_models, dataset_hash),
        **_latest("yufeng_taxonomy_v2_strict", general_models, dataset_hash),
    }
    if not results:
        raise SystemExit("没有找到与当前数据集哈希和 YuFeng taxonomy v2 协议匹配的完整结果")

    lines = [
        "# YuFeng 原生标签空间模型对比", "",
        f"- 数据集：`{DATASET.relative_to(ROOT)}`",
        f"- SHA256：`{dataset_hash}`",
        "- 分布：1000（UNSAFE 600 / SAFE 250 / IRRELEVANT 150）",
        "- 标签空间：完整 YuFeng 29 类；UNSAFE 细分类重点评测 12 类，每类 50 条。",
        "- YuFeng 使用 native 模板；通用模型使用含义一致的完整分类提示词。", "",
        "| 模型 | 协议 | 代码准确率 | 风险检测准确率 | UNSAFE Recall | False Pass | False Block | 风险 Macro F1 | INVALID | 平均延迟 | P95 |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    rows = []
    for model, (path, payload) in results.items():
        metrics = payload["metrics"]
        detection = metrics["risk_detection"]
        row = {
            "model": model,
            "protocol": payload["adapter_protocol"],
            "path": path,
            "metrics": metrics,
        }
        rows.append(row)
        lines.append(
            f"| {model} | {row['protocol']} | {_pct(metrics['overall_accuracy'])} | "
            f"{_pct(detection['accuracy'])} | {_pct(detection['classes']['UNSAFE']['recall'])} | "
            f"{_pct(detection['false_pass_rate'])} | {_pct(detection['false_block_rate'])} | "
            f"{_pct(metrics['risk_type']['macro_f1'])} | {_pct(metrics['invalid_rate'])} | "
            f"{metrics['average_latency_ms']:.1f}ms | {metrics['p95_latency_ms']:.1f}ms |"
        )

    lines += ["", "## UNSAFE 各风险代码 Recall", "",
              "| 模型 | " + " | ".join(SELECTED_EVAL_RISKS) + " |",
              "|---|" + "---:|" * len(SELECTED_EVAL_RISKS)]
    for row in rows:
        classes = row["metrics"]["risk_type"]["classes"]
        lines.append(f"| {row['model']} | " + " | ".join(_pct(classes[risk]["recall"])
                                                           for risk in SELECTED_EVAL_RISKS) + " |")

    lines += ["", "## 结果文件", ""]
    for row in rows:
        lines.append(f"- `{row['model']}`：`{row['path'].relative_to(ROOT)}`")
    lines += ["", "说明：SAFE 与 IRRELEVANT 的 YuFeng 真值都为 `sec`；本报告不把二者路由能力计入模型排名。", ""]
    REPORT.write_text("\n".join(lines), encoding="utf-8")
    print(REPORT)


if __name__ == "__main__":
    main()
