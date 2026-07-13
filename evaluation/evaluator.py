from __future__ import annotations

import json
from pathlib import Path

from pydantic import ValidationError

from app.inference import OOSDetector
from app.schemas import DatasetItem, PredictionResult


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


def evaluate_dataset(items: list[DatasetItem], detector: OOSDetector) -> list[dict]:
    details = []
    for item in items:
        result: PredictionResult = detector.predict(item.text)
        details.append({**item.model_dump(), **result.model_dump(exclude={"text"})})
    return details
