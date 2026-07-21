import json
from collections import Counter

import pytest

from app.config import PROJECT_ROOT
from evaluation.evaluator import load_dataset, validate_dataset_contract


def write_rows(tmp_path, rows):
    path = tmp_path / "data.jsonl"
    path.write_text("\n".join(rows), encoding="utf-8")
    return path


def valid_row(**changes):
    row = {
        "id": "safe_001", "text": "向前走", "label": "SAFE", "original_label": "PASS",
        "category": "safe", "eval_subset": "safety_core", "risk_type": None,
        "irrelevant_type": None, "scene_type": "clean", "difficulty": "easy",
        "source": "manual", "source_id": None, "template_family": "safe_move",
    }
    row.update(changes)
    return row


def test_load_valid_binary_jsonl(tmp_path):
    items = load_dataset(write_rows(tmp_path, [json.dumps(valid_row(), ensure_ascii=False)]))
    assert items[0].label == "SAFE"
    unsafe = valid_row(label="UNSAFE", original_label="BLOCK", category="unsafe", risk_type="violence")
    assert load_dataset(write_rows(tmp_path, [json.dumps(unsafe, ensure_ascii=False)]))[0].label == "UNSAFE"


@pytest.mark.parametrize("row", [
    {"id": "a", "text": "x", "label": "SAFE"},
    valid_row(label="PASS"),
    valid_row(label="SAFE", original_label="BLOCK"),
    valid_row(label="UNSAFE", original_label="BLOCK", category="unsafe", risk_type=None),
])
def test_rejects_missing_fields_invalid_labels_and_mappings(tmp_path, row):
    with pytest.raises(ValueError, match="第 1 行"):
        load_dataset(write_rows(tmp_path, [json.dumps(row, ensure_ascii=False)]))


def test_rejects_broken_json(tmp_path):
    with pytest.raises(ValueError, match="第 1 行"):
        load_dataset(write_rows(tmp_path, ['{"id":']))


def test_binary_dataset_distribution_and_preserved_fields():
    items = load_dataset(PROJECT_ROOT / "datasets/raw/sample_guard_safety_binary_eval.jsonl")
    assert len(items) == 1000
    assert len({item.id for item in items}) == 1000
    assert Counter(item.label for item in items) == {"SAFE": 500, "UNSAFE": 500}
    assert Counter(item.eval_subset for item in items) == {"safety_core": 900, "noise_robustness": 100}
    assert Counter(item.label for item in items if item.eval_subset == "safety_core") == {"SAFE": 400, "UNSAFE": 500}
    assert all(item.label == "SAFE" for item in items if item.eval_subset == "noise_robustness")
    assert all(item.risk_type for item in items if item.original_label == "BLOCK")
    assert all(item.irrelevant_type for item in items if item.original_label == "IRRELEVANT")
    validate_dataset_contract(items)
