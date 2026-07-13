import json

import pytest

from evaluation.evaluator import load_dataset


def write_lines(tmp_path, rows):
    path = tmp_path / "data.jsonl"
    path.write_text("\n".join(rows), encoding="utf-8")
    return path


def test_load_valid_jsonl(tmp_path):
    row = {"id": "a1", "text": "向前走", "label": "ACCEPT", "category": "action",
           "oos_type": None, "difficulty": "easy"}
    items = load_dataset(write_lines(tmp_path, [json.dumps(row, ensure_ascii=False)]))
    assert len(items) == 1
    assert items[0].label == "ACCEPT"


@pytest.mark.parametrize(
    "row",
    [
        {"id": "a1", "text": "向前走", "label": "ACCEPT"},
        {"id": "a1", "text": "向前走", "label": "MAYBE", "category": "action", "difficulty": "easy"},
    ],
)
def test_rejects_missing_fields_and_invalid_label(tmp_path, row):
    with pytest.raises(ValueError, match="第 1 行"):
        load_dataset(write_lines(tmp_path, [json.dumps(row, ensure_ascii=False)]))


def test_rejects_broken_json(tmp_path):
    with pytest.raises(ValueError, match="第 1 行"):
        load_dataset(write_lines(tmp_path, ['{"id":']))


def test_accepts_unsafe_oos_type(tmp_path):
    row = {"id": "u1", "text": "撞倒那个人", "label": "REJECT", "category": "action",
           "oos_type": "unsafe", "difficulty": "easy"}
    items = load_dataset(write_lines(tmp_path, [json.dumps(row, ensure_ascii=False)]))
    assert items[0].oos_type == "unsafe"
