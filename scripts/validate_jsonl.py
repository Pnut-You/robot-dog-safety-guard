#!/usr/bin/env python3
import argparse

from dataset_pipeline import validate_outputs
from generate_guard_binary_dataset import OUTPUT as GUARD_BINARY_OUTPUT, validate as validate_guard_binary_rows
from generate_multiclass_dataset import OUTPUT as MULTICLASS_OUTPUT, validate as validate_multiclass_rows

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="验证安全评测 JSONL")
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--binary", action="store_true", help="验证第一阶段二分类评测集")
    group.add_argument("--multiclass", action="store_true", help="验证输入安全三分类评测集")
    args = parser.parse_args()
    if args.multiclass:
        import json
        rows = [json.loads(line) for line in MULTICLASS_OUTPUT.read_text(encoding="utf-8").splitlines() if line.strip()]
        result = validate_multiclass_rows(rows)
    else:
        if args.binary:
            import json
            rows = [json.loads(line) for line in GUARD_BINARY_OUTPUT.read_text(encoding="utf-8").splitlines() if line.strip()]
            result = validate_guard_binary_rows(rows)
        else:
            result = validate_outputs()
    print("验证通过：" + ", ".join(f"{k}={v}" for k, v in result.items()))
