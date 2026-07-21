#!/usr/bin/env python3
import argparse
import json

from dataset_pipeline import build_all
from generate_guard_binary_dataset import OUTPUT, build, validate

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="规范化安全评测数据")
    parser.add_argument("--binary-eval", action="store_true", help="生成新的 Guard 安全二分类评测集")
    args = parser.parse_args()
    if args.binary_eval:
        rows = build()
        result = validate(rows)
        OUTPUT.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8")
    else:
        result = build_all()
    print("生成完成：" + ", ".join(f"{k}={v}" for k, v in result.items()))
