#!/usr/bin/env python3
import argparse

from dataset_pipeline import build_all, build_binary_evaluation

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="规范化安全评测数据")
    parser.add_argument("--binary-eval", action="store_true", help="从只读 sample_safety_new.jsonl 生成二分类评测集")
    args = parser.parse_args()
    result = build_binary_evaluation() if args.binary_eval else build_all()
    print("生成完成：" + ", ".join(f"{k}={v}" for k, v in result.items()))
