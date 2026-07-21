#!/usr/bin/env python3
from dataset_pipeline import run_deduplication

if __name__ == "__main__":
    result = run_deduplication()
    print("去重检查：" + ", ".join(f"{k}={v}" for k, v in result.items()))
    if any(result.values()):
        raise SystemExit(1)
