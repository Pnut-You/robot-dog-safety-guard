#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

from openai import APIConnectionError, APIStatusError, APITimeoutError, OpenAI

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from app.config import get_model_config  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description="检查本地 vLLM OpenAI Compatible API")
    parser.add_argument("--model", help="models.yaml 中的模型键；默认使用 active_model")
    args = parser.parse_args()
    config = get_model_config(args.model)
    client = OpenAI(base_url=config.base_url, api_key=config.effective_api_key, timeout=10)
    print(f"检查服务: {config.base_url}（{config.served_model_name}）")
    try:
        model_ids = [model.id for model in client.models.list().data]
    except (APIConnectionError, APITimeoutError, APIStatusError) as exc:
        print(f"服务检查失败: {exc}", file=sys.stderr)
        return 1
    if config.served_model_name not in model_ids:
        print(f"服务可访问，但未找到目标模型。服务模型: {model_ids}", file=sys.stderr)
        return 1
    print("vLLM 服务正常。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
