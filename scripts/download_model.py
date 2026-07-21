#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

from dotenv import load_dotenv
from huggingface_hub import snapshot_download

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from app.config import PROJECT_ROOT, get_model_config  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description="下载 Hugging Face 模型到配置的本地目录")
    parser.add_argument("--model", help="models.yaml 中的模型键；默认使用 active_model")
    args = parser.parse_args()
    load_dotenv(PROJECT_ROOT / ".env")
    try:
        config = get_model_config(args.model)
    except (OSError, ValueError) as exc:
        print(f"模型配置读取失败: {exc}", file=sys.stderr)
        return 1
    destination = config.resolved_local_path
    if destination.exists():
        if not destination.is_dir():
            print(f"错误: 目标路径已存在但不是目录: {destination}", file=sys.stderr)
            return 1
        weight_files = list(destination.glob("*.safetensors")) if destination.is_dir() else []
        if destination.is_dir() and (destination / "config.json").is_file() and weight_files:
            print(f"模型目录已存在，跳过下载: {destination}")
            return 0
        print(f"检测到未完成的模型下载，将继续下载: {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    print(f"下载 {config.model_id} 到 {destination}")
    try:
        snapshot_download(
            repo_id=config.model_id,
            local_dir=destination,
            token=os.getenv("HF_TOKEN") or None,
            endpoint=os.getenv("HF_ENDPOINT") or None,
        )
    except Exception as exc:
        if destination.exists() and not any(destination.iterdir()):
            destination.rmdir()
        print(f"模型下载失败: {exc}", file=sys.stderr)
        return 1
    print("模型下载完成。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
