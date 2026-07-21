from __future__ import annotations

import os
from pathlib import Path

import yaml
from dotenv import load_dotenv
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, ValidationError

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "configs" / "models.yaml"
load_dotenv(PROJECT_ROOT / ".env")


class ModelConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    display_name: str
    model_id: str
    local_path: Path
    served_model_name: str
    host: str
    port: int = Field(ge=1, le=65535)
    api_key: str
    dtype: str
    max_model_len: int = Field(gt=0)
    gpu_memory_utilization: float = Field(gt=0, le=1)
    enforce_eager: bool = False
    native_guard: bool = False
    guard_family: Literal["none", "yufeng", "qwen3guard", "llama_guard", "singguard"] = "none"

    @property
    def resolved_local_path(self) -> Path:
        return self.local_path if self.local_path.is_absolute() else PROJECT_ROOT / self.local_path

    @property
    def base_url(self) -> str:
        configured = os.getenv("VLLM_BASE_URL")
        return configured or f"http://{self.host}:{self.port}/v1"

    @property
    def effective_api_key(self) -> str:
        return os.getenv("VLLM_API_KEY") or self.api_key


class ModelsConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    active_model: str
    models: dict[str, ModelConfig]

    def get_model(self, name: str | None = None) -> ModelConfig:
        key = name or self.active_model
        if key not in self.models:
            available = ", ".join(sorted(self.models))
            raise ValueError(f"模型配置不存在: {key}（可用: {available}）")
        return self.models[key]


def load_models_config(path: Path | str = DEFAULT_CONFIG_PATH) -> ModelsConfig:
    config_path = Path(path)
    if not config_path.is_file():
        raise FileNotFoundError(f"模型配置文件不存在: {config_path}")
    try:
        data = yaml.safe_load(config_path.read_text(encoding="utf-8"))
        config = ModelsConfig.model_validate(data)
        config.get_model()
        return config
    except (yaml.YAMLError, ValidationError, ValueError) as exc:
        raise ValueError(f"模型配置无效: {config_path}: {exc}") from exc


def get_model_config(name: str | None = None) -> ModelConfig:
    return load_models_config().get_model(name)
