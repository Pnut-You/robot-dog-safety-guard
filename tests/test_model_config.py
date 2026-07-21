from pathlib import Path

import pytest

from app.config import load_models_config


def test_default_and_named_model_configs():
    config = load_models_config()

    assert config.active_model == "yufeng_xguard_0_6b"
    assert config.get_model().served_model_name == "yufeng-xguard-reason-0.6b"
    assert config.get_model().native_guard is True
    assert config.get_model().guard_family == "yufeng"
    assert config.get_model("qwen2_5_1_5b").served_model_name == "qwen2.5-1.5b-instruct"
    assert config.get_model("qwen3guard_gen_0_6b").guard_family == "qwen3guard"
    assert config.get_model("llama_guard_3_1b").guard_family == "llama_guard"
    assert "qwen3_4b_instruct_2507_awq" not in config.models


def test_unknown_model_has_clear_error():
    config = load_models_config()

    with pytest.raises(ValueError, match="模型配置不存在: unknown"):
        config.get_model("unknown")


def test_model_paths_are_project_relative():
    config = load_models_config()

    assert all(not model.local_path.is_absolute() for model in config.models.values())
    assert all(isinstance(model.resolved_local_path, Path) for model in config.models.values())
