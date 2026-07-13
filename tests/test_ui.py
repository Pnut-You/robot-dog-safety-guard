from pathlib import Path

from unittest.mock import patch

from streamlit.testing.v1 import AppTest


def test_ui_script_renders_active_model_without_import_errors():
    ui_path = Path(__file__).resolve().parents[1] / "app" / "ui.py"

    with patch("app.inference.check_vllm_server", return_value=(True, "vLLM 已连接")):
        app = AppTest.from_file(str(ui_path), default_timeout=10).run()

    assert not app.exception
    assert app.title[0].value == "机器狗拒识模型测试"
    assert "Qwen2.5 3B Instruct" in app.caption[0].value
    assert "http://127.0.0.1:8000/v1" in app.caption[0].value
    assert app.radio[0].value == "Few-shot"
