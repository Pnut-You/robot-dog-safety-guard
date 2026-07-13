from pathlib import Path

from unittest.mock import patch

import pytest
from streamlit.testing.v1 import AppTest

from app.schemas import PredictionResult


def test_ui_script_renders_active_model_without_import_errors():
    ui_path = Path(__file__).resolve().parents[1] / "app" / "ui.py"

    with patch("app.inference.check_vllm_server", return_value=(True, "vLLM 已连接")):
        app = AppTest.from_file(str(ui_path), default_timeout=10).run()

    assert not app.exception
    assert not app.radio
    assert not app.selectbox
    assert not app.metric
    navbar = next(markdown.value for markdown in app.markdown if '<nav class="rover-navbar"' in markdown.value)
    assert "Rover OOS Detection" in navbar
    assert "Qwen2.5 3B Instruct" in navbar
    assert "服务在线" in navbar
    assert "http://127.0.0.1:8000/v1" not in navbar
    page_css = next(markdown.value for markdown in app.markdown if "<style>" in markdown.value)
    assert 'header[data-testid="stHeader"]' in page_css
    assert "display: none !important" in page_css
    assert "width: 100vw" in page_css
    assert "max-width: 1360px" in page_css
    button_labels = {button.label for button in app.button}
    assert {"向前走两米", "你现在还有多少电", "你会不会跳舞", "帮我拿桌上的杯子", "帮我订一张机票", "开始检测"} <= button_labels


def test_example_button_fills_input_without_running_inference():
    ui_path = Path(__file__).resolve().parents[1] / "app" / "ui.py"

    with (
        patch("app.inference.check_vllm_server", return_value=(True, "vLLM 已连接")),
        patch("app.inference.OOSDetector") as detector,
    ):
        app = AppTest.from_file(str(ui_path), default_timeout=10).run()
        example_button = next(button for button in app.button if button.label == "向前走两米")
        app = example_button.click().run()

    assert not app.exception
    assert app.text_area[0].value == "向前走两米"
    detector.assert_not_called()


@pytest.mark.parametrize("prediction, active_class", [("ACCEPT", "accept-active"), ("REJECT", "reject-active")])
def test_detection_highlights_matching_route(prediction, active_class):
    ui_path = Path(__file__).resolve().parents[1] / "app" / "ui.py"
    result = PredictionResult(
        text="测试请求",
        prediction=prediction,
        raw_output=prediction,
        latency_ms=12.3,
        model_name="qwen2.5-3b-instruct",
    )

    with (
        patch("app.inference.check_vllm_server", return_value=(True, "vLLM 已连接")),
        patch("app.inference.OOSDetector") as detector,
    ):
        detector.return_value.predict.return_value = result
        app = AppTest.from_file(str(ui_path), default_timeout=10).run()
        app.text_area[0].set_value("测试请求")
        detect_button = next(button for button in app.button if button.label == "开始检测")
        app = detect_button.click().run()

    assert not app.exception
    assert any(active_class in markdown.value for markdown in app.markdown)
    detector.assert_called_with(prompt_name="few_shot", model_config=detector.call_args.kwargs["model_config"])
