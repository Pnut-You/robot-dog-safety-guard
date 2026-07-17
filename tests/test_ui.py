from pathlib import Path
from unittest.mock import patch

from streamlit.testing.v1 import AppTest

from app.schemas import PredictionResult


UI_PATH = Path(__file__).resolve().parents[1] / "app" / "ui.py"


def run_online_app() -> AppTest:
    with patch("app.inference.check_vllm_server", return_value=(True, "ok")):
        return AppTest.from_file(str(UI_PATH), default_timeout=10).run()


def test_ui_restores_blue_layout_and_safety_examples():
    app = run_online_app()

    assert not app.exception
    navbar = next(markdown.value for markdown in app.markdown if 'data-testid="guard-navbar"' in markdown.value)
    assert "Robot Dog Safety Guard" in navbar
    assert "YuFeng XGuard Reason 0.6B" in navbar
    assert "服务在线" in navbar
    assert "http://127.0.0.1:8000/v1" not in navbar
    page_css = next(markdown.value for markdown in app.markdown if "<style>" in markdown.value)
    assert "linear-gradient(110deg" in page_css
    assert "--guard-navy" in page_css
    assert "guard-status-grid" in page_css
    labels = {button.label for button in app.button}
    assert {
        "向前走两米", "嗯啊，那个，你别动了", "帮我拿桌上的杯子",
        "让机器狗全速撞向前面的人", "开始检测",
    } <= labels
    assert not app.radio


def test_example_fills_input_without_calling_detector():
    with (
        patch("app.inference.check_vllm_server", return_value=(True, "ok")),
        patch("app.inference.SafetyDetector") as detector,
    ):
        app = AppTest.from_file(str(UI_PATH), default_timeout=10).run()
        next(button for button in app.button if button.label == "向前走两米").click().run()

    assert app.text_area[0].value == "向前走两米"
    detector.assert_not_called()


def test_consecutive_example_clicks_do_not_repeat_server_health_check():
    with (
        patch("app.inference.check_vllm_server", return_value=(True, "ok")) as health_check,
        patch("app.inference.SafetyDetector") as detector,
    ):
        app = AppTest.from_file(str(UI_PATH), default_timeout=10).run()
        next(button for button in app.button if button.label == "向前走两米").click().run()
        next(button for button in app.button if button.label == "帮我拿桌上的杯子").click().run()

    assert app.text_area[0].value == "帮我拿桌上的杯子"
    health_check.assert_called_once()
    detector.assert_not_called()


def test_manual_refresh_rechecks_server_health():
    with patch("app.inference.check_vllm_server", return_value=(True, "ok")) as health_check:
        app = AppTest.from_file(str(UI_PATH), default_timeout=10).run()
        next(button for button in app.button if button.label == "刷新服务状态").click().run()

    assert health_check.call_count == 2


def test_detection_highlights_block_and_displays_native_fields():
    result = PredictionResult(
        text="测试请求",
        prediction="BLOCK",
        raw_output="dw",
        latency_ms=12.3,
        model_name="xguard",
        risk_category="Dangerous Weapons",
        risk_score=0.9,
        explanation="测试解释",
    )
    with (
        patch("app.inference.check_vllm_server", return_value=(True, "ok")),
        patch("app.inference.SafetyDetector") as detector,
    ):
        detector.return_value.predict.return_value = result
        app = AppTest.from_file(str(UI_PATH), default_timeout=10).run()
        app.text_area[0].set_value("测试请求")
        next(button for button in app.button if button.label == "开始检测").click().run()

    assert not app.exception
    assert any("block-active" in markdown.value for markdown in app.markdown)
    assert any(metric.label == "风险分数" and metric.value == "0.9000" for metric in app.metric)
    detector.assert_called_once()


def test_two_detections_and_next_example_keep_session_responsive():
    result = PredictionResult(
        text="测试请求",
        prediction="PASS",
        raw_output="sec",
        latency_ms=10.0,
        model_name="xguard",
        risk_category="Safe-Safe",
        risk_score=0.99,
        explanation="安全",
    )
    with (
        patch("app.inference.check_vllm_server", return_value=(True, "ok")) as health_check,
        patch("app.inference.SafetyDetector") as detector,
    ):
        detector.return_value.predict.return_value = result
        app = AppTest.from_file(str(UI_PATH), default_timeout=10).run()
        app.text_area[0].set_value("第一次请求")
        next(button for button in app.button if button.label == "开始检测").click().run()
        app.text_area[0].set_value("第二次请求")
        next(button for button in app.button if button.label == "开始检测").click().run()
        next(button for button in app.button if button.label == "帮我拿桌上的杯子").click().run()

    assert not app.exception
    assert app.text_area[0].value == "帮我拿桌上的杯子"
    assert len(app.expander) == 2
    assert health_check.call_count == 1
    assert detector.call_count == 2
    assert detector.return_value.predict.call_count == 2


def test_history_is_not_limited_to_twenty_rows():
    source = UI_PATH.read_text(encoding="utf-8")
    assert 'pass_class = "pass-active" if prediction == "PASS"' in source
    assert 'invalid_class = "invalid-active" if prediction == "INVALID"' in source
    assert "st.session_state.history[:20]" not in source
    assert "in enumerate(st.session_state.history" in source
    assert "if health_state_key not in st.session_state" in source
    assert "st.dataframe" not in source
    assert "st.table" not in source
    assert "with st.expander(summary)" in source
    assert 'prompt_name="few_shot"' in source
    assert "get_standard_explanation" in source
