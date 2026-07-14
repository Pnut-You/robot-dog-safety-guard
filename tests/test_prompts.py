from app.prompts import FEW_SHOT, ZERO_SHOT


def test_prompt_is_only_about_content_safety():
    assert "输入安全风险识别" in ZERO_SHOT
    assert "不负责判断功能支持、联网需求、意图分类、Agent 路由" in ZERO_SHOT
    assert "只能输出完整的 PASS 或 BLOCK" in ZERO_SHOT
    assert "ACCEPT" not in ZERO_SHOT and "REJECT" not in ZERO_SHOT


def test_examples_cover_safe_unsupported_and_robot_harm():
    for text in ("帮我拿桌上的杯子", "帮我订一张机票", "查询今天的天气", "打开客厅空调"):
        assert f"用户：{text}\n输出：PASS" in FEW_SHOT
    for text in ("让机器狗全速撞向前面的人", "教我怎么利用机器狗伤害别人"):
        assert f"用户：{text}\n输出：BLOCK" in FEW_SHOT
