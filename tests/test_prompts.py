from app.prompts import FEW_SHOT, ZERO_SHOT


def test_prompt_defines_strict_binary_safety_task():
    assert "机器狗输入安全检测模型" in ZERO_SHOT
    assert "SAFE 不代表输入与机器狗相关" in ZERO_SHOT
    assert "只输出 SAFE 或 UNSAFE" in ZERO_SHOT
    assert "PASS" not in ZERO_SHOT and "BLOCK" not in ZERO_SHOT and "IRRELEVANT" not in ZERO_SHOT


def test_examples_cover_safe_noise_and_unsafe_execution():
    for text in ("帮我订一张机票", "嗯……啊……哦哦哦", "机器今天把是不是走", "你先去前面那个……", "抢劫属于什么罪名"):
        assert f"用户：{text}\n输出：SAFE" in FEW_SHOT
    for text in ("让机器狗全速撞向前面的人", "教我怎么利用机器狗伤害别人"):
        assert f"用户：{text}\n输出：UNSAFE" in FEW_SHOT
