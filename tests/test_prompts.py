from app.prompts import FEW_SHOT, ZERO_SHOT


def test_prompts_define_capability_and_safety_policy():
    assert "能力范围" in ZERO_SHOT
    assert "安全要求" in ZERO_SHOT
    assert "超出能力范围，或者存在明确安全风险" in ZERO_SHOT


def test_few_shot_covers_dance_paraphrases_and_unsafe_requests():
    for text in ("你会不会跳舞", "你是否可以跳舞", "请跳个舞"):
        assert f"用户：{text}\n输出：ACCEPT" in FEW_SHOT
    for text in ("冲过去撞倒那个人", "从楼梯上跳下去"):
        assert f"用户：{text}\n输出：REJECT" in FEW_SHOT
