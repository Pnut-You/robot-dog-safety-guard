from evaluation.metrics import calculate_metrics, calculate_native_safety_metrics


def row(expected, predicted, subset="safety_core", risk=None, irrelevant=None, difficulty="easy", latency=10):
    return {"expected": expected, "predicted": predicted, "eval_subset": subset, "risk_type": risk,
            "irrelevant_type": irrelevant, "difficulty": difficulty, "latency_ms": latency}


def test_three_metric_sections_and_invalid_is_strict_error():
    rows = [
        row("SAFE", "SAFE"), row("SAFE", "UNSAFE", difficulty="medium"),
        row("UNSAFE", "UNSAFE", risk="robot_harm", difficulty="easy"),
        row("UNSAFE", "SAFE", risk="violence", difficulty="medium"),
        row("UNSAFE", "INVALID", risk="violence", difficulty="hard"),
        row("SAFE", "SAFE", "noise_robustness", irrelevant="pure_filler", latency=20),
        row("SAFE", "UNSAFE", "noise_robustness", irrelevant="noise", latency=30),
        row("SAFE", "INVALID", "noise_robustness", irrelevant="noise", latency=40),
    ]
    metrics = calculate_metrics(rows)
    core = metrics["safety_core"]
    assert core["strict_accuracy"] == 2 / 5
    assert core["invalid_count"] == 1
    assert core["false_pass_rate"] == 1 / 3
    assert core["false_block_rate"] == 0.5
    assert core["unsafe_recall"] == 1 / 3
    assert core["confusion_matrix"]["UNSAFE"]["INVALID"] == 1
    assert core["risk_types"]["violence"]["count"] == 2
    assert core["robot_related_danger"]["unsafe_recall"] == 1.0
    noise = metrics["noise_robustness"]
    assert noise["safe_acceptance_rate"] == 1 / 3
    assert noise["noise_false_block_rate"] == 1 / 3
    assert noise["invalid_rate"] == 1 / 3
    assert noise["irrelevant_types"]["noise"]["invalid_count"] == 1
    assert metrics["full_dataset"]["strict_accuracy"] == 3 / 8


def test_native_safety_metrics_keep_project_labels_and_native_codes_separate():
    rows = [
        {**row("SAFE", "SAFE", difficulty="easy"), "original_label": "SAFE", "native_risk_code": "sec"},
        {**row("SAFE", "SAFE", difficulty="medium"), "original_label": "IRRELEVANT", "native_risk_code": "sec"},
        {**row("UNSAFE", "UNSAFE", risk="harm_others", difficulty="hard"),
         "original_label": "UNSAFE", "native_risk_code": "ph"},
        {**row("UNSAFE", "SAFE", risk="self_harm", difficulty="hard"),
         "original_label": "UNSAFE", "native_risk_code": "sec"},
    ]
    metrics = calculate_native_safety_metrics(rows)
    assert metrics["strict_accuracy"] == .75
    assert metrics["unsafe_recall"] == .5
    assert metrics["original_labels"]["IRRELEVANT"]["safe_recall"] == 1
    assert metrics["risk_types"]["self_harm"]["false_pass_rate"] == 1
    assert metrics["native_risk_code_distribution"] == {"sec": 3, "ph": 1}
