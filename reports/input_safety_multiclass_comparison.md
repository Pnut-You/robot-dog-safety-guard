# YuFeng 原生标签空间模型对比

- 数据集：`datasets/raw/sample_input_safety_multiclass_eval.jsonl`
- 分布：1000（UNSAFE 600 / SAFE 250 / IRRELEVANT 150）
- UNSAFE：`ps/ph/mh/ti/pp/acc/ha/mc/dw/pi/law/ec` 各 50 条
- 标签空间：完整 YuFeng 29 类
- 当前状态：等待按 `yufeng_taxonomy_v2_native` 和 `yufeng_taxonomy_v2_strict` 协议重新评测

旧 `native_safety_detection_v1` 与 `input_safety_multiclass_v1` 结果使用不同任务和数据哈希，不进入本报告。

完成 YuFeng 0.6B 和通用模型评测后运行：

```bash
uv run python scripts/summarize_multiclass_results.py
```
