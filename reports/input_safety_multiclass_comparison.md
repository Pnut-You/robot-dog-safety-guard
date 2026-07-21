# 本地输入安全模型双轨重测

- 数据集：`datasets/raw/sample_input_safety_multiclass_eval.jsonl`
- SHA256：`a8b2dd90d029c32e4d1fe364367148e9093b3e753c3fc4f8e09f440ef8db7573`
- 样本：1000（SAFE 400 / UNSAFE 400 / IRRELEVANT 200）
- 推理：temperature=0 / top_p=1 / concurrency=1 / warmup=1（不计延迟）
- 部署：RTX 4060 Ti 8GB；四个模型均使用 eager 模式
- 口径：专用 Guard 与项目三分类分轨展示，不跨轨道直接排名

## Guard 原生风险检测

该轨道把项目 SAFE 与 IRRELEVANT 合并为无风险，只衡量风险是否检出。

| 模型 | Accuracy | Macro F1 | UNSAFE Recall | 漏检率 | 误拦截率 | INVALID | 平均延迟 | P95 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| yufeng-xguard-reason-0.6b | 97.50% | 97.38% | 94.75% | 5.25% | 0.67% | 0.00% | 13.4ms | 15.1ms |
| qwen3guard-gen-0.6b | 95.20% | 94.90% | 88.50% | 11.50% | 0.33% | 0.00% | 84.4ms | 110.5ms |

### Guard 对项目风险场景的检出率

这些数值只表示是否检出风险，不表示原生风险代码能映射为项目 risk_type。

| 模型 | legal | self_harm | harm_others | bypass | dangerous_operation | other |
|---|---:|---:|---:|---:|---:|---:|
| yufeng-xguard-reason-0.6b | 98.33% | 100.00% | 97.78% | 90.00% | 85.00% | 100.00% |
| qwen3guard-gen-0.6b | 98.33% | 100.00% | 100.00% | 90.00% | 51.25% | 100.00% |

## 项目严格三分类

| 模型 | Accuracy | Macro F1 | SAFE R | UNSAFE R | IRRELEVANT R | Risk Acc | INVALID | 平均延迟 | P95 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| qwen2.5-1.5b-instruct | 71.20% | 75.47% | 50.00% | 93.00% | 70.00% | 56.00% | 13.60% | 241.7ms | 437.7ms |
| qwen2.5-3b-instruct | 91.70% | 91.14% | 87.75% | 93.25% | 96.50% | 62.75% | 0.00% | 365.3ms | 422.0ms |

### 三分类业务指标

| 模型 | 危险漏检 | 安全误拦截 | 噪声误进入 | dangerous_operation R | harm_others R | self_harm R |
|---|---:|---:|---:|---:|---:|---:|
| qwen2.5-1.5b-instruct | 2.25% | 17.00% | 1.50% | 93.75% | 90.00% | 42.86% |
| qwen2.5-3b-instruct | 6.75% | 2.75% | 3.50% | 70.00% | 95.56% | 75.71% |

## 部署参数与结果文件

| 模型 | max_model_len | GPU利用率 | eager | 结果文件 |
|---|---:|---:|---:|---|
| yufeng-xguard-reason-0.6b | 4096 | 0.8 | True | `results/native_safety/yufeng-xguard-reason-0.6b_sample_input_safety_multiclass_eval_native_20260721_103117.json` |
| qwen3guard-gen-0.6b | 4096 | 0.75 | True | `results/native_safety/qwen3guard-gen-0.6b_sample_input_safety_multiclass_eval_native_20260721_103331.json` |
| qwen2.5-1.5b-instruct | 4096 | 0.8 | True | `results/input_safety_multiclass/qwen2.5-1.5b-instruct_sample_input_safety_multiclass_eval_20260721_103826.json` |
| qwen2.5-3b-instruct | 768 | 0.8 | True | `results/input_safety_multiclass/qwen2.5-3b-instruct_sample_input_safety_multiclass_eval_20260721_104736.json` |

## 结论

- Guard 原生风险检测中，YuFeng 的危险召回和漏检率优于 Qwen3Guard，且延迟显著更低。
- Qwen3Guard 的主要短板是 dangerous_operation；不能仅凭伤人、自残场景的高召回判断整体效果。
- 项目三分类中，Qwen2.5-3B 明显优于1.5B，且没有 INVALID；当前是更合适的端到端三分类候选。
- Qwen2.5-3B 的六类风险归因准确率仍只有约63%，如果业务依赖 risk_type，需要继续优化提示词或增加第二阶段分类器。
- 不能将 YuFeng 的97%风险二分类准确率与Qwen2.5-3B的92%三分类准确率直接比较，它们解决的任务不同。
