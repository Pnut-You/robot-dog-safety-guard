# 数据质量报告

- 原始 sample：1000 条
- 清洗后保留：428 条
- 转人工审核：572 条
- template_family_cap_exceeded：362 条
- normalized_duplicate：10 条
- source_dependent_irrelevant_removed：140 条
- synthetic_unsupported_template_removed：60 条

原文件未被修改。能力外但安全的请求不再作为 IRRELEVANT；模板族最多保留 3 个旧样本。

## 参考池来源

- `CCAC2024-FS_Moderation`：200 条
- `CValues`：100 条
- `Chinese-Do-Not-Answer`：150 条
- `Safety-Prompts`：251 条
- `XGuard-Train-Open-200K`：494 条

参考池合计 1195 条；公开原始标签只作为候选信息，不确定项不自动映射。人工审核文件合计 572 条。


# 标签定义修订记录

本次直接更新现有派生数据；公开原始数据和 `datasets/raw/sample_safety.jsonl` 未修改。

## 旧 evaluation 中 400 条 IRRELEVANT 的去向

- `retained_as_irrelevant`：100
- `relabelled_pass`：20
- `relabelled_block`：0
- `manual_review`：30
- `excluded_source_templates`：250

其中 250 条依赖旁人、电视、电话或背景来源的模板从正式评测集排除；纯文本评测不再依据假设声源判断。

## 新定义

IRRELEVANT 仅包括 pure_filler、noise、asr_garbage、incomplete、repetition 和 unintelligible；所有文本均非空。
