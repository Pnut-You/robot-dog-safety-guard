# 去重与泄漏报告

- duplicate_ids：0
- duplicate_texts：0
- normalized_duplicates：0
- label_conflicts：0
- near_duplicate_pairs_across_families：0
- reference_text_leaks：0
- reference_family_leaks：0

近似检查采用规范化中文 3-gram Jaccard，相似阈值 0.88；同一 template_family 内的预期语义变体不计为跨族重复。
