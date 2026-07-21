#!/usr/bin/env python3
from dataset_pipeline import run_audit

if __name__ == "__main__":
    audits = run_audit()
    print(f"审计完成：{len(audits)} 个数据文件，报告 reports/dataset_audit.md")
