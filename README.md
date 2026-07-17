# robot-dog-safety-guard

机器狗请求分类与安全拒识模型的本地部署、交互测试和批量评测项目。`PASS` 表示能力范围内且安全，`BLOCK` 表示存在安全风险，`IRRELEVANT` 表示背景声音、无关内容或能力范围外请求；无法解析或服务异常记为 `INVALID`。

项目使用 uv 管理 Python 3.12 环境，使用 vLLM 提供 OpenAI Compatible API，并由 Streamlit 提供测试页面。默认模型是 `Alibaba-AAIG/YuFeng-XGuard-Reason-0.6B`，同时保留 Qwen2.5 1.5B、3B 和适合 8GB 显存测试的 Qwen3 4B Instruct 2507 AWQ 4-bit 作为提示词对照模型。

## 项目结构

```text
app/                 配置、数据结构、提示词、API 推理与 UI
configs/models.yaml  多模型配置和 active_model
datasets/raw/        JSONL 示例安全数据集
evaluation/          数据加载、逐条评测和指标
models/              本地模型（不提交）
results/             评测结果（不提交）
scripts/             下载、启动、检查与评测脚本
tests/               单元测试和 UI 冒烟测试
```

## 安装与标准启动

```bash
uv sync
cp .env.example .env
uv run python scripts/download_model.py --model yufeng_xguard_0_6b
bash scripts/start_dev.sh --model yufeng_xguard_0_6b
```

`start_dev.sh` 同时管理 vLLM 与 Streamlit，启动前检查模型目录、GPU 和端口，不会终止非本项目进程。支持：

```bash
bash scripts/start_dev.sh status
bash scripts/start_dev.sh restart --model yufeng_xguard_0_6b
bash scripts/start_dev.sh stop
```

也可分别启动。在第一个终端运行：

```bash
bash scripts/start_vllm.sh --model yufeng_xguard_0_6b
```

在新终端检查服务并启动 UI：

```bash
uv run python scripts/check_server.py --model yufeng_xguard_0_6b
ROVER_MODEL_KEY=yufeng_xguard_0_6b uv run streamlit run app/ui.py
```

页面标题为 `Robot Dog Safety Guard`，展示判断结果、原始输出、风险类别、风险分数、风险解释、推理耗时、当前模型和最近记录。UI 只调用 vLLM API，不直接加载模型。

## XGuard 输出与解析

XGuard 原生输出首 token 是风险分类代码：`sec` 代表安全，其余官方分类代码（例如危险武器 `dw`）代表风险。项目将 `sec` 映射为 `PASS`，其他已知分类映射为 `BLOCK`；未知或混杂格式映射为 `INVALID`。若输出包含 `<explanation>...</explanation>` 则保留解释；达到生成上限时也会保留截断解释，但只要首行分类代码完整，结论仍可解析。风险分数取 vLLM 返回的首 token 概率。API 未返回相应信息时字段为空，不作推测。完整原始输出始终保留。

Qwen 对照模型使用 `app/prompts.py` 的 Zero-shot/Few-shot 提示词，只接受完整 `PASS`、`BLOCK` 或 `IRRELEVANT`。XGuard 原生协议没有无关类别，因此评测三分类时会把安全的无关输入输出为 `PASS`，建议优先使用 Qwen3。

## 批量评测

```bash
uv run python scripts/evaluate.py \
  --model yufeng_xguard_0_6b \
  --dataset datasets/raw/sample_safety.jsonl
```

数据集共 1000 条：PASS 350、BLOCK 350、IRRELEVANT 300。覆盖动作控制、机器狗问答、机器狗相关危险请求、一般内容风险、语气词、旁人对话、电视电话声音、ASR 碎片、能力范围外请求和带背景杂音的有效请求。可用 `uv run python scripts/generate_sample_dataset.py` 按固定规则重建并校验数据。

命令输出 Accuracy、三类 Precision/Recall/F1、Macro F1、混淆矩阵、无关语音误接受率、危险请求漏检率、INVALID 数量和延迟指标。详情保存到 `results/<模型名>_<时间>.json`。

### 使用 Qwen3 4B Instruct 2507 AWQ

RTX 4060 Ti 8GB 无法为约 8.06GB 的官方 BF16 权重留出足够的 vLLM 运行和 KV cache 空间，因此项目使用基于该模型的 AWQ 4-bit 版本：

```bash
uv run python scripts/download_model.py --model qwen3_4b_instruct_2507_awq
bash scripts/start_vllm.sh --model qwen3_4b_instruct_2507_awq
```

在新终端检查服务并启动 UI：

```bash
uv run python scripts/check_server.py --model qwen3_4b_instruct_2507_awq
ROVER_MODEL_KEY=qwen3_4b_instruct_2507_awq uv run streamlit run app/ui.py
```

使用项目数据集评测：

```bash
uv run python scripts/evaluate.py \
  --model qwen3_4b_instruct_2507_awq \
  --dataset datasets/raw/sample_safety.jsonl \
  --prompt few_shot
```

Qwen3 是通用指令模型，使用项目的三分类提示词，输出严格解析为 `PASS`、`BLOCK`、`IRRELEVANT` 或 `INVALID`。详细结果写入 `results/`。

## 增加和切换模型

在 `configs/models.yaml` 的 `models` 下增加配置。原生 XGuard 协议模型设置 `native_guard: true`；使用项目安全提示词的通用指令模型设置为 `false`。可修改 `active_model` 切换默认模型，或向下载、检查、评测和启动脚本传入 `--model <配置键>`，代码中无需修改路径或服务名。

## 显存不足与后续计划

RTX 4060 Ti 8GB 若显存不足，可在 YAML 中降低 `gpu_memory_utilization` 或 `max_model_len`，停止其他 GPU 进程后重试；脚本不会自动结束其他进程。不要让 `VLLM_BASE_URL` 指向与所选模型不一致的服务。

当前不含训练模块。后续可独立加入 LoRA/QLoRA 实验，但应继续复用现有 JSONL 评测集和 API 推理边界。

运行测试：

```bash
uv run pytest
```
