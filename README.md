# robot-dog-safety-guard

机器狗第一阶段输入安全检测模型的本地部署、交互测试和批量评测项目。`SAFE` 表示没有明确安全风险，`UNSAFE` 表示存在现实危害或危险执行意图，无法严格解析的模型输出记为 `INVALID`。第一阶段不判断业务意图、能力范围或输入是否与机器狗相关；`SAFE` 不代表可以直接执行。

项目使用 uv 管理 Python 3.12 环境，使用 vLLM 提供 OpenAI Compatible API，并由 Streamlit 提供测试页面。当前候选模型为 YuFeng-XGuard-Reason-0.6B、Qwen3Guard-Gen-0.6B、Llama-Guard-3-1B、Qwen2.5-1.5B-Instruct 和 Qwen2.5-3B-Instruct。

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
bash scripts/start_dev.sh switch --model qwen3guard_gen_0_6b
bash scripts/start_dev.sh restart --model yufeng_xguard_0_6b
bash scripts/start_dev.sh stop
```

推荐使用 `switch --model` 切换模型。它会先确认目标模型的配置和全部权重分片完整，再自动停止本项目上一个 vLLM/UI、释放端口和显存并加载新模型。目标模型不完整时不会停止当前服务；非本项目占用的端口也不会被自动终止。

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

## 二分类输出与解析

`strict` 协议让所有候选模型使用 `app/prompts.py` 中含义一致的二分类提示词，并固定 `temperature=0`、`top_p=1`、`max_tokens=4`。解析器只接受规范化后的完整 `SAFE` 或 `UNSAFE`；原生 Guard 格式、解释文字和空输出均为 `INVALID`。

`native` 协议仅支持三个专用 Guard 模型，使用模型自带的 chat template，并把官方原生结论映射成 SAFE/UNSAFE。Qwen3Guard 的 `Controversial` 按安全优先原则映射为 `UNSAFE`，原始输出和类别仍保存在逐条结果中。两套结果衡量的能力不同，必须分别查看，不能混为同一排名。

## 批量评测

```bash
uv run python scripts/evaluate.py \
  --model qwen2_5_1_5b \
  --protocol strict
```

默认评测集为 `datasets/raw/sample_safety_binary_eval.jsonl`，共 1000 条：SAFE 600、UNSAFE 400。核心安全集 800 条（400/400）是模型排名依据；噪声鲁棒性集 200 条全部期望 SAFE。二分类文件可通过 `uv run python scripts/normalize_datasets.py --binary-eval` 从只读的 `sample_safety_new.jsonl` 无损转换，并用 `uv run python scripts/validate_jsonl.py --binary` 校验。

评测默认预热 1 次且不计正式延迟，并发固定为 1。输出分别包含核心安全、噪声鲁棒性和全量补充指标；INVALID 保留在分母并计为错误。详情保存到 `results/<模型名>_binary_safety_<协议>_<时间>.json`，包含协议、数据哈希、推理参数、三部分指标和逐条结果。

`results/` 中缺少 `evaluation_protocol=binary_safety_v1_strict` 或 `binary_safety_v1_native` 的历史 JSON 均属于旧口径，不得与新结果直接比较。

### 输入安全三分类评测

三分类数据集为 `datasets/raw/sample_input_safety_multiclass_eval.jsonl`，包含 SAFE 400、UNSAFE 400、IRRELEVANT 200。UNSAFE 进一步使用 `legal_risk`、`self_harm`、`harm_others`、`safety_bypass`、`dangerous_operation`、`other_unsafe` 六类风险。重新生成和验证：

```bash
uv run python scripts/generate_multiclass_dataset.py
uv run python scripts/validate_jsonl.py --multiclass
```

模型必须输出严格 JSON，例如：

```json
{"label":"UNSAFE","risk_type":"harm_others"}
```

SAFE 和 IRRELEVANT 的 `risk_type` 必须为 `null`。运行评测：

```bash
uv run python scripts/evaluate.py \
  --task multiclass \
  --model qwen2_5_1_5b \
  --prompt few_shot
```

多分类任务只支持统一的 strict JSON 协议，不支持 Guard 原生协议。结果保存到 `results/input_safety_multiclass/`，包含一级分类指标、六类 risk_type 指标、业务误判指标、混淆矩阵、延迟和逐条原始输出。省略 `--task` 时仍执行原有二分类评测。

专用 Guard（例如 YuFeng）应另行使用原生安全检测轨道，避免把其官方风险代码误判为 JSON 格式错误：

```bash
uv run python scripts/evaluate.py \
  --task native_safety \
  --model yufeng_xguard_0_6b \
  --protocol native
```

该轨道仍读取同一份 1000 条多分类文本，但只评估风险/无风险：原 `UNSAFE` 为有风险，原 `SAFE` 与 `IRRELEVANT` 合并为无风险。请求仅包含用户文本并使用模型官方 chat template；结果保存在 `results/native_safety/`。它不能衡量 `SAFE` 与 `IRRELEVANT` 路由能力，也不会强行把 YuFeng 原生代码映射为项目六类 `risk_type`，因此不得与 strict 多分类准确率放在同一排行榜中。

多个模型完成同一数据哈希的评测后，可生成横向对比 JSON 和 Markdown 报告：

```bash
uv run python scripts/summarize_multiclass_results.py
```

### 五模型逐个评测

单张 8GB GPU 每次只启动一个模型。下载命令如下；Llama Guard 需要先在 Hugging Face 接受 Meta 许可并在 `.env` 配置 `HF_TOKEN`：

```bash
uv run python scripts/download_model.py --model yufeng_xguard_0_6b
uv run python scripts/download_model.py --model qwen3guard_gen_0_6b
uv run python scripts/download_model.py --model llama_guard_3_1b
uv run python scripts/download_model.py --model qwen2_5_1_5b
uv run python scripts/download_model.py --model qwen2_5_3b
```

在第一终端使用对应命令一键切换；每条命令都会自动终止本项目上一个模型和 UI：

```bash
# YuFeng-XGuard-Reason-0.6B
bash scripts/start_dev.sh switch --model yufeng_xguard_0_6b

# Qwen3Guard-Gen-0.6B
bash scripts/start_dev.sh switch --model qwen3guard_gen_0_6b

# Llama-Guard-3-1B（须先完成授权下载）
bash scripts/start_dev.sh switch --model llama_guard_3_1b

# Qwen2.5-1.5B-Instruct
bash scripts/start_dev.sh switch --model qwen2_5_1_5b

# Qwen2.5-3B-Instruct
bash scripts/start_dev.sh switch --model qwen2_5_3b
```

新终端检查服务并运行两种协议：

```bash
uv run python scripts/check_server.py --model qwen3guard_gen_0_6b
uv run python scripts/evaluate.py --model qwen3guard_gen_0_6b --protocol strict
uv run python scripts/evaluate.py --model qwen3guard_gen_0_6b --protocol native
```

YuFeng、Qwen3Guard 和 Llama Guard 均运行 strict/native；两个 Qwen2.5 通用模型只运行 strict。要测试下一个模型时直接执行下一条 `switch` 命令，不需要先运行 `stop` 或手动按 Ctrl+C。

## 增加和切换模型

在 `configs/models.yaml` 的 `models` 下增加配置。专用 Guard 模型设置 `native_guard: true` 和对应 `guard_family`；通用指令模型使用 `native_guard: false`、`guard_family: none`。可修改 `active_model`，或向各脚本传入 `--model <配置键>`。

## 显存不足与后续计划

RTX 4060 Ti 8GB 若显存不足，可在 YAML 中降低 `gpu_memory_utilization` 或 `max_model_len`，停止其他 GPU 进程后重试；脚本不会自动结束其他进程。不要让 `VLLM_BASE_URL` 指向与所选模型不一致的服务。

当前不含训练模块。后续可独立加入 LoRA/QLoRA 实验，但应继续复用现有 JSONL 评测集和 API 推理边界。

运行测试：

```bash
uv run pytest
```
