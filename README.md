# rover-oos-detection

机器狗请求拒识模型的本地部署、交互测试与批量评测项目。模型将动作控制及机器狗状态、能力、使用方法问答判为 `ACCEPT`，将范围外请求判为 `REJECT`；任何其他模型输出均严格记为 `INVALID`。当前不包含训练、数据库、Redis 或 Docker，后续可在不影响推理评测结构的前提下扩展 LoRA/QLoRA。

## 项目结构

```text
app/          配置、提示词、API 推理、Streamlit UI
configs/      多模型 YAML 配置
datasets/     JSONL 原始数据和处理数据占位目录
evaluation/   数据加载、批量执行和指标计算
models/       本地模型（不提交 Git）
results/      评测结果（不提交 Git）
scripts/      下载、服务启动、健康检查、评测入口
tests/        输出解析和数据集校验测试
```

## 安装与标准启动

要求 Ubuntu 24.04、Python 3.12、可用 NVIDIA 驱动及 CUDA 环境。项目保留 1.5B 基线，默认活动模型为 Qwen2.5 3B；RTX 4060 Ti 8GB 同一时间只运行一个模型。

```bash
uv sync
cp .env.example .env
uv run python scripts/download_model.py --model qwen2_5_1_5b
bash scripts/start_vllm.sh
```

默认 3B 模型的首次准备流程为：

```bash
uv run python scripts/download_model.py --model qwen2_5_3b
bash scripts/start_vllm.sh --model qwen2_5_3b
```

下载脚本使用 Hugging Face Hub，将模型下载到 `configs/models.yaml` 的 `local_path`。私有或受限模型可在 `.env` 设置 `HF_TOKEN`，镜像站可设置 `HF_ENDPOINT`。目标目录已经非空时脚本会明确跳过，不会覆盖；推理和启动脚本也不会静默下载模型。

启动脚本会检查模型目录、uv 环境中的 vLLM、NVIDIA GPU 和监听端口，然后在前台执行 `vllm serve`。它不会结束已有进程，也不会后台运行。
默认使用 vLLM 的标准 PyTorch 采样器，避免仅安装 NVIDIA 驱动、未安装完整 CUDA Toolkit 时 FlashInfer JIT 查找 `nvcc` 失败；需要启用 FlashInfer 采样器时可设置 `VLLM_USE_FLASHINFER_SAMPLER=1`。
3B 模型在 8GB 显卡上启用 `enforce_eager`，避免 CUDA Graph 捕获的临时显存峰值；该参数按模型配置，不影响 1.5B 基线。

## 检查服务与启动 UI

推荐本地调试使用统一前台启动器，它会依次启动并监管 vLLM 与 Streamlit：

```bash
bash scripts/start_dev.sh --model qwen2_5_3b
```

浏览器访问 `http://localhost:8501`。保持该终端运行；按 `Ctrl+C` 会同时清理两个服务。UI 异常退出或连续健康检查失败时会自动重启（最多连续 5 次），vLLM 异常退出时才清理整套服务。重复执行启动命令会先优雅停止旧实例，再启动新实例，不会因本项目已经占用 8501 而直接失败。

统一管理命令如下：

```bash
bash scripts/start_dev.sh status
bash scripts/start_dev.sh stop
bash scripts/start_dev.sh restart --model qwen2_5_3b
```

不要只终止 Streamlit 子进程，因为监管脚本会自动恢复它。需要完整停止时使用 `stop` 或在启动终端按 `Ctrl+C`。如果端口被其他项目占用，启动器只报告占用进程，不会擅自终止。

保持 vLLM 终端运行，在新终端执行：

```bash
uv run python scripts/check_server.py
uv run streamlit run app/ui.py
```

vLLM 是 API 服务，不提供根路径网页，因此浏览器访问 `http://127.0.0.1:8000/` 返回 404 属于正常现象；`/v1` 请求未携带密钥时返回 401 也属于正常鉴权行为。可用以下命令查看当前服务模型：

```bash
curl -H "Authorization: Bearer local-token" \
  http://127.0.0.1:8000/v1/models
```

实际测试页面由 Streamlit 提供，默认地址为 `http://localhost:8501`。
页面脚本会根据自身位置解析项目根目录，因此即使从其他工作目录启动，也能正确导入项目模块。

页面使用最大 1200px 的居中响应式布局，浅蓝导航栏显示当前模型和服务状态，不展示内部 API 地址；快速样例直接平铺在页面中，点击后填入输入框。UI 固定使用效果更稳定的 Few-shot，并通过 ACCEPT/REJECT 双状态卡高亮检测结果，同时展示原始输出、耗时和最近记录。Zero-shot 仍保留给命令行批量评测做对照。UI 只调用 OpenAI Compatible API，不在进程中加载模型。

## 批量评测

```bash
uv run python scripts/evaluate.py \
  --model qwen2_5_3b \
  --dataset datasets/raw/sample_oos.jsonl
```

默认使用 Few-shot，可用 `--prompt zero_shot` 运行 Zero-shot 对照。命令输出 Accuracy、两类 Precision/Recall/F1、False Accept Rate、False Reject Rate、INVALID 数量、平均与 P95 延迟，并将逐条详情写入 `results/<模型名>_<时间>.json`。

JSONL 每行必须包含 `id`、`text`、`label`、`category`、`oos_type`、`difficulty`；`label` 只能是 `ACCEPT` 或 `REJECT`。运行测试：
`oos_type` 可为 `near`、`far`、`unsafe` 或 `null`。当前策略采用能力范围与安全双门控：只有能力范围内且安全的请求才接受；明确伤人、撞击、破坏、危险地形或违法侵入的请求拒绝。

```bash
uv run pytest
```

## 增加和切换模型

在 `configs/models.yaml` 的 `models` 下增加一项，完整填写 Hugging Face ID、本地相对路径、服务名称、端口和推理参数。下载、启动、服务检查和评测都可通过 `--model <配置键>` 指定模型；省略参数时使用顶层 `active_model`。UI 使用活动模型配置。

两个模型复用一张 GPU 和端口 8000，必须串行运行。停止当前服务后，可分别执行：

```bash
bash scripts/start_vllm.sh --model qwen2_5_3b
bash scripts/start_vllm.sh --model qwen2_5_1_5b
```

如果不同模型服务使用另一个地址或密钥，可在 `.env` 用 `VLLM_BASE_URL` 和 `VLLM_API_KEY` 覆盖当前活动配置。通常切换活动模型时应同步修改或清空这两个覆盖值，避免仍连接旧服务。模型路径和名称均来自 YAML，而非业务代码。

## 显存不足处理

先停止 vLLM，再在 YAML 中逐步降低 `gpu_memory_utilization` 或 `max_model_len`；同时确认没有其他 GPU 进程占用显存。仍不足时应增加一个更小或量化模型配置并切换 `active_model`。不要直接修改业务代码，也不要在同一张 8GB 显卡上同时启动多个模型服务。

## 后续微调计划

当前版本只做基线模型选型和提示词评测。积累并清洗真实误判数据后，可另行增加 LoRA/QLoRA 训练流程和适配器配置；训练产物作为新的模型配置接入现有 vLLM/API/评测链路。本项目当前刻意不创建训练模块。

## 启动命令

bash scripts/start_dev.sh --model qwen2_5_3b
