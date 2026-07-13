请帮我初始化一个机器狗拒识模型选型与评测项目。

项目名称：

```text
rover-oos-detection
```

## 一、项目目标

当前先完成以下功能：

1. 本地下载并部署 `Qwen/Qwen2.5-1.5B-Instruct`；
2. 使用 vLLM 提供 OpenAI Compatible API；
3. 使用 Streamlit 提供基础测试页面；
4. 输入用户文本，输出：

   * `ACCEPT`
   * `REJECT`
5. 支持后续频繁增加和切换其他模型；
6. 支持 JSONL 数据集批量评测；
7. 当前不做微调，只预留后续 LoRA/QLoRA 扩展空间；
8. 使用 uv 管理环境和依赖。

运行环境：

```text
Ubuntu 24.04
Python 3.12
RTX 4060 Ti 8GB
uv
vLLM
Streamlit
```

## 二、拒识任务

机器狗当前支持：

1. 动作控制：前进、后退、转向、停止、站立、坐下、趴下、握手、跳舞等；
2. 问答：机器狗状态、能力、使用方法及允许范围内的问答。

示例：

```text
向前走两米 → ACCEPT
你现在还有多少电 → ACCEPT
你会不会跳舞 → ACCEPT
帮我拿桌上的杯子 → REJECT
帮我订一张机票 → REJECT
```

模型只能输出完整的 `ACCEPT` 或 `REJECT`。

去除首尾空格和换行后，如果不是这两个值，则标记为 `INVALID`。

## 三、项目目录

目录保持简洁：

```text
rover-oos-detection/
├── app/
│   ├── config.py
│   ├── schemas.py
│   ├── prompts.py
│   ├── inference.py
│   └── ui.py
├── configs/
│   └── models.yaml
├── datasets/
│   ├── raw/sample_oos.jsonl
│   └── processed/.gitkeep
├── evaluation/
│   ├── evaluator.py
│   └── metrics.py
├── models/.gitkeep
├── results/.gitkeep
├── scripts/
│   ├── download_model.py
│   ├── start_vllm.sh
│   ├── check_server.py
│   └── evaluate.py
├── tests/
│   ├── test_output_parser.py
│   └── test_dataset_loader.py
├── .env.example
├── .gitignore
├── .python-version
├── pyproject.toml
├── uv.lock
└── README.md
```

不要创建数据库、Redis、Docker和训练模块。

## 四、多模型配置

使用 `configs/models.yaml` 管理模型：

```yaml
active_model: qwen2_5_1_5b

models:
  qwen2_5_1_5b:
    display_name: Qwen2.5 1.5B Instruct
    model_id: Qwen/Qwen2.5-1.5B-Instruct
    local_path: models/Qwen2.5-1.5B-Instruct
    served_model_name: qwen2.5-1.5b-instruct
    host: 127.0.0.1
    port: 8000
    api_key: local-token
    dtype: auto
    max_model_len: 4096
    gpu_memory_utilization: 0.80
```

要求：

* 下载、部署、推理和 UI 统一读取该配置；
* 通过 `active_model` 切换模型；
* 不要在代码中写死模型路径和名称；
* 模型不存在时给出明确错误，不要静默下载。

## 五、模型下载与 vLLM 部署

实现模型下载脚本：

```bash
uv run python scripts/download_model.py --model qwen2_5_1_5b
```

要求：

* 使用 Hugging Face Hub；
* 下载到配置中的 `local_path`；
* 支持 `HF_TOKEN` 和 `HF_ENDPOINT`；
* 已存在时不要重复覆盖。

`.env.example`：

```env
HF_TOKEN=
HF_ENDPOINT=
VLLM_API_KEY=local-token
VLLM_BASE_URL=http://127.0.0.1:8000/v1
```

实现 `scripts/start_vllm.sh`，使用本地模型执行：

```bash
vllm serve
```

需要读取配置并传入：

```text
--served-model-name
--host
--port
--api-key
--dtype
--max-model-len
--gpu-memory-utilization
```

启动前检查：

* 模型目录；
* vLLM 命令；
* NVIDIA GPU；
* 端口占用。

脚本使用：

```bash
set -euo pipefail
```

不要自动终止其他进程，也不要后台启动。

## 六、推理模块

`app/inference.py` 使用 OpenAI Python 客户端调用本地 vLLM，不要再次加载模型。

提供：

```python
class OOSDetector:
    def predict(self, text: str) -> PredictionResult:
        ...
```

返回结构：

```python
class PredictionResult(BaseModel):
    text: str
    prediction: Literal["ACCEPT", "REJECT", "INVALID"]
    raw_output: str
    latency_ms: float
    model_name: str
    error: str | None = None
```

要求：

* `temperature=0`；
* `max_tokens` 较小；
* 校验空输入；
* 捕获连接失败、超时和服务异常；
* 严格解析模型输出。

## 七、提示词

提示词统一放在 `app/prompts.py`，提供：

```text
zero_shot
few_shot
```

Zero-shot 核心内容：

```text
你是机器狗系统的请求拒识模块。

机器狗支持动作控制，以及机器狗状态、能力和使用方法相关问答。

属于支持范围时输出 ACCEPT。
超出支持范围时输出 REJECT。

只能输出 ACCEPT 或 REJECT，不得解释。
```

Few-shot 增加少量正反样例。

## 八、Streamlit UI

实现 `app/ui.py`：

```bash
uv run streamlit run app/ui.py
```

页面标题：

```text
机器狗拒识模型测试
```

页面包含：

* 当前模型；
* vLLM 地址；
* Zero-shot/Few-shot 选择；
* 输入框；
* 检测按钮；
* ACCEPT、REJECT 或 INVALID；
* 原始输出；
* 推理耗时；
* 最近测试记录；
* 快速测试示例。

UI 只能调用 vLLM API，不能直接加载模型。

## 九、示例数据与评测

创建 `datasets/raw/sample_oos.jsonl`，放置约 20～30 条演示数据，覆盖：

```text
ACCEPT/action
ACCEPT/qa
REJECT/near
REJECT/far
```

格式示例：

```json
{"id":"action_001","text":"向前走两米","label":"ACCEPT","category":"action","oos_type":null,"difficulty":"easy"}
{"id":"qa_001","text":"你现在还有多少电","label":"ACCEPT","category":"qa","oos_type":null,"difficulty":"easy"}
{"id":"near_oos_001","text":"帮我拿桌上的杯子","label":"REJECT","category":"unknown","oos_type":"near","difficulty":"hard"}
{"id":"far_oos_001","text":"帮我订一张机票","label":"REJECT","category":"unknown","oos_type":"far","difficulty":"easy"}
```

实现批量评测：

```bash
uv run python scripts/evaluate.py \
  --dataset datasets/raw/sample_oos.jsonl
```

输出：

* Accuracy；
* ACCEPT Precision、Recall、F1；
* REJECT Precision、Recall、F1；
* False Accept Rate；
* False Reject Rate；
* INVALID 数量；
* 平均延迟；
* P95 延迟。

详细结果保存到 `results/`，文件名包含模型名称和时间。

## 十、依赖、测试与 README

使用 uv 管理依赖，包含：

```text
vllm
openai
streamlit
pydantic
pydantic-settings
pyyaml
python-dotenv
huggingface-hub
scikit-learn
pandas
pytest
```

测试至少覆盖：

1. `ACCEPT`、`REJECT`、大小写、空格、换行、长回答、空输出和非法输出；
2. 合法 JSONL、缺少字段、非法标签和损坏 JSON。

运行：

```bash
uv run pytest
```

`.gitignore` 忽略：

```text
.venv/
.env
__pycache__/
models/*
results/*
datasets/processed/*
```

但保留对应 `.gitkeep` 文件。

README 使用中文，说明：

* 项目介绍；
* 项目结构；
* 安装依赖；
* 下载模型；
* 启动 vLLM；
* 检查服务；
* 启动 UI；
* 批量评测；
* 切换模型；
* 显存不足处理；
* 后续微调计划。

标准启动流程：

```bash
uv sync
cp .env.example .env
uv run python scripts/download_model.py --model qwen2_5_1_5b
bash scripts/start_vllm.sh
```

新终端执行：

```bash
uv run python scripts/check_server.py
uv run streamlit run app/ui.py
```

## 十一、验收要求

请实际检查：

1. 项目和核心文件已创建；
2. 模型配置可以读取；
3. 下载和 vLLM 脚本可运行；
4. UI 能调用本地 vLLM；
5. 输出解析严格；
6. 示例数据可以批量评测；
7. 测试通过；
8. 代码没有绝对路径；
9. 后续切换模型主要通过 YAML 完成。

如果当前环境无法下载模型或启动 GPU 服务，请明确说明，不要虚构成功。

最后汇报：

1. 创建了哪些核心文件；
2. 如何下载和启动模型；
3. 如何启动 UI；
4. 如何运行评测；
5. 如何增加和切换模型；
6. 哪些步骤已验证；
7. 哪些步骤尚未验证。
