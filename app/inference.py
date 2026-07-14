from __future__ import annotations

import math
import re
import time
from dataclasses import dataclass

from openai import APIConnectionError, APIStatusError, APITimeoutError, OpenAI

from app.config import ModelConfig, get_model_config
from app.prompts import get_prompt
from app.schemas import PredictionResult

# YuFeng-XGuard native one-token taxonomy. ``sec`` is the only safe label;
# every other documented taxonomy token represents a risk.
RISK_CATEGORIES = {
    "sec": "安全",
    "pc": "违法犯罪－色情违禁品",
    "dc": "违法犯罪－毒品犯罪",
    "dw": "违法犯罪－危险武器",
    "pi": "违法犯罪－财产侵害",
    "ec": "违法犯罪－经济犯罪",
    "ac": "仇恨言论－辱骂诅咒",
    "def": "仇恨言论－诽谤",
    "ti": "仇恨言论－威胁恐吓",
    "cy": "仇恨言论－网络欺凌",
    "ph": "身心健康－身体健康",
    "mh": "身心健康－心理健康",
    "se": "伦理道德－社会伦理",
    "sci": "伦理道德－科学伦理",
    "pp": "数据隐私－个人隐私",
    "cs": "数据隐私－商业秘密",
    "acc": "网络安全－访问控制",
    "mc": "网络安全－恶意代码",
    "ha": "网络安全－黑客攻击",
    "ps": "网络安全－物理安全",
    "ter": "极端主义－暴力恐怖活动",
    "sd": "极端主义－社会秩序破坏",
    "ext": "极端主义－极端思想倾向",
    "fin": "不当建议－金融",
    "med": "不当建议－医疗",
    "law": "不当建议－法律",
    "cm": "未成年人风险－腐化未成年人",
    "ma": "未成年人风险－虐待与剥削",
    "md": "未成年人风险－违法失范",
}


@dataclass(frozen=True)
class ParsedPrediction:
    prediction: str
    risk_category: str | None = None
    explanation: str | None = None


def parse_guard_output(raw_output: str) -> ParsedPrediction:
    """Parse either standardized labels or XGuard's documented native output."""
    normalized = raw_output.strip()
    if normalized in {"PASS", "BLOCK"}:
        return ParsedPrediction(normalized)
    match = re.fullmatch(r"([a-z]+)(?:\s*<explanation>\s*(.*?)(?:\s*</explanation>)?)?", normalized, re.DOTALL)
    if not match or match.group(1) not in RISK_CATEGORIES:
        return ParsedPrediction("INVALID")
    code, explanation = match.groups()
    return ParsedPrediction(
        "PASS" if code == "sec" else "BLOCK",
        RISK_CATEGORIES[code],
        explanation.removesuffix("</explanation>").strip() if explanation and explanation.strip() else None,
    )


def parse_prediction(raw_output: str) -> str:
    return parse_guard_output(raw_output).prediction


def get_standard_explanation(prediction: str, risk_category: str | None) -> str:
    if prediction == "PASS":
        return "未检测到明显安全风险，该请求可以进入后续处理流程。"
    if prediction == "BLOCK":
        category = risk_category or "未知类别"
        return f"检测到“{category}”相关安全风险，建议拦截该请求。"
    return "模型输出无法解析或服务异常，无法完成安全判断。"


def check_vllm_server(config: ModelConfig, timeout: float = 3.0) -> tuple[bool, str]:
    client = OpenAI(base_url=config.base_url, api_key=config.effective_api_key, timeout=timeout)
    try:
        model_ids = [model.id for model in client.models.list().data]
    except APIConnectionError as exc:
        return False, f"无法连接 vLLM：{exc}"
    except APITimeoutError as exc:
        return False, f"vLLM 健康检查超时：{exc}"
    except APIStatusError as exc:
        return False, f"vLLM 健康检查失败（HTTP {exc.status_code}）：{exc}"
    if config.served_model_name not in model_ids:
        return False, f"服务模型不匹配：期望 {config.served_model_name}，实际 {model_ids}"
    return True, f"vLLM 已连接：{config.served_model_name}"


class SafetyDetector:
    def __init__(self, prompt_name: str = "few_shot", model_config: ModelConfig | None = None, timeout: float = 30.0) -> None:
        self.config = model_config or get_model_config()
        self.prompt_name = prompt_name
        self.client = OpenAI(base_url=self.config.base_url, api_key=self.config.effective_api_key, timeout=timeout)

    def predict(self, text: str) -> PredictionResult:
        if not text or not text.strip():
            return self._result(text, "", 0.0, "输入不能为空")
        started = time.perf_counter()
        raw_output = ""
        risk_score: float | None = None
        error: str | None = None
        try:
            messages = [{"role": "user", "content": text.strip()}]
            if not self.config.native_guard:
                messages.insert(0, {"role": "system", "content": get_prompt(self.prompt_name)})
            response = self.client.chat.completions.create(
                model=self.config.served_model_name,
                messages=messages,
                temperature=0,
                max_tokens=200 if self.config.native_guard else 8,
                logprobs=self.config.native_guard,
                top_logprobs=20 if self.config.native_guard else None,
            )
            choice = response.choices[0]
            raw_output = choice.message.content or ""
            if self.config.native_guard and choice.logprobs and choice.logprobs.content:
                risk_score = math.exp(choice.logprobs.content[0].logprob)
        except APIConnectionError as exc:
            error = f"无法连接 vLLM 服务: {exc}"
        except APITimeoutError as exc:
            error = f"vLLM 请求超时: {exc}"
        except APIStatusError as exc:
            error = f"vLLM 服务异常（HTTP {exc.status_code}）: {exc}"
        except (IndexError, AttributeError, TypeError, ValueError) as exc:
            error = f"vLLM 响应格式异常: {exc}"
        return self._result(text, raw_output, (time.perf_counter() - started) * 1000, error, risk_score)

    def _result(self, text: str, raw: str, latency: float, error: str | None, risk_score: float | None = None) -> PredictionResult:
        parsed = parse_guard_output(raw)
        return PredictionResult(text=text, prediction=parsed.prediction, raw_output=raw, latency_ms=latency,
                                model_name=self.config.served_model_name, risk_category=parsed.risk_category,
                                risk_score=risk_score, explanation=parsed.explanation, error=error)


# Compatibility for existing callers while the public task name changes.
OOSDetector = SafetyDetector
