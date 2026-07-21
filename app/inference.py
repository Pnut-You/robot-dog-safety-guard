from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Literal

from openai import APIConnectionError, APIStatusError, APITimeoutError, OpenAI

from app.config import ModelConfig, get_model_config
from app.prompts import get_multiclass_prompt, get_prompt
from app.schemas import MULTICLASS_RISK_TYPES, PredictionResult

@dataclass(frozen=True)
class ParsedPrediction:
    prediction: str
    risk_category: str | None = None
    explanation: str | None = None
    format_valid: bool = True
    parse_error: str | None = None


def parse_guard_output(raw_output: str) -> ParsedPrediction:
    """Normalize a single binary label and reject all non-exact output."""
    normalized = raw_output.strip()
    quote_pairs = {'"': '"', "'": "'", "“": "”", "‘": "’"}
    if len(normalized) >= 2 and normalized[0] in quote_pairs and normalized[-1] == quote_pairs[normalized[0]]:
        normalized = normalized[1:-1].strip()
    normalized = normalized.upper()
    if normalized.endswith((".", "。")):
        normalized = normalized[:-1].rstrip()
    if normalized == "SAFE":
        return ParsedPrediction("SAFE")
    if normalized == "UNSAFE":
        return ParsedPrediction("UNSAFE")
    return ParsedPrediction("INVALID")


def parse_prediction(raw_output: str) -> str:
    return parse_guard_output(raw_output).prediction


def parse_multiclass_output(raw_output: str) -> ParsedPrediction:
    """Parse the project JSON contract while preserving a valid primary label."""
    import json

    try:
        value = json.loads(raw_output.strip())
    except (json.JSONDecodeError, TypeError):
        return ParsedPrediction("INVALID", format_valid=False, parse_error="输出不是合法 JSON")
    if not isinstance(value, dict):
        return ParsedPrediction("INVALID", format_valid=False, parse_error="JSON 顶层必须是对象")
    label_value = value.get("label")
    if not isinstance(label_value, str):
        return ParsedPrediction("INVALID", format_valid=False, parse_error="缺少合法 label")
    label = label_value.strip().upper()
    if label not in {"SAFE", "UNSAFE", "IRRELEVANT"}:
        return ParsedPrediction("INVALID", format_valid=False, parse_error="label 非法")

    errors: list[str] = []
    if set(value) != {"label", "risk_type"}:
        errors.append("字段必须且只能包含 label 和 risk_type")
    risk = value.get("risk_type")
    normalized_risk = risk.strip().lower() if isinstance(risk, str) else None
    if label == "UNSAFE":
        if normalized_risk not in MULTICLASS_RISK_TYPES:
            errors.append("UNSAFE 缺少合法 risk_type")
            normalized_risk = None
    elif risk is not None:
        errors.append("SAFE/IRRELEVANT 的 risk_type 必须为 null")
        normalized_risk = None
    return ParsedPrediction(label, risk_category=normalized_risk, format_valid=not errors,
                            parse_error="；".join(errors) or None)


YUFENG_RISK_CODES = {
    "pc", "dc", "dw", "pi", "ec", "ac", "def", "ti", "cy", "ph", "mh", "se", "sci",
    "pp", "cs", "acc", "mc", "ha", "ps", "ter", "sd", "ext", "fin", "med", "law", "cm", "ma", "md",
}


def parse_native_guard_output(raw_output: str, guard_family: str) -> ParsedPrediction:
    """Parse only documented native guard formats; unknown output remains INVALID."""
    normalized = raw_output.strip()
    if guard_family == "yufeng":
        code = normalized.splitlines()[0].strip().lower() if normalized else ""
        if code == "sec":
            return ParsedPrediction("SAFE", risk_category="sec")
        if code in YUFENG_RISK_CODES:
            return ParsedPrediction("UNSAFE", risk_category=code)
        return ParsedPrediction("INVALID")
    if guard_family == "qwen3guard":
        lines = [line.strip() for line in normalized.splitlines() if line.strip()]
        if not lines:
            return ParsedPrediction("INVALID")
        header = lines[0].lower()
        category = None
        for line in lines[1:]:
            if line.lower().startswith("categories:"):
                category = line.split(":", 1)[1].strip() or None
                break
        if header == "safety: safe":
            return ParsedPrediction("SAFE", risk_category=category)
        if header in {"safety: unsafe", "safety: controversial"}:
            return ParsedPrediction("UNSAFE", risk_category=category)
        return ParsedPrediction("INVALID")
    if guard_family == "llama_guard":
        lines = [line.strip() for line in normalized.splitlines() if line.strip()]
        if not lines:
            return ParsedPrediction("INVALID")
        if lines[0].lower() == "safe":
            return ParsedPrediction("SAFE")
        if lines[0].lower() == "unsafe":
            return ParsedPrediction("UNSAFE", risk_category=lines[1] if len(lines) > 1 else None)
        return ParsedPrediction("INVALID")
    if guard_family == "singguard":
        import re

        lines = [line.strip() for line in normalized.splitlines() if line.strip()]
        if not lines:
            return ParsedPrediction("INVALID")
        label = lines[0].lower()
        category_match = re.search(r"<answer>\s*(.*?)\s*</answer>", normalized, flags=re.IGNORECASE | re.DOTALL)
        category = category_match.group(1).strip() if category_match else None
        if label == "safe":
            return ParsedPrediction("SAFE", risk_category=category)
        if label == "unsafe":
            return ParsedPrediction("UNSAFE", risk_category=category)
        return ParsedPrediction("INVALID")
    return ParsedPrediction("INVALID")


def get_standard_explanation(prediction: str, risk_category: str | None) -> str:
    if prediction == "SAFE":
        return "未检测到明确安全风险；这不代表输入与机器狗相关或可以直接执行。"
    if prediction == "UNSAFE":
        return "检测到明确安全风险，建议拦截该输入。"
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
    def __init__(self, prompt_name: str = "few_shot", model_config: ModelConfig | None = None,
                 timeout: float = 30.0, protocol: Literal["strict", "native"] = "strict",
                 task: Literal["binary", "multiclass"] = "binary") -> None:
        self.config = model_config or get_model_config()
        self.prompt_name = prompt_name
        self.protocol = protocol
        self.task = task
        if task == "multiclass" and protocol != "strict":
            raise ValueError("多分类任务只支持 strict JSON 协议")
        if protocol == "native" and not self.config.native_guard:
            raise ValueError(f"模型 {self.config.served_model_name} 不支持 native Guard 协议")
        self.client = OpenAI(base_url=self.config.base_url, api_key=self.config.effective_api_key, timeout=timeout)

    def predict(self, text: str) -> PredictionResult:
        if not text or not text.strip():
            return self._result(text, "", 0.0, "输入不能为空")
        started = time.perf_counter()
        raw_output = ""
        error: str | None = None
        try:
            if self.task == "multiclass":
                messages = [
                    {"role": "system", "content": get_multiclass_prompt(self.prompt_name)},
                    {"role": "user", "content": text.strip()},
                ]
                max_tokens = 64
            elif self.protocol == "native":
                if self.config.guard_family == "singguard":
                    messages = [{"role": "user", "content": [{"type": "text", "text": text.strip()}]}]
                    max_tokens = 64
                else:
                    messages = [{"role": "user", "content": text.strip()}]
                    max_tokens = 1 if self.config.guard_family == "yufeng" else 32
            else:
                messages = [
                    {"role": "system", "content": get_prompt(self.prompt_name)},
                    {"role": "user", "content": text.strip()},
                ]
                max_tokens = 4
            request_kwargs = dict(
                model=self.config.served_model_name,
                messages=messages,
                temperature=0,
                top_p=1,
                max_tokens=max_tokens,
            )
            if self.protocol == "native" and self.config.guard_family == "singguard":
                request_kwargs["extra_body"] = {"chat_template_kwargs": {"thinking_type": "fast"}}
            response = self.client.chat.completions.create(**request_kwargs)
            choice = response.choices[0]
            raw_output = choice.message.content or ""
        except APIConnectionError as exc:
            error = f"无法连接 vLLM 服务: {exc}"
        except APITimeoutError as exc:
            error = f"vLLM 请求超时: {exc}"
        except APIStatusError as exc:
            error = f"vLLM 服务异常（HTTP {exc.status_code}）: {exc}"
        except (IndexError, AttributeError, TypeError, ValueError) as exc:
            error = f"vLLM 响应格式异常: {exc}"
        return self._result(text, raw_output, (time.perf_counter() - started) * 1000, error)

    def _result(self, text: str, raw: str, latency: float, error: str | None) -> PredictionResult:
        if self.task == "multiclass":
            parsed = parse_multiclass_output(raw)
        else:
            parsed = (parse_native_guard_output(raw, self.config.guard_family)
                      if self.protocol == "native" else parse_guard_output(raw))
        return PredictionResult(text=text, prediction=parsed.prediction, raw_output=raw, latency_ms=latency,
                                model_name=self.config.served_model_name, risk_category=parsed.risk_category,
                                predicted_risk_type=parsed.risk_category, format_valid=parsed.format_valid,
                                parse_error=parsed.parse_error, risk_score=None, explanation=None, error=error)


# Compatibility for existing callers while the public task name changes.
OOSDetector = SafetyDetector
