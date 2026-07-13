from __future__ import annotations

import time

from openai import APIConnectionError, APIStatusError, APITimeoutError, OpenAI

from app.config import ModelConfig, get_model_config
from app.prompts import get_prompt
from app.schemas import PredictionResult


def parse_prediction(raw_output: str) -> str:
    normalized = raw_output.strip()
    return normalized if normalized in {"ACCEPT", "REJECT"} else "INVALID"


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


class OOSDetector:
    def __init__(
        self,
        prompt_name: str = "few_shot",
        model_config: ModelConfig | None = None,
        timeout: float = 30.0,
    ) -> None:
        self.config = model_config or get_model_config()
        self.prompt_name = prompt_name
        self.client = OpenAI(
            base_url=self.config.base_url,
            api_key=self.config.effective_api_key,
            timeout=timeout,
        )

    def predict(self, text: str) -> PredictionResult:
        if not text or not text.strip():
            return self._result(text, "", 0.0, "输入不能为空")

        started = time.perf_counter()
        raw_output = ""
        error: str | None = None
        try:
            response = self.client.chat.completions.create(
                model=self.config.served_model_name,
                messages=[
                    {"role": "system", "content": get_prompt(self.prompt_name)},
                    {"role": "user", "content": text.strip()},
                ],
                temperature=0,
                max_tokens=8,
            )
            raw_output = response.choices[0].message.content or ""
        except APIConnectionError as exc:
            error = f"无法连接 vLLM 服务: {exc}"
        except APITimeoutError as exc:
            error = f"vLLM 请求超时: {exc}"
        except APIStatusError as exc:
            error = f"vLLM 服务异常（HTTP {exc.status_code}）: {exc}"
        except (IndexError, AttributeError, ValueError) as exc:
            error = f"vLLM 响应格式异常: {exc}"
        latency_ms = (time.perf_counter() - started) * 1000
        return self._result(text, raw_output, latency_ms, error)

    def _result(self, text: str, raw: str, latency: float, error: str | None) -> PredictionResult:
        return PredictionResult(
            text=text,
            prediction=parse_prediction(raw),
            raw_output=raw,
            latency_ms=latency,
            model_name=self.config.served_model_name,
            error=error,
        )
