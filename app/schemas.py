from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


class PredictionResult(BaseModel):
    text: str
    prediction: Literal["PASS", "BLOCK", "IRRELEVANT", "INVALID"]
    raw_output: str
    latency_ms: float = Field(ge=0)
    model_name: str
    risk_category: str | None = None
    risk_score: float | None = None
    explanation: str | None = None
    error: str | None = None


class DatasetItem(BaseModel):
    model_config = ConfigDict(extra="allow")
    id: str = Field(min_length=1)
    text: str = Field(min_length=1)
    label: Literal["PASS", "BLOCK", "IRRELEVANT"]
    category: str = Field(min_length=1)
    risk_type: str | None = None
    difficulty: str = Field(min_length=1)
    source: str = Field(min_length=1)
