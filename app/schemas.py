from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


class PredictionResult(BaseModel):
    text: str
    prediction: Literal["ACCEPT", "REJECT", "INVALID"]
    raw_output: str
    latency_ms: float = Field(ge=0)
    model_name: str
    error: str | None = None


class DatasetItem(BaseModel):
    model_config = ConfigDict(extra="allow")
    id: str = Field(min_length=1)
    text: str = Field(min_length=1)
    label: Literal["ACCEPT", "REJECT"]
    category: str = Field(min_length=1)
    oos_type: Literal["near", "far", "unsafe"] | None = None
    difficulty: str = Field(min_length=1)
