from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator


class PredictionResult(BaseModel):
    text: str
    prediction: Literal["SAFE", "UNSAFE", "IRRELEVANT", "INVALID"]
    raw_output: str
    latency_ms: float = Field(ge=0)
    model_name: str
    risk_category: str | None = None
    risk_score: float | None = None
    explanation: str | None = None
    predicted_risk_type: str | None = None
    format_valid: bool = True
    parse_error: str | None = None
    error: str | None = None


class DatasetItem(BaseModel):
    model_config = ConfigDict(extra="allow")
    id: str = Field(min_length=1)
    text: str = Field(min_length=1)
    label: Literal["SAFE", "UNSAFE"]
    original_label: Literal["PASS", "BLOCK", "IRRELEVANT"]
    category: Literal["safe", "unsafe"]
    eval_subset: Literal["safety_core", "noise_robustness"]
    risk_type: str | None = None
    irrelevant_type: str | None = None
    scene_type: str = Field(min_length=1)
    difficulty: Literal["easy", "medium", "hard"]
    source: str = Field(min_length=1)
    source_id: str | None = None
    template_family: str = Field(min_length=1)

    @model_validator(mode="after")
    def validate_binary_mapping(self) -> "DatasetItem":
        expected = {
            "PASS": ("SAFE", "safe", "safety_core"),
            "BLOCK": ("UNSAFE", "unsafe", "safety_core"),
            "IRRELEVANT": ("SAFE", "safe", "noise_robustness"),
        }[self.original_label]
        if (self.label, self.category, self.eval_subset) != expected:
            raise ValueError(f"original_label={self.original_label} 的二分类映射不一致")
        if self.original_label == "BLOCK" and not self.risk_type:
            raise ValueError("原 BLOCK 样本必须保留 risk_type")
        if self.original_label == "IRRELEVANT" and not self.irrelevant_type:
            raise ValueError("原 IRRELEVANT 样本必须保留 irrelevant_type")
        return self


MULTICLASS_RISK_TYPES = {
    "legal_risk", "self_harm", "harm_others", "safety_bypass",
    "dangerous_operation", "other_unsafe",
}


class MulticlassDatasetItem(BaseModel):
    model_config = ConfigDict(extra="forbid")
    id: str = Field(min_length=1)
    text: str = Field(min_length=1)
    label: Literal["SAFE", "UNSAFE", "IRRELEVANT"]
    risk_type: str | None = None
    difficulty: Literal["easy", "medium", "hard"]
    source: Literal["synthetic_reviewed"]

    @model_validator(mode="after")
    def validate_risk_type(self) -> "MulticlassDatasetItem":
        if self.label == "UNSAFE" and self.risk_type not in MULTICLASS_RISK_TYPES:
            raise ValueError("UNSAFE 样本必须填写合法 risk_type")
        if self.label != "UNSAFE" and self.risk_type is not None:
            raise ValueError("SAFE/IRRELEVANT 的 risk_type 必须为 null")
        return self
