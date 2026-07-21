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

YUFENG_ALIGNED_RISK_TYPES = {
    "ps", "ph", "mh", "ti", "pp", "acc", "ha", "mc", "dw", "pi", "law", "ec",
}


class MulticlassDatasetItem(BaseModel):
    model_config = ConfigDict(extra="forbid")
    id: str = Field(min_length=1)
    text: str = Field(min_length=1)
    label: Literal["SAFE", "UNSAFE", "IRRELEVANT"]
    safety_label: Literal["SAFE", "UNSAFE"]
    risk_type: str | None = None
    guard_label: str = Field(min_length=1)
    difficulty: Literal["easy", "medium", "hard"]
    source: Literal["synthetic_reviewed"]

    @model_validator(mode="after")
    def validate_risk_type(self) -> "MulticlassDatasetItem":
        if self.label == "UNSAFE":
            if self.safety_label != "UNSAFE":
                raise ValueError("UNSAFE 路由标签必须映射为 safety_label=UNSAFE")
            if self.risk_type not in YUFENG_ALIGNED_RISK_TYPES or self.guard_label != self.risk_type:
                raise ValueError("UNSAFE 必须填写一致且合法的 YuFeng risk_type/guard_label")
        elif self.safety_label != "SAFE" or self.risk_type is not None or self.guard_label != "sec":
            raise ValueError("SAFE/IRRELEVANT 必须映射为 safety_label=SAFE、risk_type=null、guard_label=sec")
        return self
