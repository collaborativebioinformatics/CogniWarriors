"""HTTP request models for the Training Head API."""

from __future__ import annotations

from typing import Dict, Optional

from pydantic import BaseModel, Field


class InitializeRequest(BaseModel):
    round: int = Field(..., ge=0)
    model_version: str = Field(default="global")
    weights: Optional[str] = None
    weights_format: str = Field(default="torch_state_dict_base64")


class TrainRequest(BaseModel):
    round: Optional[int] = Field(default=None, ge=0)
    model_version: Optional[str] = None
    epochs: Optional[int] = Field(default=None, ge=1)
    batch_size: Optional[int] = Field(default=None, ge=1)
    learning_rate: Optional[float] = Field(default=None, gt=0)


class EvaluateRequest(BaseModel):
    split: str = Field(default="validation", pattern="^(validation|train)$")


class MetricsResponse(BaseModel):
    center_id: str
    round: int
    metrics: Dict[str, float]

