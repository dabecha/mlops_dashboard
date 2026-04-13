from __future__ import annotations

from datetime import datetime
from typing import Dict, Literal, Optional

from pydantic import BaseModel, Field


class ProjectCreate(BaseModel):
    name: str = Field(..., min_length=1, max_length=100)
    description: Optional[str] = Field(None, max_length=500)
    task_type: Literal["classification", "regression"] = "classification"


class ProjectResponse(BaseModel):
    id: int
    name: str
    description: Optional[str]
    task_type: str
    created_at: datetime

    model_config = {"from_attributes": True}


class InferenceLogCreate(BaseModel):
    project_name: str
    request_id: Optional[str] = None
    prediction: float
    actual_label: Optional[float] = None
    confidence: Optional[float] = None
    response_time_ms: float = Field(..., ge=0)
    is_error: bool = False
    error_message: Optional[str] = None
    feature_values: Optional[Dict[str, float]] = None
    timestamp: Optional[datetime] = None


class InferenceLogResponse(BaseModel):
    id: int
    project_id: int
    timestamp: datetime
    request_id: Optional[str]
    prediction: float
    actual_label: Optional[float]
    confidence: Optional[float]
    response_time_ms: float
    is_error: bool
    error_message: Optional[str]

    model_config = {"from_attributes": True}
