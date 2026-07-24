from __future__ import annotations

from datetime import datetime
from typing import Any, Dict, Literal, Optional

import json

from pydantic import BaseModel, Field, field_validator


class ProjectCreate(BaseModel):
    project_name: str = Field(..., min_length=1, max_length=100)
    description: Optional[str] = Field(None, max_length=500)
    task_type: Literal["binary", "multi-class", "multi-label", "regression"] = "binary"


class ProjectResponse(BaseModel):
    project_id: str
    project_name: str
    description: Optional[str]
    task_type: str
    created_at: datetime

    model_config = {"from_attributes": True}


class DeployedModelCreate(BaseModel):
    project_id: str
    model_version: Optional[str] = None
    feature_dtypes: Optional[Dict[str, str]] = None
    feature_importance: Optional[Dict[str, float]] = None
    is_activate: bool = True


class DeployedModelResponse(BaseModel):
    model_id: str
    project_id: str
    model_version: Optional[str]
    feature_importance: Optional[Dict[str, float]]
    is_activate: bool
    created_at: datetime

    model_config = {"from_attributes": True}

    @field_validator("feature_importance", mode="before")
    @classmethod
    def parse_feature_importance(cls, v):
        if isinstance(v, str):
            return json.loads(v)
        return v


class ReferenceLogCreate(BaseModel):
    model_id: Optional[str] = None
    feature_values: Optional[Dict[str, Any]] = None
    actual_values: Optional[float] = None


class ReferenceLogResponse(BaseModel):
    log_id: str
    project_id: str
    model_id: Optional[str]
    actual_values: Optional[float]

    model_config = {"from_attributes": True}


class InferenceLogCreate(BaseModel):
    project_name: str
    batch_log_id: str = Field(..., min_length=1, max_length=100)
    model_id: Optional[str] = None
    prediction_values: float
    actual_values: Optional[float] = None
    is_error: bool = False
    feature_values: Optional[Dict[str, Any]] = None
    feature_dtypes: Optional[Dict[str, str]] = None
    request_timestamp: Optional[datetime] = None


class ProjectConfigUpdate(BaseModel):
    drift_window_size: int = Field(100, ge=10, le=10000)
    psi_warning: float = Field(0.10, ge=0.0, le=1.0)
    psi_alert: float = Field(0.25, ge=0.0, le=2.0)
    ks_alpha: float = Field(0.05, ge=0.0, le=1.0)
    metric_name: str = Field("Accuracy", min_length=1, max_length=50)
    metric_warning: float = Field(75.0)
    metric_alert: float = Field(60.0)


class ProjectConfigResponse(ProjectConfigUpdate):
    project_id: str
    updated_at: datetime

    model_config = {"from_attributes": True}


class BulkDeleteLogsRequest(BaseModel):
    log_ids: list[str]
    project_id: Optional[str] = None  # dataiku モードで対象データセット特定に使用


class InferenceLogResponse(BaseModel):
    log_id: str
    project_id: str
    batch_log_id: str
    request_timestamp: datetime
    model_id: Optional[str]
    prediction_values: float
    actual_values: Optional[float]
    is_error: bool

    model_config = {"from_attributes": True}
