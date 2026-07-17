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
    project_id: int
    project_name: str
    description: Optional[str]
    task_type: str
    created_at: datetime

    model_config = {"from_attributes": True}


class DeployedModelCreate(BaseModel):
    project_id: int
    model_version: Optional[str] = None
    feature_dtypes: Optional[Dict[str, str]] = None
    feature_importance: Optional[Dict[str, float]] = None
    is_activate: bool = True


class DeployedModelResponse(BaseModel):
    model_id: int
    project_id: int
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
    model_id: Optional[int] = None
    feature_values: Optional[Dict[str, Any]] = None
    actual_values: Optional[float] = None


class ReferenceLogResponse(BaseModel):
    ref_id: int
    project_id: int
    model_id: Optional[int]
    actual_values: Optional[float]

    model_config = {"from_attributes": True}


class InferenceLogCreate(BaseModel):
    project_name: str
    batch_log_id: Optional[str] = None
    request_id: Optional[str] = None
    model_id: Optional[int] = None
    prediction_values: float
    actual_values: Optional[float] = None
    response_time_ms: float = Field(..., ge=0)
    is_error: bool = False
    feature_values: Optional[Dict[str, Any]] = None
    feature_dtypes: Optional[Dict[str, str]] = None
    request_timestamp: Optional[datetime] = None


class ProjectConfigUpdate(BaseModel):
    drift_window_size: int = Field(100, ge=10, le=10000)
    psi_warning: float = Field(0.10, ge=0.0, le=1.0)
    psi_alert: float = Field(0.25, ge=0.0, le=2.0)
    ks_alpha: float = Field(0.05, ge=0.0, le=1.0)
    accuracy_warning: float = Field(75.0, ge=0.0, le=100.0)
    accuracy_alert: float = Field(60.0, ge=0.0, le=100.0)
    mae_warning: Optional[float] = None
    mae_alert: Optional[float] = None


class ProjectConfigResponse(ProjectConfigUpdate):
    project_id: int
    updated_at: datetime

    model_config = {"from_attributes": True}


class BulkDeleteLogsRequest(BaseModel):
    log_ids: list[int]


# ─── Agent Monitoring Schemas ────────────────────────────────────────────────

class AgentProjectCreate(BaseModel):
    project_name: str = Field(..., min_length=1, max_length=100)
    description: Optional[str] = Field(None, max_length=500)
    agent_type: Literal["llm", "tool", "multi"] = "tool"
    model_name: Optional[str] = Field(None, max_length=100)


class AgentProjectResponse(BaseModel):
    project_id: int
    project_name: str
    description: Optional[str]
    agent_type: str
    model_name: Optional[str]
    created_at: datetime

    model_config = {"from_attributes": True}


class AgentRefDataCreate(BaseModel):
    feature_values: Optional[Dict[str, float]] = None
    feature_importance: Optional[Dict[str, float]] = None


class AgentRefDataResponse(BaseModel):
    ref_id: int
    project_id: int
    created_at: datetime

    model_config = {"from_attributes": True}


class AgentTaskCreate(BaseModel):
    project_name: str
    session_id: Optional[str] = None
    task_name: Optional[str] = None
    status: Literal["success", "failure", "error"] = "success"
    quality_score: Optional[float] = Field(None, ge=0.0, le=1.0)
    total_input_tokens: Optional[int] = Field(None, ge=0)
    total_output_tokens: Optional[int] = Field(None, ge=0)
    total_cost_usd: Optional[float] = Field(None, ge=0.0)
    total_steps: int = 0
    duration_ms: Optional[float] = Field(None, ge=0.0)
    started_at: Optional[datetime] = None
    feature_values: Optional[Dict[str, float]] = None
    error_message: Optional[str] = None


class AgentTaskResponse(BaseModel):
    task_id: int
    project_id: int
    session_id: Optional[str]
    task_name: Optional[str]
    status: str
    quality_score: Optional[float]
    total_input_tokens: Optional[int]
    total_output_tokens: Optional[int]
    total_cost_usd: Optional[float]
    total_steps: int
    duration_ms: Optional[float]
    started_at: datetime

    model_config = {"from_attributes": True}


class AgentStepCreate(BaseModel):
    step_type: Literal["llm_call", "tool_call", "agent_call"]
    step_name: Optional[str] = Field(None, max_length=200)
    input_tokens: Optional[int] = Field(None, ge=0)
    output_tokens: Optional[int] = Field(None, ge=0)
    cost_usd: Optional[float] = Field(None, ge=0.0)
    duration_ms: Optional[float] = Field(None, ge=0.0)
    status: Literal["success", "failure", "error"] = "success"
    started_at: Optional[datetime] = None


class AgentStepResponse(BaseModel):
    step_id: int
    task_id: int
    project_id: int
    step_type: str
    step_name: Optional[str]
    input_tokens: Optional[int]
    output_tokens: Optional[int]
    cost_usd: Optional[float]
    duration_ms: Optional[float]
    status: str
    started_at: datetime

    model_config = {"from_attributes": True}


class AgentConfigUpdate(BaseModel):
    drift_window_size: int = Field(100, ge=10, le=10000)
    psi_warning: float = Field(0.10, ge=0.0, le=1.0)
    psi_alert: float = Field(0.25, ge=0.0, le=2.0)
    success_rate_warning: float = Field(90.0, ge=0.0, le=100.0)
    success_rate_alert: float = Field(75.0, ge=0.0, le=100.0)
    quality_warning: Optional[float] = None
    quality_alert: Optional[float] = None


class AgentConfigResponse(AgentConfigUpdate):
    project_id: int
    updated_at: datetime

    model_config = {"from_attributes": True}


class BulkDeleteTasksRequest(BaseModel):
    task_ids: list[int]


class InferenceLogResponse(BaseModel):
    log_id: int
    project_id: int
    batch_log_id: Optional[str]
    request_timestamp: datetime
    request_id: Optional[str]
    model_id: Optional[int]
    prediction_values: float
    actual_values: Optional[float]
    response_time_ms: float
    is_error: bool

    model_config = {"from_attributes": True}
