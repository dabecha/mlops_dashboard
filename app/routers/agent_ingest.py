from __future__ import annotations

import json
from datetime import datetime

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from ..database import get_db
from ..models import AgentConfig, AgentProject, AgentRefData, AgentStep, AgentTask
from ..schemas import (
    AgentConfigResponse,
    AgentConfigUpdate,
    AgentProjectCreate,
    AgentProjectResponse,
    AgentRefDataCreate,
    AgentRefDataResponse,
    AgentStepCreate,
    AgentStepResponse,
    AgentTaskCreate,
    AgentTaskResponse,
    BulkDeleteTasksRequest,
)
from ..services.agent_config import AGENT_DEFAULTS

router = APIRouter(prefix="/agent-api", tags=["agent-ingest"])


# ── Projects ──────────────────────────────────────────────────────────────────

@router.post("/projects", response_model=AgentProjectResponse, status_code=201)
def create_agent_project(body: AgentProjectCreate, db: Session = Depends(get_db)):
    if db.query(AgentProject).filter(AgentProject.project_name == body.project_name).first():
        raise HTTPException(status_code=400, detail="同名のエージェントプロジェクトが既に存在します")
    project = AgentProject(**body.model_dump())
    db.add(project)
    db.commit()
    db.refresh(project)
    return project


@router.get("/projects", response_model=list[AgentProjectResponse])
def list_agent_projects(db: Session = Depends(get_db)):
    return db.query(AgentProject).order_by(AgentProject.created_at).all()


@router.delete("/projects/{project_id}", status_code=204)
def delete_agent_project(project_id: int, db: Session = Depends(get_db)):
    project = db.get(AgentProject, project_id)
    if not project:
        raise HTTPException(status_code=404, detail="エージェントプロジェクトが見つかりません")
    db.delete(project)
    db.commit()


# ── Reference Data ────────────────────────────────────────────────────────────

@router.post("/projects/{project_id}/ref-data", response_model=AgentRefDataResponse, status_code=201)
def register_ref_data(project_id: int, body: AgentRefDataCreate, db: Session = Depends(get_db)):
    if not db.get(AgentProject, project_id):
        raise HTTPException(status_code=404, detail="エージェントプロジェクトが見つかりません")
    ref = AgentRefData(
        project_id=project_id,
        feature_values=json.dumps(body.feature_values) if body.feature_values else None,
        feature_importance=json.dumps(body.feature_importance) if body.feature_importance else None,
    )
    db.add(ref)
    db.commit()
    db.refresh(ref)
    return ref


# ── Tasks ─────────────────────────────────────────────────────────────────────

@router.post("/tasks", response_model=AgentTaskResponse, status_code=201)
def log_agent_task(body: AgentTaskCreate, db: Session = Depends(get_db)):
    project = db.query(AgentProject).filter(AgentProject.project_name == body.project_name).first()
    if not project:
        raise HTTPException(status_code=404, detail=f"プロジェクト '{body.project_name}' が見つかりません")
    data = body.model_dump(exclude={"project_name"})
    task = AgentTask(
        project_id=project.project_id,
        session_id=data["session_id"],
        task_name=data["task_name"],
        status=data["status"],
        quality_score=data["quality_score"],
        total_input_tokens=data["total_input_tokens"],
        total_output_tokens=data["total_output_tokens"],
        total_cost_usd=data["total_cost_usd"],
        total_steps=data["total_steps"],
        duration_ms=data["duration_ms"],
        started_at=data["started_at"] or datetime.utcnow(),
        feature_values=json.dumps(data["feature_values"]) if data["feature_values"] else None,
        error_message=data["error_message"],
    )
    db.add(task)
    db.commit()
    db.refresh(task)
    return task


@router.delete("/tasks/{task_id}", status_code=204)
def delete_agent_task(task_id: int, db: Session = Depends(get_db)):
    task = db.get(AgentTask, task_id)
    if not task:
        raise HTTPException(status_code=404, detail="タスクが見つかりません")
    db.delete(task)
    db.commit()


@router.post("/tasks/bulk-delete", status_code=200)
def bulk_delete_agent_tasks(body: BulkDeleteTasksRequest, db: Session = Depends(get_db)):
    count = (
        db.query(AgentTask)
        .filter(AgentTask.task_id.in_(body.task_ids))
        .delete(synchronize_session=False)
    )
    db.commit()
    return {"deleted": count}


# ── Steps ─────────────────────────────────────────────────────────────────────

@router.post("/tasks/{task_id}/steps", response_model=AgentStepResponse, status_code=201)
def log_agent_step(task_id: int, body: AgentStepCreate, db: Session = Depends(get_db)):
    task = db.get(AgentTask, task_id)
    if not task:
        raise HTTPException(status_code=404, detail="タスクが見つかりません")
    step = AgentStep(
        task_id=task_id,
        project_id=task.project_id,
        step_type=body.step_type,
        step_name=body.step_name,
        input_tokens=body.input_tokens,
        output_tokens=body.output_tokens,
        cost_usd=body.cost_usd,
        duration_ms=body.duration_ms,
        status=body.status,
        started_at=body.started_at or datetime.utcnow(),
    )
    db.add(step)
    db.commit()
    db.refresh(step)
    return step


# ── Config ────────────────────────────────────────────────────────────────────

@router.get("/projects/{project_id}/config", response_model=AgentConfigResponse)
def get_agent_config(project_id: int, db: Session = Depends(get_db)):
    if not db.get(AgentProject, project_id):
        raise HTTPException(status_code=404, detail="エージェントプロジェクトが見つかりません")
    cfg = db.query(AgentConfig).filter(AgentConfig.project_id == project_id).first()
    if cfg:
        return cfg
    return AgentConfigResponse(project_id=project_id, updated_at=datetime.utcnow(), **AGENT_DEFAULTS)


@router.put("/projects/{project_id}/config", response_model=AgentConfigResponse)
def upsert_agent_config_route(project_id: int, body: AgentConfigUpdate, db: Session = Depends(get_db)):
    if not db.get(AgentProject, project_id):
        raise HTTPException(status_code=404, detail="エージェントプロジェクトが見つかりません")
    from ..services.agent_config import upsert_agent_config
    cfg = upsert_agent_config(db, project_id, body.model_dump())
    return cfg
