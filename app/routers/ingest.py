from __future__ import annotations

import json
from datetime import datetime

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from ..database import get_db
from ..models import InferenceLog, Project, ProjectConfig
from ..schemas import (
    InferenceLogCreate,
    InferenceLogResponse,
    ProjectConfigResponse,
    ProjectConfigUpdate,
    ProjectCreate,
    ProjectResponse,
)
from ..services.config import DEFAULTS

router = APIRouter(prefix="/api", tags=["ingest"])


@router.post("/projects", response_model=ProjectResponse, status_code=201)
def create_project(body: ProjectCreate, db: Session = Depends(get_db)):
    if db.query(Project).filter(Project.name == body.name).first():
        raise HTTPException(status_code=400, detail="同名のプロジェクトが既に存在します")
    project = Project(**body.model_dump())
    db.add(project)
    db.commit()
    db.refresh(project)
    return project


@router.get("/projects", response_model=list[ProjectResponse])
def list_projects(db: Session = Depends(get_db)):
    return db.query(Project).order_by(Project.created_at).all()


@router.delete("/projects/{project_id}", status_code=204)
def delete_project(project_id: int, db: Session = Depends(get_db)):
    project = db.get(Project, project_id)
    if not project:
        raise HTTPException(status_code=404, detail="プロジェクトが見つかりません")
    db.delete(project)
    db.commit()


@router.get("/projects/{project_id}/config", response_model=ProjectConfigResponse)
def get_config(project_id: int, db: Session = Depends(get_db)):
    if not db.get(Project, project_id):
        raise HTTPException(status_code=404, detail="プロジェクトが見つかりません")
    cfg = db.query(ProjectConfig).filter(ProjectConfig.project_id == project_id).first()
    if cfg:
        return cfg
    # 未設定時はデフォルト値を返す（DB に保存はしない）
    return ProjectConfigResponse(project_id=project_id, updated_at=datetime.utcnow(), **DEFAULTS)


@router.put("/projects/{project_id}/config", response_model=ProjectConfigResponse)
def upsert_config(project_id: int, body: ProjectConfigUpdate, db: Session = Depends(get_db)):
    if not db.get(Project, project_id):
        raise HTTPException(status_code=404, detail="プロジェクトが見つかりません")
    cfg = db.query(ProjectConfig).filter(ProjectConfig.project_id == project_id).first()
    if cfg:
        for k, v in body.model_dump().items():
            setattr(cfg, k, v)
        cfg.updated_at = datetime.utcnow()
    else:
        cfg = ProjectConfig(project_id=project_id, updated_at=datetime.utcnow(), **body.model_dump())
        db.add(cfg)
    db.commit()
    db.refresh(cfg)
    return cfg


@router.post("/infer", response_model=InferenceLogResponse, status_code=201)
def log_inference(body: InferenceLogCreate, db: Session = Depends(get_db)):
    project = db.query(Project).filter(Project.name == body.project_name).first()
    if not project:
        raise HTTPException(
            status_code=404, detail=f"プロジェクト '{body.project_name}' が見つかりません"
        )

    log = InferenceLog(
        project_id=project.id,
        timestamp=body.timestamp or datetime.utcnow(),
        request_id=body.request_id,
        prediction=body.prediction,
        actual_label=body.actual_label,
        confidence=body.confidence,
        response_time_ms=body.response_time_ms,
        is_error=body.is_error,
        error_message=body.error_message,
        feature_values=json.dumps(body.feature_values) if body.feature_values else None,
    )
    db.add(log)
    db.commit()
    db.refresh(log)
    return log
