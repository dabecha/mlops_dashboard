from __future__ import annotations

import json
from datetime import datetime

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from ..database import get_db
from ..models import InferenceLog, Project
from ..schemas import InferenceLogCreate, InferenceLogResponse, ProjectCreate, ProjectResponse

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
