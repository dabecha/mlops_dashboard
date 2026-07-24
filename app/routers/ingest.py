from __future__ import annotations

import json
from datetime import datetime

from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.orm import Session

from ..database import get_db
from ..models import DeployedModel, InferenceLog, Project, ProjectConfig, ReferenceLog
from ..schemas import (
    BulkDeleteLogsRequest,
    DeployedModelCreate,
    DeployedModelResponse,
    InferenceLogCreate,
    InferenceLogResponse,
    ProjectConfigResponse,
    ProjectConfigUpdate,
    ProjectCreate,
    ProjectResponse,
    ReferenceLogCreate,
    ReferenceLogResponse,
)
from ..services.config import DEFAULTS
from ..settings import settings
from ..logging_utils import log_call

router = APIRouter(prefix="/api", tags=["ingest"])


@router.post("/projects", response_model=ProjectResponse, status_code=201)
@log_call
def create_project(body: ProjectCreate, db: Session = Depends(get_db)):
    if db.query(Project).filter(Project.project_name == body.project_name).first():
        raise HTTPException(status_code=400, detail="同名のプロジェクトが既に存在します")
    project = Project(**body.model_dump())
    db.add(project)
    db.commit()
    db.refresh(project)
    return project


@router.get("/projects", response_model=list[ProjectResponse])
@log_call
def list_projects(db: Session = Depends(get_db)):
    return db.query(Project).order_by(Project.created_at).all()


@router.delete("/projects/{project_id}", status_code=204)
@log_call
def delete_project(project_id: str, db: Session = Depends(get_db)):
    if settings.is_dataiku:
        from ..dataiku_client import delete_project as dku_delete_project
        if not dku_delete_project(project_id):
            raise HTTPException(status_code=404, detail="プロジェクトが見つかりません")
        return
    project = db.get(Project, project_id)
    if not project:
        raise HTTPException(status_code=404, detail="プロジェクトが見つかりません")
    db.delete(project)
    db.commit()


@router.post("/projects/{project_id}/models", response_model=DeployedModelResponse, status_code=201)
@log_call
def register_deployed_model(
    project_id: str,
    body: DeployedModelCreate,
    db: Session = Depends(get_db),
):
    if not db.get(Project, project_id):
        raise HTTPException(status_code=404, detail="プロジェクトが見つかりません")
    model = DeployedModel(
        project_id=project_id,
        model_version=body.model_version,
        feature_dtypes=json.dumps(body.feature_dtypes) if body.feature_dtypes else None,
        feature_importance=json.dumps(body.feature_importance) if body.feature_importance else None,
        is_activate=body.is_activate,
    )
    db.add(model)
    db.commit()
    db.refresh(model)
    return model


@router.post("/projects/{project_id}/reference-logs", response_model=ReferenceLogResponse, status_code=201)
@log_call
def register_reference_log(
    project_id: str,
    body: ReferenceLogCreate,
    db: Session = Depends(get_db),
):
    if not db.get(Project, project_id):
        raise HTTPException(status_code=404, detail="プロジェクトが見つかりません")
    ref_log = ReferenceLog(
        project_id=project_id,
        model_id=body.model_id,
        feature_values=json.dumps(body.feature_values) if body.feature_values else None,
        actual_values=body.actual_values,
    )
    db.add(ref_log)
    db.commit()
    db.refresh(ref_log)
    return ref_log


@log_call
def _check_project_exists(db: Session, project_id: str) -> bool:
    """モードに応じてプロジェクト存在確認を行う。"""
    if settings.is_dataiku:
        from ..dataiku_client import get_project_by_id
        return get_project_by_id(project_id) is not None
    return db.get(Project, project_id) is not None


@router.get("/projects/{project_id}/config", response_model=ProjectConfigResponse)
@log_call
def get_config(project_id: str, db: Session = Depends(get_db)):
    if not _check_project_exists(db, project_id):
        raise HTTPException(status_code=404, detail="プロジェクトが見つかりません")
    cfg = db.query(ProjectConfig).filter(ProjectConfig.project_id == project_id).first()
    if cfg:
        return cfg
    return ProjectConfigResponse(project_id=project_id, updated_at=datetime.utcnow(), **DEFAULTS)


@router.put("/projects/{project_id}/config", response_model=ProjectConfigResponse)
@log_call
def upsert_config(project_id: str, body: ProjectConfigUpdate, db: Session = Depends(get_db)):
    if not _check_project_exists(db, project_id):
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


@router.delete("/logs/{log_id}", status_code=204)
@log_call
def delete_log(
    log_id: str,
    project_id: str | None = Query(None),
    db: Session = Depends(get_db),
):
    if settings.is_dataiku:
        if project_id is None:
            raise HTTPException(status_code=400, detail="dataiku モードでは project_id が必要です")
        from ..dataiku_client import delete_inference_logs
        if delete_inference_logs(project_id, [log_id]) == 0:
            raise HTTPException(status_code=404, detail="ログが見つかりません")
        return
    log = db.get(InferenceLog, log_id)
    if not log:
        raise HTTPException(status_code=404, detail="ログが見つかりません")
    db.delete(log)
    db.commit()


@router.post("/logs/bulk-delete", status_code=200)
@log_call
def bulk_delete_logs(body: BulkDeleteLogsRequest, db: Session = Depends(get_db)):
    if settings.is_dataiku:
        if body.project_id is None:
            raise HTTPException(status_code=400, detail="dataiku モードでは project_id が必要です")
        from ..dataiku_client import delete_inference_logs
        count = delete_inference_logs(body.project_id, body.log_ids)
        return {"deleted": count}
    count = (
        db.query(InferenceLog)
        .filter(InferenceLog.log_id.in_(body.log_ids))
        .delete(synchronize_session=False)
    )
    db.commit()
    return {"deleted": count}


@router.post("/infer", response_model=InferenceLogResponse, status_code=201)
@log_call
def log_inference(body: InferenceLogCreate, db: Session = Depends(get_db)):
    project = db.query(Project).filter(Project.project_name == body.project_name).first()
    if not project:
        raise HTTPException(
            status_code=404, detail=f"プロジェクト '{body.project_name}' が見つかりません"
        )

    log = InferenceLog(
        project_id=project.project_id,
        batch_log_id=body.batch_log_id,
        request_timestamp=body.request_timestamp or datetime.utcnow(),
        request_id=body.request_id,
        model_id=body.model_id,
        prediction_values=body.prediction_values,
        actual_values=body.actual_values,
        response_time_ms=body.response_time_ms,
        is_error=body.is_error,
        feature_values=json.dumps(body.feature_values) if body.feature_values else None,
        feature_dtypes=json.dumps(body.feature_dtypes) if body.feature_dtypes else None,
    )
    db.add(log)
    db.commit()
    db.refresh(log)
    return log
