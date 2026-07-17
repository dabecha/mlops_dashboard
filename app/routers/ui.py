from __future__ import annotations

import os
from datetime import datetime as dt

from fastapi import APIRouter, Depends, Query, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from sqlalchemy.orm import Session

from ..database import get_db
from ..models import AgentProject, InferenceLog, Project
from ..services import config as config_svc
from ..services import drift as drift_svc
from ..services import metrics as metrics_svc
from ..settings import settings

_LOG_PAGE_SIZE = 50

router = APIRouter(tags=["ui"])

_TEMPLATE_DIR = os.path.join(os.path.dirname(__file__), "..", "templates")
templates = Jinja2Templates(directory=_TEMPLATE_DIR)


def _get_projects(db: Session) -> list:
    """モードに応じてプロジェクト一覧を取得する。"""
    if settings.is_dataiku:
        from ..dataiku_client import get_projects
        return get_projects()
    return db.query(Project).order_by(Project.created_at).all()


def _get_project(db: Session, project_id: int):
    """モードに応じて単一プロジェクトを取得する。"""
    if settings.is_dataiku:
        from ..dataiku_client import get_project_by_id
        return get_project_by_id(project_id)
    return db.get(Project, project_id)


@router.get("/", response_class=HTMLResponse)
async def index(
    request: Request,
    project_id: int | None = Query(None),
    db: Session = Depends(get_db),
):
    projects = _get_projects(db)
    initial_project_id = project_id or (projects[0].project_id if projects else None)
    return templates.TemplateResponse(
        request, "index.html", {"projects": projects, "initial_project_id": initial_project_id}
    )


@router.get("/summary", response_class=HTMLResponse)
async def summary_page(request: Request, db: Session = Depends(get_db)):
    projects = _get_projects(db)
    return templates.TemplateResponse(request, "summary.html", {"projects": projects})


@router.get("/ui/summary-panels", response_class=HTMLResponse)
async def summary_panels(
    request: Request,
    hours: int = Query(24),
    db: Session = Depends(get_db),
):
    projects = _get_projects(db)
    rows = []
    for p in projects:
        cfg = config_svc.get_config(db, p.project_id)
        summary = metrics_svc.get_summary(db, p.project_id, hours)
        accuracy = metrics_svc.get_latest_accuracy(db, p.project_id, hours=168, task_type=p.task_type)
        drift = drift_svc.detect_drift(
            db, p.project_id,
            window_size=cfg["drift_window_size"],
            psi_warning=cfg["psi_warning"],
            psi_alert=cfg["psi_alert"],
            ks_alpha=cfg["ks_alpha"],
        )
        rows.append({"project": p, "summary": summary, "accuracy": accuracy, "drift": drift, "config": cfg})

    return templates.TemplateResponse(
        request,
        "partials/summary_panels.html",
        {"rows": rows, "hours": hours},
    )


@router.get("/ui/panels", response_class=HTMLResponse)
async def all_panels(
    request: Request,
    project_id: int = Query(...),
    hours: int = Query(24),
    days: int = Query(7),
    db: Session = Depends(get_db),
):
    project = _get_project(db, project_id)
    if not project:
        return HTMLResponse("<p class='text-gray-400 text-center py-8'>プロジェクトが見つかりません</p>")

    cfg = config_svc.get_config(db, project_id)
    summary = metrics_svc.get_summary(db, project_id, hours)
    latency = metrics_svc.get_latency_distribution(db, project_id, hours)
    accuracy = metrics_svc.get_accuracy_over_time(db, project_id, days, project.task_type)
    drift = drift_svc.detect_drift(
        db, project_id,
        window_size=cfg["drift_window_size"],
        psi_warning=cfg["psi_warning"],
        psi_alert=cfg["psi_alert"],
        ks_alpha=cfg["ks_alpha"],
    )
    feature_drift = drift_svc.detect_feature_drift(
        db, project_id,
        window_size=cfg["drift_window_size"],
        psi_warning=cfg["psi_warning"],
        psi_alert=cfg["psi_alert"],
    )

    return templates.TemplateResponse(
        request,
        "partials/all_panels.html",
        {
            "project": project,
            "summary": summary,
            "latency": latency,
            "accuracy": accuracy,
            "drift": drift,
            "feature_drift": feature_drift,
            "config": cfg,
            "hours": hours,
            "days": days,
        },
    )


@router.get("/manage", response_class=HTMLResponse)
async def manage_page(request: Request, db: Session = Depends(get_db)):
    projects = _get_projects(db)
    agent_projects = db.query(AgentProject).order_by(AgentProject.created_at).all()
    return templates.TemplateResponse(
        request, "manage.html", {
            "projects": projects,
            "agent_projects": agent_projects,
            "app_mode": settings.app_mode.value,
        }
    )


@router.get("/ui/project-list", response_class=HTMLResponse)
async def project_list(request: Request, db: Session = Depends(get_db)):
    projects = _get_projects(db)
    return templates.TemplateResponse(
        request, "partials/project_list.html", {"projects": projects}
    )


@router.get("/ui/log-list", response_class=HTMLResponse)
async def log_list(
    request: Request,
    project_id: int = Query(...),
    from_dt: str | None = Query(None),
    to_dt: str | None = Query(None),
    request_id_filter: str | None = Query(None),
    is_error: str | None = Query(None),
    page: int = Query(1),
    db: Session = Depends(get_db),
):
    q = db.query(InferenceLog).filter(InferenceLog.project_id == project_id)
    if from_dt:
        try:
            q = q.filter(InferenceLog.request_timestamp >= dt.fromisoformat(from_dt))
        except ValueError:
            pass
    if to_dt:
        try:
            q = q.filter(InferenceLog.request_timestamp <= dt.fromisoformat(to_dt))
        except ValueError:
            pass
    if request_id_filter:
        q = q.filter(InferenceLog.request_id.contains(request_id_filter))
    if is_error == "true":
        q = q.filter(InferenceLog.is_error == True)  # noqa: E712
    elif is_error == "false":
        q = q.filter(InferenceLog.is_error == False)  # noqa: E712

    total = q.count()
    logs = (
        q.order_by(InferenceLog.request_timestamp.desc())
        .offset((page - 1) * _LOG_PAGE_SIZE)
        .limit(_LOG_PAGE_SIZE)
        .all()
    )

    return templates.TemplateResponse(
        request,
        "partials/log_list.html",
        {
            "logs": logs,
            "total": total,
            "page": page,
            "page_size": _LOG_PAGE_SIZE,
            "project_id": project_id,
        },
    )


@router.get("/ui/config-modal/{project_id}", response_class=HTMLResponse)
async def config_modal(
    request: Request,
    project_id: int,
    db: Session = Depends(get_db),
):
    project = _get_project(db, project_id)
    if not project:
        return HTMLResponse("<p class='text-red-400 p-4'>プロジェクトが見つかりません</p>")
    cfg = config_svc.get_config(db, project_id)
    return templates.TemplateResponse(
        request,
        "partials/config_modal.html",
        {"project": project, "config": cfg},
    )
