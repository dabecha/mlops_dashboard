from __future__ import annotations

import os

from fastapi import APIRouter, Depends, Query, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from sqlalchemy.orm import Session

from ..database import get_db
from ..models import Project
from ..services import drift as drift_svc
from ..services import metrics as metrics_svc

router = APIRouter(tags=["ui"])

_TEMPLATE_DIR = os.path.join(os.path.dirname(__file__), "..", "templates")
templates = Jinja2Templates(directory=_TEMPLATE_DIR)


@router.get("/", response_class=HTMLResponse)
async def index(
    request: Request,
    project_id: int | None = Query(None),
    db: Session = Depends(get_db),
):
    projects = db.query(Project).order_by(Project.created_at).all()
    # サマリーページからの直リンク対応: project_id が指定されていれば先頭に移動
    initial_project_id = project_id or (projects[0].id if projects else None)
    return templates.TemplateResponse(
        request, "index.html", {"projects": projects, "initial_project_id": initial_project_id}
    )


@router.get("/summary", response_class=HTMLResponse)
async def summary_page(request: Request, db: Session = Depends(get_db)):
    projects = db.query(Project).order_by(Project.created_at).all()
    return templates.TemplateResponse(request, "summary.html", {"projects": projects})


@router.get("/ui/summary-panels", response_class=HTMLResponse)
async def summary_panels(
    request: Request,
    hours: int = Query(24),
    drift_window: int = Query(100),
    db: Session = Depends(get_db),
):
    projects = db.query(Project).order_by(Project.created_at).all()
    rows = []
    for p in projects:
        summary = metrics_svc.get_summary(db, p.id, hours)
        accuracy = metrics_svc.get_latest_accuracy(db, p.id, hours=168, task_type=p.task_type)
        drift = drift_svc.detect_drift(db, p.id, drift_window)
        rows.append({"project": p, "summary": summary, "accuracy": accuracy, "drift": drift})

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
    drift_window: int = Query(100),
    db: Session = Depends(get_db),
):
    project = db.get(Project, project_id)
    if not project:
        return HTMLResponse("<p class='text-gray-400 text-center py-8'>プロジェクトが見つかりません</p>")

    summary = metrics_svc.get_summary(db, project_id, hours)
    latency = metrics_svc.get_latency_distribution(db, project_id, hours)
    accuracy = metrics_svc.get_accuracy_over_time(db, project_id, days, project.task_type)
    drift = drift_svc.detect_drift(db, project_id, drift_window)

    return templates.TemplateResponse(
        request,
        "partials/all_panels.html",
        {
            "project": project,
            "summary": summary,
            "latency": latency,
            "accuracy": accuracy,
            "drift": drift,
            "hours": hours,
            "days": days,
        },
    )
