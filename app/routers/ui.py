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
async def index(request: Request, db: Session = Depends(get_db)):
    projects = db.query(Project).order_by(Project.created_at).all()
    return templates.TemplateResponse(request, "index.html", {"projects": projects})


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
