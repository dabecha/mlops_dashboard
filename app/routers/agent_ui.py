from __future__ import annotations

import os
from datetime import datetime as dt

from fastapi import APIRouter, Depends, Query, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from sqlalchemy.orm import Session

from ..database import get_db
from ..models import AgentProject, AgentTask
from ..services import agent_config as agent_config_svc
from ..services import agent_drift as agent_drift_svc
from ..services import agent_metrics as agent_metrics_svc

router = APIRouter(tags=["agent-ui"])

_TEMPLATE_DIR = os.path.join(os.path.dirname(__file__), "..", "templates")
templates = Jinja2Templates(directory=_TEMPLATE_DIR)

_TASK_PAGE_SIZE = 50


@router.get("/agents", response_class=HTMLResponse)
async def agents_page(
    request: Request,
    project_id: int | None = Query(None),
    db: Session = Depends(get_db),
):
    projects = db.query(AgentProject).order_by(AgentProject.created_at).all()
    initial_project_id = project_id or (projects[0].project_id if projects else None)
    return templates.TemplateResponse(
        request,
        "agents.html",
        {"projects": projects, "initial_project_id": initial_project_id},
    )


@router.get("/agents/summary", response_class=HTMLResponse)
async def agents_summary_page(request: Request, db: Session = Depends(get_db)):
    projects = db.query(AgentProject).order_by(AgentProject.created_at).all()
    return templates.TemplateResponse(request, "agents_summary.html", {"projects": projects})


@router.get("/ui/agent-panels", response_class=HTMLResponse)
async def agent_panels(
    request: Request,
    project_id: int = Query(...),
    hours: int = Query(24),
    days: int = Query(7),
    db: Session = Depends(get_db),
):
    project = db.get(AgentProject, project_id)
    if not project:
        return HTMLResponse("<p class='text-gray-400 text-center py-8'>エージェントプロジェクトが見つかりません</p>")

    cfg = agent_config_svc.get_agent_config(db, project_id)
    summary = agent_metrics_svc.get_agent_summary(db, project_id, hours)
    cost_chart = agent_metrics_svc.get_cost_over_time(db, project_id, days)
    quality_chart = agent_metrics_svc.get_quality_over_time(db, project_id, days)
    duration_dist = agent_metrics_svc.get_duration_distribution(db, project_id, hours)
    step_analysis = agent_metrics_svc.get_step_analysis(db, project_id, hours)
    feature_drift = agent_drift_svc.detect_agent_feature_drift(
        db, project_id,
        window_size=cfg["drift_window_size"],
        psi_warning=cfg["psi_warning"],
        psi_alert=cfg["psi_alert"],
    )

    return templates.TemplateResponse(
        request,
        "partials/agent_panels.html",
        {
            "project": project,
            "summary": summary,
            "cost_chart": cost_chart,
            "quality_chart": quality_chart,
            "duration_dist": duration_dist,
            "step_analysis": step_analysis,
            "feature_drift": feature_drift,
            "config": cfg,
            "hours": hours,
            "days": days,
        },
    )


@router.get("/ui/agent-summary-panels", response_class=HTMLResponse)
async def agent_summary_panels(
    request: Request,
    hours: int = Query(24),
    db: Session = Depends(get_db),
):
    projects = db.query(AgentProject).order_by(AgentProject.created_at).all()
    rows = []
    for p in projects:
        cfg = agent_config_svc.get_agent_config(db, p.project_id)
        summary = agent_metrics_svc.get_agent_summary(db, p.project_id, hours)
        quality = agent_metrics_svc.get_quality_over_time(db, p.project_id, days=7)
        drift = agent_drift_svc.detect_agent_feature_drift(
            db, p.project_id,
            window_size=cfg["drift_window_size"],
            psi_warning=cfg["psi_warning"],
            psi_alert=cfg["psi_alert"],
        )
        latest_quality = quality["values"][-1] if quality["values"] else None
        rows.append({
            "project": p,
            "summary": summary,
            "latest_quality": latest_quality,
            "drift": drift,
            "config": cfg,
        })
    return templates.TemplateResponse(
        request,
        "partials/agent_summary_panels.html",
        {"rows": rows, "hours": hours},
    )


@router.get("/ui/agent-config-modal/{project_id}", response_class=HTMLResponse)
async def agent_config_modal(
    request: Request,
    project_id: int,
    db: Session = Depends(get_db),
):
    project = db.get(AgentProject, project_id)
    if not project:
        return HTMLResponse("<p class='text-red-400 p-4'>エージェントプロジェクトが見つかりません</p>")
    cfg = agent_config_svc.get_agent_config(db, project_id)
    return templates.TemplateResponse(
        request,
        "partials/agent_config_modal.html",
        {"project": project, "config": cfg},
    )


@router.get("/ui/agent-project-list", response_class=HTMLResponse)
async def agent_project_list(request: Request, db: Session = Depends(get_db)):
    projects = db.query(AgentProject).order_by(AgentProject.created_at).all()
    return templates.TemplateResponse(
        request, "partials/agent_project_list.html", {"projects": projects}
    )


@router.get("/ui/agent-task-list", response_class=HTMLResponse)
async def agent_task_list(
    request: Request,
    project_id: int = Query(...),
    from_dt: str | None = Query(None),
    to_dt: str | None = Query(None),
    task_name_filter: str | None = Query(None),
    status_filter: str | None = Query(None),
    page: int = Query(1),
    db: Session = Depends(get_db),
):
    q = db.query(AgentTask).filter(AgentTask.project_id == project_id)
    if from_dt:
        try:
            q = q.filter(AgentTask.started_at >= dt.fromisoformat(from_dt))
        except ValueError:
            pass
    if to_dt:
        try:
            q = q.filter(AgentTask.started_at <= dt.fromisoformat(to_dt))
        except ValueError:
            pass
    if task_name_filter:
        q = q.filter(AgentTask.task_name.contains(task_name_filter))
    if status_filter and status_filter in ("success", "failure", "error"):
        q = q.filter(AgentTask.status == status_filter)

    total = q.count()
    tasks = (
        q.order_by(AgentTask.started_at.desc())
        .offset((page - 1) * _TASK_PAGE_SIZE)
        .limit(_TASK_PAGE_SIZE)
        .all()
    )

    return templates.TemplateResponse(
        request,
        "partials/agent_task_list.html",
        {
            "tasks": tasks,
            "total": total,
            "page": page,
            "page_size": _TASK_PAGE_SIZE,
            "project_id": project_id,
        },
    )
