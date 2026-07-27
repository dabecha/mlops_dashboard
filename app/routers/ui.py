from __future__ import annotations

import logging
import os
from datetime import datetime as dt

from fastapi import APIRouter, Depends, Query, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from sqlalchemy.orm import Session

from ..database import get_db
from ..exceptions import AppError, NotFoundError
from ..formatting import format_duration, period_label
from ..metrics_catalog import is_higher_better, metrics_for_task
from ..models import InferenceLog, Project
from ..services import config as config_svc
from ..services import drift as drift_svc
from ..services import metrics as metrics_svc
from ..settings import settings
from ..logging_utils import log_call

_LOG_PAGE_SIZE = 50

logger = logging.getLogger(__name__)
router = APIRouter(tags=["ui"])

_TEMPLATE_DIR = os.path.join(os.path.dirname(__file__), "..", "templates")
templates = Jinja2Templates(directory=_TEMPLATE_DIR)
templates.env.filters["duration"] = format_duration
templates.env.globals["metrics_for_task"] = metrics_for_task
templates.env.globals["metric_higher_better"] = is_higher_better
templates.env.globals["period_label"] = period_label


@log_call
def _parse_iso_dt(value: str | None) -> dt | None:
    """ISO 形式の日時文字列を datetime に変換する。無効・空なら None を返す。"""
    if not value:
        return None
    try:
        return dt.fromisoformat(value)
    except ValueError:
        return None


@log_call
def _get_projects(db: Session) -> list:
    """モードに応じてプロジェクト一覧を取得する。"""
    if settings.is_dataiku:
        from ..dataiku_client import get_projects
        return get_projects()
    logger.info("_get_projects: local_dev モードで DB からプロジェクト一覧を取得")
    projects = db.query(Project).order_by(Project.created_at).all()
    logger.info("_get_projects: %d 件取得", len(projects))
    return projects


@log_call
def _get_project(db: Session, project_id: str):
    """モードに応じて単一プロジェクトを取得する。"""
    if settings.is_dataiku:
        from ..dataiku_client import get_project_by_id
        return get_project_by_id(project_id)
    return db.get(Project, project_id)


@router.get("/detail", response_class=HTMLResponse)
@log_call
async def index(
    request: Request,
    project_id: str | None = Query(None),
    db: Session = Depends(get_db),
):
    projects = _get_projects(db)
    initial_project_id = project_id or (projects[0].project_id if projects else None)
    return templates.TemplateResponse(
        request, "index.html", {"projects": projects, "initial_project_id": initial_project_id}
    )


@router.get("/", response_class=HTMLResponse)
@router.get("/summary", response_class=HTMLResponse)
@log_call
async def summary_page(request: Request, db: Session = Depends(get_db)):
    projects = _get_projects(db)
    return templates.TemplateResponse(request, "summary.html", {"projects": projects})


@router.get("/ui/summary-panels", response_class=HTMLResponse)
@log_call
async def summary_panels(
    request: Request,
    hours: int = Query(24),
    db: Session = Depends(get_db),
):
    projects = _get_projects(db)
    rows = []
    for p in projects:
        # プロジェクト単位で例外を隔離し、不具合のあるプロジェクトだけを
        # エラー表示、正常なプロジェクトは通常表示する。
        try:
            if settings.is_dataiku:
                from ..dataiku_client import check_project_datasets
                check_project_datasets(p.project_id)
            cfg = config_svc.get_config(db, p.project_id)
            summary = metrics_svc.get_summary(db, p.project_id, hours)
            accuracy = metrics_svc.get_latest_accuracy(
                db, p.project_id, hours=hours, task_type=p.task_type,
                metric_name=cfg["metric_name"], threshold=cfg["classification_threshold"],
            )
            drift = drift_svc.detect_drift(
                db, p.project_id,
                window_size=cfg["drift_window_size"],
                psi_warning=cfg["psi_warning"],
                psi_alert=cfg["psi_alert"],
                ks_alpha=cfg["ks_alpha"],
            )
            rows.append({
                "project": p, "summary": summary, "accuracy": accuracy,
                "drift": drift, "config": cfg, "error": None,
            })
        except AppError as exc:
            logger.warning("サマリー: プロジェクト %s の取得に失敗: %s", p.project_id, exc.log_message)
            rows.append({"project": p, "error": exc.user_message})
        except Exception as exc:
            logger.error("サマリー: プロジェクト %s で想定外エラー", p.project_id, exc_info=exc)
            rows.append({"project": p, "error": "データ取得中にエラーが発生しました"})

    return templates.TemplateResponse(
        request,
        "partials/summary_panels.html",
        {"rows": rows, "hours": hours},
    )


@router.get("/ui/panels", response_class=HTMLResponse)
@log_call
async def all_panels(
    request: Request,
    project_id: str = Query(...),
    hours: int = Query(24),
    db: Session = Depends(get_db),
):
    project = _get_project(db, project_id)
    if not project:
        raise NotFoundError("プロジェクトが見つかりません")

    if settings.is_dataiku:
        from ..dataiku_client import check_project_datasets
        check_project_datasets(project_id)

    cfg = config_svc.get_config(db, project_id)
    summary = metrics_svc.get_summary(db, project_id, hours)
    latency = metrics_svc.get_latency_distribution(db, project_id, hours)
    accuracy = metrics_svc.get_accuracy_over_time(
        db, project_id, hours, project.task_type,
        metric_name=cfg["metric_name"], threshold=cfg["classification_threshold"],
    )
    drift = drift_svc.detect_drift(
        db, project_id,
        window_size=cfg["drift_window_size"],
        psi_warning=cfg["psi_warning"],
        psi_alert=cfg["psi_alert"],
        ks_alpha=cfg["ks_alpha"],
    )
    # PSI 推移のみ選択期間に連動（判定は drift_window_size 基準で期間と独立）
    drift_trend = drift_svc.get_drift_over_time(
        db, project_id, hours=hours,
        window_size=cfg["drift_window_size"],
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
            "drift_trend": drift_trend,
            "feature_drift": feature_drift,
            "config": cfg,
            "hours": hours,
        },
    )


@router.get("/manage", response_class=HTMLResponse)
@log_call
async def manage_page(request: Request, db: Session = Depends(get_db)):
    projects = _get_projects(db)
    return templates.TemplateResponse(
        request, "manage.html", {
            "projects": projects,
            "app_mode": settings.app_mode.value,
        }
    )


@router.get("/ui/project-list", response_class=HTMLResponse)
@log_call
async def project_list(request: Request, db: Session = Depends(get_db)):
    projects = _get_projects(db)
    return templates.TemplateResponse(
        request, "partials/project_list.html", {"projects": projects}
    )


@router.get("/ui/log-list", response_class=HTMLResponse)
@log_call
async def log_list(
    request: Request,
    project_id: str = Query(...),
    from_dt: str | None = Query(None),
    to_dt: str | None = Query(None),
    is_error: str | None = Query(None),
    page: int = Query(1),
    db: Session = Depends(get_db),
):
    parsed_from = _parse_iso_dt(from_dt)
    parsed_to = _parse_iso_dt(to_dt)
    error_filter = True if is_error == "true" else (False if is_error == "false" else None)

    if settings.is_dataiku:
        from ..dataiku_client import get_inference_logs_page
        logs, total = get_inference_logs_page(
            project_id,
            from_dt=parsed_from,
            to_dt=parsed_to,
            is_error=error_filter,
            page=page,
            page_size=_LOG_PAGE_SIZE,
        )
    else:
        q = db.query(InferenceLog).filter(InferenceLog.project_id == project_id)
        if parsed_from is not None:
            q = q.filter(InferenceLog.request_timestamp >= parsed_from)
        if parsed_to is not None:
            q = q.filter(InferenceLog.request_timestamp <= parsed_to)
        if error_filter is not None:
            q = q.filter(InferenceLog.is_error == error_filter)

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
@log_call
async def config_modal(
    request: Request,
    project_id: str,
    db: Session = Depends(get_db),
):
    project = _get_project(db, project_id)
    if not project:
        raise NotFoundError("プロジェクトが見つかりません")
    cfg = config_svc.get_config(db, project_id)
    return templates.TemplateResponse(
        request,
        "partials/config_modal.html",
        {"project": project, "config": cfg},
    )
