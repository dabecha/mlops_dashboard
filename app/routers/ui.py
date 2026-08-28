from __future__ import annotations

import calendar
import logging
import os
from datetime import date, datetime as dt, time, timedelta

from fastapi import APIRouter, Depends, Query, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from sqlalchemy.orm import Session

from ..database import get_db
from ..exceptions import AppError, NotFoundError, ValidationError
from ..formatting import format_duration
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
# ダッシュボード自動更新の設定（.env の AUTO_REFRESH_* ）。
# 間隔は管理ページから変更でき、以下は未設定ブラウザでの初期値として使われる。
templates.env.globals["auto_refresh_enabled"] = settings.auto_refresh_enabled
templates.env.globals["auto_refresh_summary_seconds"] = settings.auto_refresh_summary_seconds
templates.env.globals["auto_refresh_detail_seconds"] = settings.auto_refresh_detail_seconds


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
def _one_month_ago(d: date) -> date:
    """1カ月前の同日を返す（月末を超える場合は前月末日に丸める）。"""
    y, m = (d.year - 1, 12) if d.month == 1 else (d.year, d.month - 1)
    last_day = calendar.monthrange(y, m)[1]
    return date(y, m, min(d.day, last_day))


@log_call
def _resolve_period(from_date: str | None, to_date: str | None) -> tuple[date, date]:
    """From / To の日付文字列（YYYY-MM-DD）を解決する。

    未指定・無効な場合のデフォルトは To = 今日、From = To の1カ月前。
    From > To の場合は ValidationError を送出する。
    """
    def parse(value: str | None) -> date | None:
        if not value:
            return None
        try:
            return date.fromisoformat(value)
        except ValueError:
            return None

    to_d = parse(to_date) or date.today()
    from_d = parse(from_date) or _one_month_ago(to_d)
    if from_d > to_d:
        raise ValidationError("期間の From には To 以前の日付を指定してください")
    return from_d, to_d


@log_call
def _period_datetimes(from_d: date, to_d: date) -> tuple[dt, dt]:
    """日付の組を集計用の datetime 範囲（From 0時 以上、To 翌日0時 未満）に変換する。"""
    return dt.combine(from_d, time.min), dt.combine(to_d + timedelta(days=1), time.min)


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
    default_from, default_to = _resolve_period(None, None)
    return templates.TemplateResponse(
        request, "index.html", {
            "projects": projects,
            "initial_project_id": initial_project_id,
            "default_from": default_from.isoformat(),
            "default_to": default_to.isoformat(),
        }
    )


@router.get("/", response_class=HTMLResponse)
@router.get("/summary", response_class=HTMLResponse)
@log_call
async def summary_page(request: Request, db: Session = Depends(get_db)):
    projects = _get_projects(db)
    default_from, default_to = _resolve_period(None, None)
    return templates.TemplateResponse(
        request, "summary.html", {
            "projects": projects,
            "default_from": default_from.isoformat(),
            "default_to": default_to.isoformat(),
        }
    )


@router.get("/ui/summary-panels", response_class=HTMLResponse)
@log_call
async def summary_panels(
    request: Request,
    from_date: str | None = Query(None),
    to_date: str | None = Query(None),
    db: Session = Depends(get_db),
):
    from_d, to_d = _resolve_period(from_date, to_date)
    from_dt, to_dt = _period_datetimes(from_d, to_d)
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
            summary = metrics_svc.get_summary(db, p.project_id, from_dt, to_dt)
            accuracy = metrics_svc.get_latest_accuracy(
                db, p.project_id, from_dt, to_dt, task_type=p.task_type,
                metric_name=cfg["metric_name"], threshold=cfg["classification_threshold"],
            )
            # ターゲットドリフト（実績値）を最重要指標として先に計算
            target_drift = drift_svc.detect_target_drift(
                db, p.project_id,
                window_size=cfg["drift_window_size"],
                psi_warning=cfg["psi_warning"],
                psi_alert=cfg["psi_alert"],
                ks_alpha=cfg["ks_alpha"],
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
                "target_drift": target_drift, "drift": drift,
                "config": cfg, "error": None,
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
        {"rows": rows, "from_date": from_d.isoformat(), "to_date": to_d.isoformat()},
    )


@router.get("/ui/panels", response_class=HTMLResponse)
@log_call
async def all_panels(
    request: Request,
    project_id: str = Query(...),
    from_date: str | None = Query(None),
    to_date: str | None = Query(None),
    db: Session = Depends(get_db),
):
    from_d, to_d = _resolve_period(from_date, to_date)
    from_dt, to_dt = _period_datetimes(from_d, to_d)

    project = _get_project(db, project_id)
    if not project:
        raise NotFoundError("プロジェクトが見つかりません")

    if settings.is_dataiku:
        from ..dataiku_client import check_project_datasets
        check_project_datasets(project_id)

    cfg = config_svc.get_config(db, project_id)
    summary = metrics_svc.get_summary(db, project_id, from_dt, to_dt)
    latency = metrics_svc.get_latency_distribution(db, project_id, from_dt, to_dt)
    accuracy = metrics_svc.get_accuracy_over_time(
        db, project_id, from_dt, to_dt, project.task_type,
        metric_name=cfg["metric_name"], threshold=cfg["classification_threshold"],
    )
    # ターゲットドリフト（実績値）を最重要指標として先に計算
    target_drift = drift_svc.detect_target_drift(
        db, project_id,
        window_size=cfg["drift_window_size"],
        psi_warning=cfg["psi_warning"],
        psi_alert=cfg["psi_alert"],
        ks_alpha=cfg["ks_alpha"],
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
        db, project_id, from_dt, to_dt,
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
            "target_drift": target_drift,
            "drift": drift,
            "drift_trend": drift_trend,
            "feature_drift": feature_drift,
            "config": cfg,
            "from_date": from_d.isoformat(),
            "to_date": to_d.isoformat(),
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
