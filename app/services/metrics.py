from __future__ import annotations

from collections import defaultdict
from datetime import datetime, timedelta

import numpy as np
from sqlalchemy.orm import Session

from ..formatting import format_duration
from ..metrics_catalog import compute_metric, resolve_metric
from ..models import InferenceLog
from ..settings import settings
from ..logging_utils import log_call


@log_call
def _is_classification(task_type: str) -> bool:
    return task_type != "regression"


@log_call
def _batch_latencies_ms(rows) -> list[float]:
    """(batch_log_id, request_timestamp, updated_at) の反復から、
    batch_log_id 単位の応答時間(ms)のリストを返す。

    応答時間(batch) = ( max(updated_at) − min(request_timestamp) ) × 1000
    """
    span: dict = {}
    for batch_id, req_ts, upd in rows:
        if req_ts is None or upd is None:
            continue
        s = span.get(batch_id)
        if s is None:
            span[batch_id] = [req_ts, upd]
        else:
            if req_ts < s[0]:
                s[0] = req_ts
            if upd > s[1]:
                s[1] = upd
    return [(s[1] - s[0]).total_seconds() * 1000.0 for s in span.values()]


@log_call
def _dku_batch_latencies_ms(df) -> list[float]:
    """Dataiku DataFrame から batch_log_id 単位の応答時間(ms)のリストを返す。"""
    if df.empty or "batch_log_id" not in df.columns or "updated_at" not in df.columns:
        return []
    g = df.groupby("batch_log_id").agg(
        start=("request_timestamp", "min"), end=("updated_at", "max")
    )
    return ((g["end"] - g["start"]).dt.total_seconds() * 1000.0).tolist()


# ── SQLite 実装 ──────────────────────────────────────────────────────────────

@log_call
def get_summary(db: Session, project_id: str, hours: int = 24) -> dict:
    if settings.is_dataiku:
        return _dku_get_summary(project_id, hours)

    since = datetime.utcnow() - timedelta(hours=hours)
    logs = (
        db.query(InferenceLog)
        .filter(InferenceLog.project_id == project_id, InferenceLog.request_timestamp >= since)
        .all()
    )

    empty = {
        "total_requests": 0,
        "error_count": 0,
        "error_rate": 0.0,
        "avg_latency_ms": 0.0,
        "p50_latency_ms": 0.0,
        "p95_latency_ms": 0.0,
        "p99_latency_ms": 0.0,
    }
    if not logs:
        return empty

    total = len(logs)
    errors = sum(1 for l in logs if l.is_error)
    latencies = sorted(_batch_latencies_ms(
        (l.batch_log_id, l.request_timestamp, l.updated_at) for l in logs
    ))
    n_batches = len(latencies)

    def pct(data: list[float], p: float) -> float:
        idx = min(int(len(data) * p / 100), len(data) - 1)
        return data[idx]

    return {
        "total_requests": total,
        "error_count": errors,
        "error_rate": round(errors / total * 100, 2),
        "avg_latency_ms": round(sum(latencies) / n_batches, 2) if n_batches else 0.0,
        "p50_latency_ms": round(pct(latencies, 50), 2) if n_batches else 0.0,
        "p95_latency_ms": round(pct(latencies, 95), 2) if n_batches else 0.0,
        "p99_latency_ms": round(pct(latencies, 99), 2) if n_batches else 0.0,
    }


@log_call
def get_latency_distribution(db: Session, project_id: str, hours: int = 24) -> dict:
    if settings.is_dataiku:
        return _dku_get_latency_distribution(project_id, hours)

    since = datetime.utcnow() - timedelta(hours=hours)
    rows = (
        db.query(InferenceLog.batch_log_id, InferenceLog.request_timestamp, InferenceLog.updated_at)
        .filter(
            InferenceLog.project_id == project_id,
            InferenceLog.request_timestamp >= since,
            InferenceLog.is_error == False,  # noqa: E712
        )
        .all()
    )
    latencies = _batch_latencies_ms(rows)
    if not latencies:
        return {"labels": [], "counts": []}

    arr = np.array(latencies)
    counts, edges = np.histogram(arr, bins=20)
    labels = [f"{format_duration(edges[i])}–{format_duration(edges[i+1])}" for i in range(len(edges) - 1)]
    return {"labels": labels, "counts": counts.tolist()}


@log_call
def get_latest_accuracy(
    db: Session, project_id: str, hours: int = 168, task_type: str = "binary",
    metric_name: str | None = None, threshold: float = 0.5,
) -> dict:
    if settings.is_dataiku:
        return _dku_get_latest_accuracy(project_id, hours, task_type, metric_name, threshold)

    metric = resolve_metric(metric_name, task_type)
    since = datetime.utcnow() - timedelta(hours=hours)
    logs = (
        db.query(InferenceLog)
        .filter(
            InferenceLog.project_id == project_id,
            InferenceLog.request_timestamp >= since,
            InferenceLog.actual_values != None,  # noqa: E711
            InferenceLog.is_error == False,  # noqa: E712
        )
        .all()
    )
    if not logs:
        return {"value": None, "metric_name": metric, "sample_count": 0}

    y_true = np.array([l.actual_value for l in logs], dtype=float)
    y_pred = np.array([l.prediction_value for l in logs], dtype=float)
    return {
        "value": compute_metric(metric, y_true, y_pred, threshold),
        "metric_name": metric,
        "sample_count": len(logs),
    }


@log_call
def get_accuracy_over_time(
    db: Session, project_id: str, hours: int = 168, task_type: str = "binary",
    metric_name: str | None = None, threshold: float = 0.5,
) -> dict:
    if settings.is_dataiku:
        return _dku_get_accuracy_over_time(project_id, hours, task_type, metric_name, threshold)

    metric = resolve_metric(metric_name, task_type)
    since = datetime.utcnow() - timedelta(hours=hours)
    logs = (
        db.query(InferenceLog)
        .filter(
            InferenceLog.project_id == project_id,
            InferenceLog.request_timestamp >= since,
            InferenceLog.actual_values != None,  # noqa: E711
        )
        .order_by(InferenceLog.request_timestamp)
        .all()
    )

    if not logs:
        return {"labels": [], "data": [], "metric_name": metric}

    daily: dict[str, list] = defaultdict(list)
    for log in logs:
        daily[log.request_timestamp.strftime("%Y-%m-%d")].append(log)

    labels = sorted(daily.keys())
    values: list[float | None] = []
    for day in labels:
        day_logs = daily[day]
        y_true = np.array([l.actual_value for l in day_logs], dtype=float)
        y_pred = np.array([l.prediction_value for l in day_logs], dtype=float)
        values.append(compute_metric(metric, y_true, y_pred, threshold))

    return {"labels": labels, "data": values, "metric_name": metric}


# ── Dataiku 実装 ─────────────────────────────────────────────────────────────

@log_call
def _dku_get_summary(project_id: str, hours: int) -> dict:
    from ..dataiku_client import get_inference_logs_df

    since = datetime.utcnow() - timedelta(hours=hours)
    df = get_inference_logs_df(project_id, since)

    empty = {
        "total_requests": 0, "error_count": 0, "error_rate": 0.0,
        "avg_latency_ms": 0.0, "p50_latency_ms": 0.0,
        "p95_latency_ms": 0.0, "p99_latency_ms": 0.0,
    }
    if df.empty:
        return empty

    total = len(df)
    errors = int(df["is_error"].sum())
    latencies = sorted(_dku_batch_latencies_ms(df))
    n_batches = len(latencies)

    def pct(data: list, p: float) -> float:
        if not data:
            return 0.0
        idx = min(int(len(data) * p / 100), len(data) - 1)
        return float(data[idx])

    return {
        "total_requests": total,
        "error_count": errors,
        "error_rate": round(errors / total * 100, 2),
        "avg_latency_ms": round(sum(latencies) / n_batches, 2) if n_batches else 0.0,
        "p50_latency_ms": round(pct(latencies, 50), 2),
        "p95_latency_ms": round(pct(latencies, 95), 2),
        "p99_latency_ms": round(pct(latencies, 99), 2),
    }


@log_call
def _dku_get_latency_distribution(project_id: str, hours: int) -> dict:
    from ..dataiku_client import get_inference_logs_df

    since = datetime.utcnow() - timedelta(hours=hours)
    df = get_inference_logs_df(project_id, since)
    df = df[~df["is_error"]]
    latencies = _dku_batch_latencies_ms(df)
    if not latencies:
        return {"labels": [], "counts": []}

    arr = np.array(latencies)
    counts, edges = np.histogram(arr, bins=20)
    labels = [f"{format_duration(edges[i])}–{format_duration(edges[i+1])}" for i in range(len(edges) - 1)]
    return {"labels": labels, "counts": counts.tolist()}


@log_call
def _dku_get_latest_accuracy(project_id: str, hours: int, task_type: str, metric_name: str | None = None, threshold: float = 0.5) -> dict:
    from ..dataiku_client import get_inference_logs_df

    metric = resolve_metric(metric_name, task_type)
    since = datetime.utcnow() - timedelta(hours=hours)
    df = get_inference_logs_df(project_id, since)

    df = df[df["actual_values"].notna() & ~df["is_error"]]
    if df.empty:
        return {"value": None, "metric_name": metric, "sample_count": 0}

    y_true = df["actual_values"].to_numpy(dtype=float)
    y_pred = df["prediction_values"].to_numpy(dtype=float)
    return {
        "value": compute_metric(metric, y_true, y_pred, threshold),
        "metric_name": metric,
        "sample_count": len(df),
    }


@log_call
def _dku_get_accuracy_over_time(project_id: str, hours: int, task_type: str, metric_name: str | None = None, threshold: float = 0.5) -> dict:
    from ..dataiku_client import get_inference_logs_df

    metric = resolve_metric(metric_name, task_type)
    since = datetime.utcnow() - timedelta(hours=hours)
    df = get_inference_logs_df(project_id, since)

    df = df[df["actual_values"].notna()].sort_values("request_timestamp")
    if df.empty:
        return {"labels": [], "data": [], "metric_name": metric}

    df["_day"] = df["request_timestamp"].dt.strftime("%Y-%m-%d")
    labels = sorted(df["_day"].unique())
    values: list[float | None] = []

    for day in labels:
        group = df[df["_day"] == day]
        y_true = group["actual_values"].to_numpy(dtype=float)
        y_pred = group["prediction_values"].to_numpy(dtype=float)
        values.append(compute_metric(metric, y_true, y_pred, threshold))

    return {"labels": labels, "data": values, "metric_name": metric}
