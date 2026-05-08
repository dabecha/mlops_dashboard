from __future__ import annotations

from collections import defaultdict
from datetime import datetime, timedelta

import numpy as np
from sqlalchemy.orm import Session

from ..models import InferenceLog


def _is_classification(task_type: str) -> bool:
    return task_type != "regression"


def get_summary(db: Session, project_id: int, hours: int = 24) -> dict:
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
    latencies = sorted(l.response_time_ms for l in logs)

    def pct(data: list[float], p: float) -> float:
        idx = min(int(len(data) * p / 100), len(data) - 1)
        return data[idx]

    return {
        "total_requests": total,
        "error_count": errors,
        "error_rate": round(errors / total * 100, 2),
        "avg_latency_ms": round(sum(latencies) / total, 2),
        "p50_latency_ms": round(pct(latencies, 50), 2),
        "p95_latency_ms": round(pct(latencies, 95), 2),
        "p99_latency_ms": round(pct(latencies, 99), 2),
    }


def get_latency_distribution(db: Session, project_id: int, hours: int = 24) -> dict:
    since = datetime.utcnow() - timedelta(hours=hours)
    rows = (
        db.query(InferenceLog.response_time_ms)
        .filter(
            InferenceLog.project_id == project_id,
            InferenceLog.request_timestamp >= since,
            InferenceLog.is_error == False,  # noqa: E712
        )
        .all()
    )
    latencies = [r[0] for r in rows]
    if not latencies:
        return {"labels": [], "counts": []}

    arr = np.array(latencies)
    counts, edges = np.histogram(arr, bins=20)
    labels = [f"{edges[i]:.0f}–{edges[i+1]:.0f}" for i in range(len(edges) - 1)]
    return {"labels": labels, "counts": counts.tolist()}


def get_latest_accuracy(
    db: Session, project_id: int, hours: int = 168, task_type: str = "binary"
) -> dict:
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
    metric_name = "Accuracy (%)" if _is_classification(task_type) else "MAE"
    if not logs:
        return {"value": None, "metric_name": metric_name, "sample_count": 0}

    if _is_classification(task_type):
        correct = sum(1 for l in logs if (l.prediction_values > 0.5) == bool(l.actual_values))
        return {
            "value": round(correct / len(logs) * 100, 1),
            "metric_name": metric_name,
            "sample_count": len(logs),
        }
    else:
        mae = sum(abs(l.prediction_values - l.actual_values) for l in logs) / len(logs)
        return {
            "value": round(mae, 3),
            "metric_name": metric_name,
            "sample_count": len(logs),
        }


def get_accuracy_over_time(
    db: Session, project_id: int, days: int = 7, task_type: str = "binary"
) -> dict:
    since = datetime.utcnow() - timedelta(days=days)
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

    metric_name = "Accuracy (%)" if _is_classification(task_type) else "MAE"
    if not logs:
        return {"labels": [], "data": [], "metric_name": metric_name}

    daily: dict[str, list] = defaultdict(list)
    for log in logs:
        daily[log.request_timestamp.strftime("%Y-%m-%d")].append(log)

    labels = sorted(daily.keys())
    values: list[float] = []
    for day in labels:
        day_logs = daily[day]
        if _is_classification(task_type):
            correct = sum(1 for l in day_logs if (l.prediction_values > 0.5) == bool(l.actual_values))
            values.append(round(correct / len(day_logs) * 100, 2))
        else:
            mae = sum(abs(l.prediction_values - l.actual_values) for l in day_logs) / len(day_logs)
            values.append(round(mae, 4))

    return {"labels": labels, "data": values, "metric_name": metric_name}
