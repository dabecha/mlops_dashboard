from __future__ import annotations

from collections import defaultdict
from datetime import datetime, timedelta

import numpy as np
from sqlalchemy.orm import Session

from ..models import InferenceLog


def get_summary(db: Session, project_id: int, hours: int = 24) -> dict:
    since = datetime.utcnow() - timedelta(hours=hours)
    logs = (
        db.query(InferenceLog)
        .filter(InferenceLog.project_id == project_id, InferenceLog.timestamp >= since)
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
            InferenceLog.timestamp >= since,
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


def get_accuracy_over_time(
    db: Session, project_id: int, days: int = 7, task_type: str = "classification"
) -> dict:
    since = datetime.utcnow() - timedelta(days=days)
    logs = (
        db.query(InferenceLog)
        .filter(
            InferenceLog.project_id == project_id,
            InferenceLog.timestamp >= since,
            InferenceLog.actual_label != None,  # noqa: E711
        )
        .order_by(InferenceLog.timestamp)
        .all()
    )

    if not logs:
        metric_name = "Accuracy (%)" if task_type == "classification" else "MAE"
        return {"labels": [], "data": [], "metric_name": metric_name}

    daily: dict[str, list] = defaultdict(list)
    for log in logs:
        daily[log.timestamp.strftime("%Y-%m-%d")].append(log)

    labels = sorted(daily.keys())
    values: list[float] = []
    for day in labels:
        day_logs = daily[day]
        if task_type == "classification":
            correct = sum(1 for l in day_logs if (l.prediction > 0.5) == bool(l.actual_label))
            values.append(round(correct / len(day_logs) * 100, 2))
        else:
            mae = sum(abs(l.prediction - l.actual_label) for l in day_logs) / len(day_logs)
            values.append(round(mae, 4))

    metric_name = "Accuracy (%)" if task_type == "classification" else "MAE"
    return {"labels": labels, "data": values, "metric_name": metric_name}
