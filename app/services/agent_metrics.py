from __future__ import annotations

from collections import defaultdict
from datetime import datetime, timedelta

import numpy as np
from sqlalchemy.orm import Session

from ..models import AgentStep, AgentTask

_EMPTY_SUMMARY = {
    "total_tasks": 0,
    "success_count": 0,
    "failure_count": 0,
    "error_count": 0,
    "success_rate": 0.0,
    "error_rate": 0.0,
    "total_cost_usd": 0.0,
    "avg_cost_usd": 0.0,
    "avg_duration_ms": 0.0,
    "p95_duration_ms": 0.0,
    "total_input_tokens": 0,
    "total_output_tokens": 0,
}


def _pct(data: list, p: float) -> float:
    if not data:
        return 0.0
    idx = min(int(len(data) * p / 100), len(data) - 1)
    return data[idx]


def get_agent_summary(db: Session, project_id: int, hours: int = 24) -> dict:
    since = datetime.utcnow() - timedelta(hours=hours)
    tasks = (
        db.query(AgentTask)
        .filter(AgentTask.project_id == project_id, AgentTask.started_at >= since)
        .all()
    )
    if not tasks:
        return dict(_EMPTY_SUMMARY)

    total = len(tasks)
    success = sum(1 for t in tasks if t.status == "success")
    failure = sum(1 for t in tasks if t.status == "failure")
    error = sum(1 for t in tasks if t.status == "error")

    durations = sorted(t.duration_ms for t in tasks if t.duration_ms is not None)
    costs = [t.total_cost_usd for t in tasks if t.total_cost_usd is not None]
    total_cost = sum(costs)

    return {
        "total_tasks": total,
        "success_count": success,
        "failure_count": failure,
        "error_count": error,
        "success_rate": round(success / total * 100, 1),
        "error_rate": round(error / total * 100, 1),
        "total_cost_usd": round(total_cost, 6),
        "avg_cost_usd": round(total_cost / len(costs), 6) if costs else 0.0,
        "avg_duration_ms": round(sum(durations) / len(durations), 1) if durations else 0.0,
        "p95_duration_ms": round(_pct(durations, 95), 1),
        "total_input_tokens": sum(t.total_input_tokens for t in tasks if t.total_input_tokens),
        "total_output_tokens": sum(t.total_output_tokens for t in tasks if t.total_output_tokens),
    }


def get_cost_over_time(db: Session, project_id: int, days: int = 7) -> dict:
    since = datetime.utcnow() - timedelta(days=days)
    tasks = (
        db.query(AgentTask)
        .filter(AgentTask.project_id == project_id, AgentTask.started_at >= since)
        .order_by(AgentTask.started_at)
        .all()
    )
    if not tasks:
        return {"labels": [], "cost_values": [], "input_token_values": [], "output_token_values": []}

    daily_cost: dict[str, float] = defaultdict(float)
    daily_in: dict[str, int] = defaultdict(int)
    daily_out: dict[str, int] = defaultdict(int)

    for t in tasks:
        day = t.started_at.strftime("%Y-%m-%d")
        if t.total_cost_usd:
            daily_cost[day] += t.total_cost_usd
        if t.total_input_tokens:
            daily_in[day] += t.total_input_tokens
        if t.total_output_tokens:
            daily_out[day] += t.total_output_tokens

    labels = sorted(set(daily_cost) | set(daily_in) | set(daily_out))
    return {
        "labels": labels,
        "cost_values": [round(daily_cost.get(d, 0.0), 6) for d in labels],
        "input_token_values": [daily_in.get(d, 0) for d in labels],
        "output_token_values": [daily_out.get(d, 0) for d in labels],
    }


def get_quality_over_time(db: Session, project_id: int, days: int = 7) -> dict:
    since = datetime.utcnow() - timedelta(days=days)
    tasks = (
        db.query(AgentTask)
        .filter(
            AgentTask.project_id == project_id,
            AgentTask.started_at >= since,
            AgentTask.quality_score != None,  # noqa: E711
            AgentTask.status == "success",
        )
        .order_by(AgentTask.started_at)
        .all()
    )
    if not tasks:
        return {"labels": [], "values": [], "sample_counts": []}

    daily: dict[str, list[float]] = defaultdict(list)
    for t in tasks:
        daily[t.started_at.strftime("%Y-%m-%d")].append(t.quality_score)

    labels = sorted(daily.keys())
    return {
        "labels": labels,
        "values": [round(sum(daily[d]) / len(daily[d]), 4) for d in labels],
        "sample_counts": [len(daily[d]) for d in labels],
    }


def get_duration_distribution(db: Session, project_id: int, hours: int = 24) -> dict:
    since = datetime.utcnow() - timedelta(hours=hours)
    rows = (
        db.query(AgentTask.duration_ms)
        .filter(
            AgentTask.project_id == project_id,
            AgentTask.started_at >= since,
            AgentTask.duration_ms != None,  # noqa: E711
        )
        .all()
    )
    durations = [r[0] for r in rows]
    if not durations:
        return {"labels": [], "counts": [], "p50": 0.0, "p95": 0.0, "p99": 0.0, "avg": 0.0}

    arr = np.array(durations)
    counts, edges = np.histogram(arr, bins=15)
    labels = [f"{edges[i]:.0f}–{edges[i+1]:.0f}" for i in range(len(edges) - 1)]

    sorted_d = sorted(durations)
    return {
        "labels": labels,
        "counts": counts.tolist(),
        "p50": round(_pct(sorted_d, 50), 1),
        "p95": round(_pct(sorted_d, 95), 1),
        "p99": round(_pct(sorted_d, 99), 1),
        "avg": round(sum(durations) / len(durations), 1),
    }


def get_step_analysis(db: Session, project_id: int, hours: int = 24) -> dict:
    since = datetime.utcnow() - timedelta(hours=hours)
    steps = (
        db.query(AgentStep)
        .filter(AgentStep.project_id == project_id, AgentStep.started_at >= since)
        .all()
    )
    if not steps:
        return {"by_type": [], "top_tools": []}

    by_type_acc: dict[str, dict] = defaultdict(lambda: {"count": 0, "success": 0, "durations": []})
    tool_acc: dict[str, dict] = defaultdict(lambda: {"count": 0, "success": 0})

    for s in steps:
        by_type_acc[s.step_type]["count"] += 1
        if s.status == "success":
            by_type_acc[s.step_type]["success"] += 1
        if s.duration_ms:
            by_type_acc[s.step_type]["durations"].append(s.duration_ms)
        if s.step_name and s.step_type in ("tool_call", "agent_call"):
            tool_acc[s.step_name]["count"] += 1
            if s.status == "success":
                tool_acc[s.step_name]["success"] += 1

    by_type = [
        {
            "step_type": stype,
            "count": d["count"],
            "success_rate": round(d["success"] / d["count"] * 100, 1),
            "avg_duration_ms": round(sum(d["durations"]) / len(d["durations"]), 1) if d["durations"] else None,
        }
        for stype, d in sorted(by_type_acc.items())
    ]

    top_tools = sorted(
        [
            {
                "name": k,
                "count": v["count"],
                "success_rate": round(v["success"] / v["count"] * 100, 1),
            }
            for k, v in tool_acc.items()
        ],
        key=lambda x: x["count"],
        reverse=True,
    )[:10]

    return {"by_type": by_type, "top_tools": top_tools}
