from __future__ import annotations

import json

import numpy as np
from sqlalchemy.orm import Session

from ..models import AgentRefData, AgentTask
from .drift import _compute_psi


def detect_agent_feature_drift(
    db: Session,
    project_id: int,
    window_size: int = 100,
    psi_warning: float = 0.10,
    psi_alert: float = 0.25,
) -> dict:
    """
    m_agent_ref_data（参照分布）と t_agent_tasks（最新ウィンドウ）の
    feature_values を特徴量ごとに PSI で比較してドリフトを検知する。
    """
    latest_ref = (
        db.query(AgentRefData)
        .filter(AgentRefData.project_id == project_id)
        .order_by(AgentRefData.created_at.desc())
        .first()
    )

    if not latest_ref:
        return {
            "status": "no_reference_data",
            "message": "参照データが登録されていません（POST /agent-api/projects/{id}/ref-data で登録）",
            "features": [],
            "drift_detected": False,
        }

    ref_rows = db.query(AgentRefData).filter(AgentRefData.project_id == project_id).all()
    ref_features = [json.loads(r.feature_values) for r in ref_rows if r.feature_values]

    if not ref_features:
        return {
            "status": "no_reference_features",
            "message": "参照データに特徴量データがありません",
            "features": [],
            "drift_detected": False,
        }

    feature_importance: dict[str, float] = {}
    if latest_ref.feature_importance:
        feature_importance = json.loads(latest_ref.feature_importance)

    current_tasks = (
        db.query(AgentTask)
        .filter(
            AgentTask.project_id == project_id,
            AgentTask.feature_values != None,  # noqa: E711
            AgentTask.status == "success",
        )
        .order_by(AgentTask.started_at.desc())
        .limit(window_size)
        .all()
    )

    if not current_tasks:
        return {
            "status": "insufficient_data",
            "message": "特徴量データが含まれる成功タスクがありません",
            "features": [],
            "drift_detected": False,
        }

    current_features = [json.loads(t.feature_values) for t in current_tasks]
    feature_names = list(ref_features[0].keys())
    results = []

    for feat in feature_names:
        ref_arr = np.array(
            [f[feat] for f in ref_features if feat in f and f[feat] is not None], dtype=float
        )
        cur_arr = np.array(
            [f[feat] for f in current_features if feat in f and f[feat] is not None], dtype=float
        )
        if len(ref_arr) < 5 or len(cur_arr) < 5:
            continue

        psi = _compute_psi(ref_arr, cur_arr)
        importance = feature_importance.get(feat, 0.0)
        psi_level = "ok" if psi < psi_warning else ("warning" if psi < psi_alert else "alert")

        results.append({
            "feature": feat,
            "psi": round(psi, 4),
            "psi_level": psi_level,
            "importance": round(float(importance), 4),
        })

    max_psi = max((r["psi"] for r in results), default=0.0)

    return {
        "status": "ok",
        "reference_count": len(ref_features),
        "current_count": len(current_features),
        "features": results,
        "drift_detected": max_psi >= psi_alert,
        "psi_warning": psi_warning,
        "psi_alert": psi_alert,
    }
