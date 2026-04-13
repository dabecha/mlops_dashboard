from __future__ import annotations

import numpy as np
from scipy import stats
from sqlalchemy.orm import Session

from ..models import InferenceLog


def _compute_psi(expected: np.ndarray, actual: np.ndarray, bins: int = 10) -> float:
    """Population Stability Index を計算する。"""
    eps = 1e-6
    lo = min(expected.min(), actual.min())
    hi = max(expected.max(), actual.max())
    if lo == hi:
        return 0.0
    edges = np.linspace(lo, hi, bins + 1)

    exp_counts, _ = np.histogram(expected, bins=edges)
    act_counts, _ = np.histogram(actual, bins=edges)

    exp_pct = (exp_counts + eps) / (len(expected) + eps * bins)
    act_pct = (act_counts + eps) / (len(actual) + eps * bins)

    return float(np.sum((act_pct - exp_pct) * np.log(act_pct / exp_pct)))


def detect_drift(db: Session, project_id: int, window_size: int = 100) -> dict:
    """
    参照ウィンドウ（最古 window_size 件）と現在ウィンドウ（最新 window_size 件）を
    KS 検定と PSI で比較してドリフトを検知する。
    """
    logs = (
        db.query(InferenceLog)
        .filter(
            InferenceLog.project_id == project_id,
            InferenceLog.is_error == False,  # noqa: E712
        )
        .order_by(InferenceLog.timestamp)
        .all()
    )

    if len(logs) < window_size * 2:
        return {
            "status": "insufficient_data",
            "message": f"ドリフト検知には最低 {window_size * 2} 件のデータが必要です（現在: {len(logs)} 件）",
            "ks_statistic": None,
            "ks_pvalue": None,
            "psi": None,
            "psi_level": None,
            "drift_detected": False,
        }

    reference = np.array([l.prediction for l in logs[:window_size]])
    current = np.array([l.prediction for l in logs[-window_size:]])

    ks_stat, ks_pvalue = stats.ks_2samp(reference, current)
    psi = _compute_psi(reference, current)

    # PSI 判定: <0.1 正常、0.1–0.25 警告、>0.25 異常
    psi_level = "ok" if psi < 0.1 else ("warning" if psi < 0.25 else "alert")
    drift_by_ks = ks_pvalue < 0.05
    drift_detected = drift_by_ks or psi_level == "alert"

    return {
        "status": "ok",
        "ks_statistic": round(ks_stat, 4),
        "ks_pvalue": round(ks_pvalue, 4),
        "psi": round(psi, 4),
        "psi_level": psi_level,
        "drift_by_ks": drift_by_ks,
        "drift_detected": drift_detected,
        "reference_count": window_size,
        "current_count": window_size,
        "message": _build_message(drift_detected, psi_level, ks_pvalue),
    }


def _build_message(drift_detected: bool, psi_level: str, ks_pvalue: float) -> str:
    if not drift_detected:
        return "ドリフトは検出されていません"
    parts = []
    if psi_level == "alert":
        parts.append("PSI > 0.25: 分布が大幅に変化しています")
    elif psi_level == "warning":
        parts.append("PSI 0.1–0.25: 軽微な分布変化を検出")
    if ks_pvalue < 0.05:
        parts.append(f"KS 検定: p = {ks_pvalue:.4f}（有意差あり）")
    return "　".join(parts)
