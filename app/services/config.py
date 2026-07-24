from __future__ import annotations

from sqlalchemy.orm import Session

from ..models import ProjectConfig
from ..logging_utils import log_call

# システムデフォルト値
DEFAULTS: dict = {
    "drift_window_size": 100,
    "psi_warning": 0.10,
    "psi_alert": 0.25,
    "ks_alpha": 0.05,
    "metric_name": "ROC-AUC",
    "metric_warning": 75.0,
    "metric_alert": 60.0,
    "classification_threshold": 0.5,
}


@log_call
def get_config(db: Session, project_id: str) -> dict:
    """プロジェクト設定を dict で返す。未登録の場合はデフォルト値を返す。"""
    cfg = db.query(ProjectConfig).filter(ProjectConfig.project_id == project_id).first()
    if cfg:
        return {k: getattr(cfg, k) for k in DEFAULTS}
    return DEFAULTS.copy()
