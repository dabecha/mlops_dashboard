from __future__ import annotations

from sqlalchemy.orm import Session

from ..metrics_catalog import is_higher_better as _metric_higher_better
from ..models import ProjectConfig
from ..settings import settings
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
    "is_higher_better": True,
    "classification_threshold": 0.5,
}


def apply_defaults(raw: dict | None) -> dict:
    """設定 dict の欠損値にデフォルト値を適用し、DEFAULTS のキーに揃える。

    is_higher_better が未設定の場合は指標カタログの向きを採用する。
    """
    raw = raw or {}
    cfg = {}
    for k, default in DEFAULTS.items():
        v = raw.get(k)
        cfg[k] = default if v is None else v
    if raw.get("is_higher_better") is None:
        cfg["is_higher_better"] = _metric_higher_better(cfg["metric_name"])
    else:
        cfg["is_higher_better"] = bool(cfg["is_higher_better"])
    return cfg


@log_call
def get_config(db: Session, project_id: str) -> dict:
    """プロジェクト設定を dict で返す。未登録の場合はデフォルト値を返す。

    dev / production モードでは Dataiku 管理プロジェクトの m_project_configs
    データセットから取得する。local_dev モードでは SQLite から取得する。
    """
    if settings.is_dataiku:
        from ..dataiku_client import get_project_config
        return apply_defaults(get_project_config(project_id))

    cfg = db.query(ProjectConfig).filter(ProjectConfig.project_id == project_id).first()
    if cfg:
        return apply_defaults({k: getattr(cfg, k) for k in DEFAULTS})
    return DEFAULTS.copy()
