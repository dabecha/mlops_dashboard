from __future__ import annotations

from datetime import datetime

from sqlalchemy.orm import Session

from ..models import AgentConfig

AGENT_DEFAULTS: dict = {
    "drift_window_size": 100,
    "psi_warning": 0.10,
    "psi_alert": 0.25,
    "success_rate_warning": 90.0,
    "success_rate_alert": 75.0,
    "quality_warning": None,
    "quality_alert": None,
}


def get_agent_config(db: Session, project_id: int) -> dict:
    cfg = db.query(AgentConfig).filter(AgentConfig.project_id == project_id).first()
    if cfg:
        return {k: getattr(cfg, k) for k in AGENT_DEFAULTS}
    return dict(AGENT_DEFAULTS)


def upsert_agent_config(db: Session, project_id: int, data: dict) -> AgentConfig:
    cfg = db.query(AgentConfig).filter(AgentConfig.project_id == project_id).first()
    if cfg:
        for k, v in data.items():
            setattr(cfg, k, v)
        cfg.updated_at = datetime.utcnow()
    else:
        cfg = AgentConfig(project_id=project_id, updated_at=datetime.utcnow(), **data)
        db.add(cfg)
    db.commit()
    db.refresh(cfg)
    return cfg
