import uuid

from sqlalchemy import Boolean, Column, DateTime, Float, ForeignKey, Integer, String, Text
from sqlalchemy.orm import relationship
from datetime import datetime

from .database import Base


def _gen_id() -> str:
    """local_dev モード用の文字列 ID を採番する（uuid4 の hex 32 文字）。

    dev / production モードでは Dataiku データセットの ID を明示指定するため
    このデフォルトは使用されない。
    """
    return uuid.uuid4().hex


class Project(Base):
    __tablename__ = "m_projects"

    project_id = Column(String(100), primary_key=True, default=_gen_id, index=True)
    project_name = Column(String(100), unique=True, nullable=False, index=True)
    description = Column(String(500), nullable=True)
    task_type = Column(String(20), default="binary")  # binary, multi-class, multi-label, regression
    created_at = Column(DateTime, default=datetime.utcnow)

    logs = relationship("InferenceLog", back_populates="project", cascade="all, delete-orphan")
    config = relationship("ProjectConfig", back_populates="project", uselist=False, cascade="all, delete-orphan")
    deployed_models = relationship("DeployedModel", back_populates="project", cascade="all, delete-orphan")
    reference_logs = relationship("ReferenceLog", back_populates="project", cascade="all, delete-orphan")


class ProjectConfig(Base):
    __tablename__ = "project_configs"

    id = Column(Integer, primary_key=True, index=True)
    project_id = Column(String(100), ForeignKey("m_projects.project_id"), unique=True, nullable=False)

    drift_window_size = Column(Integer, default=100)
    psi_warning = Column(Float, default=0.10)
    psi_alert = Column(Float, default=0.25)
    ks_alpha = Column(Float, default=0.05)

    accuracy_warning = Column(Float, default=75.0)
    accuracy_alert = Column(Float, default=60.0)

    mae_warning = Column(Float, nullable=True)
    mae_alert = Column(Float, nullable=True)

    updated_at = Column(DateTime, default=datetime.utcnow)

    project = relationship("Project", back_populates="config")


class DeployedModel(Base):
    __tablename__ = "t_deployed_models"

    model_id = Column(String(100), primary_key=True, default=_gen_id, index=True)
    project_id = Column(String(100), ForeignKey("m_projects.project_id"), nullable=False, index=True)
    model_version = Column(String(100), nullable=True)
    feature_dtypes = Column(Text, nullable=True)       # JSON: {"age": "float32", ...}
    feature_importance = Column(Text, nullable=True)   # JSON: {"age": 0.43, ...}
    is_activate = Column(Boolean, nullable=False, default=True)
    created_at = Column(DateTime, default=datetime.utcnow, index=True)

    project = relationship("Project", back_populates="deployed_models")
    logs = relationship("InferenceLog", back_populates="deployed_model")
    reference_logs = relationship("ReferenceLog", back_populates="deployed_model")


class ReferenceLog(Base):
    """特徴量ドリフト検知用の学習データ参照ログ（1行 = 1サンプル）"""
    __tablename__ = "t_reference_logs"

    log_id = Column(String(100), primary_key=True, default=_gen_id, index=True)
    project_id = Column(String(100), ForeignKey("m_projects.project_id"), nullable=False, index=True)
    model_id = Column(String(100), ForeignKey("t_deployed_models.model_id"), nullable=True, index=True)
    feature_values = Column(Text, nullable=True)   # JSON: {"age": 35.0, "amount": 5200.0}
    actual_values = Column(Float, nullable=True)

    project = relationship("Project", back_populates="reference_logs")
    deployed_model = relationship("DeployedModel", back_populates="reference_logs")


class InferenceLog(Base):
    __tablename__ = "t_inference_logs"

    log_id = Column(String(100), primary_key=True, default=_gen_id, index=True)
    project_id = Column(String(100), ForeignKey("m_projects.project_id"), nullable=False, index=True)
    batch_log_id = Column(String(100), nullable=True, index=True)
    request_timestamp = Column(DateTime, default=datetime.utcnow, index=True)
    request_id = Column(String(100), nullable=True)
    model_id = Column(String(100), ForeignKey("t_deployed_models.model_id"), nullable=True, index=True)
    prediction_values = Column(Float, nullable=False)
    actual_values = Column(Float, nullable=True)
    response_time_ms = Column(Float, nullable=False)
    is_error = Column(Boolean, default=False)
    feature_values = Column(Text, nullable=True)    # JSON: {"age": 35.0, ...}
    feature_dtypes = Column(Text, nullable=True)    # JSON: {"age": "float32", ...}
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    project = relationship("Project", back_populates="logs")
    deployed_model = relationship("DeployedModel", back_populates="logs")
