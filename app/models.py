from sqlalchemy import Boolean, Column, DateTime, Float, ForeignKey, Integer, String, Text
from sqlalchemy.orm import relationship
from datetime import datetime

from .database import Base


class Project(Base):
    __tablename__ = "m_projects"

    project_id = Column(Integer, primary_key=True, index=True)
    project_name = Column(String(100), unique=True, nullable=False, index=True)
    description = Column(String(500), nullable=True)
    task_type = Column(String(20), default="binary")  # binary, multi-class, multi-label, regression
    created_at = Column(DateTime, default=datetime.utcnow)

    logs = relationship("InferenceLog", back_populates="project", cascade="all, delete-orphan")
    config = relationship("ProjectConfig", back_populates="project", uselist=False, cascade="all, delete-orphan")
    deployed_models = relationship("DeployedModel", back_populates="project", cascade="all, delete-orphan")


class ProjectConfig(Base):
    __tablename__ = "project_configs"

    id = Column(Integer, primary_key=True, index=True)
    project_id = Column(Integer, ForeignKey("m_projects.project_id"), unique=True, nullable=False)

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
    __tablename__ = "m_deployed_models"

    model_id = Column(Integer, primary_key=True, index=True)
    project_id = Column(Integer, ForeignKey("m_projects.project_id"), nullable=False, index=True)
    model_version = Column(String(100), nullable=True)
    feature_values = Column(Text, nullable=True)    # JSON: one training sample {"age": 35.0, ...}
    feature_dtypes = Column(Text, nullable=True)    # JSON: {"age": "float32", ...}
    feature_importance = Column(Text, nullable=True)  # JSON: {"age": 0.43, ...}
    actual_values = Column(Float, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow, index=True)

    project = relationship("Project", back_populates="deployed_models")
    logs = relationship("InferenceLog", back_populates="deployed_model")


class InferenceLog(Base):
    __tablename__ = "t_inference_logs"

    log_id = Column(Integer, primary_key=True, index=True)
    project_id = Column(Integer, ForeignKey("m_projects.project_id"), nullable=False, index=True)
    request_timestamp = Column(DateTime, default=datetime.utcnow, index=True)
    request_id = Column(String(100), nullable=True)
    model_id = Column(Integer, ForeignKey("m_deployed_models.model_id"), nullable=True, index=True)
    prediction_values = Column(Float, nullable=False)
    actual_values = Column(Float, nullable=True)
    response_time_ms = Column(Float, nullable=False)
    is_error = Column(Boolean, default=False)
    feature_values = Column(Text, nullable=True)    # JSON: {"age": 35.0, ...}
    feature_dtypes = Column(Text, nullable=True)    # JSON: {"age": "float32", ...}

    project = relationship("Project", back_populates="logs")
    deployed_model = relationship("DeployedModel", back_populates="logs")
