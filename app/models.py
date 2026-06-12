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


# ─────────────────────────────────────────
#  Agent Monitoring Tables
# ─────────────────────────────────────────

class AgentProject(Base):
    __tablename__ = "m_agent_projects"

    project_id = Column(Integer, primary_key=True, index=True)
    project_name = Column(String(100), unique=True, nullable=False, index=True)
    description = Column(String(500), nullable=True)
    agent_type = Column(String(20), default="tool")   # llm | tool | multi
    model_name = Column(String(100), nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)

    tasks = relationship("AgentTask", back_populates="project", cascade="all, delete-orphan")
    ref_data = relationship("AgentRefData", back_populates="project", cascade="all, delete-orphan")
    config = relationship("AgentConfig", back_populates="project", uselist=False, cascade="all, delete-orphan")


class AgentConfig(Base):
    __tablename__ = "m_agent_configs"

    id = Column(Integer, primary_key=True, index=True)
    project_id = Column(Integer, ForeignKey("m_agent_projects.project_id"), unique=True, nullable=False)

    drift_window_size = Column(Integer, default=100)
    psi_warning = Column(Float, default=0.10)
    psi_alert = Column(Float, default=0.25)
    success_rate_warning = Column(Float, default=90.0)
    success_rate_alert = Column(Float, default=75.0)
    quality_warning = Column(Float, nullable=True)
    quality_alert = Column(Float, nullable=True)

    updated_at = Column(DateTime, default=datetime.utcnow)

    project = relationship("AgentProject", back_populates="config")


class AgentRefData(Base):
    """ドリフト検知用参照データ（1行 = 1サンプル）"""
    __tablename__ = "m_agent_ref_data"

    ref_id = Column(Integer, primary_key=True, index=True)
    project_id = Column(Integer, ForeignKey("m_agent_projects.project_id"), nullable=False, index=True)
    feature_values = Column(Text, nullable=True)      # JSON: {"prompt_len": 512.0, ...}
    feature_importance = Column(Text, nullable=True)  # JSON: {"prompt_len": 0.6, ...}
    created_at = Column(DateTime, default=datetime.utcnow, index=True)

    project = relationship("AgentProject", back_populates="ref_data")


class AgentTask(Base):
    __tablename__ = "t_agent_tasks"

    task_id = Column(Integer, primary_key=True, index=True)
    project_id = Column(Integer, ForeignKey("m_agent_projects.project_id"), nullable=False, index=True)
    session_id = Column(String(100), nullable=True, index=True)
    task_name = Column(String(200), nullable=True)
    status = Column(String(20), default="success")     # success | failure | error
    quality_score = Column(Float, nullable=True)       # 0.0 – 1.0
    total_input_tokens = Column(Integer, nullable=True)
    total_output_tokens = Column(Integer, nullable=True)
    total_cost_usd = Column(Float, nullable=True)
    total_steps = Column(Integer, default=0)
    duration_ms = Column(Float, nullable=True)
    started_at = Column(DateTime, default=datetime.utcnow, index=True)
    feature_values = Column(Text, nullable=True)       # JSON: ドリフト用数値特徴量
    error_message = Column(Text, nullable=True)

    project = relationship("AgentProject", back_populates="tasks")
    steps = relationship("AgentStep", back_populates="task", cascade="all, delete-orphan")


class AgentStep(Base):
    __tablename__ = "t_agent_steps"

    step_id = Column(Integer, primary_key=True, index=True)
    task_id = Column(Integer, ForeignKey("t_agent_tasks.task_id"), nullable=False, index=True)
    project_id = Column(Integer, ForeignKey("m_agent_projects.project_id"), nullable=False, index=True)
    step_type = Column(String(20), nullable=False)     # llm_call | tool_call | agent_call
    step_name = Column(String(200), nullable=True)     # モデル名 or ツール名
    input_tokens = Column(Integer, nullable=True)
    output_tokens = Column(Integer, nullable=True)
    cost_usd = Column(Float, nullable=True)
    duration_ms = Column(Float, nullable=True)
    status = Column(String(20), default="success")     # success | failure | error
    started_at = Column(DateTime, default=datetime.utcnow, index=True)

    task = relationship("AgentTask", back_populates="steps")
