from sqlalchemy import Boolean, Column, DateTime, Float, ForeignKey, Integer, String, Text
from sqlalchemy.orm import relationship
from datetime import datetime

from .database import Base


class Project(Base):
    __tablename__ = "projects"

    id = Column(Integer, primary_key=True, index=True)
    name = Column(String(100), unique=True, nullable=False, index=True)
    description = Column(String(500), nullable=True)
    task_type = Column(String(20), default="classification")  # "classification" or "regression"
    created_at = Column(DateTime, default=datetime.utcnow)

    logs = relationship("InferenceLog", back_populates="project", cascade="all, delete-orphan")


class InferenceLog(Base):
    __tablename__ = "inference_logs"

    id = Column(Integer, primary_key=True, index=True)
    project_id = Column(Integer, ForeignKey("projects.id"), nullable=False, index=True)
    timestamp = Column(DateTime, default=datetime.utcnow, index=True)
    request_id = Column(String(100), nullable=True)
    prediction = Column(Float, nullable=False)
    actual_label = Column(Float, nullable=True)
    confidence = Column(Float, nullable=True)
    response_time_ms = Column(Float, nullable=False)
    is_error = Column(Boolean, default=False)
    error_message = Column(Text, nullable=True)
    feature_values = Column(Text, nullable=True)  # JSON string

    project = relationship("Project", back_populates="logs")
