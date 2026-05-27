import datetime
from typing import Any

from sqlalchemy import DateTime, Float, Integer, String, Text, create_engine
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, sessionmaker

from src.common.config import settings


class Base(DeclarativeBase):
    pass


class Prediction(Base):
    __tablename__ = "predictions"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    timestamp: Mapped[datetime.datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, index=True
    )
    model_version: Mapped[str] = mapped_column(String(64), nullable=False)
    input_features: Mapped[dict[str, Any]] = mapped_column(JSONB, nullable=False)
    prediction: Mapped[int] = mapped_column(Integer, nullable=False)
    probability: Mapped[float] = mapped_column(Float, nullable=False)


class MonitoringMetric(Base):
    __tablename__ = "monitoring_metrics"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    timestamp: Mapped[datetime.datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, index=True
    )
    metric_name: Mapped[str] = mapped_column(String(128), nullable=False)
    feature_name: Mapped[str | None] = mapped_column(String(128), nullable=True)
    value: Mapped[float] = mapped_column(Float, nullable=False)
    window_start: Mapped[datetime.datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    window_end: Mapped[datetime.datetime] = mapped_column(DateTime(timezone=True), nullable=False)


class Incident(Base):
    __tablename__ = "incidents"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    opened_at: Mapped[datetime.datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, index=True
    )
    closed_at: Mapped[datetime.datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    status: Mapped[str] = mapped_column(String(32), nullable=False, default="open")
    trigger_metric: Mapped[str] = mapped_column(String(128), nullable=False)
    agent_diagnosis: Mapped[dict[str, Any] | None] = mapped_column(JSONB, nullable=True)
    agent_action: Mapped[str | None] = mapped_column(String(64), nullable=True)
    human_decision: Mapped[str | None] = mapped_column(String(32), nullable=True)
    reasoning_trace: Mapped[str | None] = mapped_column(Text, nullable=True)


engine = create_engine(settings.DATABASE_URL, pool_pre_ping=True)
SessionLocal = sessionmaker(bind=engine, autoflush=False, autocommit=False)
