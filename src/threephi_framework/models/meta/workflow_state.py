import datetime

from sqlalchemy import Boolean, DateTime, Identity, Integer, Text, text
from sqlalchemy.orm import Mapped, mapped_column

from threephi_framework.models.base import BaseModel
from threephi_framework.models.meta.meta_schema_mixin import MetaSchemaMixin


class WorkflowStateModel(MetaSchemaMixin, BaseModel):
    __tablename__ = "workflow_states"
    __table_args__ = MetaSchemaMixin.__table_args__

    id: Mapped[int] = mapped_column(Integer, Identity(always=False), primary_key=True)
    workflow: Mapped[str] = mapped_column(Text, nullable=False, unique=True)
    completed: Mapped[bool] = mapped_column(Boolean, nullable=False, server_default=text("false"))
    description: Mapped[str | None] = mapped_column(Text, nullable=True)
    # maintained by trigger; no timezone per migration DDL
    updated_at: Mapped[datetime.datetime | None] = mapped_column(DateTime(timezone=False), server_default=text("now()"))
