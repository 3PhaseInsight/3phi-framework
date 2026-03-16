import datetime
from typing import Any

from sqlalchemy import BigInteger, DateTime
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import Mapped, mapped_column

from threephi_framework.models.base import BaseModel
from threephi_framework.models.meta.meta_schema_mixin import MetaSchemaMixin


class MetaMeterModel(MetaSchemaMixin, BaseModel):
    __tablename__ = "meter"

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True)
    first_seen: Mapped[datetime.datetime] = mapped_column(DateTime)
    last_seen: Mapped[datetime.datetime] = mapped_column(DateTime)
    total_rows: Mapped[int] = mapped_column(BigInteger)
    updated_at: Mapped[datetime.datetime] = mapped_column(DateTime)
    data_quality: Mapped[dict[str, Any] | None] = mapped_column(JSONB, nullable=True)
    data_statistics: Mapped[dict[str, Any] | None] = mapped_column(JSONB, nullable=True)
    connectivity: Mapped[dict[str, Any] | None] = mapped_column(JSONB, nullable=True)
