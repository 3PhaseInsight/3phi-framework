import datetime
import uuid
from typing import Any

from sqlalchemy import DateTime, Text, UniqueConstraint, text
from sqlalchemy.dialects.postgresql import JSONB, UUID
from sqlalchemy.orm import Mapped, mapped_column

from threephi_framework.models.base import BaseModel
from threephi_framework.models.meta.meta_schema_mixin import MetaSchemaMixin


class IngestBatchModel(MetaSchemaMixin, BaseModel):
    __tablename__ = "ingest_batch"
    __table_args__ = (
        UniqueConstraint("source_file", "run_id"),
        MetaSchemaMixin.__table_args__,
    )

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, server_default=text("gen_random_uuid()")
    )
    source_file: Mapped[str] = mapped_column(Text, nullable=False)
    received_at: Mapped[datetime.datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=text("now()")
    )
    status: Mapped[str] = mapped_column(Text, nullable=False)
    error_log: Mapped[str | None] = mapped_column(Text, nullable=True)
    stats_json: Mapped[dict[str, Any] | None] = mapped_column(JSONB, nullable=True)
    run_id: Mapped[str | None] = mapped_column(Text, nullable=True)
