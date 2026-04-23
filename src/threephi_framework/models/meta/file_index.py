import datetime
import os
import uuid

from sqlalchemy import BigInteger, Date, DateTime, ForeignKey, Integer, Text, UniqueConstraint, text
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import Mapped, mapped_column

from threephi_framework.models.base import BaseModel
from threephi_framework.models.meta.meta_schema_mixin import MetaSchemaMixin

_META_SCHEMA = os.getenv("META_SCHEMA", "meta")


class FileIndexModel(MetaSchemaMixin, BaseModel):
    __tablename__ = "file_index"
    __table_args__ = (
        UniqueConstraint("dt", "shard", "seq"),
        MetaSchemaMixin.__table_args__,
    )

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, server_default=text("gen_random_uuid()")
    )
    s3_key: Mapped[str] = mapped_column(Text, nullable=False)
    dt: Mapped[datetime.date] = mapped_column(Date, nullable=False)
    shard: Mapped[int] = mapped_column(Integer, nullable=False)
    seq: Mapped[int] = mapped_column(BigInteger, nullable=False)
    ts_start: Mapped[datetime.datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    ts_end: Mapped[datetime.datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    rows: Mapped[int] = mapped_column(BigInteger, nullable=False)
    bytes: Mapped[int] = mapped_column(BigInteger, nullable=False)
    schema_version: Mapped[str] = mapped_column(Text, nullable=False)
    status: Mapped[str] = mapped_column(Text, nullable=False)
    batch_id: Mapped[uuid.UUID | None] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey(f"{_META_SCHEMA}.ingest_batch.id", ondelete="SET NULL"),
        nullable=True,
    )
    ingest_file: Mapped[str] = mapped_column(Text, nullable=False)
    created_at: Mapped[datetime.datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=text("now()")
    )
    committed_at: Mapped[datetime.datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
