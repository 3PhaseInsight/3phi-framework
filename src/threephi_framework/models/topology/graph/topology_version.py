from datetime import datetime

from sqlalchemy import Boolean, DateTime, Enum, Integer, text
from sqlalchemy.orm import Mapped, mapped_column

from threephi_framework.models.base import BaseModel
from threephi_framework.models.topology.lv_schema_mixin import LvSchemaMixin
from threephi_framework.processing_level import ProcessingLevel


class TopologyVersionModel(LvSchemaMixin, BaseModel):
    __tablename__ = "topology_version"

    version: Mapped[int] = mapped_column(Integer, primary_key=True)
    ingested_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False, server_default=text("now()"))
    is_current: Mapped[bool] = mapped_column(Boolean, nullable=False, server_default=text("false"))
    # VARCHAR(32) with CHECK constraint in DB (migration 07); validated at the
    # Python boundary via the ProcessingLevel enum here. create_constraint=False
    # leaves DDL emission to sqitch.
    processing_level: Mapped[ProcessingLevel] = mapped_column(
        Enum(
            ProcessingLevel,
            native_enum=False,
            length=32,
            create_constraint=False,
            values_callable=lambda e: [m.value for m in e],
            validate_strings=True,
        ),
        nullable=False,
        server_default=text("'raw'"),
    )

    def __repr__(self) -> str:
        return (
            f"TopologyVersion(version={self.version!r}, "
            f"ingested_at={self.ingested_at!r}, "
            f"is_current={self.is_current!r}, "
            f"processing_level={self.processing_level!r})"
        )
