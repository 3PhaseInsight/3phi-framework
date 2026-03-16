from datetime import datetime

from sqlalchemy import Boolean, DateTime, Integer, text
from sqlalchemy.orm import Mapped, mapped_column

from threephi_framework.models.base import BaseModel
from threephi_framework.models.topology.lv_schema_mixin import LvSchemaMixin


class TopologyVersionModel(LvSchemaMixin, BaseModel):
    __tablename__ = "topology_version"

    version: Mapped[int] = mapped_column(Integer, primary_key=True)
    ingested_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False, server_default=text("now()"))
    is_current: Mapped[bool] = mapped_column(Boolean, nullable=False, server_default=text("false"))

    def __repr__(self) -> str:
        return (
            f"TopologyVersion(version={self.version!r}, "
            f"ingested_at={self.ingested_at!r}, is_current={self.is_current!r})"
        )
