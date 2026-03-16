from __future__ import annotations

from sqlalchemy import BigInteger, ForeignKey, Integer
from sqlalchemy.orm import Mapped, mapped_column

from threephi_framework.models.base import BaseModel
from threephi_framework.models.topology.lv_schema_mixin import LvSchemaMixin


class TransformerModel(LvSchemaMixin, BaseModel):
    __tablename__ = "transformer"

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True)
    substation_id: Mapped[int] = mapped_column(
        BigInteger,
        ForeignKey("lv.secondary_substation.id", ondelete="CASCADE"),
        nullable=False,
    )
    capacity_kva: Mapped[int | None] = mapped_column(Integer)

    def __repr__(self) -> str:
        return f"Transformer(id={self.id!r}, capacity_kva={self.capacity_kva!r})"
