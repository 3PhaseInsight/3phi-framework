from sqlalchemy import BigInteger, ForeignKey, Integer
from sqlalchemy.orm import Mapped, mapped_column

from threephi_framework.models.base import BaseModel
from threephi_framework.models.topology.lv_schema_mixin import LvSchemaMixin


class DeliveryPointModel(LvSchemaMixin, BaseModel):
    __tablename__ = "delivery_point"

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True)
    # nullable since migration 05b: delivery points may exist without a cabinet
    cabinet_id: Mapped[int | None] = mapped_column(
        BigInteger,
        ForeignKey("lv.cabinet.id", ondelete="CASCADE"),
        nullable=True,
    )
    service_fuse_size_amps: Mapped[int | None] = mapped_column(Integer)

    def __repr__(self) -> str:
        return f"DeliveryPoint(id={self.id!r}, service_fuse_size_amps={self.service_fuse_size_amps!r})"
