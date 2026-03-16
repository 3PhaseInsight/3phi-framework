from sqlalchemy import BigInteger, ForeignKey, Integer
from sqlalchemy.orm import Mapped, mapped_column

from threephi_framework.models.base import BaseModel
from threephi_framework.models.topology.lv_schema_mixin import LvSchemaMixin


class DeliveryPointModel(LvSchemaMixin, BaseModel):
    __tablename__ = "delivery_point"

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True)
    cabinet_id: Mapped[int] = mapped_column(
        BigInteger,
        ForeignKey("lv.cabinet.id", ondelete="CASCADE"),
        nullable=False,
    )
    service_fuse_size_amps: Mapped[int | None] = mapped_column(Integer)

    def __repr__(self) -> str:
        return f"Cabinet(id={self.id!r}, service_fuse_size_amps={self.service_fuse_size_amps!r})"
