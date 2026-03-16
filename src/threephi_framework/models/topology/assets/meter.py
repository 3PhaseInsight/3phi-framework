from sqlalchemy import BigInteger, ForeignKey
from sqlalchemy.orm import Mapped, mapped_column

from threephi_framework.models.base import BaseModel
from threephi_framework.models.topology.lv_schema_mixin import LvSchemaMixin


class MeterModel(LvSchemaMixin, BaseModel):
    __tablename__ = "meter"

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True)
    delivery_point_id: Mapped[int] = mapped_column(
        BigInteger,
        ForeignKey("lv.delivery_point.id", ondelete="CASCADE"),
        nullable=False,
    )
    has_heat_pump: Mapped[bool | None]
    has_solar_panel: Mapped[bool | None]
    solar_capacity_kw: Mapped[float | None]

    def __repr__(self) -> str:
        return (
            f"Meter(id={self.id!r}, has_heat_pump={self.has_heat_pump!r}, has_solar_panel={self.has_solar_panel!r}, "
            f"solar_capacity_kw={self.solar_capacity_kw!r})"
        )
