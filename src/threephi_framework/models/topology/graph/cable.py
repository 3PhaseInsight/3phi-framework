from sqlalchemy import BigInteger, ForeignKey, Integer, Numeric, PrimaryKeyConstraint, Text
from sqlalchemy.orm import Mapped, mapped_column

from threephi_framework.models.base import BaseModel
from threephi_framework.models.topology.lv_schema_mixin import LvSchemaMixin


class CableModel(LvSchemaMixin, BaseModel):
    __tablename__ = "cable"
    __table_args__ = (
        PrimaryKeyConstraint("version", "cable_id", name="pk_lv_cable"),
        LvSchemaMixin.__table_args__,
    )

    version: Mapped[int] = mapped_column(
        Integer, ForeignKey("lv.topology_version.version", ondelete="CASCADE"), nullable=False
    )
    cable_id: Mapped[int] = mapped_column(BigInteger, nullable=False)

    cable_type: Mapped[str | None] = mapped_column(Text)
    cable_length_m: Mapped[float | None] = mapped_column(Numeric(asdecimal=False))
    phase_size: Mapped[float | None] = mapped_column(Numeric(asdecimal=False))
    phase_material: Mapped[str | None] = mapped_column(Text)
    capacity_a: Mapped[int | None] = mapped_column(Integer)
    resistance_ohm: Mapped[float | None] = mapped_column(Numeric(asdecimal=False))
    reactance_ohm: Mapped[float | None] = mapped_column(Numeric(asdecimal=False))
