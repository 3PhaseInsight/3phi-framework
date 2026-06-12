import os

from sqlalchemy import BigInteger, ForeignKey
from sqlalchemy.dialects.postgresql import ENUM
from sqlalchemy.orm import Mapped, mapped_column

from threephi_framework.models.base import BaseModel
from threephi_framework.models.meta.meta_schema_mixin import MetaSchemaMixin

_META_SCHEMA = os.getenv("META_SCHEMA", "meta")


class MetaPhaseMappingModel(MetaSchemaMixin, BaseModel):
    __tablename__ = "sm_phase_mapping"

    meter_id: Mapped[int] = mapped_column(
        BigInteger,
        ForeignKey(f"{_META_SCHEMA}.meter.id", ondelete="CASCADE"),
        primary_key=True,
    )

    sm_phase: Mapped[str] = mapped_column(
        ENUM("L1", "L2", "L3", name="phase", schema=_META_SCHEMA, create_type=False),
        primary_key=True,
    )

    feeder_phase: Mapped[str | None] = mapped_column(
        ENUM("L1", "L2", "L3", name="phase", schema=_META_SCHEMA, create_type=False),
        nullable=True,
    )

    trafo_phase: Mapped[str | None] = mapped_column(
        ENUM("L1", "L2", "L3", name="phase", schema=_META_SCHEMA, create_type=False),
        nullable=True,
    )

    true_feeder_id: Mapped[int | None] = mapped_column(
        BigInteger,
        ForeignKey("lv.feeder.id"),
        nullable=True,
    )

    true_trafo_id: Mapped[int | None] = mapped_column(
        BigInteger,
        ForeignKey("lv.transformer.id"),
        nullable=True,
    )

    likely_cabinet_id: Mapped[int | None] = mapped_column(
        BigInteger,
        ForeignKey("lv.cabinet.id"),
        nullable=True,
    )
