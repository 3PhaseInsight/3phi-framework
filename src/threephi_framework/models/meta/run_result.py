import uuid
from typing import Any

from sqlalchemy import BigInteger, Float, Integer, Text
from sqlalchemy.dialects.postgresql import ENUM, JSONB, UUID
from sqlalchemy.orm import Mapped, mapped_column, relationship

from threephi_framework.models.base import BaseModel
from threephi_framework.models.meta.meta_schema_mixin import MetaSchemaMixin
from threephi_framework.models.meta.meter import MetaMeterModel

ResultPhaseEnum = ENUM(
    "L1",
    "L2",
    "L3",
    "L1,L2",
    "L1,L3",
    "L2,L3",
    "all",
    name="result_phase",
    schema="public",
    create_type=False,
)


class RunResultModel(MetaSchemaMixin, BaseModel):
    __tablename__ = "run_result"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        primary_key=True,
    )

    dag_id: Mapped[str] = mapped_column(Text, nullable=False)
    run_id: Mapped[str] = mapped_column(Text, nullable=False)

    meter_id: Mapped[int] = mapped_column(
        BigInteger,
        nullable=True,
        index=True,
    )

    # Renamed relationship
    meter_fk: Mapped["MetaMeterModel"] = relationship(
        "MetaMeterModel",
        primaryjoin="RunResultModel.meter_id == MetaMeterModel.id",
        foreign_keys="RunResultModel.meter_id",
        lazy="selectin",
    )

    phase: Mapped[str] = mapped_column(ResultPhaseEnum, nullable=False)

    label_type: Mapped[str] = mapped_column(Text, nullable=False)
    label_value: Mapped[str] = mapped_column(Text, nullable=False)

    confidence: Mapped[float] = mapped_column(Float, nullable=False)

    source: Mapped[str] = mapped_column(Text, nullable=False)

    result: Mapped[dict[str, Any] | None] = mapped_column(JSONB, nullable=True)

    topology_version: Mapped[int | None] = mapped_column(Integer, nullable=True)

    node_id: Mapped[int | None] = mapped_column(BigInteger, nullable=True)
    edge_id: Mapped[int | None] = mapped_column(BigInteger, nullable=True)
    cable_id: Mapped[int | None] = mapped_column(BigInteger, nullable=True)
