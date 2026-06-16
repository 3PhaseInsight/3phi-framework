from sqlalchemy import BigInteger, ForeignKey, Identity, Integer, PrimaryKeyConstraint
from sqlalchemy.orm import Mapped, mapped_column

from threephi_framework.models.base import BaseModel
from threephi_framework.models.topology.lv_schema_mixin import LvSchemaMixin
from threephi_framework.models.topology.node_type_enum import NodeTypeEnum


class NodeModel(LvSchemaMixin, BaseModel):
    __tablename__ = "node"
    __table_args__ = (
        PrimaryKeyConstraint("version", "id", name="pk_lv_node"),
        LvSchemaMixin.__table_args__,
    )
    version: Mapped[int] = mapped_column(
        Integer,
        ForeignKey("lv.topology_version.version", ondelete="CASCADE"),
        nullable=False,
    )
    id: Mapped[int] = mapped_column(
        BigInteger,
        Identity(always=False),
        nullable=False,
    )
    node_type: Mapped[str] = mapped_column(NodeTypeEnum, nullable=False)
    feeder_id: Mapped[int | None] = mapped_column(
        BigInteger, ForeignKey("lv.feeder.id", ondelete="CASCADE"), nullable=True
    )
    cabinet_id: Mapped[int | None] = mapped_column(
        BigInteger, ForeignKey("lv.cabinet.id", ondelete="CASCADE"), nullable=True
    )
    delivery_point_id: Mapped[int | None] = mapped_column(
        BigInteger, ForeignKey("lv.delivery_point.id", ondelete="CASCADE"), nullable=True
    )

    __mapper_args__ = {"eager_defaults": True}

    def __repr__(self) -> str:
        return (
            f"Node(version={self.version!r}, node_id={self.id!r}, "
            f"node_type={self.node_type!r}, feeder_id={self.feeder_id!r}, "
            f"cabinet_id={self.cabinet_id!r}, delivery_point_id={self.delivery_point_id!r})"
        )
