from sqlalchemy import BigInteger, ForeignKey, ForeignKeyConstraint, Integer, PrimaryKeyConstraint
from sqlalchemy.orm import Mapped, mapped_column

from threephi_framework.models.base import BaseModel
from threephi_framework.models.topology.lv_schema_mixin import LvSchemaMixin


class EdgeCableModel(LvSchemaMixin, BaseModel):
    __tablename__ = "edge_cable"
    __table_args__ = (
        PrimaryKeyConstraint("version", "edge_id", "cable_id", name="pk_lv_edge_cable"),
        ForeignKeyConstraint(["version", "edge_id"], ["lv.edge.version", "lv.edge.id"], ondelete="CASCADE"),
        ForeignKeyConstraint(["version", "cable_id"], ["lv.cable.version", "lv.cable.cable_id"], ondelete="CASCADE"),
        LvSchemaMixin.__table_args__,
    )

    version: Mapped[int] = mapped_column(
        Integer, ForeignKey("lv.topology_version.version", ondelete="CASCADE"), nullable=False
    )
    edge_id: Mapped[int] = mapped_column(BigInteger, nullable=False)
    cable_id: Mapped[int] = mapped_column(BigInteger, nullable=False)
    seq_no: Mapped[int] = mapped_column(Integer, nullable=False, default=1)
