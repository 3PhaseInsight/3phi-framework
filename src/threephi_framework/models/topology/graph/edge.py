from sqlalchemy import (
    BigInteger,
    ForeignKey,
    ForeignKeyConstraint,
    Identity,
    Integer,
    PrimaryKeyConstraint,
)
from sqlalchemy.orm import Mapped, mapped_column

from threephi_framework.models.base import BaseModel
from threephi_framework.models.topology.lv_schema_mixin import LvSchemaMixin


class EdgeModel(LvSchemaMixin, BaseModel):
    __tablename__ = "edge"
    __table_args__ = (
        PrimaryKeyConstraint("version", "id", name="pk_lv_edge"),
        ForeignKeyConstraint(["version", "node1_id"], ["lv.node.version", "lv.node.id"], ondelete="CASCADE"),
        ForeignKeyConstraint(["version", "node2_id"], ["lv.node.version", "lv.node.id"], ondelete="CASCADE"),
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
    node1_id: Mapped[int] = mapped_column(BigInteger, nullable=False)
    node2_id: Mapped[int] = mapped_column(BigInteger, nullable=False)

    __mapper_args__ = {"eager_defaults": True}

    def __repr__(self) -> str:
        return (
            f"Edge(version={self.version!r}, edge_id={self.id!r}, "
            f"node1_id={self.node1_id!r}, node2_id={self.node2_id!r})"
        )
