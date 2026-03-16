from sqlalchemy import BigInteger, Enum, Integer
from sqlalchemy.orm import Mapped, mapped_column

from threephi_framework.models.base import BaseModel


class NodeCurrentModel(BaseModel):  # or use your Base
    __tablename__ = "node_current"
    __table_args__ = {"schema": "lv"}

    version: Mapped[int] = mapped_column(Integer, primary_key=True)
    id: Mapped[int] = mapped_column(BigInteger, primary_key=True)
    node_type: Mapped[str] = mapped_column(Enum("LvFeeder", "Cabinet", "DeliveryPoint", name="node_type"))
    feeder_id: Mapped[int | None] = mapped_column(BigInteger)
    cabinet_id: Mapped[int | None] = mapped_column(BigInteger)
    delivery_point_id: Mapped[int | None] = mapped_column(BigInteger)


class EdgeCurrentModel(BaseModel):
    __tablename__ = "edge_current"
    __table_args__ = {"schema": "lv"}

    version: Mapped[int] = mapped_column(Integer, primary_key=True)
    id: Mapped[int] = mapped_column(BigInteger, primary_key=True)
    node1_id: Mapped[int] = mapped_column(BigInteger)
    node2_id: Mapped[int] = mapped_column(BigInteger)
