from sqlalchemy import BigInteger, ForeignKey, Integer
from sqlalchemy.orm import Mapped, mapped_column

from threephi_framework.models.base import BaseModel
from threephi_framework.models.topology.lv_schema_mixin import LvSchemaMixin


class FeederModel(LvSchemaMixin, BaseModel):
    __tablename__ = "feeder"

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True)
    transformer_id: Mapped[int] = mapped_column(
        BigInteger,
        ForeignKey("lv.transformer.id", ondelete="CASCADE"),
        nullable=False,
    )
    fuse_size_amps: Mapped[int | None] = mapped_column(Integer)

    def __repr__(self) -> str:
        return f"Feeder(id={self.id!r}, fuse_size_amps={self.fuse_size_amps!r})"
