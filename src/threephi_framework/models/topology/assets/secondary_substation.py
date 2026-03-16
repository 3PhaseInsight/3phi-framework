from sqlalchemy import BigInteger, Integer
from sqlalchemy.orm import Mapped, mapped_column, relationship

from threephi_framework.models.base import BaseModel
from threephi_framework.models.topology.assets.transformer import TransformerModel
from threephi_framework.models.topology.lv_schema_mixin import LvSchemaMixin


class SecondarySubstationModel(LvSchemaMixin, BaseModel):
    __tablename__ = "secondary_substation"

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True)
    zip_code: Mapped[int | None] = mapped_column(Integer)
    transformers: Mapped[list[TransformerModel]] = relationship(lazy="selectin")

    def __repr__(self) -> str:
        return f"SecondarySubstation(id={self.id!r}, zip_code={self.zip_code!r})"
