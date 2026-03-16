from sqlalchemy import BigInteger
from sqlalchemy.orm import Mapped, mapped_column

from threephi_framework.models.base import BaseModel
from threephi_framework.models.topology.lv_schema_mixin import LvSchemaMixin


class CabinetModel(LvSchemaMixin, BaseModel):
    __tablename__ = "cabinet"

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True)

    def __repr__(self) -> str:
        return f"Cabinet(id={self.id!r})"
