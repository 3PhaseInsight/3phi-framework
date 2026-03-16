from sqlalchemy import BigInteger, text
from sqlalchemy.dialects.postgresql import insert

from threephi_framework.models.topology.assets.cabinet import CabinetModel
from threephi_framework.resources.base import BaseResource


class CabinetResource(BaseResource):
    def bulk_upsert_from_staging(self) -> None:
        select_stmt = text(r"""
          SELECT DISTINCT regexp_replace(cabinet,'^\D+\.','')::bigint AS id
          FROM st_sm_cabinet WHERE COALESCE(cabinet,'') <> ''
        """).columns(id=BigInteger)

        stmt = insert(CabinetModel).from_select(["id"], select_stmt)
        stmt = stmt.on_conflict_do_nothing(index_elements=[CabinetModel.id])
        self.s.execute(stmt)
