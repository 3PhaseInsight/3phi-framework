from sqlalchemy import BigInteger, Integer, text
from sqlalchemy.dialects.postgresql import insert

from threephi_framework.models.topology.assets.secondary_substation import SecondarySubstationModel
from threephi_framework.resources.base import BaseResource


class SecondarySubstationResource(BaseResource):
    def bulk_upsert_from_staging(self) -> None:
        select_stmt = text(r"""
          SELECT DISTINCT
            regexp_replace(secondary_substation,'^\D+\.','')::bigint AS id,
            NULLIF(zip_code_secondary_substation,'')::int            AS zip_code
          FROM st_lv_topology
          WHERE COALESCE(secondary_substation,'') <> ''
        """).columns(id=BigInteger, zip_code=Integer)

        stmt = insert(SecondarySubstationModel).from_select(["id", "zip_code"], select_stmt)
        stmt = stmt.on_conflict_do_update(
            index_elements=[SecondarySubstationModel.id],
            set_={"zip_code": stmt.excluded.zip_code},
        )

        self.s.execute(stmt)
