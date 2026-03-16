from sqlalchemy import BigInteger, Integer, text
from sqlalchemy.dialects.postgresql import insert

from threephi_framework.models.topology.assets.transformer import TransformerModel
from threephi_framework.resources.base import BaseResource


class TransformerResource(BaseResource):
    def bulk_upsert_from_staging(self) -> None:
        select_stmt = text(r"""
          SELECT DISTINCT
              regexp_replace(transformer,'^\D+\.','')::bigint          AS id,
              regexp_replace(secondary_substation,'^\D+\.','')::bigint AS substation_id,
              NULLIF(transformer_capacity,0)::int                      AS capacity_kva
          FROM st_lv_topology
          WHERE COALESCE(transformer,'') <> ''
        """).columns(id=BigInteger, substation_id=BigInteger, capacity_kva=Integer)

        stmt = insert(TransformerModel).from_select(["id", "substation_id", "capacity_kva"], select_stmt)
        stmt = stmt.on_conflict_do_update(
            index_elements=[TransformerModel.id],
            set_={
                "substation_id": stmt.excluded.substation_id,
                "capacity_kva": stmt.excluded.capacity_kva,
            },
        )
        self.s.execute(stmt)
