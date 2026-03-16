from sqlalchemy import BigInteger, Integer, text
from sqlalchemy.dialects.postgresql import insert

from threephi_framework.models.topology.assets.feeder import FeederModel
from threephi_framework.resources.base import BaseResource


class FeederResource(BaseResource):
    def bulk_upsert_from_staging(self) -> None:
        select_stmt = text(r"""
          SELECT DISTINCT
              regexp_replace(lv_feeder,'^\D+\.','')::bigint AS id,
              regexp_replace(transformer,'^\D+\.','')::bigint AS transformer_id,
              NULLIF(lv_feeder_fuse_size,0)::int AS fuse_size_amps
          FROM st_lv_topology
          WHERE COALESCE(lv_feeder,'') <> '' AND COALESCE(transformer,'') <> ''
        """).columns(id=BigInteger, transformer_id=BigInteger, fuse_size_amps=Integer)

        stmt = insert(FeederModel).from_select(["id", "transformer_id", "fuse_size_amps"], select_stmt)
        stmt = stmt.on_conflict_do_update(
            index_elements=[FeederModel.id],
            set_={
                "transformer_id": stmt.excluded.transformer_id,
                "fuse_size_amps": stmt.excluded.fuse_size_amps,
            },
        )
        self.s.execute(stmt)
