# resources/assets/delivery_point.py
from sqlalchemy import BigInteger, Integer, text
from sqlalchemy.dialects.postgresql import insert
from sqlalchemy.orm import Session

from threephi_framework.models.topology.assets.delivery_point import DeliveryPointModel


class DeliveryPointResource:
    def __init__(self, s: Session):
        self.s = s

    def bulk_upsert_from_staging(self) -> None:
        select_stmt = text(r"""
            SELECT DISTINCT
                s.delivery_point_id::bigint                                           AS id,
                CASE
                    WHEN COALESCE(s.cabinet,'') <> ''
                        THEN regexp_replace(s.cabinet,'^\D+\.','')::bigint
                    ELSE NULL
                END                                                                  AS cabinet_id,
                NULLIF(s.service_fuse_size,0)::int                                   AS service_fuse_size_amps
            FROM st_sm_cabinet s
            WHERE s.delivery_point_id IS NOT NULL
        """).columns(
            id=BigInteger,
            cabinet_id=BigInteger,
            service_fuse_size_amps=Integer,
        )

        stmt = insert(DeliveryPointModel).from_select(
            ["id", "cabinet_id", "service_fuse_size_amps"],
            select_stmt,
        )
        stmt = stmt.on_conflict_do_update(
            index_elements=[DeliveryPointModel.id],
            set_={
                "cabinet_id": stmt.excluded.cabinet_id,
                "service_fuse_size_amps": stmt.excluded.service_fuse_size_amps,
            },
        )

        self.s.execute(stmt)
