from sqlalchemy import delete
from sqlalchemy.dialects.postgresql import insert
from sqlalchemy.orm import Session

from threephi_framework.models.meta.sm_phase_mapping import MetaPhaseMappingModel
from threephi_framework.resources.base import BaseResource


class MetaPhaseMappingResource(BaseResource):
    def __init__(self, s: Session):
        super().__init__(s)

    def upsert_many(self, rows: list[dict]):
        if not rows:
            return None

        stmt = insert(MetaPhaseMappingModel).values(rows)

        update_cols = {
            "feeder_phase": stmt.excluded.feeder_phase,
            "trafo_phase": stmt.excluded.trafo_phase,
            "true_feeder_id": stmt.excluded.true_feeder_id,
            "true_trafo_id": stmt.excluded.true_trafo_id,
            "likely_cabinet_id": stmt.excluded.likely_cabinet_id,
        }

        stmt = stmt.on_conflict_do_update(
            index_elements=[
                MetaPhaseMappingModel.meter_id,
                MetaPhaseMappingModel.sm_phase,
            ],
            set_=update_cols,
        )

        return self.s.execute(stmt)

    def delete_for_meter_ids(self, meter_ids: list[int]):
        if not meter_ids:
            return None

        stmt = delete(MetaPhaseMappingModel).where(MetaPhaseMappingModel.meter_id.in_(meter_ids))

        return self.s.execute(stmt)
