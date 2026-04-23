import pandas as pd
from sqlalchemy import func, select, update
from sqlalchemy.dialects.postgresql import insert
from sqlalchemy.orm import Session

from threephi_framework.models.meta.meter import MetaMeterModel
from threephi_framework.resources.base import BaseResource


class MetaMeterResource(BaseResource):
    def __init__(self, s: Session):
        super().__init__(s)

    def update(self, meter_id: int, data: dict):
        stmt = update(MetaMeterModel).where(MetaMeterModel.id == meter_id).values(data)
        result = self.s.execute(stmt)
        self.s.commit()
        return result

    def get(self, meter_id: int) -> MetaMeterModel | None:
        return self.s.get(MetaMeterModel, meter_id)

    def get_max_total_rows(self) -> int:
        stmt = select(func.max(MetaMeterModel.total_rows))
        return self.s.execute(stmt).scalar_one()

    def upsert_meter_stats(self, df: pd.DataFrame) -> None:
        """
        Upsert meter inventory stats from a DataFrame with columns:
        id, first_seen, last_seen, total_rows.

        On conflict: keeps the earliest first_seen, the latest last_seen,
        accumulates total_rows, and refreshes updated_at.
        """
        rows = df.to_dict(orient="records")
        if not rows:
            return
        stmt = insert(MetaMeterModel).values(rows)
        stmt = stmt.on_conflict_do_update(
            index_elements=[MetaMeterModel.id],
            set_={
                "first_seen": func.least(MetaMeterModel.first_seen, stmt.excluded.first_seen),
                "last_seen": func.greatest(MetaMeterModel.last_seen, stmt.excluded.last_seen),
                "total_rows": MetaMeterModel.total_rows + stmt.excluded.total_rows,
                "updated_at": func.now(),
            },
        )
        self.s.execute(stmt)
        self.s.commit()

    def get_timeseries_info(self) -> tuple:
        # min(first_seen), max(last_seen)
        min_max_stmt = select(
            func.min(MetaMeterModel.first_seen),
            func.max(MetaMeterModel.last_seen),
        )

        # distinct meter ids
        meter_ids_stmt = select(MetaMeterModel.id).where(MetaMeterModel.total_rows > 0).distinct()

        min_ts, max_ts = self.s.execute(min_max_stmt).one()
        meter_ids = self.s.execute(meter_ids_stmt).scalars().all()

        return min_ts, max_ts, meter_ids
