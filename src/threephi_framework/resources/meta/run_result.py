import uuid
from typing import Any

from sqlalchemy import desc, select
from sqlalchemy.orm import Session

from threephi_framework.models.meta.run_result import RunResultModel
from threephi_framework.resources.base import BaseResource


class RunResultResource(BaseResource):
    def __init__(self, s: Session):
        super().__init__(s)

    def insert(self, data: dict[str, Any]) -> RunResultModel:
        """
        Insert a new run_result row and commit immediately.

        Notes:
        - Generates a UUID if `id` is not provided.
        - Raises on constraint violations (FK, NOT NULL, etc.).
        """
        payload = dict(data)
        payload.setdefault("id", uuid.uuid4())

        obj = RunResultModel(**payload)
        self.s.add(obj)

        return obj

    def get(self, result_id: uuid.UUID) -> RunResultModel | None:
        return self.s.get(RunResultModel, result_id)

    def query(
        self,
        *,
        dag_id: str | None = None,
        run_id: str | None = None,
        meter_id: int | None = None,
        phase: str | None = None,
        label_type: str | None = None,
        label_value: str | None = None,
        source: str | None = None,
        topology_version: int | None,
        node_id: int | None,
        edge_id: int | None,
        cable_id: int | None,
        min_confidence: float | None = None,
        max_confidence: float | None = None,
        limit: int | None = None,
        offset: int | None = None,
        order_by=None,
        descending: bool = False,
    ) -> list[RunResultModel]:
        """
        Flexible query helper for run_result.

        `order_by` may be:
        - a RunResultModel column
        - a list/tuple of columns
        """
        stmt = select(RunResultModel)

        if dag_id is not None:
            stmt = stmt.where(RunResultModel.dag_id == dag_id)
        if run_id is not None:
            stmt = stmt.where(RunResultModel.run_id == run_id)
        if meter_id is not None:
            stmt = stmt.where(RunResultModel.meter_id == meter_id)
        if phase is not None:
            stmt = stmt.where(RunResultModel.phase == phase)
        if label_type is not None:
            stmt = stmt.where(RunResultModel.label_type == label_type)
        if label_value is not None:
            stmt = stmt.where(RunResultModel.label_value == label_value)
        if source is not None:
            stmt = stmt.where(RunResultModel.source == source)
        if topology_version is not None:
            stmt = stmt.where(RunResultModel.topology_version == topology_version)
        if node_id is not None:
            stmt = stmt.where(RunResultModel.node_id == node_id)
        if edge_id is not None:
            stmt = stmt.where(RunResultModel.edge_id == edge_id)
        if cable_id is not None:
            stmt = stmt.where(RunResultModel.cable_id == cable_id)

        if min_confidence is not None:
            stmt = stmt.where(RunResultModel.confidence >= min_confidence)
        if max_confidence is not None:
            stmt = stmt.where(RunResultModel.confidence <= max_confidence)

        if order_by is not None:
            cols = list(order_by) if isinstance(order_by, list | tuple) else [order_by]

            if descending:
                cols = [c.desc() for c in cols]

            stmt = stmt.order_by(*cols)

        if offset is not None:
            stmt = stmt.offset(offset)
        if limit is not None:
            stmt = stmt.limit(limit)

        return list(self.s.execute(stmt).scalars().all())

    def get_latest_for_meter(
        self,
        *,
        dag_id: str,
        meter_id: int,
    ) -> list[RunResultModel]:
        """
        Return all run_result rows belonging to the most recent run_id that
        produced a row for (dag_id, meter_id), ordered by created_at DESC.

        Returns an empty list if no rows exist for that pair.

        "Most recent" is determined by RunResultModel.created_at (added in
        migration 09_run_result_created_at).
        """
        latest_run_id_stmt = (
            select(RunResultModel.run_id)
            .where(
                RunResultModel.dag_id == dag_id,
                RunResultModel.meter_id == meter_id,
            )
            .order_by(desc(RunResultModel.created_at))
            .limit(1)
        )
        latest_run_id = self.s.execute(latest_run_id_stmt).scalar_one_or_none()
        if latest_run_id is None:
            return []

        rows_stmt = (
            select(RunResultModel)
            .where(
                RunResultModel.dag_id == dag_id,
                RunResultModel.meter_id == meter_id,
                RunResultModel.run_id == latest_run_id,
            )
            .order_by(desc(RunResultModel.created_at))
        )
        return list(self.s.execute(rows_stmt).scalars().all())
