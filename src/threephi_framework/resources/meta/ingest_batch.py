import uuid
from typing import Any

from sqlalchemy import select, update
from sqlalchemy.orm import Session

from threephi_framework.models.meta.ingest_batch import IngestBatchModel
from threephi_framework.resources.base import BaseResource


class IngestBatchResource(BaseResource):
    def __init__(self, s: Session):
        super().__init__(s)

    def create(self, source_file: str, run_id: str | None = None) -> IngestBatchModel:
        obj = IngestBatchModel(
            id=uuid.uuid4(),
            source_file=source_file,
            status="pending",
            run_id=run_id,
        )
        self.s.add(obj)
        self.s.flush()
        return obj

    def get(self, batch_id: uuid.UUID) -> IngestBatchModel | None:
        return self.s.get(IngestBatchModel, batch_id)

    def get_by_source_file(self, source_file: str, run_id: str | None = None) -> IngestBatchModel | None:
        stmt = select(IngestBatchModel).where(
            IngestBatchModel.source_file == source_file,
            IngestBatchModel.run_id == run_id,
        )
        return self.s.execute(stmt).scalar_one_or_none()

    def mark_processing(self, batch_id: uuid.UUID) -> None:
        self._set_status(batch_id, "processing")

    def mark_complete(self, batch_id: uuid.UUID, stats_json: dict[str, Any] | None = None) -> None:
        values: dict[str, Any] = {"status": "complete"}
        if stats_json is not None:
            values["stats_json"] = stats_json
        self.s.execute(update(IngestBatchModel).where(IngestBatchModel.id == batch_id).values(values))

    def mark_failed(self, batch_id: uuid.UUID, error_log: str) -> None:
        self.s.execute(
            update(IngestBatchModel).where(IngestBatchModel.id == batch_id).values(status="failed", error_log=error_log)
        )

    def _set_status(self, batch_id: uuid.UUID, status: str) -> None:
        self.s.execute(update(IngestBatchModel).where(IngestBatchModel.id == batch_id).values(status=status))
