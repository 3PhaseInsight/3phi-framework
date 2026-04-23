import datetime
import uuid
from collections.abc import Callable

from sqlalchemy.orm import Session

from threephi_framework.resources.meta.file_index import FileIndexResource
from threephi_framework.resources.meta.ingest_batch import IngestBatchResource


class IngestionController:
    """
    Controller for managing ingestion state: batch lifecycle and file-index
    tracking.

    All mutating operations that span multiple tables (e.g.
    :meth:`promote_batch_to_ready`) are executed within a single transaction so
    the database is never left in a half-committed state.

    Args:
        session_factory: Factory returning new SQLAlchemy sessions.
    """

    def __init__(self, session_factory: Callable[[], Session]):
        self._sf = session_factory

    # ------------------------------------------------------------------ #
    # Batch lifecycle
    # ------------------------------------------------------------------ #

    def insert_batch(self, source_file: str, run_id: str) -> uuid.UUID:
        s = self._sf()
        batch = IngestBatchResource(s).create(source_file=source_file, run_id=run_id)
        s.commit()
        return batch.id

    def mark_batch_failed(self, batch_id: uuid.UUID, error_log: str) -> None:
        s = self._sf()
        IngestBatchResource(s).mark_failed(batch_id, error_log)
        s.commit()

    # ------------------------------------------------------------------ #
    # File index
    # ------------------------------------------------------------------ #

    def get_current_max_seq_for_ring(self, dt: datetime.date, shard: int) -> int:
        return FileIndexResource(self._sf()).get_max_seq(dt, shard)

    def upsert_file_index(
        self,
        *,
        s3_key: str,
        dt: datetime.date,
        shard: int,
        seq: int,
        ts_start: datetime.datetime,
        ts_end: datetime.datetime,
        rows: int,
        bytes: int,
        schema_version: str,
        status: str,
        batch_id: uuid.UUID | None,
        ingest_file: str,
        committed_at: datetime.datetime | None = None,
    ) -> uuid.UUID:
        s = self._sf()
        file_id = FileIndexResource(s).upsert(
            s3_key=s3_key,
            dt=dt,
            shard=shard,
            seq=seq,
            ts_start=ts_start,
            ts_end=ts_end,
            rows=rows,
            bytes=bytes,
            schema_version=schema_version,
            status=status,
            batch_id=batch_id,
            ingest_file=ingest_file,
            committed_at=committed_at,
        )
        s.commit()
        return file_id

    def promote_batch_to_ready(self, batch_id: uuid.UUID) -> None:
        """Mark all staged file-index entries as ready and the batch as complete, atomically."""
        s = self._sf()
        FileIndexResource(s).mark_ready(batch_id)
        IngestBatchResource(s).mark_complete(batch_id)
        s.commit()
