import datetime
import uuid

from sqlalchemy import func, select, update
from sqlalchemy.dialects.postgresql import insert
from sqlalchemy.orm import Session

from threephi_framework.models.meta.file_index import FileIndexModel
from threephi_framework.resources.base import BaseResource


class FileIndexResource(BaseResource):
    def __init__(self, s: Session):
        super().__init__(s)

    def upsert(
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
        stmt = insert(FileIndexModel).values(
            id=uuid.uuid4(),
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
        stmt = stmt.on_conflict_do_update(
            index_elements=[FileIndexModel.dt, FileIndexModel.shard, FileIndexModel.seq],
            set_={
                "s3_key": stmt.excluded.s3_key,
                "ts_start": stmt.excluded.ts_start,
                "ts_end": stmt.excluded.ts_end,
                "rows": stmt.excluded.rows,
                "bytes": stmt.excluded.bytes,
                "schema_version": stmt.excluded.schema_version,
                "status": stmt.excluded.status,
                "batch_id": stmt.excluded.batch_id,
                "ingest_file": stmt.excluded.ingest_file,
                # preserve existing committed_at if incoming value is NULL
                "committed_at": func.coalesce(stmt.excluded.committed_at, FileIndexModel.committed_at),
            },
        ).returning(FileIndexModel.id)
        result = self.s.execute(stmt)
        return result.scalar_one()

    def get_max_seq(self, dt: datetime.date, shard: int) -> int:
        stmt = (
            select(func.coalesce(func.max(FileIndexModel.seq), 0))
            .where(FileIndexModel.dt == dt)
            .where(FileIndexModel.shard == shard)
        )
        return self.s.execute(stmt).scalar_one()

    def mark_ready(self, batch_id: uuid.UUID) -> None:
        self.s.execute(
            update(FileIndexModel)
            .where(FileIndexModel.batch_id == batch_id)
            .where(FileIndexModel.status == "staged")
            .values(status="ready", committed_at=func.now())
        )
