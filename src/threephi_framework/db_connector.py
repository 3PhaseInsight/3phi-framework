import logging
import uuid
from contextlib import contextmanager

import sqlalchemy
from sqlalchemy import MetaData, Table, func, select, text, update
from sqlalchemy.dialects.postgresql import insert
from sqlalchemy.engine import Result
from sqlalchemy.exc import OperationalError

from threephi_framework.db import db

DASK_NPARTITIONS = 20  # scale this up when running on full dataset and beefier machine

logger = logging.getLogger(__name__)


class DBConnector:
    def __init__(self):
        self.dsn = None
        self.engine: sqlalchemy.engine.Engine = self._create_engine()
        self.metadata = MetaData()

    @contextmanager
    def transaction(self):
        with self.engine.begin() as conn:
            yield conn  # a transactional Connection

    @property
    def url_str(self) -> str:
        # for engine creation inside dask workers
        return self.engine.url.render_as_string(hide_password=False)

    def _define_tables(self):
        self.workflow_states = Table("workflow_states", self.metadata, autoload_with=self.engine)
        self.ingest_batch = Table("ingest_batch", self.metadata, autoload_with=self.engine)
        self.file_index = Table("file_index", self.metadata, autoload_with=self.engine)
        self.meter = Table("meter", self.metadata, autoload_with=self.engine)

    def _create_engine(self) -> sqlalchemy.engine.Engine:
        return db.get_engine()

    def check_db_connection(self):
        try:
            with self.engine.begin() as conn:
                result = conn.execute(select(1)).scalar()
                return result == 1
        except OperationalError as e:
            print(f"Database connection failed: {e}")
            return False

    def start_workflow(self, workflow, description=None):
        logger.info(f"Starting workflow: {workflow}")
        stmt = (
            insert(self.workflow_states)
            .values(workflow=workflow, completed=False, description=description)
            .on_conflict_do_nothing(index_elements=["workflow"])
        )

        with self.engine.begin() as conn:
            conn.execute(stmt)

    def complete_workflow(self, workflow):
        logger.info(f"Completing workflow: {workflow}")
        stmt = update(self.workflow_states).where(self.workflow_states.c.workflow == workflow).values(completed=True)

        with self.engine.begin() as conn:
            conn.execute(stmt)

    def is_workflow_completed(self, workflow) -> bool:
        stmt = select(self.workflow_states.c.completed).where(self.workflow_states.c.workflow == workflow)
        with self.engine.begin() as conn:
            result = conn.execute(stmt).scalar_one_or_none()
            if result is None:
                return False
            return bool(result)

    def insert_batch(self, sourcefile_name: str, run_id: str) -> uuid.UUID:
        new_batch_id = str(uuid.uuid4())
        query = (
            insert(self.ingest_batch)
            .values(id=new_batch_id, source_file=sourcefile_name, status="in_progress", run_id=run_id)
            .returning(self.ingest_batch.c.id)
        )
        with self.engine.begin() as conn:
            id = conn.execute(query).scalar_one()
        return id

    def get_current_max_seq_for_ring(self, dt: str, shard: int) -> int:
        with self.engine.begin() as conn:
            max_seq = conn.execute(
                select(func.coalesce(func.max(self.file_index.c.seq), 0))
                .where(self.file_index.c.dt == dt)
                .where(self.file_index.c.shard == shard)
            ).scalar_one()
        return max_seq

    def upsert_file_index(
            self,
            s3_key: str,
            dt: str,
            shard: int,
            seq: int,
            ts_start: str,
            ts_end: str,
            rows: int,
            bytes: int,
            schema_version: str,
            status: str,
            batch_id: uuid.UUID,
            ingest_file: str,
            committed_at: str | None = None,
    ) -> uuid.UUID:
        new_file_id = str(uuid.uuid4())
        query = insert(self.file_index).values(
            id=new_file_id,
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

        query = query.on_conflict_do_update(
            index_elements=["dt", "shard", "seq"],
            set_={
                "s3_key": query.excluded.s3_key,
                "ts_start": query.excluded.ts_start,
                "ts_end": query.excluded.ts_end,
                "rows": query.excluded.rows,
                "bytes": query.excluded.bytes,
                "schema_version": query.excluded.schema_version,
                "status": query.excluded.status,
                "batch_id": query.excluded.batch_id,
                "ingest_file": query.excluded.ingest_file,
                # keep existing committed_at if incoming is NULL
                "committed_at": func.coalesce(query.excluded.committed_at, self.file_index.c.committed_at),
            },
        ).returning(self.file_index.c.id)
        with self.engine.begin() as conn:
            id = conn.execute(query).scalar_one()
        return id

    def promote_batch_to_ready(self, batch_id: uuid.UUID):
        """
        Update status of rows in file_index, set committed_at timestamp,
        mark batch as processed
        :param batch_id: UUID of the batch
        :return:
        """
        file_index_query = (
            update(self.file_index)
            .where(self.file_index.c.id == batch_id)
            .where(self.file_index.c.status == "staged")
            .values(status="ready", committed_at=func.now())
        )
        batch_query = update(self.ingest_batch).where(self.ingest_batch.c.id == batch_id).values(status="processed")
        with self.engine.begin() as conn:
            conn.execute(file_index_query)
            conn.execute(batch_query)

    def upsert_meter_stats(self, agg_df):
        """
        Upserting meter stats into meter inventory table
        :param agg_df: a pandas dataframe with the columns: id, first_seen, last_seen, total_rows
        :return:
        """
        rows = agg_df.to_dict(orient="records")
        with self.engine.begin() as conn:
            stmt = insert(self.meter).values(rows)
            stmt = stmt.on_conflict_do_update(
                index_elements=["id"],
                set_={
                    "first_seen": func.least(self.meter.c.first_seen, stmt.excluded.first_seen),
                    "last_seen": func.greatest(self.meter.c.last_seen, stmt.excluded.last_seen),
                    "total_rows": self.meter.c.total_rows + stmt.excluded.total_rows,
                    "updated_at": func.now(),
                },
            )
            conn.execute(stmt)

    def get_meters_for_substation(self, substation_id: int) -> list[dict]:
        sql = text("""
            WITH RECURSIVE
                cur AS (
                  SELECT version
                  FROM lv.topology_version
                  WHERE is_current
                ),
                seed AS (
                  SELECT n.version, n.node_id
                  FROM lv.lv_node n
                  JOIN cur              ON cur.version = n.version
                  JOIN lv.lv_feeder f   ON n.feeder_id = f.feeder_id
                  JOIN lv.transformer t ON f.transformer_id = t.transformer_id
                  WHERE n.node_type = 'LvFeeder'
                    AND t.substation_id = :substation_id
                ),
                walk(version, node_id, path) AS (
                  -- ANCHOR
                  SELECT s.version, s.node_id, ARRAY[s.node_id]::bigint[]
                  FROM seed s

                  UNION ALL

                  -- Single recursive step (undirected)
                  SELECT
                    w.version,
                    CASE
                      WHEN e.node1_id = w.node_id THEN e.node2_id
                      ELSE e.node1_id
                    END AS next_node_id,
                    w.path || CASE
                      WHEN e.node1_id = w.node_id THEN e.node2_id
                      ELSE e.node1_id
                    END
                  FROM walk w
                  JOIN lv.lv_edge e
                    ON e.version = w.version
                   AND (e.node1_id = w.node_id OR e.node2_id = w.node_id)
                  WHERE NOT (
                    CASE
                      WHEN e.node1_id = w.node_id THEN e.node2_id
                      ELSE e.node1_id
                    END = ANY (w.path)
                  )
                )
                SELECT DISTINCT m.*
                FROM walk w
                JOIN lv.lv_node n
                  ON n.version = w.version
                 AND n.node_id = w.node_id
                 AND n.node_type = 'Cabinet'
                JOIN lv.delivery_point dp
                  ON dp.cabinet_id = n.cabinet_id
                JOIN lv.meter m
                  ON m.delivery_point_id = dp.delivery_point_id
                ORDER BY m.meter_id;
        """)

        params = {"substation_id": int(substation_id)}

        with self.engine.begin() as conn:
            result: Result = conn.execute(sql, params)
            rows = [dict(r._mapping) for r in result.fetchall()]  # fetch while conn is open
        return rows