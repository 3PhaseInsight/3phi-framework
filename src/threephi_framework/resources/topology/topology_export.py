from __future__ import annotations

from typing import TYPE_CHECKING

import pandas as pd
from sqlalchemy import text

from threephi_framework.resources.base import BaseResource
from threephi_framework.schemas.v1.topology import (
    lv_topology_dtype,
    sm_cabinet_dtype,
)

if TYPE_CHECKING:
    from sqlalchemy.orm import Session


class TopologyExportResource(BaseResource):
    """
    Resource that knows how to 'reverse' the ingest and reconstruct
    the topology and sm_cabinet CSV-like dataframes from the DB.
    """

    def __init__(self, session: Session):
        super().__init__(session)

    # ------------------------------------------------------------------
    # 1) Export topology (lv_topology_* schema)
    # ------------------------------------------------------------------
    def get_topology_pdf(self) -> pd.DataFrame:
        """
        Reconstruct the *current* topology in the lv_topology_* schema.

        Columns (matching lv_topology_dtype):
            secondary_substation
            zip_code_secondary_substation
            transformer
            transformer_capacity
            lv_feeder
            lv_feeder_fuse_size
            node1
            node2
            cable_id
            cable_type
            cable_length
            phase_size
            phase_material
            cable_capacity
            resistance
            reactance
        """

        sql = text(
            r"""
            WITH RECURSIVE
                -- Seed: all feeder nodes in the current topology, with their feeder_id
                seed AS (
                    SELECT
                    n.id        AS node_id,
                    n.feeder_id AS feeder_id,
                    ARRAY[n.id]::bigint[] AS path
                    FROM lv.node_current n
                    WHERE n.node_type = 'LvFeeder'
                ),
                -- Walk the undirected graph from all feeder nodes, carrying feeder_id
                walk AS (
                    -- anchor
                    SELECT node_id, feeder_id, path
                    FROM seed
                    UNION ALL
                    -- recursive step
                    SELECT
                    CASE WHEN e.node1_id = w.node_id
                        THEN e.node2_id
                        ELSE e.node1_id
                    END AS next_node_id,
                    w.feeder_id,
                    w.path || CASE WHEN e.node1_id = w.node_id
                                    THEN e.node2_id
                                    ELSE e.node1_id
                                END
                    FROM walk w
                    JOIN lv.edge_current e
                    ON e.node1_id = w.node_id OR e.node2_id = w.node_id
                    WHERE NOT (
                    CASE WHEN e.node1_id = w.node_id
                        THEN e.node2_id
                        ELSE e.node1_id
                    END = ANY (w.path)
                    )
                ),
                -- Final mapping: each node -> its feeder_id
                node_feeder AS (
                    SELECT DISTINCT node_id, feeder_id
                    FROM walk
                )

                SELECT
                    -- ===== hierarchy with prefixes (matching your CSV) =====
                    ('SecondarySubstation.' || ss.id::text)          AS secondary_substation,
                    ss.zip_code::text                                AS zip_code_secondary_substation,
                    ('Transformer.' || t.id::text)                   AS transformer,
                    t.capacity_kva::float8                           AS transformer_capacity,
                    ('LvFeeder.' || f.id::text)                      AS lv_feeder,
                    f.fuse_size_amps::float8                         AS lv_feeder_fuse_size,

                    -- ===== node1 / node2 with type prefixes =====
                    CASE
                        WHEN n1.node_type = 'LvFeeder' AND n1.feeder_id IS NOT NULL
                            THEN 'LvFeeder.' || n1.feeder_id::text
                        WHEN n1.node_type = 'Cabinet' AND n1.cabinet_id IS NOT NULL
                            THEN 'Cabinet.' || n1.cabinet_id::text
                        ELSE n1.node_type::text || '.' || n1.id::text
                    END AS node1,
                    CASE
                        WHEN n2.node_type = 'LvFeeder' AND n2.feeder_id IS NOT NULL
                            THEN 'LvFeeder.' || n2.feeder_id::text
                        WHEN n2.node_type = 'Cabinet' AND n2.cabinet_id IS NOT NULL
                            THEN 'Cabinet.' || n2.cabinet_id::text
                        ELSE n2.node_type::text || '.' || n2.id::text
                    END                                              AS node2,

                    -- ===== physical cable with LvCable.* ids =====
                    ('LvCable.' || c.cable_id::text)                 AS cable_id,
                    c.cable_type::text                               AS cable_type,
                    c.cable_length_m::float8                         AS cable_length,
                    c.phase_size::float8                             AS phase_size,
                    c.phase_material::text                           AS phase_material,
                    c.capacity_a::float8                             AS cable_capacity,
                    c.resistance_ohm::float8                         AS resistance,
                    c.reactance_ohm::float8                          AS reactance

                FROM lv.edge_current e
                JOIN lv.node_current n1
                ON n1.id = e.node1_id
                JOIN lv.node_current n2
                ON n2.id = e.node2_id

                LEFT JOIN node_feeder nf1 ON nf1.node_id = e.node1_id
                LEFT JOIN node_feeder nf2 ON nf2.node_id = e.node2_id
                LEFT JOIN lv.feeder f     ON f.id = COALESCE(nf1.feeder_id, nf2.feeder_id)
                LEFT JOIN lv.transformer t ON t.id = f.transformer_id
                LEFT JOIN lv.secondary_substation ss ON ss.id = t.substation_id


                -- one row per physical cable on each edge
                LEFT JOIN lv.edge_cable ec ON ec.edge_id = e.id
                LEFT JOIN lv.cable c
                ON c.cable_id = ec.cable_id
                AND c.version  = ec.version
                ;
            """
        )

        conn = self.s.connection()
        pdf = pd.read_sql(sql, conn)

        # Enforce the same column order and dtypes as when reading the CSV
        # (First ensure the columns are ordered like lv_topology_dtype)
        pdf = pdf[list(lv_topology_dtype.keys())]
        pdf = pdf.astype(lv_topology_dtype)
        # If you use lv_topology_types as a second stage (like in ingest), you can:
        # pdf = pdf.astype(lv_topology_types)

        return pdf

    # ------------------------------------------------------------------
    # 2) Export sm_cabinet (sm_cabinet_* schema)
    # ------------------------------------------------------------------
    def get_sm_cabinet_pdf(self) -> pd.DataFrame:
        """
        Reconstruct the smart-meter–cabinet mapping in sm_cabinet_* schema.

        Columns:
            meter_number
            delivery_point_id
            cabinet
            lv_feeder
            has_heat_pump
            has_solar_panel
            capacity_solar_panel
            service_fuse_size
        """

        sql = text(
            r"""
            WITH RECURSIVE
                -- Seed: start from all feeder nodes in the current topology
                seed AS (
                    SELECT
                    n.id        AS node_id,
                    n.feeder_id AS feeder_id,
                    ARRAY[n.id]::bigint[] AS path
                    FROM lv.node_current n
                    WHERE n.node_type = 'LvFeeder'
                ),
                -- Walk: undirected graph traversal from all feeders, carrying feeder_id
                walk AS (
                    -- Anchor: start from the feeder nodes themselves
                    SELECT node_id, feeder_id, path
                    FROM seed

                    UNION ALL

                    -- Recursive step: step from current node to its neighbors
                    SELECT
                    CASE WHEN e.node1_id = w.node_id
                        THEN e.node2_id
                        ELSE e.node1_id
                    END AS next_node_id,
                    w.feeder_id,
                    w.path || CASE WHEN e.node1_id = w.node_id
                                    THEN e.node2_id
                                    ELSE e.node1_id
                                END
                    FROM walk w
                    JOIN lv.edge_current e
                    ON e.node1_id = w.node_id OR e.node2_id = w.node_id
                    WHERE NOT (
                    CASE WHEN e.node1_id = w.node_id
                        THEN e.node2_id
                        ELSE e.node1_id
                    END = ANY (w.path)
                    )
                ),
                -- Node → feeder mapping
                node_feeder AS (
                    SELECT DISTINCT node_id, feeder_id
                    FROM walk
                ),
                -- Cabinet → feeder mapping (restrict to Cabinet nodes)
                cabinet_feeder AS (
                    SELECT DISTINCT
                    n.cabinet_id AS cabinet_id,
                    nf.feeder_id AS feeder_id
                    FROM lv.node_current n
                    JOIN node_feeder nf
                    ON nf.node_id = n.id
                    WHERE n.node_type = 'Cabinet'
                )

                SELECT
                -- numeric ids as float64-compatible
                m.id::float8                         AS meter_number,
                dp.id::float8                        AS delivery_point_id,

                -- Cabinet.* style like in the original CSV
                ('Cabinet.' || c.id::text)          AS cabinet,

                -- optional feeder label; can be left NULL if not needed
                CASE
                    WHEN cf.feeder_id IS NOT NULL
                        THEN 'LvFeeder.' || cf.feeder_id::text
                    ELSE NULL
                END                                  AS lv_feeder,

                m.has_heat_pump AS has_heat_pump,
                m.has_solar_panel AS has_solar_panel,
                m.solar_capacity_kw::float8          AS capacity_solar_panel,
                dp.service_fuse_size_amps::float8    AS service_fuse_size

                FROM lv.meter m
                JOIN lv.delivery_point dp
                ON dp.id = m.delivery_point_id
                JOIN lv.cabinet c
                ON c.id = dp.cabinet_id
                LEFT JOIN cabinet_feeder cf
                ON cf.cabinet_id = c.id
                ;

            """
        )

        conn = self.s.connection()
        pdf = pd.read_sql(sql, conn)

        # Order and dtypes like the CSV ingest
        pdf = pdf[list(sm_cabinet_dtype.keys())]
        pdf = pdf.astype(sm_cabinet_dtype)
        # If you cast to logical types via sm_cabinet_types in ingest, do the same here:
        # pdf = pdf.astype(sm_cabinet_types)

        return pdf

    # ------------------------------------------------------------------
    # 3) Export topology at an explicit version (lv_topology_* schema)
    # ------------------------------------------------------------------
    def get_topology_at_version(self, version: int) -> pd.DataFrame:
        """Same output schema as get_topology_pdf() but filtered to an explicit version."""
        sql = text(
            r"""
            WITH RECURSIVE
                seed AS (
                    SELECT
                    n.id        AS node_id,
                    n.feeder_id AS feeder_id,
                    ARRAY[n.id]::bigint[] AS path
                    FROM lv.node n
                    WHERE n.node_type = 'LvFeeder'
                      AND n.version = :version
                ),
                walk AS (
                    SELECT node_id, feeder_id, path
                    FROM seed
                    UNION ALL
                    SELECT
                    CASE WHEN e.node1_id = w.node_id
                        THEN e.node2_id
                        ELSE e.node1_id
                    END AS next_node_id,
                    w.feeder_id,
                    w.path || CASE WHEN e.node1_id = w.node_id
                                    THEN e.node2_id
                                    ELSE e.node1_id
                                END
                    FROM walk w
                    JOIN lv.edge e
                    ON (e.node1_id = w.node_id OR e.node2_id = w.node_id)
                       AND e.version = :version
                    WHERE NOT (
                    CASE WHEN e.node1_id = w.node_id
                        THEN e.node2_id
                        ELSE e.node1_id
                    END = ANY (w.path)
                    )
                ),
                node_feeder AS (
                    SELECT DISTINCT node_id, feeder_id
                    FROM walk
                )

                SELECT
                    ('SecondarySubstation.' || ss.id::text)          AS secondary_substation,
                    ss.zip_code::text                                AS zip_code_secondary_substation,
                    ('Transformer.' || t.id::text)                   AS transformer,
                    t.capacity_kva::float8                           AS transformer_capacity,
                    ('LvFeeder.' || f.id::text)                      AS lv_feeder,
                    f.fuse_size_amps::float8                         AS lv_feeder_fuse_size,

                    CASE
                        WHEN n1.node_type = 'LvFeeder' AND n1.feeder_id IS NOT NULL
                            THEN 'LvFeeder.' || n1.feeder_id::text
                        WHEN n1.node_type = 'Cabinet' AND n1.cabinet_id IS NOT NULL
                            THEN 'Cabinet.' || n1.cabinet_id::text
                        ELSE n1.node_type::text || '.' || n1.id::text
                    END AS node1,
                    CASE
                        WHEN n2.node_type = 'LvFeeder' AND n2.feeder_id IS NOT NULL
                            THEN 'LvFeeder.' || n2.feeder_id::text
                        WHEN n2.node_type = 'Cabinet' AND n2.cabinet_id IS NOT NULL
                            THEN 'Cabinet.' || n2.cabinet_id::text
                        ELSE n2.node_type::text || '.' || n2.id::text
                    END                                              AS node2,

                    ('LvCable.' || c.cable_id::text)                 AS cable_id,
                    c.cable_type::text                               AS cable_type,
                    c.cable_length_m::float8                         AS cable_length,
                    c.phase_size::float8                             AS phase_size,
                    c.phase_material::text                           AS phase_material,
                    c.capacity_a::float8                             AS cable_capacity,
                    c.resistance_ohm::float8                         AS resistance,
                    c.reactance_ohm::float8                          AS reactance

                FROM lv.edge e
                JOIN lv.node n1
                ON n1.id = e.node1_id AND n1.version = :version
                JOIN lv.node n2
                ON n2.id = e.node2_id AND n2.version = :version

                LEFT JOIN node_feeder nf1 ON nf1.node_id = e.node1_id
                LEFT JOIN node_feeder nf2 ON nf2.node_id = e.node2_id
                LEFT JOIN lv.feeder f     ON f.id = COALESCE(nf1.feeder_id, nf2.feeder_id)
                LEFT JOIN lv.transformer t ON t.id = f.transformer_id
                LEFT JOIN lv.secondary_substation ss ON ss.id = t.substation_id

                LEFT JOIN lv.edge_cable ec ON ec.edge_id = e.id AND ec.version = :version
                LEFT JOIN lv.cable c
                ON c.cable_id = ec.cable_id
                AND c.version = :version

                WHERE e.version = :version
                ;
            """
        ).bindparams(version=version)

        conn = self.s.connection()
        pdf = pd.read_sql(sql, conn)
        pdf = pdf[list(lv_topology_dtype.keys())]
        pdf = pdf.astype(lv_topology_dtype)
        return pdf

    # ------------------------------------------------------------------
    # 4) Export sm_cabinet at an explicit version (sm_cabinet_* schema)
    # ------------------------------------------------------------------
    def get_sm_cabinet_at_version(self, version: int) -> pd.DataFrame:
        """Same output schema as get_sm_cabinet_pdf() but filtered to an explicit version."""
        sql = text(
            r"""
            WITH RECURSIVE
                seed AS (
                    SELECT
                    n.id        AS node_id,
                    n.feeder_id AS feeder_id,
                    ARRAY[n.id]::bigint[] AS path
                    FROM lv.node n
                    WHERE n.node_type = 'LvFeeder'
                      AND n.version = :version
                ),
                walk AS (
                    SELECT node_id, feeder_id, path
                    FROM seed
                    UNION ALL
                    SELECT
                    CASE WHEN e.node1_id = w.node_id
                        THEN e.node2_id
                        ELSE e.node1_id
                    END AS next_node_id,
                    w.feeder_id,
                    w.path || CASE WHEN e.node1_id = w.node_id
                                    THEN e.node2_id
                                    ELSE e.node1_id
                                END
                    FROM walk w
                    JOIN lv.edge e
                    ON (e.node1_id = w.node_id OR e.node2_id = w.node_id)
                       AND e.version = :version
                    WHERE NOT (
                    CASE WHEN e.node1_id = w.node_id
                        THEN e.node2_id
                        ELSE e.node1_id
                    END = ANY (w.path)
                    )
                ),
                node_feeder AS (
                    SELECT DISTINCT node_id, feeder_id
                    FROM walk
                ),
                cabinet_feeder AS (
                    SELECT DISTINCT
                    n.cabinet_id AS cabinet_id,
                    nf.feeder_id AS feeder_id
                    FROM lv.node n
                    JOIN node_feeder nf
                    ON nf.node_id = n.id
                    WHERE n.node_type = 'Cabinet'
                      AND n.version = :version
                )

                SELECT
                m.id::float8                         AS meter_number,
                dp.id::float8                        AS delivery_point_id,
                ('Cabinet.' || c.id::text)          AS cabinet,
                CASE
                    WHEN cf.feeder_id IS NOT NULL
                        THEN 'LvFeeder.' || cf.feeder_id::text
                    ELSE NULL
                END                                  AS lv_feeder,
                m.has_heat_pump AS has_heat_pump,
                m.has_solar_panel AS has_solar_panel,
                m.solar_capacity_kw::float8          AS capacity_solar_panel,
                dp.service_fuse_size_amps::float8    AS service_fuse_size

                FROM lv.meter m
                JOIN lv.delivery_point dp
                ON dp.id = m.delivery_point_id
                JOIN lv.cabinet c
                ON c.id = dp.cabinet_id
                LEFT JOIN cabinet_feeder cf
                ON cf.cabinet_id = c.id
                ;
            """
        ).bindparams(version=version)

        conn = self.s.connection()
        pdf = pd.read_sql(sql, conn)
        pdf = pdf[list(sm_cabinet_dtype.keys())]
        pdf = pdf.astype(sm_cabinet_dtype)
        return pdf
