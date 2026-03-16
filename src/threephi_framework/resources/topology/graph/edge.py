import logging

from sqlalchemy import text
from sqlalchemy.orm import Session


class EdgeResource:
    def __init__(self, s: Session):
        self.s = s

    def group_edges_in_staging(self) -> None:
        self.s.execute(text("DROP TABLE IF EXISTS st_edges_grouped"))
        self.s.execute(
            text("""
            CREATE TEMP TABLE st_edges_grouped AS
            WITH topo_edges AS (
                SELECT
                    node1 AS node_code1,
                    node2 AS node_code2,
                    cable_id::text AS cable_id
                FROM st_lv_topology
                WHERE node1 IS NOT NULL AND node2 IS NOT NULL
                    AND node1 <> '' AND node2 <> ''
                    AND node1 <> node2
            ),
            inferred_dp_feeder_edges AS (
                SELECT DISTINCT
                    s.lv_feeder AS node_code1,
                    ('DeliveryPoint.' || s.delivery_point_id::text) AS node_code2,
                    NULL::text AS cable_id
                FROM st_sm_cabinet s
                WHERE s.delivery_point_id IS NOT NULL
                    AND COALESCE(s.cabinet, '') = ''
                    AND COALESCE(s.lv_feeder, '') <> ''
            ),
            dp_edges_via_cabinet AS (
                SELECT DISTINCT
                    TRIM(s.cabinet) AS node_code1,
                    ('DeliveryPoint.' || s.delivery_point_id::text) AS node_code2,
                    NULL::text AS cable_id
                FROM st_sm_cabinet s
                WHERE s.delivery_point_id IS NOT NULL
                    AND COALESCE(s.cabinet,'') <> ''
            ),
            all_edges AS (
                SELECT * FROM topo_edges
                UNION ALL
                SELECT * FROM inferred_dp_feeder_edges
                UNION ALL
                SELECT * FROM dp_edges_via_cabinet
            )
            SELECT
                LEAST(node_code1, node_code2)    AS n1_code,
                GREATEST(node_code1, node_code2) AS n2_code,
                STRING_AGG(cable_id, '||' ORDER BY cable_id) FILTER (WHERE cable_id IS NOT NULL) AS cable_ids
                FROM all_edges
                GROUP BY 1, 2;
        """)
        )

    def log_dp_edges_in_staging(self, sample: int = 10) -> None:
        """
        Must be called after group_edges_in_staging() in the same session,
        because st_edges_grouped is a TEMP table.
        """

        totals = (
            self.s.execute(
                text("""
            SELECT
              COUNT(*) AS total_edges,
              COUNT(*) FILTER (
                WHERE n1_code LIKE 'DeliveryPoint.%' OR n2_code LIKE 'DeliveryPoint.%'
              ) AS dp_edges
            FROM st_edges_grouped;
        """)
            )
            .mappings()
            .one()
        )

        print(f"[INFO] st_edges_grouped: total_edges={totals['total_edges']}, dp_edges={totals['dp_edges']}")

        # Breakdown by counterpart type (based on code prefix)
        breakdown = (
            self.s.execute(
                text("""
            SELECT
              CASE
                WHEN (n1_code LIKE 'DeliveryPoint.%' AND n2_code LIKE 'LvFeeder.%')
                  OR (n2_code LIKE 'DeliveryPoint.%' AND n1_code LIKE 'LvFeeder.%')
                  THEN 'DP-Feeder'
                WHEN (n1_code LIKE 'DeliveryPoint.%' AND n2_code LIKE 'Cabinet.%')
                  OR (n2_code LIKE 'DeliveryPoint.%' AND n1_code LIKE 'Cabinet.%')
                  THEN 'DP-Cabinet'
                WHEN (n1_code LIKE 'DeliveryPoint.%' OR n2_code LIKE 'DeliveryPoint.%')
                  THEN 'DP-Other'
                ELSE 'No-DP'
              END AS edge_kind,
              COUNT(*) AS cnt
            FROM st_edges_grouped
            GROUP BY 1
            ORDER BY cnt DESC;
        """)
            )
            .mappings()
            .all()
        )

        for row in breakdown:
            print(f"[INFO] st_edges_grouped breakdown: {row['edge_kind']}={row['cnt']}")

        # Optional samples (useful to validate formatting)
        if sample and int(totals["dp_edges"]) > 0:
            rows = (
                self.s.execute(
                    text("""
                SELECT n1_code, n2_code, cable_ids
                FROM st_edges_grouped
                WHERE n1_code LIKE 'DeliveryPoint.%' OR n2_code LIKE 'DeliveryPoint.%'
                LIMIT :sample;
            """),
                    {"sample": int(sample)},
                )
                .mappings()
                .all()
            )

            logging.info("[INFO] sample DP edges (n1_code, n2_code, cable_ids):")
            for r in rows:
                logging.info(f"  {r['n1_code']}  <->  {r['n2_code']}   cables={r['cable_ids']}")

    def log_edge_resolution_counts(self, version: int) -> None:
        totals = (
            self.s.execute(
                text("""
            SELECT
              (SELECT COUNT(*) FROM st_edges_grouped) AS total_edges,
              (SELECT COUNT(*)
                 FROM st_edges_grouped g
                 JOIN st_node_map m1 ON m1.version=:v AND m1.node_code=g.n1_code
                 JOIN st_node_map m2 ON m2.version=:v AND m2.node_code=g.n2_code
              ) AS resolvable_edges
        """),
                {"v": version},
            )
            .mappings()
            .one()
        )

        dropped = totals["total_edges"] - totals["resolvable_edges"]

        breakdown = (
            self.s.execute(
                text("""
            SELECT
              COUNT(*) FILTER (WHERE g.n1_code LIKE 'DeliveryPoint.%' AND m1.node_id IS NULL)
            + COUNT(*) FILTER (WHERE g.n2_code LIKE 'DeliveryPoint.%' AND m2.node_id IS NULL)
                AS missing_deliverypoint_endpoints,

              COUNT(*) FILTER (WHERE g.n1_code LIKE 'LvFeeder.%' AND m1.node_id IS NULL)
            + COUNT(*) FILTER (WHERE g.n2_code LIKE 'LvFeeder.%' AND m2.node_id IS NULL)
                AS missing_feeder_endpoints,

              COUNT(*) FILTER (WHERE g.n1_code LIKE 'Cabinet.%' AND m1.node_id IS NULL)
            + COUNT(*) FILTER (WHERE g.n2_code LIKE 'Cabinet.%' AND m2.node_id IS NULL)
                AS missing_cabinet_endpoints,

              COUNT(*) FILTER (WHERE m1.node_id IS NULL AND m2.node_id IS NULL)
                AS missing_both_endpoints
            FROM st_edges_grouped g
            LEFT JOIN st_node_map m1 ON m1.version=:v AND m1.node_code=g.n1_code
            LEFT JOIN st_node_map m2 ON m2.version=:v AND m2.node_code=g.n2_code
            WHERE m1.node_id IS NULL OR m2.node_id IS NULL
        """),
                {"v": version},
            )
            .mappings()
            .one()
        )

        logging.info(
            "[INFO] Edge resolution:"
            f" total={totals['total_edges']}, resolvable={totals['resolvable_edges']}, dropped={dropped};"
            f" missing_endpoints: dp={breakdown['missing_deliverypoint_endpoints']},"
            f" feeder={breakdown['missing_feeder_endpoints']},"
            f" cabinet={breakdown['missing_cabinet_endpoints']},"
            f" both={breakdown['missing_both_endpoints']}"
        )

    def log_dp_edge_drop_reasons(self, version: int, sample: int = 20) -> None:
        """
        Diagnose inferred DP edges in staging vs resolvability in st_node_map.
        Call after st_edges_grouped is created AND after st_node_map exists.
        """

        # Count staged DP edges
        dp_edges = self.s.execute(
            text("""
            SELECT COUNT(*) AS dp_edges
            FROM st_edges_grouped
            WHERE n1_code LIKE 'DeliveryPoint.%' OR n2_code LIKE 'DeliveryPoint.%';
        """)
        ).scalar_one()

        # Count staged DP edges where feeder endpoint is missing in node_map
        dp_missing_feeder = self.s.execute(
            text("""
            SELECT COUNT(*) AS dp_edges_missing_feeder
            FROM st_edges_grouped g
            LEFT JOIN st_node_map m1 ON m1.version=:v AND m1.node_code=g.n1_code
            LEFT JOIN st_node_map m2 ON m2.version=:v AND m2.node_code=g.n2_code
            WHERE (g.n1_code LIKE 'DeliveryPoint.%' OR g.n2_code LIKE 'DeliveryPoint.%')
              AND (
                (g.n1_code LIKE 'LvFeeder.%' AND m1.node_id IS NULL) OR
                (g.n2_code LIKE 'LvFeeder.%' AND m2.node_id IS NULL)
              );
        """),
            {"v": version},
        ).scalar_one()

        # Count staged DP edges where DP endpoint is missing in node_map (should be 0)
        dp_missing_dp = self.s.execute(
            text("""
            SELECT COUNT(*) AS dp_edges_missing_dp
            FROM st_edges_grouped g
            LEFT JOIN st_node_map m1 ON m1.version=:v AND m1.node_code=g.n1_code
            LEFT JOIN st_node_map m2 ON m2.version=:v AND m2.node_code=g.n2_code
            WHERE (g.n1_code LIKE 'DeliveryPoint.%' OR g.n2_code LIKE 'DeliveryPoint.%')
              AND (
                (g.n1_code LIKE 'DeliveryPoint.%' AND m1.node_id IS NULL) OR
                (g.n2_code LIKE 'DeliveryPoint.%' AND m2.node_id IS NULL)
              );
        """),
            {"v": version},
        ).scalar_one()

        # Count staged DP edges where cabinet endpoint is missing (if you later add DP-Cabinet)
        dp_missing_cabinet = self.s.execute(
            text("""
            SELECT COUNT(*) AS dp_edges_missing_cabinet
            FROM st_edges_grouped g
            LEFT JOIN st_node_map m1 ON m1.version=:v AND m1.node_code=g.n1_code
            LEFT JOIN st_node_map m2 ON m2.version=:v AND m2.node_code=g.n2_code
            WHERE (g.n1_code LIKE 'DeliveryPoint.%' OR g.n2_code LIKE 'DeliveryPoint.%')
              AND (
                (g.n1_code LIKE 'Cabinet.%' AND m1.node_id IS NULL) OR
                (g.n2_code LIKE 'Cabinet.%' AND m2.node_id IS NULL)
              );
        """),
            {"v": version},
        ).scalar_one()

        print(
            "[INFO] DP edge diagnostics:"
            f" staged_dp_edges={dp_edges},"
            f" missing_feeder={dp_missing_feeder},"
            f" missing_dp={dp_missing_dp},"
            f" missing_cabinet={dp_missing_cabinet}"
        )

        # Sample a few DP edges that are missing feeder resolution
        if sample and dp_missing_feeder:
            rows = (
                self.s.execute(
                    text("""
                SELECT g.n1_code, g.n2_code
                FROM st_edges_grouped g
                LEFT JOIN st_node_map m1 ON m1.version=:v AND m1.node_code=g.n1_code
                LEFT JOIN st_node_map m2 ON m2.version=:v AND m2.node_code=g.n2_code
                WHERE (g.n1_code LIKE 'DeliveryPoint.%' OR g.n2_code LIKE 'DeliveryPoint.%')
                  AND (
                    (g.n1_code LIKE 'LvFeeder.%' AND m1.node_id IS NULL) OR
                    (g.n2_code LIKE 'LvFeeder.%' AND m2.node_id IS NULL)
                  )
                LIMIT :sample;
            """),
                    {"v": version, "sample": sample},
                )
                .mappings()
                .all()
            )

            print("[INFO] sample DP edges with missing feeder endpoint:")
            for r in rows:
                print(f"  {r['n1_code']}  <->  {r['n2_code']}")

    def bulk_insert_from_staging(self, version: int) -> None:
        self.s.execute(
            text("""
          INSERT INTO lv.edge (version, node1_id, node2_id)
          SELECT :v, m1.node_id, m2.node_id
          FROM st_edges_grouped g
          JOIN st_node_map m1 ON m1.version=:v AND m1.node_code = g.n1_code
          JOIN st_node_map m2 ON m2.version=:v AND m2.node_code = g.n2_code
          ON CONFLICT DO NOTHING
        """),
            {"v": version},
        )

        self.s.execute(text("DROP TABLE IF EXISTS st_edge_map"))
        self.s.execute(
            text("""
          CREATE TEMP TABLE st_edge_map AS
          SELECT :v AS version,
                 LEAST(e.node1_id, e.node2_id) AS n1_id,
                 GREATEST(e.node1_id, e.node2_id) AS n2_id,
                 e.id
          FROM lv.edge e
          WHERE e.version = :v
        """),
            {"v": version},
        )
