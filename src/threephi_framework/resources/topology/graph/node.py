from sqlalchemy import text
from sqlalchemy.orm import Session


class NodeResource:
    def __init__(self, s: Session):
        self.s = s

    def prepare_node_helpers_in_staging(self) -> None:
        self.s.execute(text("DROP TABLE IF EXISTS st_nodes_raw"))
        self.s.execute(
            text(r"""
          CREATE TEMP TABLE st_nodes_raw AS
            -- 1) Feeder/Cabinet nodes from topology file
            SELECT DISTINCT
              node_code,
              CASE
                WHEN node_code ~ '^LvFeeder\.' THEN 'LvFeeder'::lv.node_type
                WHEN node_code ~ '^Cabinet\.'  THEN 'Cabinet'::lv.node_type
              END AS node_type,
              CASE
                WHEN node_code ~ '^LvFeeder\.' THEN regexp_replace(node_code,'^\D+\.','')::bigint
                WHEN node_code ~ '^Cabinet\.'  THEN regexp_replace(node_code,'^\D+\.','')::bigint
              END AS asset_id
            FROM (
              SELECT node1 AS node_code FROM st_lv_topology
              UNION ALL
              SELECT node2 AS node_code FROM st_lv_topology
            ) t
            WHERE node_code IS NOT NULL AND node_code <> ''
              AND (node_code ~ '^LvFeeder\.' OR node_code ~ '^Cabinet\.')

            UNION ALL

            -- 2) DeliveryPoint nodes from SM cabinet file
            SELECT DISTINCT
              ('DeliveryPoint.' || s.delivery_point_id::text)  AS node_code,
              'DeliveryPoint'::lv.node_type                    AS node_type,
              s.delivery_point_id::bigint                      AS asset_id
            FROM st_sm_cabinet s
            WHERE s.delivery_point_id IS NOT NULL;
        """)
        )

        self.s.execute(text("DROP TABLE IF EXISTS st_node_map"))
        self.s.execute(
            text("""
          CREATE TEMP TABLE st_node_map (
            node_code text PRIMARY KEY,
            version   int,
            node_id   bigint
          )
        """)
        )

    def bulk_insert_feeder_nodes_in_staging(self, version: int) -> None:
        self.s.execute(
            text("""
          INSERT INTO lv.node (version, node_type, feeder_id, cabinet_id, delivery_point_id)
          SELECT :v, 'LvFeeder', f.id, NULL, NULL
          FROM st_nodes_raw r
          JOIN lv.feeder f ON f.id = r.asset_id
          WHERE r.node_type = 'LvFeeder'
          ON CONFLICT (version, feeder_id) DO NOTHING
        """),
            {"v": version},
        )

        self.s.execute(
            text("""
          INSERT INTO st_node_map(node_code, version, node_id)
          SELECT r.node_code, :v, n.id
          FROM st_nodes_raw r
          JOIN lv.feeder f ON f.id = r.asset_id
          JOIN lv.node   n ON n.version=:v AND n.feeder_id=f.id
          WHERE r.node_type = 'LvFeeder'
        """),
            {"v": version},
        )

    def bulk_insert_cabinet_nodes_in_staging(self, version: int) -> None:
        self.s.execute(
            text("""
          INSERT INTO lv.node (version, node_type, feeder_id, cabinet_id, delivery_point_id)
          SELECT :v, 'Cabinet', NULL, c.id, NULL
          FROM st_nodes_raw r
          JOIN lv.cabinet c ON c.id = r.asset_id
          WHERE r.node_type = 'Cabinet'
          ON CONFLICT (version, cabinet_id) DO NOTHING
        """),
            {"v": version},
        )

        self.s.execute(
            text("""
          INSERT INTO st_node_map(node_code, version, node_id)
          SELECT r.node_code, :v, n.id
          FROM st_nodes_raw r
          JOIN lv.cabinet c ON c.id = r.asset_id
          JOIN lv.node   n ON n.version=:v AND n.cabinet_id=c.id
          WHERE r.node_type = 'Cabinet'
        """),
            {"v": version},
        )

    def bulk_insert_delivery_point_nodes_in_staging(self, version: int) -> None:
        self.s.execute(
            text("""
          INSERT INTO lv.node (version, node_type, feeder_id, cabinet_id, delivery_point_id)
          SELECT :v, 'DeliveryPoint', NULL, NULL, dp.id
          FROM st_nodes_raw r
          JOIN lv.delivery_point dp ON dp.id = r.asset_id
          WHERE r.node_type = 'DeliveryPoint'
          ON CONFLICT (version, delivery_point_id) WHERE delivery_point_id IS NOT NULL
          DO NOTHING
        """),
            {"v": version},
        )

        self.s.execute(
            text("""
          INSERT INTO st_node_map(node_code, version, node_id)
          SELECT r.node_code, :v, n.id
          FROM st_nodes_raw r
          JOIN lv.delivery_point dp ON dp.id = r.asset_id
          JOIN lv.node   n ON n.version=:v AND n.delivery_point_id=dp.id
          WHERE r.node_type = 'DeliveryPoint'
        """),
            {"v": version},
        )
