from sqlalchemy import BigInteger, Boolean, Integer, Numeric, literal, select, text, true, union_all
from sqlalchemy.dialects.postgresql import insert

from threephi_framework.models.topology.assets.delivery_point import DeliveryPointModel
from threephi_framework.models.topology.assets.feeder import FeederModel
from threephi_framework.models.topology.assets.meter import MeterModel
from threephi_framework.models.topology.assets.transformer import TransformerModel
from threephi_framework.models.topology.utilities import EdgeCurrentModel, NodeCurrentModel
from threephi_framework.resources.base import BaseResource


class MeterResource(BaseResource):
    def bulk_upsert_from_staging(self) -> None:
        # Join delivery_point to ensure FK exists; filters out orphan rows
        select_stmt = text(r"""
              SELECT DISTINCT
                  s.meter_number::bigint               AS id,
                  s.delivery_point_id::bigint          AS delivery_point_id,
                  s.has_heat_pump                      AS has_heat_pump,
                  s.has_solar_panel                    AS has_solar_panel,
                  NULLIF(s.capacity_solar_panel,0)     AS solar_capacity_kw
              FROM st_sm_cabinet s
              JOIN lv.delivery_point d
                ON d.id = s.delivery_point_id
              WHERE s.delivery_point_id IS NOT NULL
                AND s.meter_number IS NOT NULL
            """).columns(
            id=BigInteger,
            delivery_point_id=BigInteger,
            has_heat_pump=Boolean,
            has_solar_panel=Boolean,
            solar_capacity_kw=Numeric(asdecimal=False),
        )

        stmt = insert(MeterModel).from_select(
            ["id", "delivery_point_id", "has_heat_pump", "has_solar_panel", "solar_capacity_kw"],
            select_stmt,
        )

        stmt = stmt.on_conflict_do_update(
            index_elements=[MeterModel.id],
            set_={
                "delivery_point_id": stmt.excluded.delivery_point_id,
                "has_heat_pump": stmt.excluded.has_heat_pump,
                "has_solar_panel": stmt.excluded.has_solar_panel,
                "solar_capacity_kw": stmt.excluded.solar_capacity_kw,
            },
        )

        self.s.execute(stmt)

    def get_meters_for_substation(self, substation_id: int) -> list[dict]:
        """
        Return all meters associated with a given substation, walking the *current* LV topology.
        Uses the convenience views: lv.lv_node_current and lv.lv_edge_current.
        """
        sql = text("""
            WITH RECURSIVE
              cur AS (
                SELECT version
                FROM lv.topology_version
                WHERE is_current
              ),
              seed AS (
                SELECT n.version, n.id AS node_id
                FROM lv.node n
                JOIN cur                ON cur.version = n.version
                JOIN lv.feeder f        ON n.feeder_id = f.id
                JOIN lv.transformer t   ON f.transformer_id = t.id
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
                  CASE WHEN e.node1_id = w.node_id THEN e.node2_id ELSE e.node1_id END AS next_node_id,
                  w.path || CASE WHEN e.node1_id = w.node_id THEN e.node2_id ELSE e.node1_id END
                FROM walk w
                JOIN lv.edge e
                  ON e.version = w.version
                 AND (e.node1_id = w.node_id OR e.node2_id = w.node_id)
                WHERE NOT (
                  (CASE WHEN e.node1_id = w.node_id THEN e.node2_id ELSE e.node1_id END) = ANY (w.path)
                )
              )
            SELECT DISTINCT m.*
            FROM walk w
            JOIN lv.node n
            ON n.version = w.version
            AND n.id = w.node_id
            AND n.node_type = 'DeliveryPoint'
            JOIN lv.delivery_point dp
            ON dp.id = n.delivery_point_id
            JOIN lv.meter m
            ON m.delivery_point_id = dp.id
            ORDER BY m.id;
        """)

        result = self.s.execute(sql, {"substation_id": int(substation_id)})
        return [dict(r) for r in result.mappings().all()]

    def get_meters_for_delivery_point(self, delivery_point_id: int) -> list[dict]:
        """
        Returns meters connected to a given delivery point.

        :param delivery_point_id: A single delivery point ID
        :type delivery_point_id: int
        :return: A list of meter objects
        :rtype: list[dict[Any, Any]]
        """
        stmt = select(*MeterModel.__table__.c).where(MeterModel.delivery_point_id == delivery_point_id)

        result = self.s.execute(stmt.order_by(MeterModel.id))
        return [dict(r) for r in result.mappings().all()]

    def get_meters_for_cabinet(self, cabinet_id: int) -> list[dict]:
        """
        Returns meters connected to a given cabinet.

        :param cabinet_id: A single cabinet ID
        :type cabinet_id: int
        :return: A list of meter objects
        :rtype: list[dict[Any, Any]]
        """
        stmt = (
            select(*MeterModel.__table__.c)
            .join(DeliveryPointModel, MeterModel.delivery_point_id == DeliveryPointModel.id)
            .where(DeliveryPointModel.cabinet_id == cabinet_id)
            .order_by(MeterModel.id)
        )

        result = self.s.execute(stmt.order_by(MeterModel.id))
        return [dict(r) for r in result.mappings().all()]

    def get_meters_for_feeder(self, feeder_id: int) -> list[dict]:
        """
        Docstring for get_meters_for_feeder

        :param feeder_id: A single feeder ID
        :type feeder_id: int
        :return: A list of meter objects
        :rtype: list[dict[Any, Any]]
        """

        sql = text("""
            WITH RECURSIVE
              cur AS (
                SELECT version
                FROM lv.topology_version
                WHERE is_current
              ),
              seed AS (
                SELECT n.version, n.id AS node_id
                FROM lv.node n
                JOIN cur ON cur.version = n.version
                WHERE n.node_type='LvFeeder' AND n.feeder_id=:feeder_id
              ),
              walk(version, node_id, path) AS (
                -- ANCHOR
                SELECT s.version, s.node_id, ARRAY[s.node_id]::bigint[]
                FROM seed s
                UNION ALL
                -- Single recursive step (undirected)
                SELECT
                  w.version,
                  CASE WHEN e.node1_id = w.node_id THEN e.node2_id ELSE e.node1_id END AS next_node_id,
                  w.path || CASE WHEN e.node1_id = w.node_id THEN e.node2_id ELSE e.node1_id END
                FROM walk w
                JOIN lv.edge e
                  ON e.version = w.version
                 AND (e.node1_id = w.node_id OR e.node2_id = w.node_id)
                WHERE NOT (
                  (CASE WHEN e.node1_id = w.node_id THEN e.node2_id ELSE e.node1_id END) = ANY (w.path)
                )
              )
            SELECT DISTINCT m.*
            FROM walk w
            JOIN lv.node n
            ON n.version = w.version
            AND n.id = w.node_id
            AND n.node_type = 'DeliveryPoint'
            JOIN lv.meter m
            ON m.delivery_point_id = n.delivery_point_id
            ORDER BY m.id;
        """)

        result = self.s.execute(sql, {"feeder_id": int(feeder_id)})
        return [dict(r._mapping) for r in result.fetchall()]

    def get_meters(self, has_heat_pump: bool | None = None, has_solar_panel: bool | None = None) -> list[dict]:
        stmt = select(*MeterModel.__table__.c)

        if has_heat_pump is not None:
            stmt = stmt.where(MeterModel.has_heat_pump.is_(has_heat_pump))
        if has_solar_panel is not None:
            stmt = stmt.where(MeterModel.has_solar_panel.is_(has_solar_panel))

        result = self.s.execute(stmt.order_by(MeterModel.id))
        return [dict(r) for r in result.mappings().all()]

    def get_topology_chain_for_meter(self, meter_id: int):
        # meter -> delivery point (MODEL id)
        dp_id = (
            select(MeterModel.delivery_point_id.label("delivery_point_id"))
            .where(MeterModel.id == meter_id)
            .cte("dp_id")
        )

        # delivery point -> dp node (NODE id for traversal)
        dp_node = (
            select(NodeCurrentModel.id.label("dp_node_id"))
            .select_from(dp_id)
            .join(NodeCurrentModel, NodeCurrentModel.delivery_point_id == dp_id.c.delivery_point_id)
            .cte("dp_node")
        )

        # 1-hop neighbors of DP node (undirected) [NODE ids]
        nbrs_1 = union_all(
            select(EdgeCurrentModel.node2_id.label("node_id"))
            .select_from(dp_node)
            .where(EdgeCurrentModel.node1_id == dp_node.c.dp_node_id),
            select(EdgeCurrentModel.node1_id.label("node_id"))
            .select_from(dp_node)
            .where(EdgeCurrentModel.node2_id == dp_node.c.dp_node_id),
        ).cte("nbrs_1")

        # --- Direct feeders: DP <-> LvFeeder ---
        direct_paths = (
            select(
                NodeCurrentModel.feeder_id.label("feeder_id"),  # MODEL id
                literal(None).cast(Integer).label("cabinet_id"),  # MODEL id (nullable)
            )
            .select_from(nbrs_1)
            .join(NodeCurrentModel, NodeCurrentModel.id == nbrs_1.c.node_id)
            .where(NodeCurrentModel.node_type == "LvFeeder")
            .cte("direct_paths")
        )

        # --- Adjacent cabinets: DP <-> Cabinet ---
        # keep cabinet NODE id for traversal and cabinet MODEL id for return
        adjacent_cabinets = (
            select(
                NodeCurrentModel.id.label("cab_node_id"),  # NODE id
                NodeCurrentModel.cabinet_id.label("cabinet_id"),  # MODEL id
            )
            .select_from(nbrs_1)
            .join(NodeCurrentModel, NodeCurrentModel.id == nbrs_1.c.node_id)
            .where(NodeCurrentModel.node_type == "Cabinet")
            .cte("adjacent_cabinets")
        )

        # 2-hop neighbors of cabinet nodes (undirected), carrying cabinet MODEL id
        cab_nbrs = union_all(
            select(
                EdgeCurrentModel.node2_id.label("node_id"),
                adjacent_cabinets.c.cabinet_id.label("cabinet_id"),
            )
            .select_from(adjacent_cabinets)
            .where(EdgeCurrentModel.node1_id == adjacent_cabinets.c.cab_node_id),
            select(
                EdgeCurrentModel.node1_id.label("node_id"),
                adjacent_cabinets.c.cabinet_id.label("cabinet_id"),
            )
            .select_from(adjacent_cabinets)
            .where(EdgeCurrentModel.node2_id == adjacent_cabinets.c.cab_node_id),
        ).cte("cab_nbrs")

        # --- Cabinet feeders: DP <-> Cabinet <-> LvFeeder ---
        cabinet_paths = (
            select(
                NodeCurrentModel.feeder_id.label("feeder_id"),  # MODEL id
                cab_nbrs.c.cabinet_id.label("cabinet_id"),  # MODEL id
            )
            .select_from(cab_nbrs)
            .join(NodeCurrentModel, NodeCurrentModel.id == cab_nbrs.c.node_id)
            .where(NodeCurrentModel.node_type == "LvFeeder")
            .cte("cabinet_paths")
        )

        # Combine both path types
        all_paths = (
            select(direct_paths.c.feeder_id, direct_paths.c.cabinet_id)
            .union_all(select(cabinet_paths.c.feeder_id, cabinet_paths.c.cabinet_id))
            .cte("all_paths")
        )

        # Final projection (MODEL ids only)
        stmt = (
            select(
                TransformerModel.substation_id.label("secondary_substation_id"),
                FeederModel.transformer_id.label("transformer_id"),
                all_paths.c.feeder_id.label("feeder_id"),
                all_paths.c.cabinet_id.label("cabinet_id"),
                dp_id.c.delivery_point_id.label("delivery_point_id"),
            )
            .select_from(dp_id)
            .join(all_paths, true())  # fan-out
            .join(FeederModel, FeederModel.id == all_paths.c.feeder_id)
            .join(TransformerModel, TransformerModel.id == FeederModel.transformer_id)
            .distinct()
        )

        return self.s.execute(stmt).all()
