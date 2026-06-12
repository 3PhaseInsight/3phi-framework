from sqlalchemy import BigInteger, Boolean, Numeric, select, text
from sqlalchemy.dialects.postgresql import insert

from threephi_framework.models.topology.assets.delivery_point import DeliveryPointModel
from threephi_framework.models.topology.assets.meter import MeterModel
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

    def get_meters_for_transformer(self, transformer_id: int) -> list[dict]:
        """
        Return all meters associated with a given transformer, walking the *current* LV topology.

        Mirrors :meth:`get_meters_for_substation` but seeds the walk at the feeders that
        belong directly to the transformer, rather than at every feeder under a substation.

        :param transformer_id: A single transformer ID
        :type transformer_id: int
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
                JOIN cur                ON cur.version = n.version
                JOIN lv.feeder f        ON n.feeder_id = f.id
                WHERE n.node_type = 'LvFeeder'
                  AND f.transformer_id = :transformer_id
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

        result = self.s.execute(sql, {"transformer_id": int(transformer_id)})
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
        sql = text("""
            WITH RECURSIVE
              cur AS (
                SELECT version
                FROM lv.topology_version
                WHERE is_current
              ),

              dp AS (
                SELECT
                    m.id AS meter_id,
                    m.delivery_point_id
                FROM lv.meter m
                WHERE m.id = :meter_id
              ),

              dp_node AS (
                SELECT
                    n.version,
                    n.id AS node_id,
                    n.delivery_point_id
                FROM lv.node n
                JOIN cur
                  ON cur.version = n.version
                JOIN dp
                  ON dp.delivery_point_id = n.delivery_point_id
                WHERE n.node_type = 'DeliveryPoint'
              ),

              walk(version, node_id, path, cabinet_id, depth) AS (
                -- Anchor: start at the delivery point node
                SELECT
                    d.version,
                    d.node_id,
                    ARRAY[d.node_id]::bigint[] AS path,
                    NULL::bigint AS cabinet_id,
                    0 AS depth
                FROM dp_node d

                UNION ALL

                -- Recursive step: walk undirected through edges
                SELECT
                    w.version,

                    CASE
                        WHEN e.node1_id = w.node_id THEN e.node2_id
                        ELSE e.node1_id
                    END AS next_node_id,

                    w.path || CASE
                        WHEN e.node1_id = w.node_id THEN e.node2_id
                        ELSE e.node1_id
                    END AS path,

                    COALESCE(
                        w.cabinet_id,
                        CASE
                            WHEN n.node_type = 'Cabinet' THEN n.cabinet_id
                            ELSE NULL
                        END
                    ) AS cabinet_id,

                    w.depth + 1 AS depth
                FROM walk w
                JOIN lv.edge e
                  ON e.version = w.version
                AND (
                      e.node1_id = w.node_id
                      OR e.node2_id = w.node_id
                )
                JOIN lv.node n
                  ON n.version = w.version
                AND n.id = CASE
                        WHEN e.node1_id = w.node_id THEN e.node2_id
                        ELSE e.node1_id
                    END
                WHERE NOT (
                    CASE
                        WHEN e.node1_id = w.node_id THEN e.node2_id
                        ELSE e.node1_id
                    END = ANY(w.path)
                )
              ),

              feeder_paths AS (
                SELECT DISTINCT
                    n.feeder_id,
                    w.cabinet_id,
                    w.depth
                FROM walk w
                JOIN lv.node n
                  ON n.version = w.version
                AND n.id = w.node_id
                WHERE n.node_type = 'LvFeeder'
              )

            SELECT
                secondary_substation_id,
                transformer_id,
                feeder_id,
                cabinet_id,
                delivery_point_id
            FROM (
                SELECT DISTINCT
                    t.substation_id AS secondary_substation_id,
                    f.transformer_id AS transformer_id,
                    fp.feeder_id AS feeder_id,
                    fp.cabinet_id AS cabinet_id,
                    dp.delivery_point_id AS delivery_point_id,
                    fp.depth AS depth
                FROM feeder_paths fp
                JOIN lv.feeder f
                  ON f.id = fp.feeder_id
                JOIN lv.transformer t
                  ON t.id = f.transformer_id
                JOIN dp ON true
            ) q
            ORDER BY depth;
        """)

        return self.s.execute(sql, {"meter_id": int(meter_id)}).all()
