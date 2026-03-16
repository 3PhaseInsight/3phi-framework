from sqlalchemy import BigInteger, Float, Integer, Text, text
from sqlalchemy.dialects.postgresql import insert
from sqlalchemy.orm import Session

from threephi_framework.models.topology.graph.cable import CableModel


class CableResource:
    def __init__(self, s: Session):
        self.s = s

    def upsert_cables_from_staging(self, version: int) -> None:
        select_stmt = text(r"""
          SELECT DISTINCT
            :v AS version,
            regexp_replace(cable_id,'^\D+\.','')::bigint AS cable_id,
            cable_type,
            cable_length                         AS cable_length_m,
            NULLIF(phase_size,0)                 AS phase_size,
            phase_material,
            NULLIF(cable_capacity,0)             AS capacity_a,
            resistance                           AS resistance_ohm,
            reactance                            AS reactance_ohm
          FROM st_lv_topology
          WHERE COALESCE(cable_id,'') <> ''
        """).columns(
            version=Integer,
            cable_id=BigInteger,
            cable_type=Text,
            cable_length_m=Float,
            phase_size=Float,
            phase_material=Text,
            capacity_a=Integer,
            resistance_ohm=Float,
            reactance_ohm=Float,
        )

        # need to target Core table for bulk stuff
        t = CableModel.__table__

        ins = insert(t).from_select(
            [
                "version",
                "cable_id",
                "cable_type",
                "cable_length_m",
                "phase_size",
                "phase_material",
                "capacity_a",
                "resistance_ohm",
                "reactance_ohm",
            ],
            select_stmt,
        )

        ins = ins.on_conflict_do_update(
            index_elements=[t.c.version, t.c.cable_id],
            set_={
                "cable_type": ins.excluded.cable_type,
                "cable_length_m": ins.excluded.cable_length_m,
                "phase_size": ins.excluded.phase_size,
                "phase_material": ins.excluded.phase_material,
                "capacity_a": ins.excluded.capacity_a,
                "resistance_ohm": ins.excluded.resistance_ohm,
                "reactance_ohm": ins.excluded.reactance_ohm,
            },
        )

        self.s.execute(ins, {"v": version})
