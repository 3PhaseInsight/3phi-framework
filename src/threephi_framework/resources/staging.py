import pandas as pd
from sqlalchemy import text
from sqlalchemy.orm import Session


class StagingResource:
    def __init__(self, s: Session):
        self.s = s

    def create_temp_tables(self) -> None:
        # Core can generate DDL, but temp tables are easiest via text() here
        self.s.execute(text("DROP TABLE IF EXISTS st_lv_topology"))
        self.s.execute(
            text("""
            CREATE TEMP TABLE st_lv_topology(
              secondary_substation text,
              zip_code_secondary_substation text,
              transformer text,
              transformer_capacity bigint,
              lv_feeder text,
              lv_feeder_fuse_size bigint,
              node1 text,
              node2 text,
              cable_id text,
              cable_type text,
              cable_length double precision,
              phase_size bigint,
              phase_material text,
              cable_capacity bigint,
              resistance double precision,
              reactance double precision
            ) ON COMMIT DROP
        """)
        )

        self.s.execute(text("DROP TABLE IF EXISTS st_sm_cabinet"))
        self.s.execute(
            text("""
            CREATE TEMP TABLE st_sm_cabinet(
              meter_number bigint,
              delivery_point_id bigint,
              cabinet text,
              lv_feeder text,
              has_heat_pump boolean,
              has_solar_panel boolean,
              capacity_solar_panel double precision,
              service_fuse_size bigint
            ) ON COMMIT DROP
        """)
        )

    def load(self, topology_pdf: pd.DataFrame, sm_cab_pdf: pd.DataFrame) -> None:
        topology_pdf.to_sql("st_lv_topology", self.s.connection(), if_exists="append", index=False)
        sm_cab_pdf.to_sql("st_sm_cabinet", self.s.connection(), if_exists="append", index=False)
