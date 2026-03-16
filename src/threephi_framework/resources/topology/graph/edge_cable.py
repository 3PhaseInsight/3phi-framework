from sqlalchemy import text
from sqlalchemy.orm import Session


class EdgeCableResource:
    def __init__(self, s: Session):
        self.s = s

    def build_edge_cables_in_staging(self, version: int) -> None:
        self.s.execute(
            text(r"""
          DROP TABLE IF EXISTS st_edge_cables;
          CREATE TEMP TABLE st_edge_cables AS
          SELECT
            :v AS version,
            m1.node_id AS n1_id,
            m2.node_id AS n2_id,
            cable_text.cable_code,
            cable_text.seq_no
          FROM st_edges_grouped g
          JOIN st_node_map m1 ON m1.version=:v AND m1.node_code=g.n1_code
          JOIN st_node_map m2 ON m2.version=:v AND m2.node_code=g.n2_code
          CROSS JOIN LATERAL regexp_split_to_table(g.cable_ids, '\|\|')
            WITH ORDINALITY AS cable_text(cable_code, seq_no)
          WHERE COALESCE(g.cable_ids, '') <> '';
        """),
            {"v": version},
        )

    def bulk_upsert_from_staging(self) -> None:
        self.s.execute(
            text(r"""
          INSERT INTO lv.edge_cable (version, edge_id, cable_id, seq_no)
          SELECT s.version,
                 em.id AS edge_id,
                 regexp_replace(s.cable_code,'^\D+\.','')::bigint AS cable_id,
                 s.seq_no::int
          FROM st_edge_cables s
          JOIN st_edge_map em
            ON em.version=s.version
           AND em.n1_id = LEAST(s.n1_id, s.n2_id)
           AND em.n2_id = GREATEST(s.n1_id, s.n2_id)
          ON CONFLICT (version, edge_id, cable_id) DO UPDATE
            SET seq_no = EXCLUDED.seq_no
        """)
        )
