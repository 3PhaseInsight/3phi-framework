from sqlalchemy import text
from sqlalchemy.orm import Session


class SanityResource:
    def __init__(self, s: Session):
        self.s = s

    def edges_have_nodes(self, version: int) -> None:
        row = self.s.execute(
            text("""
          SELECT 1
          FROM lv.edge e
          LEFT JOIN lv.node n1 ON n1.version = e.version AND n1.id = e.node1_id
          LEFT JOIN lv.node n2 ON n2.version = e.version AND n2.id = e.node2_id
          WHERE e.version = :v AND (n1.id IS NULL OR n2.id IS NULL)
          LIMIT 1
        """),
            {"v": version},
        ).fetchone()
        if row:
            raise RuntimeError(f"Some edges reference missing nodes for version {version}")
