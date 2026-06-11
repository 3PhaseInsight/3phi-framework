from collections.abc import Iterable

from sqlalchemy import BigInteger, Integer, select, text
from sqlalchemy.dialects.postgresql import insert

from threephi_framework.models.topology.assets.secondary_substation import SecondarySubstationModel
from threephi_framework.resources.base import BaseResource


class SecondarySubstationResource(BaseResource):
    def get_zip_codes_for_substations(self, substation_ids: Iterable[int]) -> dict[int, int | None]:
        """Return a ``{substation_id: zip_code}`` lookup for the given substations.

        Args:
            substation_ids: Secondary-substation IDs to look up.

        Returns:
            dict[int, int | None]: Zip code per substation ID. Substations with no
            stored zip code map to ``None``; IDs not present in the table are omitted.
        """
        ids = list(substation_ids)
        if not ids:
            return {}
        stmt = select(SecondarySubstationModel.id, SecondarySubstationModel.zip_code).where(
            SecondarySubstationModel.id.in_(ids)
        )
        return {row.id: row.zip_code for row in self.s.execute(stmt)}

    def bulk_upsert_from_staging(self) -> None:
        select_stmt = text(r"""
          SELECT DISTINCT
            regexp_replace(secondary_substation,'^\D+\.','')::bigint AS id,
            NULLIF(zip_code_secondary_substation,'')::int            AS zip_code
          FROM st_lv_topology
          WHERE COALESCE(secondary_substation,'') <> ''
        """).columns(id=BigInteger, zip_code=Integer)

        stmt = insert(SecondarySubstationModel).from_select(["id", "zip_code"], select_stmt)
        stmt = stmt.on_conflict_do_update(
            index_elements=[SecondarySubstationModel.id],
            set_={"zip_code": stmt.excluded.zip_code},
        )

        self.s.execute(stmt)
