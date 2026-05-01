from sqlalchemy import func, insert, select, update

from threephi_framework.models.topology.graph.topology_version import TopologyVersionModel
from threephi_framework.processing_level import ProcessingLevel
from threephi_framework.resources.base import BaseResource


class TopologyVersionResource(BaseResource):
    def allocate_next_version(self, processing_level: ProcessingLevel = ProcessingLevel.RAW) -> int:
        next_ver = self.s.execute(select(func.coalesce(func.max(TopologyVersionModel.version), 0) + 1)).scalar_one()
        self.s.execute(
            insert(TopologyVersionModel).values(
                version=next_ver,
                is_current=False,
                processing_level=str(processing_level),
            )
        )
        self._log_info(f"Allocated topology version {next_ver} at level '{processing_level}'")
        return next_ver

    def get_latest_version_at_level(self, level: ProcessingLevel) -> int | None:
        """Return the highest version number at the given processing level, or None."""
        return self.s.execute(
            select(func.max(TopologyVersionModel.version)).where(TopologyVersionModel.processing_level == str(level))
        ).scalar_one_or_none()

    def flip_current_to(self, version: int) -> None:
        self.s.execute(update(TopologyVersionModel).where(TopologyVersionModel.is_current).values(is_current=False))
        self.s.execute(
            update(TopologyVersionModel).where(TopologyVersionModel.version == version).values(is_current=True)
        )
        self._log_info(f"Set version {version} as current topology snapshot")
