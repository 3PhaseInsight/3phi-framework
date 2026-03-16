from sqlalchemy import func, insert, select, update

from threephi_framework.models.topology.graph.topology_version import TopologyVersionModel
from threephi_framework.resources.base import BaseResource


class TopologyVersionResource(BaseResource):
    def allocate_next_version(self) -> int:
        next_ver = self.s.execute(select(func.coalesce(func.max(TopologyVersionModel.version), 0) + 1)).scalar_one()
        self.s.execute(insert(TopologyVersionModel).values(version=next_ver, is_current=False))
        self._log_info(f"Allocated new topology version {next_ver}")
        return next_ver

    def flip_current_to(self, version: int) -> None:
        self.s.execute(update(TopologyVersionModel).where(TopologyVersionModel.is_current).values(is_current=False))
        self.s.execute(
            update(TopologyVersionModel).where(TopologyVersionModel.version == version).values(is_current=True)
        )
        self._log_info(f"Set version {version} as current topology snapshot")
