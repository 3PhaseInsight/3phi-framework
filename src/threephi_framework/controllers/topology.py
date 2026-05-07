from __future__ import annotations

import logging
from collections.abc import Callable

import dask.dataframe as dd
import pandas as pd
from sqlalchemy.orm.session import Session

from threephi_framework.resources.sanity import SanityResource
from threephi_framework.resources.staging import StagingResource
from threephi_framework.resources.topology.assets.cabinet import CabinetResource
from threephi_framework.resources.topology.assets.delivery_point import DeliveryPointResource
from threephi_framework.resources.topology.assets.feeder import FeederResource
from threephi_framework.resources.topology.assets.meter import MeterResource
from threephi_framework.resources.topology.assets.secondary_substation import SecondarySubstationResource
from threephi_framework.resources.topology.assets.transformer import TransformerResource
from threephi_framework.resources.topology.graph.cable import CableResource
from threephi_framework.resources.topology.graph.edge import EdgeResource
from threephi_framework.resources.topology.graph.edge_cable import EdgeCableResource
from threephi_framework.resources.topology.graph.node import NodeResource
from threephi_framework.resources.topology.graph.topology_version import TopologyVersionResource
from threephi_framework.resources.topology.topology_export import TopologyExportResource
from threephi_framework.schemas.v1.topology import (
    lv_topology_dtype,
    lv_topology_types,
    sm_cabinet_dtype,
    sm_cabinet_types,
)


def _set_up_logger():
    if not logging.getLogger().hasHandlers():
        logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


class TopologyController:
    """Controller for ingesting and querying low-voltage topology data.

    This controller coordinates reading topology inputs, ingesting them
    into the database, managing topology versions, and exposing helper
    methods to query meters within the topology.

    Args:
        session_factory (Callable[[], Session]): Factory that returns new
            SQLAlchemy sessions.
    """

    def __init__(self, session_factory: Callable[[], Session]):
        self._sf = session_factory
        self.meter_resource = MeterResource(self._sf())
        _set_up_logger()

    @staticmethod
    def read_topology(path) -> dd.DataFrame:
        """Read LV topology data from a CSV file into a Dask DataFrame.

        The schema is enforced using predefined dtypes and type mappings
        from ``lv_topology_dtype`` and ``lv_topology_types``.

        Args:
            path: Path to the topology CSV file.

        Returns:
            dd.DataFrame: Dask DataFrame containing the topology data.
        """
        # TODO: maybe adapt to platform architecture instead of reading from raw csv
        ddf = dd.read_csv(path, dtype=lv_topology_dtype, assume_missing=True)
        ddf = ddf.astype(lv_topology_types)
        return ddf

    @staticmethod
    def read_sm_cab(path) -> dd.DataFrame:
        """Read smart-meter–cabinet mapping from a CSV file into a Dask DataFrame.

        The schema is enforced using predefined dtypes and type mappings
        from ``sm_cabinet_dtype`` and ``sm_cabinet_types``.

        Args:
            path: Path to the smart-meter–cabinet mapping CSV file.

        Returns:
            dd.DataFrame: Dask DataFrame containing the mapping data.
        """
        # TODO: maybe adapt to platform architecture instead of reading from raw csv
        ddf = dd.read_csv(
            path,
            dtype=sm_cabinet_dtype,
            assume_missing=True,
            true_values=["true", "True", "TRUE"],
            false_values=["false", "False", "FALSE"],
        )
        ddf = ddf.astype(sm_cabinet_types)
        return ddf

    def ingest(self, topology_ddf: dd.DataFrame, sm_cab_ddf: dd.DataFrame) -> int:
        """Ingest topology and smart-meter–cabinet data into the topology schema.

        This method:
        - Computes Dask DataFrames into pandas DataFrames.
        - Loads data into staging tables.
        - Upserts topology assets (substations, transformers, feeders, etc.).
        - Builds nodes, edges, cables, and edge-cable relations.
        - Runs sanity checks.
        - Allocates and activates a new topology version.

        Args:
            topology_ddf (dd.DataFrame): LV topology data.
            sm_cab_ddf (dd.DataFrame): Smart-meter–cabinet mapping data.

        Returns:
            int: The newly allocated topology version identifier.
        """
        topo_pdf: pd.DataFrame = topology_ddf.compute()
        sm_pdf: pd.DataFrame = sm_cab_ddf.compute()

        with self._sf() as s, s.begin():
            tv = TopologyVersionResource(s)
            staging = StagingResource(s)
            substations = SecondarySubstationResource(s)
            transformers = TransformerResource(s)
            feeders = FeederResource(s)
            cabinets = CabinetResource(s)
            delivery_points = DeliveryPointResource(s)
            meters = MeterResource(s)

            nodes = NodeResource(s)
            edges = EdgeResource(s)
            cables = CableResource(s)
            edge_cables = EdgeCableResource(s)
            sanity = SanityResource(s)

            logging.info("Allocating next Topology Version")
            version = tv.allocate_next_version()

            logging.info("Creating temporary tables and loading files")
            staging.create_temp_tables()
            staging.load(topo_pdf, sm_pdf)

            logging.info("Upserting static assets")
            substations.bulk_upsert_from_staging()
            transformers.bulk_upsert_from_staging()
            feeders.bulk_upsert_from_staging()
            cabinets.bulk_upsert_from_staging()
            delivery_points.bulk_upsert_from_staging()
            meters.bulk_upsert_from_staging()

            logging.info("Inserting and Upserting versioned nodes and edges")
            nodes.prepare_node_helpers_in_staging()
            nodes.bulk_insert_feeder_nodes_in_staging(version)
            nodes.bulk_insert_cabinet_nodes_in_staging(version)
            nodes.bulk_insert_delivery_point_nodes_in_staging(version)

            edges.group_edges_in_staging()
            # edges.log_edge_resolution_counts(version)
            # edges.log_dp_edges_in_staging(20)
            # edges.log_dp_edge_drop_reasons(version)
            edges.bulk_insert_from_staging(version)

            cables.upsert_cables_from_staging(version)
            edge_cables.build_edge_cables_in_staging(version)
            edge_cables.bulk_upsert_from_staging()

            logging.info("Sanity Checking")
            sanity.edges_have_nodes(version)
            logging.info("Flipping new version to current")
            tv.flip_current_to(version)

            return version

    def get_meters_for_substation(self, id: int):
        """Retrieve all meters associated with a given substation.

        This method queries the current LV topology and returns all meters
        linked to the specified substation.

        Args:
            id (int): Unique identifier of the substation.

        Returns:
            list[dict]: A list of meter records associated with the substation.
        """
        return self.meter_resource.get_meters_for_substation(id)

    def get_meters_for_node(self, node_id: int, node_type: str) -> list[dict] | None:
        """Retrieve all meters associated with a given node.

        This method queries the current LV topology and returns all meters
        linked to the specified node.

        Args:
            node_id (int): ID of the node.
            node_type (str): The node_type, must be one of "delivery_point", "cabinet" or "lv_feeder".

        Returns:
            list[dict] | None: A list of meter records associated with the node
            or None if an invalid node_type was given.
        """
        if node_type == "delivery_point":
            return self.meter_resource.get_meters_for_delivery_point(node_id)
        elif node_type == "cabinet":
            return self.meter_resource.get_meters_for_cabinet(node_id)
        elif node_type == "lv_feeder":
            return self.meter_resource.get_meters_for_feeder(node_id)
        else:
            logging.warning("Invalid node_type, valid node_type's are delivery_point, cabinet or lv_feeder.")
            return None

    def get_meters(
        self,
        has_heat_pump: bool | None = None,
        has_solar_panel: bool | None = None,
    ) -> list[dict]:
        """Retrieve meter IDs filtered by device characteristics.

        If no filters are provided, all available meters are returned.

        Args:
            has_heat_pump (bool | None, optional): If True, only include meters
                with a heat pump. If False, exclude them. If None, do not
                filter by heat pump. Defaults to None.
            has_solar_panel (bool | None, optional): If True, only include
                meters with a solar panel. If False, exclude them. If None,
                do not filter by solar panel. Defaults to None.

        Returns:
            list[dict]: List of meter objects matching the given filters.
        """
        meters = self.meter_resource.get_meters(has_heat_pump, has_solar_panel)
        return meters

    def export_topology(
        self,
        as_dask: bool = False,
    ) -> dd.DataFrame | pd.DataFrame:
        """Return the *current* LV topology in the lv_topology_* dataframe format.

        Only the version flagged ``is_current`` in ``lv.topology_version`` is returned.
        Historical versions are retained in the database but are not yet accessible through
        this API. To recover a previous version, call
        ``TopologyVersionResource.flip_current_to(version)`` directly.
        """
        with self._sf() as s:
            topo_export = TopologyExportResource(s)
            pdf = topo_export.get_topology_pdf()

        if as_dask:
            return dd.from_pandas(pdf, npartitions=1)
        return pdf

    def export_sm_cabinet(
        self,
        as_dask: bool = False,
    ) -> dd.DataFrame | pd.DataFrame:
        """Return the meter–cabinet mapping in the sm_cabinet_* dataframe format.

        See :meth:`export_topology` for version-access limitations.
        """
        with self._sf() as s:
            topo_export = TopologyExportResource(s)
            pdf = topo_export.get_sm_cabinet_pdf()

        if as_dask:
            return dd.from_pandas(pdf, npartitions=1)
        return pdf
