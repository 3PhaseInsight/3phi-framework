from __future__ import annotations

import logging
from collections.abc import Callable

import dask.dataframe as dd
import pandas as pd
from sqlalchemy.orm.session import Session

from threephi_framework.processing_level import ProcessingLevel
from threephi_framework.resources.meta.meter import MetaMeterResource
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


def _build_topology_map(
    meter_ids: list[int],
    chains_by_meter: dict[int, list],
    zip_by_substation: dict[int, int | None],
    meters_with_data: set[int],
) -> dict[int, dict]:
    """Assemble the per-meter topology map for :meth:`TopologyController.get_topology_map_for_transformer`.

    Pure transformation with no I/O: it stitches together data that has already been
    fetched from the database into the ``{meter_id: {labeled topology fields}}`` shape.

    When a meter resolves to multiple topology paths, the first chain row is used,
    matching the convention in :meth:`MetaController.get_sm_characterization`. Meters
    without any topology chain still appear, with ``None`` topology fields.

    Args:
        meter_ids: Meters to include, in the desired output order.
        chains_by_meter: Mapping of meter ID to the rows returned by
            ``MeterResource.get_topology_chain_for_meter``.
        zip_by_substation: Mapping of secondary-substation ID to its zip code.
        meters_with_data: Meter IDs that have timeseries data (``total_rows > 0``).

    Returns:
        dict[int, dict]: ``{meter_id: {"Zip Code", "Secondary Substation ID",
        "Transformer ID", "Feeder ID", "Cabinet ID", "Has data"}}``.
    """
    topology_map: dict[int, dict] = {}
    for meter_id in meter_ids:
        chain = chains_by_meter.get(meter_id) or []
        first = chain[0] if chain else None
        topology_map[meter_id] = {
            "Zip Code": zip_by_substation.get(first.secondary_substation_id) if first else None,
            "Secondary Substation ID": first.secondary_substation_id if first else None,
            "Transformer ID": first.transformer_id if first else None,
            "Feeder ID": first.feeder_id if first else None,
            "Cabinet ID": first.cabinet_id if first else None,
            "Has data": meter_id in meters_with_data,
        }
    return topology_map


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

    def ingest(
        self,
        topology_ddf: dd.DataFrame,
        sm_cab_ddf: dd.DataFrame,
        processing_level: ProcessingLevel = ProcessingLevel.RAW,
    ) -> int:
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
            processing_level (ProcessingLevel): The processing level of the data being
                ingested. Defaults to RAW.

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
            version = tv.allocate_next_version(processing_level)

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
        with self._sf() as s:
            return MeterResource(s).get_meters_for_substation(id)

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
        with self._sf() as s:
            meter_resource = MeterResource(s)
            if node_type == "delivery_point":
                return meter_resource.get_meters_for_delivery_point(node_id)
            elif node_type == "cabinet":
                return meter_resource.get_meters_for_cabinet(node_id)
            elif node_type == "lv_feeder":
                return meter_resource.get_meters_for_feeder(node_id)
            else:
                logging.warning("Invalid node_type, valid node_type's are delivery_point, cabinet or lv_feeder.")
                return None

    def get_topology_map_for_transformer(self, transformer_id: int) -> dict[int, dict]:
        """Build a per-meter topology map for every meter under a transformer.

        Finds all meters connected to the transformer (via
        :meth:`MeterResource.get_meters_for_transformer`), resolves each meter's
        topology chain, and combines the results into a single ``{meter_id: {...}}``
        dictionary.

        Each meter entry contains its zip code, the upstream topology IDs (secondary
        substation, transformer, feeder, cabinet) and whether the meter has timeseries
        data. When a meter resolves to more than one topology path, the first is used.

        Args:
            transformer_id (int): Unique identifier of the transformer.

        Returns:
            dict[int, dict]: ``{meter_id: {"Zip Code", "Secondary Substation ID",
            "Transformer ID", "Feeder ID", "Cabinet ID", "Has data"}}`` for every
            meter under the transformer.
        """
        with self._sf() as s:
            meter_resource = MeterResource(s)
            meta_meter_resource = MetaMeterResource(s)
            substation_resource = SecondarySubstationResource(s)

            meters = meter_resource.get_meters_for_transformer(transformer_id)
            meter_ids = [meter["id"] for meter in meters]

            chains_by_meter = {
                meter_id: meter_resource.get_topology_chain_for_meter(meter_id=meter_id) for meter_id in meter_ids
            }

            substation_ids = {
                row.secondary_substation_id
                for chain in chains_by_meter.values()
                for row in chain
                if row.secondary_substation_id is not None
            }
            zip_by_substation = substation_resource.get_zip_codes_for_substations(substation_ids)
            meters_with_data = meta_meter_resource.get_meter_ids_with_data(meter_ids)

        return _build_topology_map(meter_ids, chains_by_meter, zip_by_substation, meters_with_data)

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
        with self._sf() as s:
            return MeterResource(s).get_meters(has_heat_pump, has_solar_panel)

    def export_topology(
        self,
        level: ProcessingLevel | None = None,
        as_dask: bool = False,
    ) -> dd.DataFrame | pd.DataFrame:
        """Return the LV topology in the lv_topology_* dataframe format.

        If ``level`` is ``None``, returns the version flagged ``is_current`` (default
        behaviour). If ``level`` is given, returns the most recently ingested version
        at that processing level regardless of which version is currently active.

        Historical versions are retained in the database but are not accessible by
        arbitrary version number through this API. To recover a specific version, call
        ``TopologyVersionResource.flip_current_to(version)`` directly.

        Args:
            level (ProcessingLevel | None): Processing level to query. If None, the
                current version is returned. Defaults to None.
            as_dask (bool): If True, wrap the result in a single-partition Dask
                DataFrame. Defaults to False.

        Returns:
            dd.DataFrame | pd.DataFrame: Topology data at the requested level.

        Raises:
            ValueError: If ``level`` is given but no version at that level exists.
        """
        with self._sf() as s:
            topo_export = TopologyExportResource(s)
            if level is None:
                pdf = topo_export.get_topology_pdf()
            else:
                tv = TopologyVersionResource(s)
                version = tv.get_latest_version_at_level(level)
                if version is None:
                    raise ValueError(f"No topology version found at processing level '{level}'.")
                pdf = topo_export.get_topology_at_version(version)

        if as_dask:
            return dd.from_pandas(pdf, npartitions=1)
        return pdf

    def export_sm_cabinet(
        self,
        level: ProcessingLevel | None = None,
        as_dask: bool = False,
    ) -> dd.DataFrame | pd.DataFrame:
        """Return the meter–cabinet mapping in the sm_cabinet_* dataframe format.

        Follows the same level-selection logic as :meth:`export_topology`.

        Args:
            level (ProcessingLevel | None): Processing level to query. If None, the
                current version is returned. Defaults to None.
            as_dask (bool): If True, wrap the result in a single-partition Dask
                DataFrame. Defaults to False.

        Returns:
            dd.DataFrame | pd.DataFrame: Meter–cabinet mapping data.

        Raises:
            ValueError: If ``level`` is given but no version at that level exists.
        """
        with self._sf() as s:
            topo_export = TopologyExportResource(s)
            if level is None:
                pdf = topo_export.get_sm_cabinet_pdf()
            else:
                tv = TopologyVersionResource(s)
                version = tv.get_latest_version_at_level(level)
                if version is None:
                    raise ValueError(f"No topology version found at processing level '{level}'.")
                pdf = topo_export.get_sm_cabinet_at_version(version)

        if as_dask:
            return dd.from_pandas(pdf, npartitions=1)
        return pdf

    def get_topology_chain_for_meter(self, meter_id: int):
        """Helper method to retrieve the topology chain for a given meter ID.
        Args:
            meter_id (int): ID of the meter to retrieve the topology chain for.
        Returns:
            dict | None: The topology chain for the given meter, or None if no chain is found.
        """
        with self._sf() as s:
            return MeterResource(s).get_topology_chain_for_meter(meter_id)
