"""DataExtractor — compatibility proxy over the framework controllers.

Historically this class read CSV exports and hand-rolled parquet layouts directly
from object storage. The canonical storage has since moved to:

- LV topology + meter/cabinet relations: PostgreSQL (``lv`` schema), accessed via
  :class:`~threephi_framework.controllers.topology.TopologyController`
- Timeseries: dt/shard-partitioned parquet in object storage, accessed via
  :class:`~threephi_framework.controllers.time_series.TimeSeriesController`
- Meter inventory / workflow state: ``meta`` schema via
  :class:`~threephi_framework.controllers.meta.MetaController`

To keep existing data apps working without a rewrite, the legacy method surface is
preserved, but every method now delegates to the controllers above. Notable
behavioral notes for callers migrating from the CSV-era implementation:

- Per-meter dataframes are assembled on demand from the canonical store; there is
  no per-SM parquet cache to "load" anymore, so ``load_*`` and ``extract_*``
  return the same data and the ``use_existing_raw_sm_profiles`` /
  ``overwrite*`` flags are accepted but ignored.
- The wide per-meter frames are indexed by the ``timestamp`` column of the
  canonical parquet schema (the old CSVs called it ``timestamp_dst``).
- ``topology_processing_level`` ("raw" / "cleaned" / "cleaned_and_corrected")
  selects the most recent topology *version* ingested at that
  :class:`~threephi_framework.processing_level.ProcessingLevel`.
- Profiles ("raw_profiles" / "cleaned_profiles" /
  "cleaned_and_phase_corrected_profiles") map to the timeseries processing
  levels RAW / CLEANED / CLEANED_AND_CORRECTED. The cleaned levels require the
  flags / corrections / phase-map artifacts to exist in object storage.
"""

import datetime as dt
import glob
import logging
import os
import re
import uuid
from pathlib import Path

import dask.dataframe as dd
import numpy as np
import pandas as pd

import threephi_framework.db.db as threephi_db
from threephi_framework.controllers.ingestion import IngestionController
from threephi_framework.controllers.meta import MetaController
from threephi_framework.controllers.time_series import TimeSeriesController
from threephi_framework.controllers.topology import TopologyController
from threephi_framework.data_extractor.schemas.phase_measurements.v1 import (
    VERSION,
    PhaseMeasurementsCsvSchema,
    PhaseMeasurementsParquetSchema,
)
from threephi_framework.object_storage.base_connector import BaseConnector
from threephi_framework.object_storage.factory import create_connector
from threephi_framework.processing_level import ProcessingLevel
from threephi_framework.util.util import v1_get_shard_for_meter_id

# Legacy output layout — still used for the optional `save=True` parquet/JSON dumps
BATCH = "second_batch"
SOURCE_DATA_DIR = f"Data/sourcedata/{BATCH}"
PROCESSED_DATA_DIR = f"Data/processed_data/{BATCH}"
ALLOWED_PROFILES = [
    "raw_profiles",
    "cleaned_profiles",
    "cleaned_and_phase_corrected_profiles",
]

# profile name (legacy) → timeseries processing level
PROFILE_TO_LEVEL = {
    "raw_profiles": ProcessingLevel.RAW,
    "cleaned_profiles": ProcessingLevel.CLEANED,
    "cleaned_and_phase_corrected_profiles": ProcessingLevel.CLEANED_AND_CORRECTED,
}

# topology_processing_level name (legacy) → topology processing level
TOPOLOGY_LEVELS = {
    "raw": ProcessingLevel.RAW,
    "cleaned": ProcessingLevel.CLEANED,
    "cleaned_and_corrected": ProcessingLevel.CLEANED_AND_CORRECTED,
}

logger = logging.getLogger(__name__)


def _topology_level(topology_processing_level: str) -> ProcessingLevel:
    try:
        return TOPOLOGY_LEVELS[topology_processing_level]
    except KeyError:
        raise ValueError("topology_processing_level has to be 'raw', 'cleaned' or 'cleaned_and_corrected'.") from None


def _profile_level(profile: str) -> ProcessingLevel:
    if profile not in PROFILE_TO_LEVEL:
        raise ValueError(f"'profile' needs to be in {ALLOWED_PROFILES}")
    return PROFILE_TO_LEVEL[profile]


class DataExtractor:
    def __init__(
        self,
        phase_measurements_dir: str = "phase_measurements/raw",
        connector: BaseConnector | None = None,
        backend: str | None = None,
    ):
        """
        Args:
            phase_measurements_dir: Dataset root of the raw timeseries within the
                bucket/container.
            connector: Optional injected object-storage connector rooted at
                ``phase_measurements_dir``. Takes precedence over ``backend``.
            backend: Optional backend name ("s3" / "azure") for the connector
                factory when no connector is injected.
        """
        # Schemas of the canonical timeseries layout
        self.phase_measurements_csv_schema = PhaseMeasurementsCsvSchema()
        self.phase_measurements_parquet_schema = PhaseMeasurementsParquetSchema()

        # Connector rooted at the raw timeseries dataset. The attribute keeps its
        # historical name `s3_connector` for backwards compatibility, but may hold
        # any BaseConnector implementation; `connector` is the preferred alias.
        self.connector = connector or create_connector(phase_measurements_dir, backend=backend)
        self.s3_connector = self.connector

        # Controllers — the actual implementations behind this proxy
        self.meta_controller = MetaController(threephi_db.new_session)
        self.ingestion_controller = IngestionController(threephi_db.new_session)
        self.topology_controller = TopologyController(threephi_db.new_session)
        ts_base = phase_measurements_dir.removesuffix("/raw")
        self.time_series_controller = TimeSeriesController(self.connector.with_data_dir(ts_base))

        # Kept as `s3_base` for backwards compatibility; holds the storage root of
        # whatever backend the connector targets (e.g. "s3://3phi" or "az://3phi").
        self.s3_base = self.connector.storage_base
        self.sourcedata_dir = f"{self.s3_base}/{SOURCE_DATA_DIR}"
        self.processed_data_dir = f"{self.s3_base}/{PROCESSED_DATA_DIR}"

        # Cached timeseries metadata (filled lazily from the meta DB)
        self.min_timestamp = None
        self.max_timestamp = None
        self.expected_timestamps = None
        self.id_list_of_sms_with_data = None

        # Cached topology mappings, keyed by topology_processing_level
        self._cabinet_sm_mappings: dict[str, dict] = {}
        self._topology_sm_mappings: dict[str, dict] = {}
        self._sm_topology_mappings: dict[str, dict] = {}

        # Legacy aliases kept for code that read these attributes directly
        self.raw_cabinet_sm_mapping_dict = None
        self.cleaned_cabinet_sm_mapping_dict = None
        self.cleaned_and_corrected_cabinet_sm_mapping_dict = None
        self.raw_topology_dict = None
        self.raw_topology_dict_reversed = None
        self.cleaned_topology_dict = None
        self.cleaned_topology_dict_reversed = None
        self.cleaned_and_corrected_topology_dict = None
        self.cleaned_and_corrected_topology_dict_reversed = None

        # Column names of the canonical layout (kept public — used by data apps)
        self.meter_number_col = "meter_number"
        self.timestamp_col = "timestamp"
        self.cabinet_col = "cabinet"
        self.sec_substation_col = "secondary_substation"
        self.transformer_col = "transformer"
        self.lv_feeder_col = "lv_feeder"
        self.node_1_col = "node1"
        self.node_2_col = "node2"
        self.voltage_col = "voltage_l"
        self.current_col = "current_l"
        self.active_power_p14_col = "active_power_p14_l"
        self.active_power_p23_col = "active_power_p23_l"
        self.reactive_power_q12_col = "reactive_power_q12_l"
        self.reactive_power_q34_col = "reactive_power_q34_l"
        self.v_phase_unbalance_col = "v_unbalance_l"
        self.v_unbalance_col = "v_unbalance"
        self.i_phase_unbalance_col = "i_unbalance_l"
        self.i_unbalance_col = "i_unbalance"
        self.cabinet_name = "Cabinet"

    # ------------------------------------------------------------------ #
    # Canonical (v1) ingestion and access methods
    # ------------------------------------------------------------------ #

    def v1_csv_to_parquet_partitions(self, csv_path, csv_file_pattern, bucket_dest_path):
        """
        Method to transform the raw source csv files to partitioned parquet files.
        :param csv_path: Path where raw csv files are stored (local file storage)
        :param csv_file_pattern: Naming pattern of raw csv files
        :param bucket_dest_path: Destination path where to store parquet files on bucket
        :return:
        """
        csv_ts_col = self.phase_measurements_csv_schema.timestamp_col
        parquet_ts_col = self.phase_measurements_parquet_schema.timestamp_col
        parquet_meter_col = self.phase_measurements_parquet_schema.meter_col

        csv_path_pattern = os.path.join(csv_path, csv_file_pattern)
        logger.info(f"Reading CSVs from {csv_path_pattern}")
        csv_files = glob.glob(csv_path_pattern)

        def year_month_key(p: str):
            name = Path(p).name
            m = re.search(r"(\d{4})_(\d{1,2})(?=\.csv$)", name)
            if not m:
                # push anything that doesn't match to the end, tiebreak by name
                return float("inf"), float("inf"), name
            y, mth = map(int, m.groups())
            return y, mth

        csv_files = sorted(csv_files, key=year_month_key)
        logger.info(f"Found CSV files to process: {csv_files}")

        schema_version = VERSION
        run_id = dt.datetime.now(dt.UTC).strftime("%Y%m%d-%H%M%S") + "-" + uuid.uuid4().hex[:8]
        staging_path = f"{bucket_dest_path}/staging/run={run_id}"
        ready_path = bucket_dest_path

        logger.info(f"Starting run {run_id}")

        for csv_path in csv_files:
            logger.info(f"Processing {csv_path}")
            # ---------------------------------
            # Create batch row for file
            # ---------------------------------
            batch_id = self.ingestion_controller.insert_batch(csv_path, run_id)

            # ---------------------------------
            # Read in CSV,
            # normalize timestamp to UTC,
            # rename timestamp column,
            # type important cols,
            # drop rows with invalid timestamps,
            # drop rows without meter_number
            # ---------------------------------

            dask_df = dd.read_csv(csv_path, parse_dates=[csv_ts_col])
            dask_df[csv_ts_col] = dd.to_datetime(dask_df[csv_ts_col], utc=True, errors="coerce")
            dask_df = dask_df.rename(columns={csv_ts_col: parquet_ts_col})
            dask_df[parquet_meter_col] = dask_df[parquet_meter_col].astype("string")
            dask_df = dask_df.dropna(subset=[parquet_ts_col])
            dask_df = dask_df.dropna(subset=[parquet_meter_col])

            # ---------------------------------
            # Get stats for meter inventory table
            # ---------------------------------
            stats_workflow = f"{csv_path}_stats"
            if not self.meta_controller.is_workflow_completed(stats_workflow):
                self.meta_controller.start_workflow(stats_workflow)

                logger.info("Computing meter inventory stats.")
                stats_ddf = dask_df[[parquet_meter_col, parquet_ts_col]]
                agg_ddf = stats_ddf.groupby(parquet_meter_col)[parquet_ts_col].agg(["min", "max", "count"])

                agg_pdf = agg_ddf.compute()
                agg_pdf = agg_pdf.rename(columns={"min": "first_seen", "max": "last_seen", "count": "total_rows"})
                agg_pdf = agg_pdf.reset_index(names=["id"])
                self.meta_controller.upsert_meter_stats(agg_pdf)
                self.meta_controller.complete_workflow(stats_workflow)
                logger.info("Successfully computed and stored stats.")

            # ---------------------------------
            # Partitioning Logic,
            # add dt and shard columns,
            # sort to improve parquet stats,
            # pass information about new cols via "meta" prop,
            # shuffle on partition keys so we only get one file per dt/shard
            # ---------------------------------

            def add_dt_and_shard(pandas_df: pd.DataFrame) -> pd.DataFrame:
                pandas_df["dt"] = pandas_df[parquet_ts_col].dt.strftime("%Y-%m-%d").astype("string")
                pandas_df["shard"] = pandas_df[parquet_meter_col].apply(v1_get_shard_for_meter_id).astype("int16")
                pandas_df = pandas_df.sort_values([parquet_meter_col, parquet_ts_col])
                return pandas_df

            meta = dask_df._meta.assign(
                dt=pd.Series([], dtype="string"),
                shard=pd.Series([], dtype="int16"),
            )

            dask_df = dask_df.map_partitions(add_dt_and_shard, meta=meta)

            # ---------------------------------
            # Prepare write settings
            # ---------------------------------

            write_kwargs = {
                "engine": "pyarrow",
                "write_index": False,
                "partition_on": ["dt", "shard"],
                "compression": "zstd",
                "write_statistics": [
                    self.phase_measurements_parquet_schema.meter_col,
                    self.phase_measurements_parquet_schema.timestamp_col,
                ],  # enables skipping of row-groups based on meter_number or timestamp
                "row_group_size": 1_000_000,  # row-group size, ideal between 64-128MB
                "use_deprecated_int96_timestamps": False,  # ensure portability of files
                # can become a bottleneck for large, continuously changing datasets
                "write_metadata_file": False,
            }

            # ---------------------------------
            # Write Parquet files to staging,
            # (execution up to here was lazy,
            # now workflow actually triggers)
            # create file index entries,
            # set batch to "Processed"
            # ---------------------------------
            self.s3_connector.write_parquet(staging_path, dask_df, **write_kwargs)

            staged_files = self.s3_connector.discover_parquet_files(staging_path)

            logger.info(f"Discovered {len(staged_files)} staged files.")
            filename_pattern = re.compile(r"(?:^|/)dt=(\d{4}-\d{2}-\d{2})/shard=(\d+)/([^/]+\.parquet)$")
            files_grouped_by_ring: dict[tuple[str, int], list[str]] = {}

            for file in staged_files:
                # make sure we are reading a partition shard file
                match = filename_pattern.search(file)
                if not match:
                    # skip unexpected file (might be a log file or something)
                    continue
                dt_str, shard_str, fname = match.groups()
                ring_key = (dt_str, int(shard_str))
                files_grouped_by_ring.setdefault(ring_key, []).append(file)

            logger.info(f"Files grouped in {len(files_grouped_by_ring.keys())} rings.")
            for (dt_str, shard), files in files_grouped_by_ring.items():
                logger.info(f"Processing ring: {dt_str}, {shard}")
                files.sort()
                seq = self.ingestion_controller.get_current_max_seq_for_ring(dt_str, shard)

                for file in files:
                    seq += 1
                    file_stats = self.s3_connector.get_parquet_file_stats(file, dask_df, parquet_ts_col)
                    # prepare the s3 key for the "ready" file
                    ready_key = file.replace(f"/staging/run={run_id}/", "/")
                    self.ingestion_controller.upsert_file_index(
                        s3_key=ready_key,
                        dt=dt_str,
                        shard=shard,
                        seq=seq,
                        ts_start=file_stats["ts_min"],
                        ts_end=file_stats["ts_max"],
                        rows=file_stats["row_count"],
                        bytes=file_stats["size_bytes"],
                        schema_version=schema_version,
                        status="staged",
                        batch_id=batch_id,
                        ingest_file=csv_path,
                    )

            # ---------------------------------
            # Promote staged files to ready,
            # update file index
            # ---------------------------------

            logger.info(f"Promoting batch from stg {staging_path} to ready {ready_path} ...")
            self.s3_connector.promote_staged_to_ready(staging_path, bucket_dest_path)
            self.ingestion_controller.promote_batch_to_ready(batch_id=batch_id)
            logger.info(f"Processed {csv_path} as batch {batch_id}.")

        logger.info("Ingestion complete.")

    def v1_get_timeseries_info(self):
        """
        Function for extracting some information from the timeseries data.

        Information:
            - min_timestamp: earliest SM measurement timestamp
            - max_timestamp: last SM measurement timestamp
            - id_list_of_sms_with_data: list of meter IDS that we have data for
        """
        return self.meta_controller.get_time_series_meta_info()

    def v1_get_single_meter_data(self, id: str) -> dd.DataFrame:
        ddf: dd.DataFrame = self.s3_connector.get_meter_data(meter_ids=[id])
        return ddf

    # ------------------------------------------------------------------ #
    # Timeseries metadata
    # ------------------------------------------------------------------ #

    def _get_timeseries_info_db(self, overwrite=None):
        logger.info("Getting timeseries info from DB")
        if overwrite:
            logger.warning("Parameter overwrite is deprecated, please don't use anymore.")
        timeseries_info = self.meta_controller.get_time_series_meta_info()
        self.min_timestamp = timeseries_info["min_timestamp"]
        self.max_timestamp = timeseries_info["max_timestamp"]
        self.id_list_of_sms_with_data = timeseries_info["id_list_of_sms_with_data"]

        # Create expected timestamps (used later for detecting and filling
        # missing time steps) if not done before
        if self.expected_timestamps is None:
            self.expected_timestamps = pd.date_range(start=self.min_timestamp, end=self.max_timestamp, freq="15min")

    def _get_timeseries_info(self, overwrite=False):
        """Legacy alias — timeseries metadata now always comes from the meta DB."""
        if self.min_timestamp is None or self.max_timestamp is None or self.id_list_of_sms_with_data is None:
            self._get_timeseries_info_db()

    def _ids_with_data(self) -> set[str]:
        self._get_timeseries_info()
        return {str(i) for i in self.id_list_of_sms_with_data}

    # ------------------------------------------------------------------ #
    # Topology mappings (built from the DB-backed topology)
    # ------------------------------------------------------------------ #

    def _export_sm_cabinet_pdf(self, topology_processing_level: str) -> pd.DataFrame:
        return self.topology_controller.export_sm_cabinet(level=_topology_level(topology_processing_level))

    def _export_topology_pdf(self, topology_processing_level: str) -> pd.DataFrame:
        return self.topology_controller.export_topology(level=_topology_level(topology_processing_level))

    def _build_cabinet_sm_mapping(self, topology_processing_level: str) -> dict:
        """``{cabinet_id: {"METER_NUMBER": [...], "AVAILABLE_METERS": [...], "MISSING_METERS": [...]}}``

        Meter IDs are strings; AVAILABLE/MISSING is determined by the meter inventory
        (``meta.meter.total_rows > 0``).
        """
        sm_cab = self._export_sm_cabinet_pdf(topology_processing_level)
        ids_with_data = self._ids_with_data()

        sm_cab = sm_cab[sm_cab[self.cabinet_col].notna() & sm_cab[self.meter_number_col].notna()].copy()
        sm_cab["cabinet_id"] = sm_cab[self.cabinet_col].astype("string").str.split(".").str[1]
        sm_cab["meter_str"] = sm_cab[self.meter_number_col].astype("Int64").astype("string")

        grouped = sm_cab.groupby("cabinet_id")["meter_str"].apply(list)

        mapping = {}
        for cabinet_id, meters in grouped.items():
            meters = [str(m) for m in meters if pd.notna(m)]
            mapping[str(cabinet_id)] = {
                "METER_NUMBER": meters,
                "AVAILABLE_METERS": [m for m in meters if m in ids_with_data],
                "MISSING_METERS": [m for m in meters if m not in ids_with_data],
            }
        return mapping

    def _build_topology_sm_mapping(self, topology_processing_level: str) -> dict:
        """Nested ``{zip: {substation: {transformer: {feeder: {cabinet: meters}}}}}`` mapping."""
        cabinet_mapping = self._get_cabinet_sm_mapping(topology_processing_level)
        topology = self._export_topology_pdf(topology_processing_level)

        topology_dict: dict = {}
        zip_groups = topology.groupby("zip_code_secondary_substation")

        for zip_code, zip_df in zip_groups:
            topology_dict[zip_code] = {}

            secondary_subs = zip_df.groupby(self.sec_substation_col)[self.transformer_col].unique().reset_index()
            trafos = zip_df.groupby(self.transformer_col)[self.lv_feeder_col].unique().reset_index()
            lv_feeder_1 = zip_df.groupby(self.lv_feeder_col)[self.node_1_col].unique().reset_index()
            lv_feeder_2 = zip_df.groupby(self.lv_feeder_col)[self.node_2_col].unique().reset_index()

            for _, sec_sub in secondary_subs.iterrows():
                sub_key = sec_sub[self.sec_substation_col]
                topology_dict[zip_code][sub_key] = {}
                for trafo in (t for t in sec_sub[self.transformer_col] if pd.notna(t)):
                    topology_dict[zip_code][sub_key][trafo] = {}
                    lv_feeder = trafos[trafos[self.transformer_col] == trafo][self.lv_feeder_col].iloc[0]
                    for lv_f in (f for f in lv_feeder if pd.notna(f)):
                        topology_dict[zip_code][sub_key][trafo][lv_f] = {}
                        node_1_boxes = lv_feeder_1[lv_feeder_1[self.lv_feeder_col] == lv_f][self.node_1_col].iloc[0]
                        node_2_boxes = lv_feeder_2[lv_feeder_2[self.lv_feeder_col] == lv_f][self.node_2_col].iloc[0]
                        node_1_boxes = [n for n in node_1_boxes if pd.notna(n) and self.cabinet_name in n]
                        node_2_boxes = [n for n in node_2_boxes if pd.notna(n) and self.cabinet_name in n]
                        for cabinet in set(node_1_boxes + node_2_boxes):
                            meters = cabinet_mapping.get(
                                cabinet.split(".")[1],
                                {"METER_NUMBER": [], "AVAILABLE_METERS": [], "MISSING_METERS": []},
                            )
                            topology_dict[zip_code][sub_key][trafo][lv_f][cabinet] = meters

        return topology_dict

    @staticmethod
    def _reverse_topology_sm_mapping(topology_sm_mapping):
        sm_to_location = {}

        for zip_code, substations in topology_sm_mapping.items():
            zip_code_str = str(zip_code)
            for substation, transformers in substations.items():
                substation_id = substation.split(".")[-1]
                for trafo, feeders in transformers.items():
                    trafo_id = trafo.split(".")[-1]
                    for feeder, cabinets in feeders.items():
                        feeder_id = feeder.split(".")[-1]
                        for cabinet, meters in cabinets.items():
                            cabinet_id = cabinet.split(".")[-1]
                            for sm_id in meters.get("METER_NUMBER", []):
                                sm_to_location[sm_id] = {
                                    "Zip Code": zip_code_str,
                                    "Secondary Substation ID": substation_id,
                                    "Transformer ID": trafo_id,
                                    "Feeder ID": feeder_id,
                                    "Cabinet ID": cabinet_id,
                                    "Has data": (sm_id in meters["AVAILABLE_METERS"]),
                                }

        return sm_to_location

    def _get_cabinet_sm_mapping(self, level: str, overwrite: bool = False) -> dict:
        if overwrite or level not in self._cabinet_sm_mappings:
            self._cabinet_sm_mappings[level] = self._build_cabinet_sm_mapping(level)
        return self._cabinet_sm_mappings[level]

    def _get_topology_sm_mapping(self, level: str, overwrite: bool = False) -> dict:
        if overwrite or level not in self._topology_sm_mappings:
            self._topology_sm_mappings[level] = self._build_topology_sm_mapping(level)
        return self._topology_sm_mappings[level]

    def _get_sm_topology_mapping(self, level: str, overwrite: bool = False) -> dict:
        if overwrite or level not in self._sm_topology_mappings:
            self._sm_topology_mappings[level] = self._reverse_topology_sm_mapping(
                self._get_topology_sm_mapping(level, overwrite)
            )
        return self._sm_topology_mappings[level]

    def _save_mapping_json(self, mapping: dict, level: str, filename: str) -> None:
        filepath = f"{self.processed_data_dir}/topology/{level}/{filename}"
        self.s3_connector.write_json(filepath, mapping)
        logger.info(f"Mapping saved to {filepath}.")

    # --- public mapping API (legacy signatures preserved) --- #

    def create_raw_cabinet_sm_mapping(self, save=True, overwrite=False, overwrite_timeseries_info=False):
        mapping = self._get_cabinet_sm_mapping("raw", overwrite)
        self.raw_cabinet_sm_mapping_dict = mapping
        if save:
            self._save_mapping_json(mapping, "raw", "Cabinet_SM_mapping.json")
        return mapping

    def create_cleaned_cabinet_sm_mapping(self, save=True, overwrite=False, overwrite_timeseries_info=False):
        mapping = self._get_cabinet_sm_mapping("cleaned", overwrite)
        self.cleaned_cabinet_sm_mapping_dict = mapping
        if save:
            self._save_mapping_json(mapping, "cleaned", "Cabinet_SM_mapping.json")
        return mapping

    def create_cleaned_and_corrected_cabinet_sm_mapping(
        self, save=True, overwrite=False, overwrite_timeseries_info=False
    ):
        mapping = self._get_cabinet_sm_mapping("cleaned_and_corrected", overwrite)
        self.cleaned_and_corrected_cabinet_sm_mapping_dict = mapping
        if save:
            self._save_mapping_json(mapping, "cleaned_and_corrected", "Cabinet_SM_mapping.json")
        return mapping

    def create_raw_topology_sm_mapping(self, save=True, overwrite=False, overwrite_timeseries_info=False) -> dict:
        mapping = self._get_topology_sm_mapping("raw", overwrite)
        self.raw_topology_dict = mapping
        if save:
            self._save_mapping_json(mapping, "raw", "Topology_SM_mapping.json")
        return mapping

    def create_cleaned_topology_sm_mapping(self, save=True, overwrite=False, overwrite_timeseries_info=False) -> dict:
        mapping = self._get_topology_sm_mapping("cleaned", overwrite)
        self.cleaned_topology_dict = mapping
        if save:
            self._save_mapping_json(mapping, "cleaned", "Topology_SM_mapping.json")
        return mapping

    def create_cleaned_and_corrected_topology_sm_mapping(
        self, save=True, overwrite=False, overwrite_timeseries_info=False
    ) -> dict:
        mapping = self._get_topology_sm_mapping("cleaned_and_corrected", overwrite)
        self.cleaned_and_corrected_topology_dict = mapping
        if save:
            self._save_mapping_json(mapping, "cleaned_and_corrected", "Topology_SM_mapping.json")
        return mapping

    def create_raw_sm_topology_mapping(self, save=True, overwrite=False, overwrite_timeseries_info=False):
        mapping = self._get_sm_topology_mapping("raw", overwrite)
        self.raw_topology_dict_reversed = mapping
        if save:
            self._save_mapping_json(mapping, "raw", "SM_Topology_mapping.json")
        return mapping

    def create_cleaned_sm_topology_mapping(self, save=True, overwrite=False, overwrite_timeseries_info=False):
        mapping = self._get_sm_topology_mapping("cleaned", overwrite)
        self.cleaned_topology_dict_reversed = mapping
        if save:
            self._save_mapping_json(mapping, "cleaned", "SM_Topology_mapping.json")
        return mapping

    def create_cleaned_and_corrected_sm_topology_mapping(
        self, save=True, overwrite=False, overwrite_timeseries_info=False
    ):
        mapping = self._get_sm_topology_mapping("cleaned_and_corrected", overwrite)
        self.cleaned_and_corrected_topology_dict_reversed = mapping
        if save:
            self._save_mapping_json(mapping, "cleaned_and_corrected", "SM_Topology_mapping.json")
        return mapping

    # ------------------------------------------------------------------ #
    # SM-ID lookups per topology entity
    # ------------------------------------------------------------------ #

    @staticmethod
    def _validate_id_list(ids, what: str) -> list[str]:
        series = pd.Series(list(ids), dtype=pd.StringDtype())
        if series.isna().any():
            raise ValueError(f"One or more {what} are invalid.")
        return series.tolist()

    def get_sm_ids_for_cabinet(
        self,
        cabinet_id,
        sm_ids_of_cabinet,
        topology_processing_level,
        overwrite_topology_info=False,
        overwrite_timeseries_info=False,
    ):
        if sm_ids_of_cabinet is None:
            mapping = self._get_cabinet_sm_mapping(topology_processing_level, overwrite_topology_info)
            cabinet_id = str(cabinet_id)
            if cabinet_id not in mapping:
                raise ValueError(f"Cabinet {cabinet_id} does not exist in the dataset.")
            sm_ids_of_cabinet = mapping[cabinet_id]["AVAILABLE_METERS"]
        return self._validate_id_list(sm_ids_of_cabinet, "sm_ids_of_cabinet")

    def get_sm_ids_for_feeder(
        self,
        feeder_id,
        sm_ids_of_feeder,
        topology_processing_level,
        overwrite_topology_info=False,
        overwrite_timeseries_info=False,
    ):
        if sm_ids_of_feeder is None:
            topology_dict = self._get_topology_sm_mapping(topology_processing_level, overwrite_topology_info)

            feeder_ids = []
            for zip_dict in topology_dict.values():
                for sub_dict in zip_dict.values():
                    for trafo_dict in sub_dict.values():
                        for lv_feeder in trafo_dict:
                            feeder_ids.append(lv_feeder.split(".")[1])

            if str(feeder_id) not in feeder_ids:
                raise ValueError(f"Feeder {feeder_id} does not exist in available data.")

            sm_ids_of_feeder = []
            for zip_dict in topology_dict.values():
                for sub_dict in zip_dict.values():
                    for trafo_dict in sub_dict.values():
                        for lv_feeder, lvfeeder in trafo_dict.items():
                            if lv_feeder.endswith(str(feeder_id)):
                                for sm_ids in lvfeeder.values():
                                    sm_ids_of_feeder.extend(sm_ids.get("AVAILABLE_METERS", []))

        return self._validate_id_list(sm_ids_of_feeder, "sm_ids_of_feeder")

    def get_sm_ids_for_transformer(
        self,
        transformer_id,
        sm_ids_of_transformer,
        topology_processing_level,
        overwrite_topology_info=False,
        overwrite_timeseries_info=False,
    ):
        if sm_ids_of_transformer is None:
            topology_dict = self._get_topology_sm_mapping(topology_processing_level, overwrite_topology_info)

            transformer_ids = []
            for zip_dict in topology_dict.values():
                for sub_dict in zip_dict.values():
                    for trafo in sub_dict:
                        transformer_ids.append(trafo.split(".")[1])

            if str(transformer_id) not in transformer_ids:
                raise ValueError(f"Transformer {transformer_id} does not exist in available data.")

            sm_ids_of_transformer = []
            for zip_dict in topology_dict.values():
                for sub_dict in zip_dict.values():
                    for trafo, trafo_dict in sub_dict.items():
                        if trafo.endswith(str(transformer_id)):
                            for lvfeeder in trafo_dict.values():
                                for sm_ids in lvfeeder.values():
                                    sm_ids_of_transformer.extend(sm_ids.get("AVAILABLE_METERS", []))

        return self._validate_id_list(sm_ids_of_transformer, "sm_ids_of_transformer")

    def get_sm_ids_for_secondary_substation(
        self,
        substation_id,
        sm_ids_of_secondary_substation,
        topology_processing_level,
        overwrite_topology_info=False,
        overwrite_timeseries_info=False,
    ):
        if sm_ids_of_secondary_substation is None:
            topology_dict = self._get_topology_sm_mapping(topology_processing_level, overwrite_topology_info)

            substation_ids = []
            for zip_dict in topology_dict.values():
                for sub in zip_dict:
                    substation_ids.append(sub.split(".")[1])

            if str(substation_id) not in substation_ids:
                raise ValueError(f"Substation {substation_id} does not exist in available data.")

            sm_ids_of_secondary_substation = []
            for zip_dict in topology_dict.values():
                for sub, sub_dict in zip_dict.items():
                    if sub.endswith(str(substation_id)):
                        for trafo_dict in sub_dict.values():
                            for lvfeeder in trafo_dict.values():
                                for sm_ids in lvfeeder.values():
                                    sm_ids_of_secondary_substation.extend(sm_ids.get("AVAILABLE_METERS", []))

        return self._validate_id_list(sm_ids_of_secondary_substation, "sm_ids_of_secondary_substation")

    # ------------------------------------------------------------------ #
    # Wide per-meter frame assembly (the core of all extract/load methods)
    # ------------------------------------------------------------------ #

    def _wide_sm_frames(
        self,
        meter_ids,
        profile: str,
        add_current: bool = False,
        add_unbalance: bool = False,
    ) -> list[pd.DataFrame | None]:
        """Build one wide (suffixed-column) 15-minute frame per meter.

        Reads the canonical store at the processing level mapped from ``profile``,
        then per meter: index by timestamp, resample to 15 minutes, reindex to the
        full expected range, suffix every column with ``_{meter_id}``, and apply
        the requested enrichment. Meters without any rows yield ``None``.
        """
        level = _profile_level(profile)
        ids = self._validate_id_list(meter_ids, "meter_ids")
        if not ids:
            return []

        self._get_timeseries_info()

        pdf = self.time_series_controller.get_time_series_data(ids, processing_level=level).compute()
        ts_col = self.phase_measurements_parquet_schema.timestamp_col
        meter_col = self.phase_measurements_parquet_schema.meter_col

        frames: list[pd.DataFrame | None] = []
        for meter_id in ids:
            sub = pdf[pdf[meter_col].astype(str) == meter_id]
            if sub.empty:
                frames.append(None)
                continue
            frames.append(self._shape_single_meter_frame(sub, meter_id, ts_col, meter_col, add_current, add_unbalance))
        return frames

    def _shape_single_meter_frame(
        self, sub: pd.DataFrame, meter_id: str, ts_col: str, meter_col: str, add_current: bool, add_unbalance: bool
    ) -> pd.DataFrame:
        meter_df = sub.drop(columns=[meter_col, "shard", "dt"], errors="ignore").copy()
        meter_df[ts_col] = pd.to_datetime(meter_df[ts_col], utc=True)
        meter_df = meter_df.sort_values(by=ts_col).set_index(ts_col)

        # Keep only numeric columns, resample to the 15-minute grid and fill the
        # full expected range (NaN outside the meter's recording period)
        meter_df = meter_df.select_dtypes(include=[np.number])
        meter_df = meter_df.resample("15min").mean()
        meter_df = meter_df.reindex(self.expected_timestamps)
        meter_df.index.name = ts_col

        # Add the SM ID to the column names
        meter_df.columns = [f"{col}_{meter_id}" for col in meter_df.columns]

        return self.enrich_meter_df(
            meter_df=meter_df,
            meter_id=meter_id,
            add_current=add_current,
            add_unbalance=add_unbalance,
        )

    # --- per-SM API --- #

    def extract_raw_dataset_for_sm(
        self,
        meter_id,
        meter_df=None,
        add_current=False,
        add_unbalance=False,
        save=False,
        overwrite_timeseries_info=False,
    ):
        """Return the wide raw-profile frame for one meter (assembled from the canonical store).

        Raises ValueError if the meter has no timeseries data. The ``meter_df``
        parameter (a pre-filtered frame from the legacy implementation) is ignored.
        """
        if pd.isna(meter_id):
            raise ValueError("Invalid meter_id.")
        meter_id = str(meter_id)
        if meter_df is not None:
            logger.warning("extract_raw_dataset_for_sm: meter_df parameter is deprecated and ignored.")

        if meter_id not in self._ids_with_data():
            raise ValueError(f"No data available in timeseries dataset for meter {meter_id}.")

        frame = self._wide_sm_frames([meter_id], "raw_profiles", add_current, add_unbalance)[0]
        if frame is None:
            raise ValueError(f"No data available in timeseries dataset for meter {meter_id}.")

        if save:
            filepath = f"{self.processed_data_dir}/individual_sm_datasets/raw_profiles/sm_{meter_id}.parquet"
            self.s3_connector.write_parquet(filepath, frame.reset_index())
            logger.info(f"Raw data of SM {meter_id} saved to {filepath}.")

        return frame

    def _load_dataset_for_sm(self, meter_id, profile, add_current, add_unbalance):
        if pd.isna(meter_id):
            raise ValueError("Invalid meter_id.")
        return self._wide_sm_frames([str(meter_id)], profile, add_current, add_unbalance)[0]

    def load_raw_dataset_for_sm(self, meter_id, add_current=False, add_unbalance=False, overwrite=False):
        """Return the wide raw-profile frame for one meter, or None if it has no data."""
        frame = self._load_dataset_for_sm(meter_id, "raw_profiles", add_current, add_unbalance)
        if frame is None:
            logger.info(f"Dataset of SM {meter_id} does not exist.")
        return frame

    def load_cleaned_dataset_for_sm(self, meter_id, add_current=False, add_unbalance=False, overwrite=False):
        """Return the wide cleaned-profile frame for one meter.

        Raises FileNotFoundError if the meter has no data (legacy behavior).
        """
        frame = self._load_dataset_for_sm(meter_id, "cleaned_profiles", add_current, add_unbalance)
        if frame is None:
            raise FileNotFoundError(f"Cleaned dataset of SM {meter_id} does not exist. Use Timeseries Cleaner first.")
        return frame

    def load_cleaned_and_phase_corrected_dataset_for_sm(
        self, meter_id, add_current=False, add_unbalance=False, overwrite=False
    ):
        """Return the wide cleaned-and-phase-corrected frame for one meter.

        Raises FileNotFoundError if the meter has no data (legacy behavior).
        """
        frame = self._load_dataset_for_sm(meter_id, "cleaned_and_phase_corrected_profiles", add_current, add_unbalance)
        if frame is None:
            raise FileNotFoundError(
                f"Cleaned and phase-corrected dataset of SM {meter_id} does not exist. "
                f"Use Timeseries Cleaner and Phase Corrector first."
            )
        return frame

    def _load_dataset_for_multiple_sm(
        self, meter_ids, profiles, add_current=False, add_unbalance=False, overwrite=False
    ):
        return self._wide_sm_frames(meter_ids, profiles, add_current, add_unbalance)

    def load_raw_dataset_for_multiple_sm(self, meter_ids, add_current=False, add_unbalance=False, overwrite=False):
        return self._load_dataset_for_multiple_sm(meter_ids, "raw_profiles", add_current, add_unbalance, overwrite)

    def load_cleaned_dataset_for_multiple_sm(self, meter_ids, add_current=False, add_unbalance=False, overwrite=False):
        return self._load_dataset_for_multiple_sm(meter_ids, "cleaned_profiles", add_current, add_unbalance, overwrite)

    def load_cleaned_and_phase_corrected_dataset_for_multiple_sm(
        self, meter_ids, add_current=False, add_unbalance=False, overwrite=False
    ):
        return self._load_dataset_for_multiple_sm(
            meter_ids,
            "cleaned_and_phase_corrected_profiles",
            add_current,
            add_unbalance,
            overwrite,
        )

    def list_partitioned_meter_ids(self, partition_dir):
        base_path = partition_dir.rstrip("/")
        entries = self.s3_connector.fs.ls(base_path)

        meter_ids = [
            entry.split(f"{self.meter_number_col}=")[-1].split("/")[0]
            for entry in entries
            if f"{self.meter_number_col}=" in entry
        ]

        return meter_ids

    def extract_raw_datasets_for_all_sm(
        self, add_current=False, add_unbalance=False, overwrite_timeseries_info=False, chunk_size=200
    ):
        """Materialize the wide raw-profile frame of every meter with data as per-SM parquet files.

        Kept for backwards compatibility; with the canonical store the per-SM files are
        no longer needed for reading (all load/extract methods assemble frames on demand).
        """
        ids = sorted(self._ids_with_data())
        out_dir = f"{self.processed_data_dir}/individual_sm_datasets/raw_profiles"
        logger.info(f"Materializing {len(ids)} per-SM raw profiles to {out_dir}")
        for start in range(0, len(ids), chunk_size):
            chunk = ids[start : start + chunk_size]
            frames = self._wide_sm_frames(chunk, "raw_profiles", add_current, add_unbalance)
            for meter_id, frame in zip(chunk, frames, strict=True):
                if frame is not None:
                    self.s3_connector.write_parquet(f"{out_dir}/sm_{meter_id}.parquet", frame.reset_index())

    # --- per-topology-entity API --- #

    def _extract_for_entity(
        self,
        *,
        sm_ids,
        profile,
        add_current,
        add_unbalance,
        save,
        save_path,
        entity_desc,
    ):
        sm_data_list = self._wide_sm_frames(sm_ids, profile, add_current, add_unbalance)
        sm_data_list = [f for f in sm_data_list if f is not None]

        if sm_data_list:
            entity_data = pd.concat(sm_data_list, axis=1)
        else:
            logger.info(f"No SM data for {entity_desc}.")
            entity_data = pd.DataFrame()

        if save:
            self.s3_connector.write_parquet(save_path, entity_data, engine="pyarrow")
            logger.info(f"Profile '{profile}': SM data of {entity_desc} saved to {save_path}.")

        return entity_data

    def extract_raw_sm_dataset_for_cabinet(
        self,
        cabinet_id,
        topology_processing_level,
        use_existing_raw_sm_profiles=True,
        sm_ids_of_cabinet=None,
        add_current=False,
        add_unbalance=False,
        save=False,
        overwrite_topology_info=False,
        overwrite_timeseries_info=False,
    ):
        return self.extract_timeseries_data_for_cabinet(
            cabinet_id=cabinet_id,
            sm_ids_of_cabinet=sm_ids_of_cabinet,
            topology_processing_level=topology_processing_level,
            add_current=add_current,
            add_unbalance=add_unbalance,
            save=save,
            overwrite_topology_info=overwrite_topology_info,
            overwrite_timeseries_info=overwrite_timeseries_info,
            profile="raw_profiles",
        )

    def extract_cleaned_sm_dataset_for_cabinet(
        self,
        cabinet_id,
        topology_processing_level,
        sm_ids_of_cabinet=None,
        add_current=False,
        add_unbalance=False,
        save=False,
        overwrite_topology_info=False,
        overwrite_timeseries_info=False,
    ):
        return self.extract_timeseries_data_for_cabinet(
            cabinet_id=cabinet_id,
            sm_ids_of_cabinet=sm_ids_of_cabinet,
            topology_processing_level=topology_processing_level,
            add_current=add_current,
            add_unbalance=add_unbalance,
            save=save,
            overwrite_topology_info=overwrite_topology_info,
            overwrite_timeseries_info=overwrite_timeseries_info,
            profile="cleaned_profiles",
        )

    def extract_cleaned_and_phase_corrected_sm_dataset_for_cabinet(
        self,
        cabinet_id,
        topology_processing_level,
        sm_ids_of_cabinet=None,
        add_current=False,
        add_unbalance=False,
        save=False,
        overwrite_topology_info=False,
        overwrite_timeseries_info=False,
    ):
        return self.extract_timeseries_data_for_cabinet(
            cabinet_id=cabinet_id,
            sm_ids_of_cabinet=sm_ids_of_cabinet,
            topology_processing_level=topology_processing_level,
            add_current=add_current,
            add_unbalance=add_unbalance,
            save=save,
            overwrite_topology_info=overwrite_topology_info,
            overwrite_timeseries_info=overwrite_timeseries_info,
            profile="cleaned_and_phase_corrected_profiles",
        )

    def extract_timeseries_data_for_cabinet(
        self,
        cabinet_id,
        sm_ids_of_cabinet,
        topology_processing_level,
        overwrite_topology_info=False,
        overwrite_timeseries_info=False,
        add_current=False,
        add_unbalance=False,
        save=False,
        profile="raw_profiles",
    ):
        if pd.isna(cabinet_id):
            raise ValueError("Invalid cabinet_id.")
        cabinet_id = str(cabinet_id)

        sm_ids = self.get_sm_ids_for_cabinet(
            cabinet_id=cabinet_id,
            sm_ids_of_cabinet=sm_ids_of_cabinet,
            topology_processing_level=topology_processing_level,
            overwrite_topology_info=overwrite_topology_info,
            overwrite_timeseries_info=overwrite_timeseries_info,
        )
        save_path = (
            f"{self.processed_data_dir}/individual_cabinet_datasets/"
            f"based_on_{topology_processing_level}_topology/{profile}/cabinet_{cabinet_id}.parquet"
        )
        return self._extract_for_entity(
            sm_ids=sm_ids,
            profile=profile,
            add_current=add_current,
            add_unbalance=add_unbalance,
            save=save,
            save_path=save_path,
            entity_desc=f"cabinet {cabinet_id}",
        )

    def extract_raw_sm_dataset_for_feeder(
        self,
        feeder_id,
        topology_processing_level,
        use_existing_raw_sm_profiles=True,
        sm_ids_of_feeder=None,
        add_current=False,
        add_unbalance=False,
        save=False,
        overwrite_topology_info=False,
        overwrite_timeseries_info=False,
    ):
        return self.extract_timeseries_data_for_feeder(
            feeder_id=feeder_id,
            sm_ids_of_feeder=sm_ids_of_feeder,
            topology_processing_level=topology_processing_level,
            overwrite_topology_info=overwrite_topology_info,
            overwrite_timeseries_info=overwrite_timeseries_info,
            add_current=add_current,
            add_unbalance=add_unbalance,
            save=save,
            profile="raw_profiles",
        )

    def extract_cleaned_sm_dataset_for_feeder(
        self,
        feeder_id,
        topology_processing_level,
        sm_ids_of_feeder=None,
        add_current=False,
        add_unbalance=False,
        save=False,
        overwrite_topology_info=False,
        overwrite_timeseries_info=False,
    ):
        return self.extract_timeseries_data_for_feeder(
            feeder_id=feeder_id,
            sm_ids_of_feeder=sm_ids_of_feeder,
            topology_processing_level=topology_processing_level,
            overwrite_topology_info=overwrite_topology_info,
            overwrite_timeseries_info=overwrite_timeseries_info,
            add_current=add_current,
            add_unbalance=add_unbalance,
            save=save,
            profile="cleaned_profiles",
        )

    def extract_cleaned_and_phase_corrected_sm_dataset_for_feeder(
        self,
        feeder_id,
        topology_processing_level,
        sm_ids_of_feeder=None,
        add_current=False,
        add_unbalance=False,
        save=False,
        overwrite_topology_info=False,
        overwrite_timeseries_info=False,
    ):
        return self.extract_timeseries_data_for_feeder(
            feeder_id=feeder_id,
            sm_ids_of_feeder=sm_ids_of_feeder,
            topology_processing_level=topology_processing_level,
            overwrite_topology_info=overwrite_topology_info,
            overwrite_timeseries_info=overwrite_timeseries_info,
            add_current=add_current,
            add_unbalance=add_unbalance,
            save=save,
            profile="cleaned_and_phase_corrected_profiles",
        )

    def extract_timeseries_data_for_feeder(
        self,
        feeder_id,
        sm_ids_of_feeder,
        topology_processing_level,
        overwrite_topology_info=False,
        overwrite_timeseries_info=False,
        add_current=False,
        add_unbalance=False,
        save=False,
        profile="raw_profiles",
    ):
        if pd.isna(feeder_id):
            raise ValueError("Invalid feeder_id.")
        feeder_id = str(feeder_id)

        sm_ids = self.get_sm_ids_for_feeder(
            feeder_id=feeder_id,
            sm_ids_of_feeder=sm_ids_of_feeder,
            topology_processing_level=topology_processing_level,
            overwrite_topology_info=overwrite_topology_info,
            overwrite_timeseries_info=overwrite_timeseries_info,
        )
        save_path = (
            f"{self.processed_data_dir}/individual_feeder_datasets/"
            f"based_on_{topology_processing_level}_topology/{profile}/{feeder_id}.parquet"
        )
        return self._extract_for_entity(
            sm_ids=sm_ids,
            profile=profile,
            add_current=add_current,
            add_unbalance=add_unbalance,
            save=save,
            save_path=save_path,
            entity_desc=f"feeder {feeder_id}",
        )

    def extract_raw_sm_dataset_for_transformer(
        self,
        transformer_id,
        topology_processing_level,
        use_existing_raw_sm_profiles=True,
        sm_ids_of_transformer=None,
        add_current=False,
        add_unbalance=False,
        save=False,
        overwrite_topology_info=False,
        overwrite_timeseries_info=False,
    ):
        return self.extract_timeseries_data_for_transformer(
            transformer_id=transformer_id,
            sm_ids_of_transformer=sm_ids_of_transformer,
            topology_processing_level=topology_processing_level,
            overwrite_topology_info=overwrite_topology_info,
            overwrite_timeseries_info=overwrite_timeseries_info,
            add_current=add_current,
            add_unbalance=add_unbalance,
            save=save,
            profile="raw_profiles",
        )

    def extract_cleaned_sm_dataset_for_transformer(
        self,
        transformer_id,
        topology_processing_level,
        sm_ids_of_transformer=None,
        add_current=False,
        add_unbalance=False,
        save=False,
        overwrite_topology_info=False,
        overwrite_timeseries_info=False,
    ):
        return self.extract_timeseries_data_for_transformer(
            transformer_id=transformer_id,
            sm_ids_of_transformer=sm_ids_of_transformer,
            topology_processing_level=topology_processing_level,
            overwrite_topology_info=overwrite_topology_info,
            overwrite_timeseries_info=overwrite_timeseries_info,
            add_current=add_current,
            add_unbalance=add_unbalance,
            save=save,
            profile="cleaned_profiles",
        )

    def extract_cleaned_and_phase_corrected_sm_dataset_for_transformer(
        self,
        transformer_id,
        topology_processing_level,
        sm_ids_of_transformer=None,
        add_current=False,
        add_unbalance=False,
        save=False,
        overwrite_topology_info=False,
        overwrite_timeseries_info=False,
    ):
        return self.extract_timeseries_data_for_transformer(
            transformer_id=transformer_id,
            sm_ids_of_transformer=sm_ids_of_transformer,
            topology_processing_level=topology_processing_level,
            overwrite_topology_info=overwrite_topology_info,
            overwrite_timeseries_info=overwrite_timeseries_info,
            add_current=add_current,
            add_unbalance=add_unbalance,
            save=save,
            profile="cleaned_and_phase_corrected_profiles",
        )

    def extract_timeseries_data_for_transformer(
        self,
        transformer_id,
        sm_ids_of_transformer,
        topology_processing_level,
        overwrite_topology_info=False,
        overwrite_timeseries_info=False,
        add_current=False,
        add_unbalance=False,
        save=False,
        profile="raw_profiles",
    ):
        if pd.isna(transformer_id):
            raise ValueError("Invalid transformer_id.")
        transformer_id = str(transformer_id)

        sm_ids = self.get_sm_ids_for_transformer(
            transformer_id=transformer_id,
            sm_ids_of_transformer=sm_ids_of_transformer,
            topology_processing_level=topology_processing_level,
            overwrite_topology_info=overwrite_topology_info,
            overwrite_timeseries_info=overwrite_timeseries_info,
        )
        save_path = (
            f"{self.processed_data_dir}/individual_transformer_datasets/"
            f"based_on_{topology_processing_level}_topology/{profile}/transformer_{transformer_id}.parquet"
        )
        return self._extract_for_entity(
            sm_ids=sm_ids,
            profile=profile,
            add_current=add_current,
            add_unbalance=add_unbalance,
            save=save,
            save_path=save_path,
            entity_desc=f"transformer {transformer_id}",
        )

    def extract_raw_sm_dataset_for_secondary_substation(
        self,
        substation_id,
        topology_processing_level,
        use_existing_raw_sm_profiles=True,
        sm_ids_of_secondary_substation=None,
        add_current=False,
        add_unbalance=False,
        save=False,
        overwrite_topology_info=False,
        overwrite_timeseries_info=False,
    ):
        return self.extract_timeseries_data_for_secondary_substation(
            substation_id=substation_id,
            sm_ids_of_secondary_substation=sm_ids_of_secondary_substation,
            topology_processing_level=topology_processing_level,
            overwrite_topology_info=overwrite_topology_info,
            overwrite_timeseries_info=overwrite_timeseries_info,
            add_current=add_current,
            add_unbalance=add_unbalance,
            save=save,
            profile="raw_profiles",
        )

    def extract_cleaned_sm_dataset_for_secondary_substation(
        self,
        substation_id,
        topology_processing_level,
        sm_ids_of_secondary_substation=None,
        add_current=False,
        add_unbalance=False,
        save=False,
        overwrite_topology_info=False,
        overwrite_timeseries_info=False,
    ):
        return self.extract_timeseries_data_for_secondary_substation(
            substation_id=substation_id,
            sm_ids_of_secondary_substation=sm_ids_of_secondary_substation,
            topology_processing_level=topology_processing_level,
            overwrite_topology_info=overwrite_topology_info,
            overwrite_timeseries_info=overwrite_timeseries_info,
            add_current=add_current,
            add_unbalance=add_unbalance,
            save=save,
            profile="cleaned_profiles",
        )

    def extract_cleaned_and_phase_corrected_sm_dataset_for_secondary_substation(
        self,
        substation_id,
        topology_processing_level,
        sm_ids_of_secondary_substation=None,
        add_current=False,
        add_unbalance=False,
        save=False,
        overwrite_topology_info=False,
        overwrite_timeseries_info=False,
    ):
        return self.extract_timeseries_data_for_secondary_substation(
            substation_id=substation_id,
            sm_ids_of_secondary_substation=sm_ids_of_secondary_substation,
            topology_processing_level=topology_processing_level,
            overwrite_topology_info=overwrite_topology_info,
            overwrite_timeseries_info=overwrite_timeseries_info,
            add_current=add_current,
            add_unbalance=add_unbalance,
            save=save,
            profile="cleaned_and_phase_corrected_profiles",
        )

    def extract_timeseries_data_for_secondary_substation(
        self,
        substation_id,
        sm_ids_of_secondary_substation,
        topology_processing_level,
        overwrite_topology_info=False,
        overwrite_timeseries_info=False,
        add_current=False,
        add_unbalance=False,
        save=False,
        profile="raw_profiles",
    ):
        if pd.isna(substation_id):
            raise ValueError("Invalid substation_id.")
        substation_id = str(substation_id)

        sm_ids = self.get_sm_ids_for_secondary_substation(
            substation_id=substation_id,
            sm_ids_of_secondary_substation=sm_ids_of_secondary_substation,
            topology_processing_level=topology_processing_level,
            overwrite_topology_info=overwrite_topology_info,
            overwrite_timeseries_info=overwrite_timeseries_info,
        )
        save_path = (
            f"{self.processed_data_dir}/individual_secondary_substation_datasets/"
            f"based_on_{topology_processing_level}_topology/{profile}/secondary_substation_{substation_id}.parquet"
        )
        return self._extract_for_entity(
            sm_ids=sm_ids,
            profile=profile,
            add_current=add_current,
            add_unbalance=add_unbalance,
            save=save,
            save_path=save_path,
            entity_desc=f"secondary substation {substation_id}",
        )

    def extract_raw_sm_dataset_for_zip(
        self,
        zip_id,
        topology_processing_level,
        use_existing_raw_sm_profiles=True,
        sm_ids_of_zip=None,
        add_current=False,
        add_unbalance=False,
        save=False,
        overwrite_topology_info=False,
        overwrite_timeseries_info=False,
    ):
        return self.extract_timeseries_data_for_zip(
            zip_id=zip_id,
            topology_processing_level=topology_processing_level,
            sm_ids_of_zip=sm_ids_of_zip,
            add_current=add_current,
            add_unbalance=add_unbalance,
            overwrite_timeseries_info=overwrite_timeseries_info,
            overwrite_topology_info=overwrite_topology_info,
            save=save,
            profile="raw_profiles",
        )

    def extract_cleaned_sm_dataset_for_zip(
        self,
        zip_id,
        topology_processing_level,
        sm_ids_of_zip=None,
        add_current=False,
        add_unbalance=False,
        overwrite_topology_info=False,
        overwrite_timeseries_info=False,
        save=False,
    ):
        return self.extract_timeseries_data_for_zip(
            zip_id=zip_id,
            topology_processing_level=topology_processing_level,
            sm_ids_of_zip=sm_ids_of_zip,
            add_current=add_current,
            add_unbalance=add_unbalance,
            overwrite_timeseries_info=overwrite_timeseries_info,
            overwrite_topology_info=overwrite_topology_info,
            save=save,
            profile="cleaned_profiles",
        )

    def extract_cleaned_and_phase_corrected_sm_dataset_for_zip(
        self,
        zip_id,
        topology_processing_level,
        sm_ids_of_zip=None,
        add_current=False,
        add_unbalance=False,
        overwrite_topology_info=False,
        overwrite_timeseries_info=False,
        save=False,
    ):
        return self.extract_timeseries_data_for_zip(
            zip_id=zip_id,
            topology_processing_level=topology_processing_level,
            sm_ids_of_zip=sm_ids_of_zip,
            add_current=add_current,
            add_unbalance=add_unbalance,
            overwrite_timeseries_info=overwrite_timeseries_info,
            overwrite_topology_info=overwrite_topology_info,
            save=save,
            profile="cleaned_and_phase_corrected_profiles",
        )

    def extract_timeseries_data_for_zip(
        self,
        zip_id,
        topology_processing_level,
        sm_ids_of_zip=None,
        add_current=False,
        add_unbalance=False,
        save=False,
        overwrite_topology_info=False,
        overwrite_timeseries_info=False,
        profile="raw_profiles",
    ):
        if pd.isna(zip_id):
            raise ValueError("Invalid zip_code.")
        zip_id = str(zip_id)

        if sm_ids_of_zip is None:
            topology_dict = self._get_topology_sm_mapping(topology_processing_level, overwrite_topology_info)
            if zip_id not in topology_dict:
                raise ValueError(f"Zip code {zip_id} does not exist in available data.")

            sm_ids_of_zip = []
            for sub_dict in topology_dict[zip_id].values():
                for trafo_dict in sub_dict.values():
                    for lvfeeder in trafo_dict.values():
                        for sm_ids in lvfeeder.values():
                            sm_ids_of_zip.extend(sm_ids.get("AVAILABLE_METERS", []))

        save_path = (
            f"{self.processed_data_dir}/individual_zip_code_datasets/"
            f"based_on_{topology_processing_level}_topology/{profile}/zip_code_{zip_id}.parquet"
        )
        return self._extract_for_entity(
            sm_ids=sm_ids_of_zip,
            profile=profile,
            add_current=add_current,
            add_unbalance=add_unbalance,
            save=save,
            save_path=save_path,
            entity_desc=f"zip code {zip_id}",
        )

    # --- attribute-filtered extraction (heat pump / PV) --- #

    def _extract_for_sm_attribute(
        self,
        *,
        attribute: str,
        out_dir_name: str,
        topology_processing_level,
        add_current,
        add_unbalance,
        chunk_size,
        profile,
    ):
        level = _profile_level(profile)  # validates the profile name
        logger.debug(f"Extracting {out_dir_name} datasets at level {level}")

        # Heat pump / PV flags live on the (unversioned) lv.meter asset table
        kwargs = {attribute: True}
        meters = self.topology_controller.get_meters(**kwargs)
        sm_ids = sorted({str(m["id"]) for m in meters} & self._ids_with_data())

        out_dir = f"{self.processed_data_dir}/{out_dir_name}/based_on_{topology_processing_level}_topology/{profile}"
        for start in range(0, len(sm_ids), chunk_size):
            chunk = sm_ids[start : start + chunk_size]
            frames = self._wide_sm_frames(chunk, profile, add_current, add_unbalance)
            for meter_id, frame in zip(chunk, frames, strict=True):
                if frame is not None:
                    self.s3_connector.write_parquet(f"{out_dir}/sm_{meter_id}.parquet", frame, engine="pyarrow")

        logger.info(f"Profile '{profile}': SM data of {out_dir_name} saved to directory {out_dir}.")

    def extract_timeseries_data_for_sm_with_heatpump(
        self,
        topology_processing_level,
        add_current=False,
        add_unbalance=False,
        overwrite_timeseries_info=False,
        chunk_size=1000,
        profile="raw_profiles",
    ):
        return self._extract_for_sm_attribute(
            attribute="has_heat_pump",
            out_dir_name="sm_with_heatpump_datasets",
            topology_processing_level=topology_processing_level,
            add_current=add_current,
            add_unbalance=add_unbalance,
            chunk_size=chunk_size,
            profile=profile,
        )

    def extract_raw_datasets_for_sm_with_heatpump(
        self,
        topology_processing_level,
        use_existing_raw_sm_profiles=True,
        add_current=False,
        add_unbalance=False,
        overwrite_timeseries_info=False,
        chunk_size=1000,
    ):
        return self.extract_timeseries_data_for_sm_with_heatpump(
            topology_processing_level=topology_processing_level,
            add_current=add_current,
            add_unbalance=add_unbalance,
            overwrite_timeseries_info=overwrite_timeseries_info,
            chunk_size=chunk_size,
            profile="raw_profiles",
        )

    def extract_cleaned_datasets_for_sm_with_heatpump(
        self,
        topology_processing_level,
        add_current=False,
        add_unbalance=False,
        overwrite_timeseries_info=False,
        chunk_size=1000,
    ):
        return self.extract_timeseries_data_for_sm_with_heatpump(
            topology_processing_level=topology_processing_level,
            add_current=add_current,
            add_unbalance=add_unbalance,
            overwrite_timeseries_info=overwrite_timeseries_info,
            chunk_size=chunk_size,
            profile="cleaned_profiles",
        )

    def extract_cleaned_and_phase_corrected_datasets_for_sm_with_heatpump(
        self,
        topology_processing_level,
        add_current=False,
        add_unbalance=False,
        overwrite_timeseries_info=False,
        chunk_size=1000,
    ):
        return self.extract_timeseries_data_for_sm_with_heatpump(
            topology_processing_level=topology_processing_level,
            add_current=add_current,
            add_unbalance=add_unbalance,
            overwrite_timeseries_info=overwrite_timeseries_info,
            chunk_size=chunk_size,
            profile="cleaned_and_phase_corrected_profiles",
        )

    def extract_timeseries_data_for_sm_with_pv(
        self,
        topology_processing_level,
        add_current=False,
        add_unbalance=False,
        overwrite_timeseries_info=False,
        chunk_size=1000,
        profile="raw_profiles",
    ):
        return self._extract_for_sm_attribute(
            attribute="has_solar_panel",
            out_dir_name="sm_with_pv_datasets",
            topology_processing_level=topology_processing_level,
            add_current=add_current,
            add_unbalance=add_unbalance,
            chunk_size=chunk_size,
            profile=profile,
        )

    def extract_raw_datasets_for_sm_with_pv(
        self,
        topology_processing_level,
        use_existing_raw_sm_profiles=True,
        add_current=False,
        add_unbalance=False,
        overwrite_timeseries_info=False,
        chunk_size=1000,
    ):
        return self.extract_timeseries_data_for_sm_with_pv(
            topology_processing_level=topology_processing_level,
            add_current=add_current,
            add_unbalance=add_unbalance,
            overwrite_timeseries_info=overwrite_timeseries_info,
            chunk_size=chunk_size,
            profile="raw_profiles",
        )

    def extract_cleaned_datasets_for_sm_with_pv(
        self,
        topology_processing_level,
        add_current=False,
        add_unbalance=False,
        overwrite_timeseries_info=False,
        chunk_size=1000,
    ):
        return self.extract_timeseries_data_for_sm_with_pv(
            topology_processing_level=topology_processing_level,
            add_current=add_current,
            add_unbalance=add_unbalance,
            overwrite_timeseries_info=overwrite_timeseries_info,
            chunk_size=chunk_size,
            profile="cleaned_profiles",
        )

    def extract_cleaned_and_phase_corrected_datasets_for_sm_with_pv(
        self,
        topology_processing_level,
        add_current=False,
        add_unbalance=False,
        overwrite_timeseries_info=False,
        chunk_size=1000,
    ):
        return self.extract_timeseries_data_for_sm_with_pv(
            topology_processing_level=topology_processing_level,
            add_current=add_current,
            add_unbalance=add_unbalance,
            overwrite_timeseries_info=overwrite_timeseries_info,
            chunk_size=chunk_size,
            profile="cleaned_and_phase_corrected_profiles",
        )

    #############################
    # RE-USABLE (CLASS) METHODS #
    #############################

    def enrich_meter_df(self, meter_df, meter_id, add_unbalance, add_current):
        """
        Applies current and/or phase unbalance enrichment to the meter DataFrame.

        :param meter_df: Input DataFrame
        :param meter_id: Identifier for the meter
        :param add_unbalance: Whether to add voltage/current phase unbalance
        :param add_current: Whether to add current (if not already included with unbalance)
        :return: Enriched DataFrame
        """
        if add_unbalance:
            meter_df = add_phase_imbalance_for_sm(
                meter_df=meter_df,
                meter_id=meter_id,
                voltage_col=self.voltage_col,
                v_phase_unbalance_col=self.v_phase_unbalance_col,
                v_unbalance_col=self.v_unbalance_col,
                current_col=self.current_col,
                i_phase_unbalance_col=self.i_phase_unbalance_col,
                i_unbalance_col=self.i_unbalance_col,
                active_power_p14_col=self.active_power_p14_col,
                active_power_p23_col=self.active_power_p23_col,
                keep_current=add_current,
            )

        if add_current and not add_unbalance:
            meter_df = add_current_for_sm(
                meter_df=meter_df,
                meter_id=meter_id,
                voltage_col=self.voltage_col,
                current_col=self.current_col,
                active_power_p14_col=self.active_power_p14_col,
                active_power_p23_col=self.active_power_p23_col,
            )

        return meter_df

    @staticmethod
    def enrich_meter_df_with(
        meter_df,
        meter_id,
        add_unbalance,
        add_current,
        voltage_col,
        v_phase_unbalance_col,
        v_unbalance_col,
        current_col,
        i_phase_unbalance_col,
        i_unbalance_col,
        active_power_p14_col,
        active_power_p23_col,
    ):
        if add_unbalance:
            # Add phase unbalance per line and IEE PVUR
            meter_df = add_phase_imbalance_for_sm(
                meter_df=meter_df,
                meter_id=meter_id,
                voltage_col=voltage_col,
                v_phase_unbalance_col=v_phase_unbalance_col,
                v_unbalance_col=v_unbalance_col,
                current_col=current_col,
                i_phase_unbalance_col=i_phase_unbalance_col,
                i_unbalance_col=i_unbalance_col,
                active_power_p14_col=active_power_p14_col,
                active_power_p23_col=active_power_p23_col,
                keep_current=add_current,
            )

        if add_current and not add_unbalance:
            meter_df = add_current_for_sm(
                meter_df=meter_df,
                meter_id=meter_id,
                voltage_col=voltage_col,
                current_col=current_col,
                active_power_p14_col=active_power_p14_col,
                active_power_p23_col=active_power_p23_col,
            )

        return meter_df


###################
# UTILITY METHODS #
###################


def filter_nans(meters):
    return [m for m in meters if pd.notna(m)]


def get_available_meters(meters, valid_ids):
    return [m for m in meters if m in valid_ids]


def get_missing_meters(meters, valid_ids):
    return [m for m in meters if m not in valid_ids]


def add_current_for_sm(
    meter_df,
    meter_id,
    voltage_col,
    current_col,
    active_power_p14_col,
    active_power_p23_col,
):
    # Determine existing phases
    existing_phases = [p for p in [1, 2, 3] if f"{voltage_col}{p}_{meter_id}" in meter_df.columns]

    for p in existing_phases:
        meter_df[f"{current_col}{p}_{meter_id}"] = (
            meter_df[f"{active_power_p14_col}{p}_{meter_id}"] - meter_df[f"{active_power_p23_col}{p}_{meter_id}"]
        ) / meter_df[f"{voltage_col}{p}_{meter_id}"]

    return meter_df


def add_phase_imbalance_for_sm(
    meter_df,
    meter_id,
    voltage_col,
    v_phase_unbalance_col,
    v_unbalance_col,
    current_col,
    i_phase_unbalance_col,
    i_unbalance_col,
    active_power_p14_col,
    active_power_p23_col,
    keep_current=False,
):
    # Determine existing phases
    existing_phases = [p for p in [1, 2, 3] if f"{voltage_col}{p}_{meter_id}" in meter_df.columns]

    # Calculate voltage phase imbalance per line
    voltage_mean = meter_df[[f"{voltage_col}{p}_{meter_id}" for p in existing_phases]].mean(axis=1)
    for p in existing_phases:
        meter_df[f"{v_phase_unbalance_col}{p}_{meter_id}"] = (
            meter_df[f"{voltage_col}{p}_{meter_id}"] - voltage_mean
        ) / voltage_mean

    # Calculate IEEE Phase Voltage Unbalance Rate (PVUR)
    meter_df[f"{v_unbalance_col}_{meter_id}"] = (
        meter_df[[f"{v_phase_unbalance_col}{p}_{meter_id}" for p in existing_phases]].abs().max(axis=1)
    )

    # Calculate absolute current phase imbalance per line
    meter_df = add_current_for_sm(
        meter_df,
        meter_id,
        voltage_col,
        current_col,
        active_power_p14_col,
        active_power_p23_col,
    )

    current_mean = meter_df[[f"{current_col}{p}_{meter_id}" for p in existing_phases]].mean(axis=1)

    for p in existing_phases:
        meter_df[f"{i_phase_unbalance_col}{p}_{meter_id}"] = (
            meter_df[f"{current_col}{p}_{meter_id}"] - current_mean
        )  # /current_mean

    # Calculate IEEE Phase Current Unbalance Rate (PCUR)
    meter_df[f"{i_unbalance_col}_{meter_id}"] = (
        meter_df[[f"{i_phase_unbalance_col}{p}_{meter_id}" for p in existing_phases]].abs().max(axis=1)
    )

    if not keep_current:
        meter_df.drop(
            columns=[f"{current_col}{p}_{meter_id}" for p in existing_phases],
            inplace=True,
        )

    return meter_df
