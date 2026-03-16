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
from dask import compute, delayed

from threephi_framework.data_extractor.schemas.phase_measurements.v1 import (
    VERSION,
    PhaseMeasurementsCsvSchema,
    PhaseMeasurementsParquetSchema,
)
from threephi_framework.db_connector import DBConnector
from threephi_framework.object_storage.s3_connector import S3Connector
from threephi_framework.util.util import v1_get_shard_for_meter_id

# Old data storage layout
BATCH = "second_batch"
SOURCE_DATA_DIR = f"Data/sourcedata/{BATCH}"
PROCESSED_DATA_DIR = f"Data/processed_data/{BATCH}"
ALLOWED_PROFILES = [
    "raw_profiles",
    "cleaned_profiles",
    "cleaned_and_phase_corrected_profiles",
]

logger = logging.getLogger(__name__)


class DataExtractor:
    def __init__(self):
        # Properties for new data storage layout
        self.phase_measurements_csv_schema = PhaseMeasurementsCsvSchema()
        self.phase_measurements_parquet_schema = PhaseMeasurementsParquetSchema()
        self.conn = None
        # Others
        # s3 extractor
        self.s3_connector = S3Connector(data_dir_path="phase_measurements/raw")
        self.db_connector = DBConnector()
        self.s3_base = "s3://3phi"
        self.sourcedata_dir = f"{self.s3_base}/{SOURCE_DATA_DIR}"
        self.processed_data_dir = f"{self.s3_base}/{PROCESSED_DATA_DIR}"

        self.db_connector.check_db_connection()

        # check that s3 destination exists
        self.processed_data_dir = f"{self.s3_base}/{PROCESSED_DATA_DIR}"

        # define file names
        self.metercabinet_file = "meter_cabinet_connection.csv"
        self.topology_file = "lv_topology.csv"
        self.cabinet_sm_mapping_file = "Cabinet_SM_mapping.json"
        self.topology_sm_mapping_file = "Topology_SM_mapping.json"
        self.sm_topology_mapping_file = "SM_Topology_mapping.json"

        # Initialize attributes for holding various data structures and metadata
        self.raw_timeseries_df = None
        self.raw_meter_and_cabinet_df = None
        self.raw_topology_df = None
        self.cleaned_meter_and_cabinet_df = None
        self.cleaned_topology_df = None
        self.cleaned_and_corrected_meter_and_cabinet_df = None
        self.cleaned_and_corrected_topology_df = None
        self.min_timestamp = None
        self.max_timestamp = None
        self.expected_timestamps = None
        self.id_list_of_sms_with_data = None
        self.raw_cabinet_sm_mapping_dict = None
        self.cleaned_cabinet_sm_mapping_dict = None
        self.cleaned_and_corrected_cabinet_sm_mapping_dict = None
        self.raw_topology_dict = None
        self.raw_topology_dict_reversed = None
        self.cleaned_topology_dict = None
        self.cleaned_topology_dict_reversed = None
        self.cleaned_and_corrected_topology_dict = None
        self.cleaned_and_corrected_topology_dict_reversed = None

        # Define correct columns
        self.meter_number_col = "meter_number"
        self.timestamp_dst_col = "timestamp_dst"
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
        self.transformer_capacity_col = "transformer_capacity"
        self.lv_feeder_fuse_size_col = "lv_feeder_fuse_size"
        self.cable_id_col = "cable_id"
        self.cable_type_col = "cable_type"
        self.cable_length_col = "cable_length"
        self.phase_size_col = "phase_size"
        self.phase_material_col = "phase_material"
        self.cable_capacity_col = "cable_capacity"
        self.resistance_col = "resistance"
        self.reactance_col = "reactance"
        self.cabinet_name = "Cabinet"

        # Define data type dicts for data loading
        self.timeseries_dtypes = {
            self.meter_number_col: pd.StringDtype(),
            self.timestamp_dst_col: "datetime64[ns]",
            f"{self.voltage_col}1": pd.Float32Dtype(),
            f"{self.voltage_col}2": pd.Float32Dtype(),
            f"{self.voltage_col}3": pd.Float32Dtype(),
            f"{self.active_power_p14_col}1": pd.Float32Dtype(),
            f"{self.active_power_p14_col}2": pd.Float32Dtype(),
            f"{self.active_power_p14_col}3": pd.Float32Dtype(),
            f"{self.active_power_p23_col}1": pd.Float32Dtype(),
            f"{self.active_power_p23_col}2": pd.Float32Dtype(),
            f"{self.active_power_p23_col}3": pd.Float32Dtype(),
            f"{self.reactive_power_q12_col}1": pd.Float32Dtype(),
            f"{self.reactive_power_q12_col}2": pd.Float32Dtype(),
            f"{self.reactive_power_q12_col}3": pd.Float32Dtype(),
            f"{self.reactive_power_q34_col}1": pd.Float32Dtype(),
            f"{self.reactive_power_q34_col}2": pd.Float32Dtype(),
            f"{self.reactive_power_q34_col}3": pd.Float32Dtype(),
            "thdu_l1": pd.Float32Dtype(),
            "thdu_l2": pd.Float32Dtype(),
            "thdu_l3": pd.Float32Dtype(),
            "thdi_l1": pd.Float32Dtype(),
            "thdi_l2": pd.Float32Dtype(),
            "thdi_l3": pd.Float32Dtype(),
        }

        self.meter_and_cabinet_dtypes = {
            self.meter_number_col: pd.StringDtype(),
            self.cabinet_col: pd.StringDtype(),
            "delivery_point_id": pd.StringDtype(),
            "lv_feeder": pd.StringDtype(),
            "has_heat_pump": pd.BooleanDtype(),
            "has_solar_panel": pd.BooleanDtype(),
            "capacity_solar_panel": pd.Float32Dtype(),
            "service_fuse_size": pd.Float32Dtype(),
        }

        self.topology_dtypes = {
            self.sec_substation_col: pd.StringDtype(),
            self.transformer_col: pd.StringDtype(),
            self.transformer_capacity_col: pd.Float32Dtype(),
            self.lv_feeder_col: pd.StringDtype(),
            self.lv_feeder_fuse_size_col: pd.Float32Dtype(),
            self.node_1_col: pd.StringDtype(),
            self.node_2_col: pd.StringDtype(),
            self.cable_id_col: pd.StringDtype(),
            self.cable_type_col: pd.StringDtype(),
            self.cable_length_col: pd.Float32Dtype(),
            self.phase_size_col: pd.Float32Dtype(),
            self.phase_material_col: pd.StringDtype(),
            self.cable_capacity_col: pd.Float32Dtype(),
            self.resistance_col: pd.Float32Dtype(),
            self.reactance_col: pd.Float32Dtype(),
            "zip_code_secondary_substation": pd.StringDtype(),
        }

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
            batch_id = self.db_connector.insert_batch(csv_path, run_id)

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
            if not self.db_connector.is_workflow_completed(stats_workflow):
                self.db_connector.start_workflow(stats_workflow)

                logger.info("Computing meter inventory stats.")
                stats_ddf = dask_df[[parquet_meter_col, parquet_ts_col]]
                agg_ddf = stats_ddf.groupby(parquet_meter_col)[parquet_ts_col].agg(["min", "max", "count"])

                agg_pdf = agg_ddf.compute()
                agg_pdf = agg_pdf.rename(columns={"min": "first_seen", "max": "last_seen", "count": "total_rows"})
                agg_pdf = agg_pdf.reset_index(names=["id"])
                self.db_connector.upsert_meter_stats(agg_pdf)
                self.db_connector.complete_workflow(stats_workflow)
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
                seq = self.db_connector.get_current_max_seq_for_ring(dt_str, shard)

                for file in files:
                    seq += 1
                    file_stats = self.s3_connector.get_parquet_file_stats(file, dask_df, parquet_ts_col)
                    # prepare the s3 key for the "ready" file
                    ready_key = file.replace(f"/staging/run={run_id}/", "/")
                    self.db_connector.upsert_file_index(
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
            self.db_connector.promote_batch_to_ready(batch_id=batch_id)
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
        return self.db_connector.get_timeseries_info()

    def v1_get_single_meter_data(self, id: str) -> dd.DataFrame:
        ddf: dd.DataFrame = self.s3_connector.get_meter_data(meter_ids=[id])
        return ddf

    def _load_raw_timeseries_data(self, filetype="parquet"):
        if filetype != "parquet":
            logger.warning(
                "Parameter filetype is deprecated, please remove. This DataExtractor only works with parquet."
            )

        # if this is executed the first time, we read data from csv and store as parquet
        workflow = "timeseries_csv_to_parquet"
        if not self.db_connector.is_workflow_completed(workflow):
            self._extract_csv_to_parquet(s3_path=self.sourcedata_dir, file_pattern="phase_measurements_*.csv")
            self.db_connector.complete_workflow(workflow)

        # if this is executed the first time, we read data from DB and store as parquet
        # workflow = "timeseries_sql_to_parquet"
        # if not self.db_connector.is_workflow_completed(workflow):
        #     self.db_connector.start_workflow(workflow)
        #     self.db_connector.stream_meter_data_from_sql_to_parquet(
        #         s3_path=f"{self.s3_base}/{SOURCE_DATA_DIR}",
        #         s3_connector=self.s3_connector,
        #     )
        #     self.db_connector.complete_workflow(workflow)

        if self.raw_timeseries_df is None:
            self.raw_timeseries_df = dd.read_parquet(
                f"{self.s3_base}/{SOURCE_DATA_DIR}",
                storage_options=self.s3_connector.get_dask_storage_options(),
                engine="pyarrow",
            )

    def _load_raw_meter_and_cabinet_data(self):
        if self.raw_meter_and_cabinet_df is None:
            filepath = f"{self.s3_base}/{SOURCE_DATA_DIR}/{self.metercabinet_file}"
            self.raw_meter_and_cabinet_df = self.s3_connector.read_csv(filepath, self.meter_and_cabinet_dtypes)

    def _load_cleaned_meter_and_cabinet_data(self):
        if self.cleaned_meter_and_cabinet_df is None:
            # TODO: Use workflow table instead of try/catch
            try:
                filepath = f"{self.s3_base}/{SOURCE_DATA_DIR}/topology/cleaned/{self.metercabinet_file}"
                self.cleaned_meter_and_cabinet_df = self.s3_connector.read_csv(filepath, self.meter_and_cabinet_dtypes)
            except FileNotFoundError as e:
                err_msg = (
                    "Cleaned version of MeterAndCabinet.csv not found. "
                    "Use Topology Cleaner first or raw MeterAndCabinet.csv"
                )
                raise FileNotFoundError(err_msg) from e

    def _load_cleaned_and_corrected_meter_and_cabinet_data(self):
        if self.cleaned_and_corrected_meter_and_cabinet_df is None:
            # TODO: Use workflow table instead of try/catch
            try:
                filepath = f"{self.s3_base}/{SOURCE_DATA_DIR}/topology/cleaned_and_corrected/{self.metercabinet_file}"
                self.cleaned_and_corrected_meter_and_cabinet_df = self.s3_connector.read_csv(
                    filepath, self.meter_and_cabinet_dtypes
                )
            except Exception as e:
                logger.error(
                    "Cleaned and corrected version of MeterAndCabinet.csv not found. "
                    "Use Topology Cleaner + Topology Corrector first or raw MeterAndCabinet.csv"
                )
                raise e

    def _load_raw_topology_data(self):
        if self.raw_topology_df is None:
            filepath = f"{self.s3_base}/{SOURCE_DATA_DIR}/{self.topology_file}"
            self.raw_topology_df = self.s3_connector.read_csv(filepath, self.topology_dtypes)

    def _load_cleaned_topology_data(self):
        if self.cleaned_topology_df is None:
            # TODO: Use workflow table instead of try/catch
            try:
                filepath = f"{self.s3_base}/{SOURCE_DATA_DIR}/topology/cleaned/{self.topology_file}"
                self.cleaned_topology_df = self.s3_connector.read_csv(filepath, self.topology_dtypes)
            except Exception as e:
                logger.error(
                    "Cleaned version of topology.csv not found. Use Topology Cleaner first or raw Topology.csv"
                )
                raise e

    def _load_cleaned_and_corrected_topology_data(self):
        if self.cleaned_and_corrected_topology_df is None:
            # TODO: Use workflow table instead of try/catch
            try:
                filepath = f"{self.s3_base}/{SOURCE_DATA_DIR}/topology/cleaned_and_corrected/{self.topology_file}"
                self.cleaned_and_corrected_topology_df = self.s3_connector.read_csv(filepath, self.topology_dtypes)
            except Exception as e:
                logger.error(
                    "Cleaned and corrected version of topology.csv not found. "
                    "Use Topology Cleaner + Topology Corrector first or raw Topology.csv"
                )
                raise e

    def _get_timeseries_info(self, overwrite=False):
        """
        Class-internal function for extracting some information from the
        timeseries data and storing it in the class.
        Information is only extracted if not already done before.

        Information:
            - min_timestamp: earliest SM measurement timestamp
            - max_timestamp: last SM measurement timestamp
            - expected_timestamps: range of expected timestamps
            - id_list_of_sms_with_data: list of meter IDS that we have data for
        """

        if self.min_timestamp is None or self.max_timestamp is None or self.id_list_of_sms_with_data is None:
            filepath = f"{self.processed_data_dir}/timeseries_info/Timeseries_info.json"

            # If overwrite = False and file already exist, load the existing one
            if self.s3_connector.exists(filepath) and not overwrite:
                timeseries_info = self.s3_connector.read_json(filepath)
                self.min_timestamp = pd.to_datetime(timeseries_info["min_timestamp"])
                self.max_timestamp = pd.to_datetime(timeseries_info["max_timestamp"])
                self.id_list_of_sms_with_data = timeseries_info["id_list_of_sms_with_data"]
            else:
                logging.info("Loading timeseries info.")
                # Load raw timeseries data into class if not already done
                try:
                    self._load_raw_timeseries_data(filetype="parquet")
                except (OSError, FileNotFoundError) as e:
                    logger.error("Error loading Timeseries Info!")
                    raise e
                logging.info("Timeseries info loaded.")

                timeseries_info = {}

                self.min_timestamp = self.raw_timeseries_df[self.timestamp_dst_col].min().compute()
                self.max_timestamp = self.raw_timeseries_df[self.timestamp_dst_col].max().compute()
                self.id_list_of_sms_with_data = (
                    self.raw_timeseries_df[self.meter_number_col].unique().compute().tolist()
                )
                self.id_list_of_sms_with_data = (
                    self.raw_timeseries_df[[self.meter_number_col]]
                    .drop_duplicates()
                    .compute()[self.meter_number_col]
                    .tolist()
                )

                timeseries_info["min_timestamp"] = str(self.min_timestamp)
                timeseries_info["max_timestamp"] = str(self.max_timestamp)
                timeseries_info["id_list_of_sms_with_data"] = self.id_list_of_sms_with_data

                self.s3_connector.write_json(filepath, timeseries_info)
                logging.info(f"Timeseries info saved to directory {filepath}.")

            # Create expected timestamps (used later for detecting and filling
            # missing time steps) if not done before
            if self.expected_timestamps is None:
                self.expected_timestamps = pd.date_range(start=self.min_timestamp, end=self.max_timestamp, freq="15min")

    def _get_timeseries_info_db(self, overwrite=None):
        logger.info("Getting timeseries info from DB")
        if overwrite:
            logger.warning("Parameter overwrite is deprecated, please don't use anymore.")
        timeseries_info = self.db_connector.get_timeseries_info()
        self.min_timestamp = timeseries_info["min_timestamp"]
        self.max_timestamp = timeseries_info["max_timestamp"]
        self.id_list_of_sms_with_data = timeseries_info["id_list_of_sms_with_data"]

        # Create expected timestamps (used later for detecting and filling
        # missing time steps) if not done before
        if self.expected_timestamps is None:
            self.expected_timestamps = pd.date_range(start=self.min_timestamp, end=self.max_timestamp, freq="15min")

    def create_raw_cabinet_sm_mapping(self, save=True, overwrite=False, overwrite_timeseries_info=False):
        filepath = f"{self.s3_base}/{PROCESSED_DATA_DIR}/topology/raw/{self.cabinet_sm_mapping_file}"
        if self.raw_cabinet_sm_mapping_dict is None:
            if self.s3_connector.exists(filepath) and not overwrite:
                self.raw_cabinet_sm_mapping_dict = self.s3_connector.read_json(filepath)
            else:
                # Load meter and cabinet data if not already done
                self._load_raw_meter_and_cabinet_data()

                # Extract list of available meters from timeseries if not already done
                self._get_timeseries_info(overwrite=overwrite_timeseries_info)

                # Create local pandas copy of meter and cabinet df
                raw_meter_and_cabinet_df = self.raw_meter_and_cabinet_df.copy().compute()

                # Just keep the cabinet numbers instead of the whole string
                raw_meter_and_cabinet_df[self.cabinet_col] = (
                    raw_meter_and_cabinet_df[self.cabinet_col].str.split(".").str[1]
                )

                # Group the SM-Cabinet mapping by the cabinet numbers -> for each
                # cabinet we now have a list of connected SMs
                cabinet_sm_mapping = (
                    raw_meter_and_cabinet_df.groupby(self.cabinet_col)[self.meter_number_col].apply(list).reset_index()
                )

                print("Example meter:", cabinet_sm_mapping[self.meter_number_col].iloc[0])
                print("Valid IDs:", list(self.id_list_of_sms_with_data)[:5])
                cabinet_sm_mapping[self.meter_number_col] = cabinet_sm_mapping[self.meter_number_col].apply(
                    lambda meters: [m for m in meters if pd.notna(m)]
                )
                cabinet_sm_mapping["AVAILABLE_METERS"] = cabinet_sm_mapping[self.meter_number_col].apply(
                    lambda meters: [m for m in meters if m in self.id_list_of_sms_with_data]
                )
                cabinet_sm_mapping["MISSING_METERS"] = cabinet_sm_mapping[self.meter_number_col].apply(
                    lambda meters: [m for m in meters if m not in self.id_list_of_sms_with_data]
                )

                self.raw_cabinet_sm_mapping_dict = {
                    row[self.cabinet_col]: {
                        "METER_NUMBER": row[self.meter_number_col],
                        "AVAILABLE_METERS": row["AVAILABLE_METERS"],
                        "MISSING_METERS": row["MISSING_METERS"],
                    }
                    for _, row in cabinet_sm_mapping.iterrows()
                }

                if save:
                    self.s3_connector.write_json(filepath, self.raw_cabinet_sm_mapping_dict)
                    logger.info(f"Raw Cabinet-Smart meter mapping saved to directory {filepath}.")

        return self.raw_cabinet_sm_mapping_dict

    def create_cleaned_cabinet_sm_mapping(self, save=True, overwrite=False, overwrite_timeseries_info=False):
        filepath = f"{self.s3_base}/{PROCESSED_DATA_DIR}/topology/cleaned/{self.cabinet_sm_mapping_file}"
        if self.cleaned_cabinet_sm_mapping_dict is None:
            if self.s3_connector.exists(filepath) and not overwrite:
                self.cleaned_cabinet_sm_mapping_dict = self.s3_connector.read_json(filepath)
            else:
                # Load meter and cabinet data if not already done
                self._load_cleaned_meter_and_cabinet_data()
                # Extract list of available meters from timeseries if not already done
                self.db_connector.get_timeseries_info()
                # Create local copy of meter and cabinet df
                cleaned_meter_and_cabinet_df = self.cleaned_meter_and_cabinet_df.copy()

                # Just keep the cabinet numbers instead of the whole string
                cleaned_meter_and_cabinet_df[self.cabinet_col] = (
                    cleaned_meter_and_cabinet_df[self.cabinet_col].str.split(".").str[1]
                )

                # Group the SM-Cabinet mapping by the cabinet numbers -> for each
                # cabinet we now have a list of connected SMs
                cabinet_sm_mapping = (
                    cleaned_meter_and_cabinet_df.groupby(self.cabinet_col)[self.meter_number_col]
                    .apply(list)
                    .reset_index()
                )
                cabinet_sm_mapping[self.meter_number_col] = cabinet_sm_mapping[self.meter_number_col].apply(filter_nans)
                cabinet_sm_mapping["AVAILABLE_METERS"] = cabinet_sm_mapping[self.meter_number_col].apply(
                    lambda meters: get_available_meters(meters, self.id_list_of_sms_with_data)
                )
                cabinet_sm_mapping["MISSING_METERS"] = cabinet_sm_mapping[self.meter_number_col].apply(
                    lambda meters: get_missing_meters(meters, self.id_list_of_sms_with_data)
                )

                self.cleaned_cabinet_sm_mapping_dict = {
                    row[self.cabinet_col]: {
                        "METER_NUMBER": row[self.meter_number_col],
                        "AVAILABLE_METERS": row["AVAILABLE_METERS"],
                        "MISSING_METERS": row["MISSING_METERS"],
                    }
                    for _, row in cabinet_sm_mapping.iterrows()
                }

                if save:
                    self.s3_connector.write_json(filepath, self.raw_cabinet_sm_mapping_dict)
                    logger.info(f"Cleaned Cabinet-Smart meter mapping saved to directory {filepath}.")

        # Return the Cabinet-SM mapping as a dictionary
        return self.cleaned_cabinet_sm_mapping_dict

    def create_cleaned_and_corrected_cabinet_sm_mapping(
        self, save=True, overwrite=False, overwrite_timeseries_info=False
    ):
        filepath = f"{self.s3_base}/{PROCESSED_DATA_DIR}/topology/cleaned_and_corrected/{self.cabinet_sm_mapping_file}"
        if self.cleaned_and_corrected_cabinet_sm_mapping_dict is None:
            if self.s3_connector.exists(filepath) and not overwrite:
                self.cleaned_cabinet_sm_mapping_dict = self.s3_connector.read_json(filepath)
            else:
                # Load meter and cabinet data if not already done
                self._load_cleaned_and_corrected_meter_and_cabinet_data()
                # Extract list of available meters from timeseries if not already done
                self.db_connector.get_timeseries_info()
                # Create local copy of meter and cabinet df
                cleaned_and_corrected_meter_and_cabinet_df = self.cleaned_and_corrected_meter_and_cabinet_df.copy()
                # Just keep the cabinet numbers instead of the whole string
                cleaned_and_corrected_meter_and_cabinet_df[self.cabinet_col] = (
                    cleaned_and_corrected_meter_and_cabinet_df[self.cabinet_col].str.split(".").str[1]
                )
                # Group the SM-Cabinet mapping by the cabinet numbers -> for each
                # cabinet we now have a list of connected SMs
                cabinet_sm_mapping = (
                    cleaned_and_corrected_meter_and_cabinet_df.groupby(self.cabinet_col)[self.meter_number_col]
                    .apply(list)
                    .reset_index()
                )
                cabinet_sm_mapping[self.meter_number_col] = cabinet_sm_mapping[self.meter_number_col].apply(filter_nans)
                cabinet_sm_mapping["AVAILABLE_METERS"] = cabinet_sm_mapping[self.meter_number_col].apply(
                    lambda meters: get_available_meters(meters, self.id_list_of_sms_with_data)
                )
                cabinet_sm_mapping["MISSING_METERS"] = cabinet_sm_mapping[self.meter_number_col].apply(
                    lambda meters: get_missing_meters(meters, self.id_list_of_sms_with_data)
                )

                self.cleaned_and_corrected_cabinet_sm_mapping_dict = {
                    row[self.cabinet_col]: {
                        "METER_NUMBER": row[self.meter_number_col],
                        "AVAILABLE_METERS": row["AVAILABLE_METERS"],
                        "MISSING_METERS": row["MISSING_METERS"],
                    }
                    for _, row in cabinet_sm_mapping.iterrows()
                }

                if save:
                    self.s3_connector.write_json(filepath, self.raw_cabinet_sm_mapping_dict)
                    logger.info(f"Cleaned and corrected Cabinet-Smart meter mapping saved to directory {filepath}.")

        # Return the Cabinet-SM mapping as a dictionary
        return self.cleaned_and_corrected_cabinet_sm_mapping_dict

    def create_raw_topology_sm_mapping(self, save=True, overwrite=False, overwrite_timeseries_info=False) -> dict:
        """
        Function to create a mapping between topology (Substations, Transformers,
        Feeders) and Smart Meters based on raw topology data.

        arguments:
            save (boolean): If True, a textfile of the mapping is saved.

        returns:
            raw_topology_dict (dict): The topology-SM mapping in dictionary format.
        """
        filepath = f"{self.s3_base}/{PROCESSED_DATA_DIR}/topology/raw/{self.topology_sm_mapping_file}"
        if self.raw_topology_dict is None:
            if self.s3_connector.exists(filepath) and not overwrite:
                self.raw_topology_dict = self.s3_connector.read_json(filepath)

            # Otherwise extract it from raw Topology.csv and MeterAndCabinet.csv
            else:
                # Add cabinet_sm_mapping_dict to class if not already done
                self.create_raw_cabinet_sm_mapping(
                    save=True,
                    overwrite=overwrite,
                    overwrite_timeseries_info=overwrite_timeseries_info,
                )
                # Load topology data if not already done
                self._load_raw_topology_data()
                # Initialize the outermost dict
                self.raw_topology_dict = {}

                zip_groups = self.raw_topology_df.compute().groupby("zip_code_secondary_substation")

                # Iterate through each zip code group
                for zip_code, zip_df in zip_groups:
                    self.raw_topology_dict[zip_code] = {}

                    # Get unique secondary substations for this zip code
                    secondary_subs = (
                        zip_df.groupby(self.sec_substation_col)[self.transformer_col].unique().reset_index()
                    )
                    trafos = zip_df.groupby(self.transformer_col)[self.lv_feeder_col].unique().reset_index()
                    lv_feeder_1 = zip_df.groupby(self.lv_feeder_col)[self.node_1_col].unique().reset_index()
                    lv_feeder_2 = zip_df.groupby(self.lv_feeder_col)[self.node_2_col].unique().reset_index()

                    # Remove NAs
                    secondary_subs[self.transformer_col] = secondary_subs[self.transformer_col].apply(
                        lambda trafos: [t for t in trafos if pd.notna(t)]
                    )
                    trafos[self.lv_feeder_col] = trafos[self.lv_feeder_col].apply(
                        lambda feeders: [f for f in feeders if pd.notna(f)]
                    )
                    lv_feeder_1[self.node_1_col] = lv_feeder_1[self.node_1_col].apply(
                        lambda nodes: [n for n in nodes if pd.notna(n)]
                    )
                    lv_feeder_2[self.node_2_col] = lv_feeder_2[self.node_2_col].apply(
                        lambda nodes: [n for n in nodes if pd.notna(n)]
                    )

                    for _, sec_sub in secondary_subs.iterrows():
                        self.raw_topology_dict[zip_code][sec_sub[self.sec_substation_col]] = {}
                        for trafo in sec_sub[self.transformer_col]:
                            self.raw_topology_dict[zip_code][sec_sub[self.sec_substation_col]][trafo] = {}
                            lv_feeder = trafos[trafos[self.transformer_col] == trafo][self.lv_feeder_col].iloc[0]
                            for lv_f in lv_feeder:
                                self.raw_topology_dict[zip_code][sec_sub[self.sec_substation_col]][trafo][lv_f] = {}
                                node_1_boxes = lv_feeder_1[lv_feeder_1[self.lv_feeder_col] == lv_f][
                                    self.node_1_col
                                ].iloc[0]
                                node_2_boxes = lv_feeder_2[lv_feeder_2[self.lv_feeder_col] == lv_f][
                                    self.node_2_col
                                ].iloc[0]
                                node_1_boxes = [
                                    item for item in node_1_boxes if pd.notna(item) and self.cabinet_name in item
                                ]
                                node_2_boxes = [
                                    item for item in node_2_boxes if pd.notna(item) and self.cabinet_name in item
                                ]
                                cabinets = set(node_1_boxes + node_2_boxes)
                                for cabinet in cabinets:
                                    try:
                                        meters = self.raw_cabinet_sm_mapping_dict[cabinet.split(".")[1]]
                                    except KeyError:
                                        meters = {
                                            "METER_NUMBER": [],
                                            "AVAILABLE_METERS": [],
                                            "MISSING_METERS": [],
                                        }
                                    self.raw_topology_dict[zip_code][sec_sub[self.sec_substation_col]][trafo][lv_f][
                                        cabinet
                                    ] = meters

                if save:
                    self.s3_connector.write_json(filepath, self.raw_topology_dict)
                    logger.info(f"Raw Topology-Smart meter mapping saved to directory {filepath}.")

        return self.raw_topology_dict

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

    def create_raw_sm_topology_mapping(self, save=True, overwrite=False, overwrite_timeseries_info=False):
        filepath = f"{self.s3_base}/{PROCESSED_DATA_DIR}/topology/raw/{self.sm_topology_mapping_file}"
        if self.raw_topology_dict_reversed is None:
            if self.s3_connector.exists(filepath) and not overwrite:
                self.raw_topology_dict_reversed = self.s3_connector.read_json(filepath)
            else:
                self.create_raw_topology_sm_mapping(
                    overwrite=overwrite,
                    overwrite_timeseries_info=overwrite_timeseries_info,
                )
                self.raw_topology_dict_reversed = self._reverse_topology_sm_mapping(self.raw_topology_dict)

                if save:
                    self.s3_connector.write_json(filepath, self.raw_topology_dict_reversed)
                    logger.info(f"Raw Smart meter-Topology mapping saved to directory {filepath}.")

        return self.raw_topology_dict_reversed

    def create_cleaned_topology_sm_mapping(self, save=True, overwrite=False, overwrite_timeseries_info=False) -> dict:
        """
        Function to create a mapping between topology (Substations, Transformers,
        Feeders) and Smart Meters based on cleaned topology data.

        arguments:
            save (boolean): If True, a textfile of the mapping is saved.

        returns:
            cleaned_topology_dict (dict): The topology-SM mapping in dictionary format.
        """
        filepath = f"{self.s3_base}/{PROCESSED_DATA_DIR}/topology/cleaned/{self.topology_sm_mapping_file}"
        if self.cleaned_topology_dict is None:
            if self.s3_connector.exists(filepath) and not overwrite:
                self.cleaned_topology_dict = self.s3_connector.read_json(filepath)

            # Otherwise extract it from cleaned Topology.csv and MeterAndCabinet.csv
            else:
                # Add cabinet_sm_mapping_dict to class if not already done
                self.create_cleaned_cabinet_sm_mapping(
                    overwrite=overwrite,
                    overwrite_timeseries_info=overwrite_timeseries_info,
                )

                # Load topology data if not already done
                self._load_cleaned_topology_data()

                # Initialize the outermost dict
                self.cleaned_topology_dict = {}

                zip_groups = self.cleaned_topology_df.groupby("zip_code_secondary_substation")

                # Iterate through each zip code group
                for zip_code, zip_df in zip_groups:
                    self.cleaned_topology_dict[zip_code] = {}

                    # Get unique secondary substations for this zip code
                    secondary_subs = (
                        zip_df.groupby(self.sec_substation_col)[self.transformer_col].unique().reset_index()
                    )
                    trafos = zip_df.groupby(self.transformer_col)[self.lv_feeder_col].unique().reset_index()
                    lv_feeder_1 = zip_df.groupby(self.lv_feeder_col)[self.node_1_col].unique().reset_index()
                    lv_feeder_2 = zip_df.groupby(self.lv_feeder_col)[self.node_2_col].unique().reset_index()

                    for _, sec_sub in secondary_subs.iterrows():
                        self.cleaned_topology_dict[zip_code][sec_sub[self.sec_substation_col]] = {}
                        for trafo in sec_sub[self.transformer_col]:
                            self.cleaned_topology_dict[zip_code][sec_sub[self.sec_substation_col]][trafo] = {}
                            lv_feeder = trafos[trafos[self.transformer_col] == trafo][self.lv_feeder_col].iloc[0]
                            for lv_f in lv_feeder:
                                self.cleaned_topology_dict[zip_code][sec_sub[self.sec_substation_col]][trafo][lv_f] = {}
                                node_1_boxes = lv_feeder_1[lv_feeder_1[self.lv_feeder_col] == lv_f][
                                    self.node_1_col
                                ].iloc[0]
                                node_2_boxes = lv_feeder_2[lv_feeder_2[self.lv_feeder_col] == lv_f][
                                    self.node_2_col
                                ].iloc[0]
                                node_1_boxes = [item for item in node_1_boxes if self.cabinet_name in item]
                                node_2_boxes = [item for item in node_2_boxes if self.cabinet_name in item]
                                cabinets = set(node_1_boxes + node_2_boxes)
                                for cabinet in cabinets:
                                    try:
                                        meters = self.cleaned_cabinet_sm_mapping_dict[cabinet.split(".")[1]]
                                    except KeyError:
                                        meters = {
                                            "METER_NUMBER": [],
                                            "AVAILABLE_METERS": [],
                                            "MISSING_METERS": [],
                                        }
                                    self.cleaned_topology_dict[zip_code][sec_sub[self.sec_substation_col]][trafo][lv_f][
                                        cabinet
                                    ] = meters

                if save:
                    self.s3_connector.write_json(filepath, self.cleaned_topology_dict)
                    logger.info(f"Cleaned Topology-Smart meter mapping saved to directory {filepath}.")

        return self.cleaned_topology_dict

    def create_cleaned_sm_topology_mapping(self, save=True, overwrite=False, overwrite_timeseries_info=False):
        filepath = f"{self.s3_base}/{PROCESSED_DATA_DIR}/topology/cleaned/{self.sm_topology_mapping_file}"
        if self.cleaned_topology_dict_reversed is None:
            if self.s3_connector.exists(filepath) and not overwrite:
                self.cleaned_topology_dict_reversed = self.s3_connector.read_json(filepath)

            else:
                self.create_cleaned_topology_sm_mapping(
                    overwrite=overwrite,
                    overwrite_timeseries_info=overwrite_timeseries_info,
                )
                self.cleaned_topology_dict_reversed = self._reverse_topology_sm_mapping(self.cleaned_topology_dict)

                if save:
                    self.s3_connector.write_json(filepath, self.cleaned_topology_dict_reversed)
                    logger.info(f"Cleaned Smart meter-Topology mapping saved to directory {filepath}.")

        return self.cleaned_topology_dict_reversed

    def create_cleaned_and_corrected_topology_sm_mapping(
        self, save=True, overwrite=False, overwrite_timeseries_info=False
    ) -> dict:
        """
        Function to create a mapping between topology (Substations, Transformers,
        Feeders) and Smart Meters based on cleaned and corrected topology data.

        arguments:
            save (boolean): If True, a textfile of the mapping is saved.

        returns:
            cleaned_and_corrected_topology_dict (dict): The topology-SM mapping in
                dictionary format.
        """
        filepath = f"{self.s3_base}/{PROCESSED_DATA_DIR}/topology/cleaned_and_corrected/{self.topology_sm_mapping_file}"
        if self.cleaned_and_corrected_topology_dict is None:
            if self.s3_connector.exists(filepath) and not overwrite:
                self.cleaned_and_corrected_topology_dict = self.s3_connector.read_json(filepath)

            else:
                # Add cleaned_and_corrected_topology_dict to class if not already done
                self.create_cleaned_and_corrected_cabinet_sm_mapping(
                    overwrite=overwrite,
                    overwrite_timeseries_info=overwrite_timeseries_info,
                )

                # Load topology data if not already done
                self._load_cleaned_and_corrected_topology_data()

                # Initialize the outermost dict
                self.cleaned_and_corrected_topology_dict = {}

                zip_groups = self.cleaned_and_corrected_topology_df.groupby("zip_code_secondary_substation")

                # Iterate through each zip code group
                for zip_code, zip_df in zip_groups:
                    self.cleaned_and_corrected_topology_dict[zip_code] = {}

                    # Get unique secondary substations for this zip code
                    secondary_subs = (
                        zip_df.groupby(self.sec_substation_col)[self.transformer_col].unique().reset_index()
                    )
                    trafos = zip_df.groupby(self.transformer_col)[self.lv_feeder_col].unique().reset_index()
                    lv_feeder_1 = zip_df.groupby(self.lv_feeder_col)[self.node_1_col].unique().reset_index()
                    lv_feeder_2 = zip_df.groupby(self.lv_feeder_col)[self.node_2_col].unique().reset_index()

                    for _, sec_sub in secondary_subs.iterrows():
                        self.cleaned_and_corrected_topology_dict[zip_code][sec_sub[self.sec_substation_col]] = {}
                        for trafo in sec_sub[self.transformer_col]:
                            self.cleaned_and_corrected_topology_dict[zip_code][sec_sub[self.sec_substation_col]][
                                trafo
                            ] = {}
                            lv_feeder = trafos[trafos[self.transformer_col] == trafo][self.lv_feeder_col].iloc[0]
                            for lv_f in lv_feeder:
                                self.cleaned_and_corrected_topology_dict[zip_code][sec_sub[self.sec_substation_col]][
                                    trafo
                                ][lv_f] = {}
                                node_1_boxes = lv_feeder_1[lv_feeder_1[self.lv_feeder_col] == lv_f][
                                    self.node_1_col
                                ].iloc[0]
                                node_2_boxes = lv_feeder_2[lv_feeder_2[self.lv_feeder_col] == lv_f][
                                    self.node_2_col
                                ].iloc[0]
                                node_1_boxes = [item for item in node_1_boxes if self.cabinet_name in item]
                                node_2_boxes = [item for item in node_2_boxes if self.cabinet_name in item]
                                cabinets = set(node_1_boxes + node_2_boxes)
                                for cabinet in cabinets:
                                    try:
                                        meters = self.cleaned_and_corrected_cabinet_sm_mapping_dict[
                                            cabinet.split(".")[1]
                                        ]
                                    except KeyError:
                                        meters = {
                                            "METER_NUMBER": [],
                                            "AVAILABLE_METERS": [],
                                            "MISSING_METERS": [],
                                        }
                                    self.cleaned_and_corrected_topology_dict[zip_code][
                                        sec_sub[self.sec_substation_col]
                                    ][trafo][lv_f][cabinet] = meters

                if save:
                    self.s3_connector.write_json(filepath, self.cleaned_and_corrected_topology_dict)
                    logger.info(f"Cleaned and Corrected Topology-Smart meter mapping saved to directory {filepath}.")

        return self.cleaned_and_corrected_topology_dict

    def create_cleaned_and_corrected_sm_topology_mapping(
        self, save=True, overwrite=False, overwrite_timeseries_info=False
    ):
        filepath = f"{self.s3_base}/{PROCESSED_DATA_DIR}/topology/cleaned_and_corrected/{self.sm_topology_mapping_file}"
        if self.cleaned_and_corrected_topology_dict_reversed is None:
            if self.s3_connector.exists(filepath) and not overwrite:
                self.cleaned_and_corrected_topology_dict_reversed = self.s3_connector.read_json(filepath)
            else:
                self.create_cleaned_and_corrected_topology_sm_mapping(
                    overwrite=overwrite,
                    overwrite_timeseries_info=overwrite_timeseries_info,
                )
                self.cleaned_and_corrected_topology_dict_reversed = self._reverse_topology_sm_mapping(
                    self.cleaned_and_corrected_topology_dict
                )

                if save:
                    self.s3_connector.write_json(filepath, self.cleaned_and_corrected_topology_dict_reversed)
                    logger.info(f"Cleaned and corrected Smart meter-Topology mapping saved to directory {filepath}.")

        return self.cleaned_and_corrected_topology_dict_reversed

    def extract_raw_dataset_for_sm(
        self,
        meter_id,
        meter_df=None,
        add_current=False,
        add_unbalance=False,
        save=False,
        overwrite_timeseries_info=False,
    ):
        """
        Function to extract timeseries data of a specific smart meter from raw timeseries data.

        arguments:
            meter_id (int): ID of the smart meter to extract data for.
            calc_unbalance (boolean): If True, phase unbalance will be calculated and included.
            save (boolean): If True, dataset will be stored as parquet file.

        returns:
            meter_df (Dataframe): Dataset of the requested smart meter
        """

        # Ensure meter_id is a valid string
        if pd.isna(meter_id):
            raise ValueError("Invalid meter_id.")
        meter_id = str(meter_id)

        if meter_df is None:
            self._load_raw_timeseries_data()

            # Save some info about the timeseries dataset if not already done
            self._get_timeseries_info()

            # Check if SM data available in timeseries dataset
            if meter_id not in self.id_list_of_sms_with_data:
                raise ValueError(f"No data available in timeseries dataset for meter {meter_id}.")

            # Extract the data of one specific SM
            meter_ddf = self.raw_timeseries_df[self.raw_timeseries_df[self.meter_number_col] == meter_id]
            meter_df = meter_ddf.compute()

        # Sort by timestamps
        meter_df[self.timestamp_dst_col] = pd.to_datetime(meter_df[self.timestamp_dst_col], utc=True)
        meter_df = meter_df.sort_values(by=self.timestamp_dst_col).reset_index(drop=True)

        # Resample to 15 minutes in case that some timestamps are little of
        meter_df = meter_df.set_index(self.timestamp_dst_col)

        # Drop the meter ID column
        meter_df.drop([self.meter_number_col], axis=1, inplace=True)
        print(meter_df.dtypes)
        logging.info(meter_df.dtypes)

        # Keep only numeric columns for resampling
        meter_df = meter_df.select_dtypes(include=[np.number])
        # Resample and take mean of numeric columns
        meter_df = meter_df.resample("15min").mean()

        # Reindex the dataframe to include all timestamps in the range
        meter_df = (
            meter_df.reindex(self.expected_timestamps).reset_index().rename(columns={"index": self.timestamp_dst_col})
        )

        # Make timestamp column the index
        meter_df.set_index(self.timestamp_dst_col, inplace=True)

        # Add the SM ID to the column names
        meter_df.columns = [col + f"_{meter_id}" for col in meter_df.columns]

        meter_df = self.enrich_meter_df(
            meter_df=meter_df,
            meter_id=meter_id,
            add_current=add_current,
            add_unbalance=add_unbalance,
        )

        if save:
            filepath = f"{self.s3_base}/{PROCESSED_DATA_DIR}/individual_sm_datasets/raw_profiles/sm_{meter_id}.parquet"
            self.s3_connector.write_parquet(filepath, meter_df.reset_index())  # reset index to prevent dask errors
            logger.info(f"Raw data of SM {meter_id} saved to {filepath}.")

        return meter_df

    def load_raw_dataset_for_sm(self, meter_id, add_current=False, add_unbalance=False, overwrite=False):
        """
        Function to load existing raw timeseries data of a specific smart meter.

        arguments:
            meter_id (int): ID of the smart meter to extract data for.
            calc_unbalance (boolean): If True, phase unbalance will be calculated and included.
            overwrite (boolean): If True, dataset will be overwritten.

        returns:
            meter_df (Dataframe): Dataset of the requested smart meter
        """

        # Ensure meter_id is a valid string
        if pd.isna(meter_id):
            raise ValueError("Invalid meter_id.")
        meter_id = str(meter_id)

        filepath = f"{self.s3_base}/{PROCESSED_DATA_DIR}/individual_sm_datasets/raw_profiles/sm_{meter_id}.parquet"

        # Check if raw SM dataset exists
        if self.s3_connector.exists(filepath):
            meter_df = self.s3_connector.read_parquet(filepath)
        else:
            logger.info(f"Dataset of SM {meter_id} does not exist.")
            return None

        meter_df = self.enrich_meter_df(
            meter_df=meter_df,
            meter_id=meter_id,
            add_current=add_current,
            add_unbalance=add_unbalance,
        )

        if overwrite:
            self.s3_connector.write_parquet(filepath, meter_df)
            logger.info(f"Raw dataset of SM {meter_id} overwritten.")

        return meter_df

    def load_cleaned_dataset_for_sm(self, meter_id, add_current=False, add_unbalance=False, overwrite=False):
        """
        Function to load existing cleaned timeseries data of a specific smart meter.

        arguments:
            meter_id (int): ID of the smart meter to extract data for.
            calc_unbalance (boolean): If True, phase unbalance will be calculated and included.
            overwrite (boolean): If True, dataset will be overwritten.

        returns:
            meter_df (Dataframe): Dataset of the requested smart meter
        """
        # Ensure meter_id is a valid string
        if pd.isna(meter_id):
            raise ValueError("Invalid meter_id.")
        meter_id = str(meter_id)

        filepath = f"{self.s3_base}/{PROCESSED_DATA_DIR}/individual_sm_datasets/cleaned_profiles/sm_{meter_id}.parquet"
        if self.s3_connector.exists(filepath):
            meter_df = self.s3_connector.read_parquet(filepath)
        else:
            err_msg = f"Cleaned dataset of SM {meter_id} does not exist. Use Profile Cleaner first."
            raise FileNotFoundError(err_msg)

        meter_df = self.enrich_meter_df(
            meter_df=meter_df,
            meter_id=meter_id,
            add_current=add_current,
            add_unbalance=add_unbalance,
        )

        if overwrite:
            self.s3_connector.write_parquet(filepath, meter_df)
            logger.info(f"Cleaned dataset of SM {meter_id} overwritten.")

        return meter_df

    def load_cleaned_and_phase_corrected_dataset_for_sm(
        self, meter_id, add_current=False, add_unbalance=False, overwrite=False
    ):
        """
        Function to load existing cleaned and phase-corrected timeseries data of a
        specific smart meter.

        arguments:
            meter_id (int): ID of the smart meter to extract data for.
            calc_unbalance (boolean): If True, phase unbalance will be calculated and included.
            overwrite (boolean): If True, dataset will be overwritten.

        returns:
            meter_df (Dataframe): Dataset of the requested smart meter
        """
        # Ensure meter_id is a valid string
        if pd.isna(meter_id):
            raise ValueError("Invalid meter_id.")
        meter_id = str(meter_id)

        filepath = (
            f"{self.s3_base}/{PROCESSED_DATA_DIR}/individual_sm_datasets/"
            f"cleaned_and_phase_corrected_profiles/sm_{meter_id}.parquet"
        )
        if self.s3_connector.exists(filepath):
            meter_df = self.s3_connector.read_parquet(filepath)
        else:
            err_msg = (
                f"Cleaned and phase-corrected dataset of SM {meter_id} does not exist. "
                f"Use Profile Cleaner and Phase Corrector first."
            )
            raise FileNotFoundError(err_msg)

        meter_df = self.enrich_meter_df(
            meter_df=meter_df,
            meter_id=meter_id,
            add_current=add_current,
            add_unbalance=add_unbalance,
        )

        if overwrite:
            self.s3_connector.write_parquet(filepath, meter_df)
            logger.info(f"Cleaned and phase-corrected dataset of SM {meter_id} overwritten.")

        return meter_df

    def _load_dataset_for_sm_for_parallel_use(
        self,
        meter_id,
        profiles,
        add_current,
        add_unbalance,
        overwrite,
        voltage_col,
        v_phase_unbalance_col,
        v_unbalance_col,
        current_col,
        i_phase_unbalance_col,
        i_unbalance_col,
        active_power_p14_col,
        active_power_p23_col,
    ):
        filepath = f"{self.s3_base}/{PROCESSED_DATA_DIR}/individual_sm_datasets/{profiles}/sm_{meter_id}.parquet"

        # Check if raw SM dataset exists
        if self.s3_connector.exists(filepath):
            meter_df = self.s3_connector.read_parquet(filepath)

            meter_df = self.enrich_meter_df_with(
                meter_df=meter_df,
                meter_id=meter_id,
                add_current=add_current,
                add_unbalance=add_unbalance,
                voltage_col=voltage_col,
                v_phase_unbalance_col=v_phase_unbalance_col,
                v_unbalance_col=v_unbalance_col,
                current_col=current_col,
                i_phase_unbalance_col=i_phase_unbalance_col,
                i_unbalance_col=i_unbalance_col,
                active_power_p14_col=active_power_p14_col,
                active_power_p23_col=active_power_p23_col,
            )

            if overwrite:
                self.s3_connector.write_parquet(filepath, meter_df)
                logger.info(f"Cleaned dataset of SM {meter_id} overwritten.")

            return meter_df

        else:
            logger.info(f"Dataset of SM {meter_id} does not exist.")
            return None

    def _load_dataset_for_multiple_sm(
        self,
        meter_ids,
        profiles,
        add_current=False,
        add_unbalance=False,
        overwrite=False,
    ):
        meter_ids = pd.Series(meter_ids, dtype=pd.StringDtype())  # Create Series from list
        if meter_ids.isna().any():
            raise ValueError("One or more meter_ids are invalid.")
        meter_ids = meter_ids.tolist()

        delayed_tasks = [
            delayed(self._load_dataset_for_sm_for_parallel_use)(
                meter_id=meter_id,
                profiles=profiles,
                add_current=add_current,
                add_unbalance=add_unbalance,
                overwrite=overwrite,
                voltage_col=self.voltage_col,
                v_phase_unbalance_col=self.v_phase_unbalance_col,
                v_unbalance_col=self.v_unbalance_col,
                current_col=self.current_col,
                i_phase_unbalance_col=self.i_phase_unbalance_col,
                i_unbalance_col=self.i_unbalance_col,
                active_power_p14_col=self.active_power_p14_col,
                active_power_p23_col=self.active_power_p23_col,
            )
            for meter_id in meter_ids
        ]
        return list(compute(*delayed_tasks))

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

    def extract_raw_datasets_for_all_sm(self, add_current=False, add_unbalance=False, overwrite_timeseries_info=False):
        """
        Function to extract timeseries data of all smart meters connected to a
        certain cabinet from raw timeseries data.

        arguments:
            cabinet_id (int): ID of the cabinet to extract data for.
            topology_processing_level (str): Processing level of the topology
                (raw, cleaned or cleaned_and_corrected)
            save (boolean): If True, dataset will be stored as parquet file.

        returns:
            cabinet_sm_data (Dataframe): Dataset of SM data for the requested cabinet
        """
        self._load_raw_timeseries_data(filetype="parquet")

        # Save some info about the timeseries dataset if not already done
        self._get_timeseries_info(overwrite=overwrite_timeseries_info)

        filepath = f"{self.s3_base}/{SOURCE_DATA_DIR}/timeseries_id_partitioned.parquet"
        self.s3_connector.write_parquet(
            path=filepath,
            df=self.raw_timeseries_df,
            engine="pyarrow",
            partition_on=[self.meter_number_col],
            write_index=False,
        )

        expected_timestamps = self.expected_timestamps
        meter_number_col = self.meter_number_col
        timestamp_dst_col = self.timestamp_dst_col
        voltage_col = self.voltage_col
        v_phase_unbalance_col = self.v_phase_unbalance_col
        v_unbalance_col = self.v_unbalance_col
        current_col = self.current_col
        i_phase_unbalance_col = self.i_phase_unbalance_col
        i_unbalance_col = self.i_unbalance_col
        active_power_p14_col = self.active_power_p14_col
        active_power_p23_col = self.active_power_p23_col

        @delayed
        def _merge_dask_partitions_of_sm_dataset(sm_id_chunk):
            for meter_id in sm_id_chunk:
                filepath_partition = f"{filepath}/{meter_number_col}={meter_id}"
                meter_df = self.s3_connector.read_parquet(filepath_partition)
                meter_df[timestamp_dst_col] = meter_df[timestamp_dst_col].dt.tz_localize("UTC")

                # Sort by timestamps
                meter_df = meter_df.sort_values(by=timestamp_dst_col).reset_index(drop=True)

                # Resample to 15 minutes in case that some timestamps are little of
                meter_df = meter_df.set_index(timestamp_dst_col)
                meter_df = meter_df.resample("15min").mean()

                # Reindex the dataframe to include all timestamps in the range
                meter_df = (
                    meter_df.reindex(expected_timestamps).reset_index().rename(columns={"index": timestamp_dst_col})
                )

                # Make timestamp column the index
                meter_df.set_index(timestamp_dst_col, inplace=True)

                # Add the SM ID to the column names
                meter_df.columns = [col + f"_{meter_id}" for col in meter_df.columns]

                meter_df = self.enrich_meter_df_with(
                    meter_df=meter_df,
                    meter_id=meter_id,
                    add_current=add_current,
                    add_unbalance=add_unbalance,
                    voltage_col=voltage_col,
                    v_phase_unbalance_col=v_phase_unbalance_col,
                    v_unbalance_col=v_unbalance_col,
                    current_col=current_col,
                    i_phase_unbalance_col=i_phase_unbalance_col,
                    i_unbalance_col=i_unbalance_col,
                    active_power_p14_col=active_power_p14_col,
                    active_power_p23_col=active_power_p23_col,
                )
                filepath_dataset = (
                    f"{self.s3_base}/{PROCESSED_DATA_DIR}/individual_sm_datasets/raw_profiles/sm_{meter_id}.parquet"
                )
                self.s3_connector.write_parquet(filepath_dataset, meter_df, compression="snappy")

        # List of unique meter numbers
        meter_ids = self.list_partitioned_meter_ids(filepath)
        sm_id_chunks = np.array_split(meter_ids, min(len(meter_ids), 16))
        delayed_tasks = [_merge_dask_partitions_of_sm_dataset(sm_id_chunk) for sm_id_chunk in sm_id_chunks]
        compute(*delayed_tasks)

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
        """
        Function to extract timeseries data of all smart meters connected to a
        certain cabinet from raw timeseries data.

        arguments:
            cabinet_id (int): ID of the cabinet to extract data for.
            topology_processing_level (str): Processing level of the topology
                (raw, cleaned or cleaned_and_corrected)
            save (boolean): If True, dataset will be stored as parquet file.

        returns:
            cabinet_sm_data (Dataframe): Dataset of SM data for the requested cabinet
        """

        # Ensure cabinet_id is a valid string
        if pd.isna(cabinet_id):
            raise ValueError("Invalid cabinet_id.")
        cabinet_id = str(cabinet_id)

        sm_ids_of_cabinet = self.get_sm_ids_for_cabinet(
            cabinet_id=cabinet_id,
            sm_ids_of_cabinet=sm_ids_of_cabinet,
            topology_processing_level=topology_processing_level,
            overwrite_topology_info=overwrite_topology_info,
            overwrite_timeseries_info=overwrite_timeseries_info,
        )

        # Ensure that all SM IDs belonging to this cabinet are valid strings
        sm_ids_of_cabinet = pd.Series(sm_ids_of_cabinet, dtype=pd.StringDtype())
        if sm_ids_of_cabinet.isna().any():
            raise ValueError("One or more sm_ids_of_cabinet are invalid.")
        sm_ids_of_cabinet = sm_ids_of_cabinet.tolist()

        if use_existing_raw_sm_profiles:
            sm_data_list = self.load_raw_dataset_for_multiple_sm(
                meter_ids=sm_ids_of_cabinet,
                add_current=add_current,
                add_unbalance=add_unbalance,
                overwrite=False,
            )

        else:
            # Initialize
            sm_data_list = []

            # Load dataset lazily if not already loaded
            try:
                self._load_raw_timeseries_data(filetype="parquet")
            except FileNotFoundError:
                self._load_raw_timeseries_data(filetype="csv")

            # Save some info about the timeseries dataset if not already done
            self._get_timeseries_info(overwrite=overwrite_timeseries_info)

            filtered_ddf = self.raw_timeseries_df[self.raw_timeseries_df[self.meter_number_col].isin(sm_ids_of_cabinet)]
            filtered_df = filtered_ddf.compute()

            # Parallelize the per-meter operations
            for sm_id, meter_df in filtered_df.groupby(self.meter_number_col):
                # Use delayed to parallelize each meter's processing
                sm_data_list.append(
                    delayed(self.extract_raw_dataset_for_sm)(
                        sm_id,
                        meter_df,
                        add_current,
                        add_unbalance,
                        save=False,
                        overwrite_timeseries_info=overwrite_timeseries_info,
                    )
                )

            # Compute everything in parallel
            sm_data_list = compute(*sm_data_list)

            """
            ### Old version where each meter ID is filtered individually (slower)
            # Loop over connected SMs and try to append their data to the list, if
            # data not available store in missing_SMs
            for sm_id in sm_ids_of_cabinet:
                sm_data = self.extract_raw_dataset_for_sm(meter_id=sm_id,
                                                        add_current=add_current,
                                                        add_unbalance=add_unbalance,
                                                        save=False)
                sm_data_list.append(sm_data)
            """

        # Combine the data of all connected SMs
        cabinet_sm_data = pd.concat(sm_data_list, axis=1)

        if save:
            filepath = (
                f"{self.s3_base}/{PROCESSED_DATA_DIR}/individual_cabinet_datasets/"
                f"based_on_{topology_processing_level}_topology/raw_profiles/"
                f"cabinet_{cabinet_id}.parquet"
            )
            self.s3_connector.write_parquet(filepath, cabinet_sm_data)
            logger.info(f"Raw SM data of cabinet {cabinet_id} saved to directory {filepath}.")

        return cabinet_sm_data

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
        """
        Function to extract timeseries data of all smart meters connected to a
        certain cabinet from raw timeseries data.

        arguments:
            cabinet_id (int): ID of the cabinet to extract data for.
            topology_processing_level (str): Processing level of the topology
                (raw, cleaned or cleaned_and_corrected)
            save (boolean): If True, dataset will be stored as parquet file.

        returns:
            cabinet_sm_data (Dataframe): Dataset of SM data for the requested cabinet
        """

        # Ensure cabinet_id is a valid string
        if pd.isna(cabinet_id):
            raise ValueError("Invalid cabinet_id.")
        cabinet_id = str(cabinet_id)

        sm_ids_of_cabinet = self.get_sm_ids_for_cabinet(
            cabinet_id=cabinet_id,
            sm_ids_of_cabinet=sm_ids_of_cabinet,
            topology_processing_level=topology_processing_level,
            overwrite_topology_info=overwrite_topology_info,
            overwrite_timeseries_info=overwrite_timeseries_info,
        )

        # Ensure that all SM IDs belonging to this cabinet are valid strings
        sm_ids_of_cabinet = pd.Series(sm_ids_of_cabinet, dtype=pd.StringDtype())
        if sm_ids_of_cabinet.isna().any():
            raise ValueError("One or more sm_ids_of_cabinet are invalid.")
        sm_ids_of_cabinet = sm_ids_of_cabinet.tolist()

        # Load datasets of all SMs associated with this cabinet
        sm_data_list = self.load_cleaned_dataset_for_multiple_sm(
            meter_ids=sm_ids_of_cabinet,
            add_current=add_current,
            add_unbalance=add_unbalance,
            overwrite=False,
        )

        # Combine the data of all connected SMs
        cabinet_sm_data = pd.concat(sm_data_list, axis=1)

        if save:
            filepath = (
                f"{self.s3_base}/{PROCESSED_DATA_DIR}/individual_cabinet_datasets/"
                f"based_on_{topology_processing_level}_topology/cleaned_profiles/"
                f"cabinet_{cabinet_id}.parquet"
            )
            self.s3_connector.write_parquet(filepath, cabinet_sm_data)
            logger.info(f"Cleaned SM data of cabinet {cabinet_id} saved to directory {filepath}.")

        return cabinet_sm_data

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
        """
        Function to extract timeseries data of all smart meters connected to a
        certain cabinet from raw timeseries data.

        arguments:
            cabinet_id (int): ID of the cabinet to extract data for.
            topology_processing_level (str): Processing level of the topology
                (raw, cleaned or cleaned_and_corrected)
            save (boolean): If True, dataset will be stored as parquet file.

        returns:
            cabinet_sm_data (Dataframe): Dataset of SM data for the requested cabinet
        """

        # Ensure cabinet_id is a valid string
        if pd.isna(cabinet_id):
            raise ValueError("Invalid cabinet_id.")
        cabinet_id = str(cabinet_id)

        sm_ids_of_cabinet = self.get_sm_ids_for_cabinet(
            cabinet_id=cabinet_id,
            sm_ids_of_cabinet=sm_ids_of_cabinet,
            topology_processing_level=topology_processing_level,
            overwrite_topology_info=overwrite_topology_info,
            overwrite_timeseries_info=overwrite_timeseries_info,
        )

        # Ensure that all SM IDs belonging to this cabinet are valid strings
        sm_ids_of_cabinet = pd.Series(sm_ids_of_cabinet, dtype=pd.StringDtype())
        if sm_ids_of_cabinet.isna().any():
            raise ValueError("One or more sm_ids_of_cabinet are invalid.")
        sm_ids_of_cabinet = sm_ids_of_cabinet.tolist()

        # Load datasets of all SMs associated with this cabinet
        sm_data_list = self.load_cleaned_and_phase_corrected_dataset_for_multiple_sm(
            meter_ids=sm_ids_of_cabinet,
            add_current=add_current,
            add_unbalance=add_unbalance,
            overwrite=False,
        )

        # Combine the data of all connected SMs
        cabinet_sm_data = pd.concat(sm_data_list, axis=1)

        if save:
            filepath = (
                f"{self.s3_base}/{PROCESSED_DATA_DIR}/individual_cabinet_datasets/"
                f"based_on_{topology_processing_level}_topology/"
                f"cleaned_and_phase_corrected_profiles/cabinet_{cabinet_id}.parquet"
            )
            self.s3_connector.write_parquet(filepath, cabinet_sm_data)
            logger.info(f"Cleaned and phase-corrected SM data of cabinet {cabinet_id} saved to directory {filepath}.")

        return cabinet_sm_data

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
        """
        Function to extract timeseries data of all smart meters connected to a
        specific feeder from raw timeseries data.

        arguments:
            feeder_id (int): ID of the feeder to extract data for.
            save (boolean): If True, dataset will be stored as parquet file.

        returns:
            feeder_sm_data (Dataframe): Dataset of SM data for the requested feeder
        """

        # Ensure feeder_id is a valid string
        if pd.isna(feeder_id):
            raise ValueError("Invalid feeder_id.")
        feeder_id = str(feeder_id)

        sm_ids_of_feeder = self.get_sm_ids_for_feeder(
            feeder_id=feeder_id,
            sm_ids_of_feeder=sm_ids_of_feeder,
            topology_processing_level=topology_processing_level,
            overwrite_topology_info=overwrite_topology_info,
            overwrite_timeseries_info=overwrite_timeseries_info,
        )

        if use_existing_raw_sm_profiles:
            sm_data_list = self.load_raw_dataset_for_multiple_sm(
                meter_ids=sm_ids_of_feeder,
                add_current=add_current,
                add_unbalance=add_unbalance,
                overwrite=False,
            )

        else:
            # Initialize list of individual SM datasets
            sm_data_list = []

            # Load dataset lazily if not already loaded
            try:
                self._load_raw_timeseries_data(filetype="parquet")
            except FileNotFoundError:
                self._load_raw_timeseries_data(filetype="csv")

            # Save some info about the timeseries dataset if not already done
            self._get_timeseries_info(overwrite=overwrite_timeseries_info)

            filtered_ddf = self.raw_timeseries_df[self.raw_timeseries_df[self.meter_number_col].isin(sm_ids_of_feeder)]
            filtered_df = filtered_ddf.compute()

            # Parallelize the per-meter operations
            for sm_id, meter_df in filtered_df.groupby(self.meter_number_col):
                # Use delayed to parallelize each meter's processing
                sm_data_list.append(
                    delayed(self.extract_raw_dataset_for_sm)(
                        sm_id,
                        meter_df,
                        add_current,
                        add_unbalance,
                        save=False,
                        overwrite_timeseries_info=overwrite_timeseries_info,
                    )
                )

            # Compute everything in parallel
            sm_data_list = compute(*sm_data_list)

            """
            ### Old version where each meter ID is filtered individually (slower)
            # Loop over connected SMs and try to append their data to the list, if
            # data not available store in missing_SMs
            for sm_id in sm_ids_of_feeder:
                sm_data_list.append(self.extract_raw_dataset_for_sm(meter_id=sm_id,
                                                                    add_current=add_current,
                                                                    add_unbalance=add_unbalance,
                                                                    save=False))
            """

        # Combine the data of all connected SMs
        feeder_sm_data = pd.concat(sm_data_list, axis=1)

        if save:
            filepath = (
                f"{self.s3_base}/{PROCESSED_DATA_DIR}/individual_feeder_datasets/"
                f"based_on_{topology_processing_level}_topology/raw_profiles/{feeder_id}.parquet"
            )
            self.s3_connector.write_parquet(filepath, feeder_sm_data)
            logger.info(f"Raw SM data of feeder {feeder_id} saved to directory {filepath}.")

        return feeder_sm_data

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
        """
        Function to extract timeseries data of all smart meters connected to a
        certain transformer from raw timeseries data.

        arguments:
            transformer_id (int): ID of the transformer to extract data for.
            save (boolean): If True, dataset will be stored as parquet file.

        returns:
            transformer_sm_data (Dataframe): Dataset of SM data for the requested transformer
        """

        # Ensure transformer_id is a valid string
        if pd.isna(transformer_id):
            raise ValueError("Invalid transformer_id.")
        transformer_id = str(transformer_id)

        sm_ids_of_transformer = self.get_sm_ids_for_transformer(
            transformer_id=transformer_id,
            sm_ids_of_transformer=sm_ids_of_transformer,
            topology_processing_level=topology_processing_level,
            overwrite_topology_info=overwrite_topology_info,
            overwrite_timeseries_info=overwrite_timeseries_info,
        )

        if use_existing_raw_sm_profiles:
            sm_data_list = self.load_raw_dataset_for_multiple_sm(
                meter_ids=sm_ids_of_transformer,
                add_current=add_current,
                add_unbalance=add_unbalance,
                overwrite=False,
            )

        else:
            # Initialize list of individual SM datasets
            sm_data_list = []

            # Load dataset lazily if not already loaded
            try:
                self._load_raw_timeseries_data(filetype="parquet")
            except FileNotFoundError:
                self._load_raw_timeseries_data(filetype="csv")

            # Save some info about the timeseries dataset if not already done
            self._get_timeseries_info(overwrite=overwrite_timeseries_info)

            filtered_ddf = self.raw_timeseries_df[
                self.raw_timeseries_df[self.meter_number_col].isin(sm_ids_of_transformer)
            ]
            filtered_df = filtered_ddf.compute()

            # Parallelize the per-meter operations
            for sm_id, meter_df in filtered_df.groupby(self.meter_number_col):
                # Use delayed to parallelize each meter's processing
                sm_data_list.append(
                    delayed(self.extract_raw_dataset_for_sm)(
                        sm_id,
                        meter_df,
                        add_current,
                        add_unbalance,
                        save=False,
                        overwrite_timeseries_info=overwrite_timeseries_info,
                    )
                )

                # Compute everything in parallel
                sm_data_list = compute(*sm_data_list)

        # Combine the data of all connected SMs
        transformer_sm_data = pd.concat(sm_data_list, axis=1)

        if save:
            filepath = (
                f"{self.s3_base}/{PROCESSED_DATA_DIR}/individual_transformer_datasets/"
                f"based_on_{topology_processing_level}_topology/raw_profiles/"
                f"transformer_{transformer_id}.parquet"
            )
            self.s3_connector.write_parquet(filepath, transformer_sm_data, engine="pyarrow")
            logger.info(f"Raw SM data of transformer {transformer_id} saved to directory {filepath}.")

        return transformer_sm_data

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
        """
        Function to extract timeseries data of all smart meters connected to a
        specific secondary substation from raw timeseries data.

        arguments:
            substation_id (int): ID of the substation to extract data for.
            save (boolean): If True, dataset will be stored as parquet file.

        returns:
            substation_sm_data (Dataframe): Dataset of SM data for the requested substation
        """

        # Ensure substation_id is a valid string
        if pd.isna(substation_id):
            raise ValueError("Invalid substation_id.")
        substation_id = str(substation_id)

        sm_ids_of_secondary_substation = self.get_sm_ids_for_secondary_substation(
            substation_id=substation_id,
            sm_ids_of_secondary_substation=sm_ids_of_secondary_substation,
            topology_processing_level=topology_processing_level,
            overwrite_topology_info=overwrite_topology_info,
            overwrite_timeseries_info=overwrite_timeseries_info,
        )

        if use_existing_raw_sm_profiles:
            sm_data_list = self.load_raw_dataset_for_multiple_sm(
                meter_ids=sm_ids_of_secondary_substation,
                add_current=add_current,
                add_unbalance=add_unbalance,
                overwrite=False,
            )

        else:
            # Initialize list of individual SM datasets
            sm_data_list = []

            # Load dataset lazily if not already loaded
            try:
                self._load_raw_timeseries_data(filetype="parquet")
            except FileNotFoundError:
                self._load_raw_timeseries_data(filetype="csv")

            # Save some info about the timeseries dataset if not already done
            self._get_timeseries_info(overwrite=overwrite_timeseries_info)

            filtered_ddf = self.raw_timeseries_df[
                self.raw_timeseries_df[self.meter_number_col].isin(sm_ids_of_secondary_substation)
            ]
            filtered_df = filtered_ddf.compute()

            # Parallelize the per-meter operations
            for sm_id, meter_df in filtered_df.groupby(self.meter_number_col):
                # Use delayed to parallelize each meter's processing
                sm_data_list.append(
                    delayed(self.extract_raw_dataset_for_sm)(
                        sm_id,
                        meter_df,
                        add_current,
                        add_unbalance,
                        save=False,
                        overwrite_timeseries_info=overwrite_timeseries_info,
                    )
                )

            # Compute everything in parallel
            sm_data_list = compute(*sm_data_list)

        # Combine the data of all connected SMs
        if sm_data_list:
            substation_sm_data = pd.concat(sm_data_list, axis=1)
        else:
            logger.info(f"No SM data for substation {substation_id}.")
            substation_sm_data = pd.DataFrame()

        if save:
            filepath = (
                f"{self.s3_base}/{PROCESSED_DATA_DIR}/individual_secondary_substation_datasets/"
                f"based_on_{topology_processing_level}_topology/raw_profiles/"
                f"secondary_substation_{substation_id}.parquet"
            )
            self.s3_connector.write_parquet(filepath, substation_sm_data, engine="pyarrow")

            logger.info(f"Raw SM data of secondary substation {substation_id} saved to directory {filepath}.")

        return substation_sm_data

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

    def get_sm_ids_for_cabinet(
        self,
        cabinet_id,
        sm_ids_of_cabinet,
        topology_processing_level,
        overwrite_topology_info,
        overwrite_timeseries_info,
    ):
        # If list of SMs belonging to this cabinet NOT provided, extract it from
        # topology information
        if sm_ids_of_cabinet is None:
            # Add Cabinet_SM_mapping_dict to class if not already done
            if topology_processing_level == "raw":
                self.create_raw_cabinet_sm_mapping(
                    overwrite=overwrite_topology_info,
                    overwrite_timeseries_info=overwrite_timeseries_info,
                )
                cabinet_sm_mapping_dict = self.raw_cabinet_sm_mapping_dict
            elif topology_processing_level == "cleaned":
                self.create_cleaned_cabinet_sm_mapping(
                    overwrite=overwrite_topology_info,
                    overwrite_timeseries_info=overwrite_timeseries_info,
                )
                cabinet_sm_mapping_dict = self.cleaned_cabinet_sm_mapping_dict
            elif topology_processing_level == "cleaned_and_corrected":
                self.create_cleaned_and_corrected_cabinet_sm_mapping(
                    overwrite=overwrite_topology_info,
                    overwrite_timeseries_info=overwrite_timeseries_info,
                )
                cabinet_sm_mapping_dict = self.cleaned_and_corrected_cabinet_sm_mapping_dict
            else:
                err_msg = "topology_processing_level has to be 'raw', 'cleaned' or 'cleaned_and_corrected'."
                raise ValueError(err_msg)

            # Check if the requested cabinet exists in the provided raw data
            if cabinet_id not in cabinet_sm_mapping_dict:
                raise ValueError(f"Cabinet {cabinet_id} does not exist in the dataset.")

            sm_ids_of_cabinet = cabinet_sm_mapping_dict[cabinet_id]["AVAILABLE_METERS"]
        return sm_ids_of_cabinet

    def get_sm_ids_for_feeder(
        self,
        feeder_id,
        sm_ids_of_feeder,
        topology_processing_level,
        overwrite_topology_info,
        overwrite_timeseries_info,
    ):
        if sm_ids_of_feeder is None:
            # Get dictionary containing the topology if not already done
            if topology_processing_level == "raw":
                self.create_raw_topology_sm_mapping(
                    overwrite=overwrite_topology_info,
                    overwrite_timeseries_info=overwrite_timeseries_info,
                )
                topology_dict = self.raw_topology_dict
            elif topology_processing_level == "cleaned":
                self.create_cleaned_topology_sm_mapping(
                    overwrite=overwrite_topology_info,
                    overwrite_timeseries_info=overwrite_timeseries_info,
                )
                topology_dict = self.cleaned_topology_dict
            elif topology_processing_level == "cleaned_and_corrected":
                self.create_cleaned_and_corrected_topology_sm_mapping(
                    overwrite=overwrite_topology_info,
                    overwrite_timeseries_info=overwrite_timeseries_info,
                )
                topology_dict = self.cleaned_and_corrected_topology_dict
            else:
                err_msg = "topology_processing_level has to be 'raw', 'cleaned' or 'cleaned_and_corrected'."
                raise ValueError(err_msg)

            # Initialize list of feeder IDs
            feeder_ids = []

            # Traverse the updated topology_dict with zip_code as outermost layer
            for zip_dict in topology_dict.values():
                for sub_dict in zip_dict.values():
                    for trafo_dict in sub_dict.values():
                        for lv_feeder in trafo_dict:
                            feeder_ids.append(lv_feeder.split(".")[1])

            # Check if feeder exists
            if str(feeder_id) not in feeder_ids:
                raise ValueError(f"Feeder {feeder_id} does not exist in available data.")

            # Collect all smart meter IDs for the requested feeder
            sm_ids_of_feeder = []

            for zip_dict in topology_dict.values():
                for sub_dict in zip_dict.values():
                    for trafo_dict in sub_dict.values():
                        for lv_feeder, lvfeeder in trafo_dict.items():
                            if lv_feeder.endswith(str(feeder_id)):
                                for sm_ids in lvfeeder.values():
                                    sm_ids_of_feeder.extend(sm_ids.get("AVAILABLE_METERS", []))

            # Ensure all SM IDs are valid strings
            sm_ids_of_feeder = pd.Series(sm_ids_of_feeder, dtype=pd.StringDtype())

        # Ensure that all SM IDs belonging to this feeder are valid strings
        sm_ids_of_feeder = pd.Series(sm_ids_of_feeder, dtype=pd.StringDtype())
        if sm_ids_of_feeder.isna().any():
            raise ValueError("One or more sm_ids_of_feeder are invalid.")
        sm_ids_of_feeder = sm_ids_of_feeder.tolist()

        return sm_ids_of_feeder

    def get_sm_ids_for_transformer(
        self,
        transformer_id,
        sm_ids_of_transformer,
        topology_processing_level,
        overwrite_topology_info,
        overwrite_timeseries_info,
    ):
        if sm_ids_of_transformer is None:
            # Get dictionary containing the topology if not already done
            if topology_processing_level == "raw":
                self.create_raw_topology_sm_mapping(
                    overwrite=overwrite_topology_info,
                    overwrite_timeseries_info=overwrite_timeseries_info,
                )
                topology_dict = self.raw_topology_dict
            elif topology_processing_level == "cleaned":
                self.create_cleaned_topology_sm_mapping(
                    overwrite=overwrite_topology_info,
                    overwrite_timeseries_info=overwrite_timeseries_info,
                )
                topology_dict = self.cleaned_topology_dict
            elif topology_processing_level == "cleaned_and_corrected":
                self.create_cleaned_and_corrected_topology_sm_mapping(
                    overwrite=overwrite_topology_info,
                    overwrite_timeseries_info=overwrite_timeseries_info,
                )
                topology_dict = self.cleaned_and_corrected_topology_dict
            else:
                err_msg = "topology_processing_level has to be 'raw', 'cleaned' or 'cleaned_and_corrected'."
                raise ValueError(err_msg)

            # Initialize list of transformer IDs
            transformer_ids = []

            # Traverse the zip_code-aware topology_dict
            for zip_dict in topology_dict.values():
                for sub_dict in zip_dict.values():
                    for trafo in sub_dict:
                        transformer_ids.append(trafo.split(".")[1])

            # Check if transformer exists
            if str(transformer_id) not in transformer_ids:
                raise ValueError(f"Transformer {transformer_id} does not exist in available data.")

            # Collect all smart meter IDs for the requested transformer
            sm_ids_of_transformer = []

            for zip_dict in topology_dict.values():
                for sub_dict in zip_dict.values():
                    for trafo, trafo_dict in sub_dict.items():
                        if trafo.endswith(str(transformer_id)):
                            for lvfeeder in trafo_dict.values():
                                for sm_ids in lvfeeder.values():
                                    sm_ids_of_transformer.extend(sm_ids.get("AVAILABLE_METERS", []))

        # Ensure that all SM IDs belonging to this transformer are valid strings
        sm_ids_of_transformer = pd.Series(sm_ids_of_transformer, dtype=pd.StringDtype())
        if sm_ids_of_transformer.isna().any():
            raise ValueError("One or more sm_ids_of_transformer are invalid.")
        return sm_ids_of_transformer.tolist()

    def get_sm_ids_for_secondary_substation(
        self,
        substation_id,
        sm_ids_of_secondary_substation,
        topology_processing_level,
        overwrite_topology_info,
        overwrite_timeseries_info,
    ):
        if sm_ids_of_secondary_substation is None:
            # Get dictionary containing the topology if not already done
            if topology_processing_level == "raw":
                self.create_raw_topology_sm_mapping(
                    overwrite=overwrite_topology_info,
                    overwrite_timeseries_info=overwrite_timeseries_info,
                )
                topology_dict = self.raw_topology_dict
            elif topology_processing_level == "cleaned":
                self.create_cleaned_topology_sm_mapping(
                    overwrite=overwrite_topology_info,
                    overwrite_timeseries_info=overwrite_timeseries_info,
                )
                topology_dict = self.cleaned_topology_dict
            elif topology_processing_level == "cleaned_and_corrected":
                self.create_cleaned_and_corrected_topology_sm_mapping(
                    overwrite=overwrite_topology_info,
                    overwrite_timeseries_info=overwrite_timeseries_info,
                )
                topology_dict = self.cleaned_and_corrected_topology_dict
            else:
                err_msg = "topology_processing_level has to be 'raw', 'cleaned' or 'cleaned_and_corrected'."
                raise ValueError(err_msg)

            # Initialize list of substation IDs
            substation_ids = []

            # Traverse the topology_dict to collect all substation IDs
            for zip_dict in topology_dict.values():
                for sub in zip_dict:
                    substation_ids.append(sub.split(".")[1])

            # Check if substation exists
            if str(substation_id) not in substation_ids:
                raise ValueError(f"Substation {substation_id} does not exist in available data.")

            # Collect all smart meter IDs for the requested substation
            sm_ids_of_secondary_substation = []

            for zip_dict in topology_dict.values():
                for sub, sub_dict in zip_dict.items():
                    if sub.endswith(str(substation_id)):
                        for trafo_dict in sub_dict.values():
                            for lvfeeder in trafo_dict.values():
                                for sm_ids in lvfeeder.values():
                                    sm_ids_of_secondary_substation.extend(sm_ids.get("AVAILABLE_METERS", []))

        # Ensure that all SM IDs belonging to this substation are valid strings
        sm_ids_of_secondary_substation = pd.Series(sm_ids_of_secondary_substation, dtype=pd.StringDtype())
        if sm_ids_of_secondary_substation.isna().any():
            raise ValueError("One or more sm_ids_of_secondary_substation are invalid.")
        return sm_ids_of_secondary_substation.tolist()

    def extract_timeseries_data_for_feeder(
        self,
        feeder_id,
        sm_ids_of_feeder,
        topology_processing_level,
        overwrite_topology_info,
        overwrite_timeseries_info,
        add_current,
        add_unbalance,
        save,
        profile,
    ):
        """
        Function to extract timeseries data of all smart meters connected to a
        specific feeder from raw timeseries data.

        arguments:
            feeder_id (int): ID of the feeder to extract data for.
            save (boolean): If True, dataset will be stored as parquet file.

        returns:
            feeder_sm_data (Dataframe): Dataset of SM data for the requested feeder
        """

        # Ensure feeder_id is a valid string
        if pd.isna(feeder_id):
            raise ValueError("Invalid feeder_id.")
        feeder_id = str(feeder_id)

        sm_ids_of_feeder = self.get_sm_ids_for_feeder(
            feeder_id=feeder_id,
            sm_ids_of_feeder=sm_ids_of_feeder,
            topology_processing_level=topology_processing_level,
            overwrite_topology_info=overwrite_topology_info,
            overwrite_timeseries_info=overwrite_timeseries_info,
        )

        # Load datasets of all SMs associated with this feeder
        sm_data_list = self.load_cleaned_and_phase_corrected_dataset_for_multiple_sm(
            meter_ids=sm_ids_of_feeder,
            add_current=add_current,
            add_unbalance=add_unbalance,
            overwrite=False,
        )

        # Combine the data of all connected SMs
        feeder_sm_data = pd.concat(sm_data_list, axis=1)

        if save:
            filepath = (
                f"{self.s3_base}/{PROCESSED_DATA_DIR}/individual_feeder_datasets/"
                f"based_on_{topology_processing_level}_topology/{profile}/{feeder_id}.parquet"
            )
            self.s3_connector.write_parquet(filepath, feeder_sm_data)
            logger.info(f"Profile '{profile}': SM data of feeder {feeder_id} saved to directory {filepath}.")

        return feeder_sm_data

    def extract_timeseries_data_for_transformer(
        self,
        transformer_id,
        sm_ids_of_transformer,
        topology_processing_level,
        overwrite_topology_info,
        overwrite_timeseries_info,
        add_current,
        add_unbalance,
        save,
        profile,
    ):
        """
        Function to extract timeseries data of all smart meters connected to certain transformer
        from raw timeseries data.

        arguments:
            transformer_id (int): ID of the transformer to extract data for.
            save (boolean): If True, dataset will be stored as parquet file.

        returns:
            transformer_sm_data (Dataframe): Dataset of SM data for the requested transformer
        """

        # Ensure transformer_id is a valid string
        if pd.isna(transformer_id):
            raise ValueError("Invalid transformer_id.")
        transformer_id = str(transformer_id)

        sm_ids_of_transformer = self.get_sm_ids_for_transformer(
            transformer_id=transformer_id,
            sm_ids_of_transformer=sm_ids_of_transformer,
            topology_processing_level=topology_processing_level,
            overwrite_topology_info=overwrite_topology_info,
            overwrite_timeseries_info=overwrite_timeseries_info,
        )

        # Load datasets of all SMs associated with this transformer
        if profile == "cleaned_profiles":
            sm_data_list = self.load_cleaned_dataset_for_multiple_sm(
                meter_ids=sm_ids_of_transformer,
                add_current=add_current,
                add_unbalance=add_unbalance,
                overwrite=False,
            )
        elif profile == "cleaned_and_phase_corrected_profiles":
            sm_data_list = self.load_cleaned_and_phase_corrected_dataset_for_multiple_sm(
                meter_ids=sm_ids_of_transformer,
                add_current=add_current,
                add_unbalance=add_unbalance,
                overwrite=False,
            )
        else:
            sm_data_list = self.load_raw_dataset_for_multiple_sm(
                meter_ids=sm_ids_of_transformer,
                add_current=add_current,
                add_unbalance=add_unbalance,
                overwrite=False,
            )

        # Combine the data of all connected SMs
        transformer_sm_data = pd.concat(sm_data_list, axis=1)

        if save:
            filepath = (
                f"{self.s3_base}/{PROCESSED_DATA_DIR}/individual_transformer_datasets/"
                f"based_on_{topology_processing_level}_topology/{profile}/"
                f"transformer_{transformer_id}.parquet"
            )
            self.s3_connector.write_parquet(filepath, transformer_sm_data, engine="pyarrow")
            logger.info(f"Profile '{profile}': SM data of transformer {transformer_id} saved to directory {filepath}.")

        return transformer_sm_data

    def extract_timeseries_data_for_secondary_substation(
        self,
        substation_id,
        sm_ids_of_secondary_substation,
        topology_processing_level,
        overwrite_topology_info,
        overwrite_timeseries_info,
        add_current,
        add_unbalance,
        save,
        profile,
    ):
        """
        Function to extract timeseries data of all smart meters connected to a
        specific secondary substation from raw timeseries data.

        arguments:
            substation_id (int): ID of the substation to extract data for.
            save (boolean): If True, dataset will be stored as parquet file.

        returns:
            substation_sm_data (Dataframe): Dataset of SM data for the requested substation
        """
        # Ensure substation_id is a valid string
        if pd.isna(substation_id):
            raise ValueError("Invalid substation_id.")
        substation_id = str(substation_id)

        sm_ids_of_secondary_substation = self.get_sm_ids_for_secondary_substation(
            substation_id=substation_id,
            sm_ids_of_secondary_substation=sm_ids_of_secondary_substation,
            topology_processing_level=topology_processing_level,
            overwrite_topology_info=overwrite_topology_info,
            overwrite_timeseries_info=overwrite_timeseries_info,
        )

        if profile == "cleaned_profiles":
            # Load datasets of all SMs associated with this substation
            sm_data_list = self.load_cleaned_dataset_for_multiple_sm(
                meter_ids=sm_ids_of_secondary_substation,
                add_current=add_current,
                add_unbalance=add_unbalance,
                overwrite=False,
            )
        elif profile == "cleaned_and_phase_corrected_profiles":
            sm_data_list = self.load_cleaned_and_phase_corrected_dataset_for_multiple_sm(
                meter_ids=sm_ids_of_secondary_substation,
                add_current=add_current,
                add_unbalance=add_unbalance,
                overwrite=False,
            )
        else:
            sm_data_list = self.load_raw_dataset_for_multiple_sm(
                meter_ids=sm_ids_of_secondary_substation,
                add_current=add_current,
                add_unbalance=add_unbalance,
                overwrite=False,
            )

        # Combine the data of all connected SMs
        if sm_data_list:
            substation_sm_data = pd.concat(sm_data_list, axis=1)
        else:
            logger.info(f"No SM data for substation {substation_id}.")
            substation_sm_data = pd.DataFrame()

        if save:
            filepath = (
                f"{self.s3_base}/{PROCESSED_DATA_DIR}/individual_secondary_substation_datasets/"
                f"based_on_{topology_processing_level}_topology/{profile}/"
                f"secondary_substation_{substation_id}.parquet"
            )
            self.s3_connector.write_parquet(filepath, substation_sm_data, engine="pyarrow")
            logger.info(
                f"Profile '{profile}': SM data of secondary substation {substation_id} saved to directory {filepath}."
            )

        return substation_sm_data

    def extract_timeseries_data_for_sm_with_heatpump(
        self,
        topology_processing_level,
        add_current=False,
        add_unbalance=False,
        overwrite_timeseries_info=False,
        chunk_size=1000,
        profile="raw_profiles",
    ):
        if profile not in ALLOWED_PROFILES:
            raise ValueError(f"'profile' needs to be in {ALLOWED_PROFILES}")
        # Load the meter and cabinet df for the requested topology processing level
        if topology_processing_level == "raw":
            self._load_raw_meter_and_cabinet_data()
            meter_and_cabinet_df = self.raw_meter_and_cabinet_df
        elif topology_processing_level == "cleaned":
            self._load_cleaned_meter_and_cabinet_data()
            meter_and_cabinet_df = self.cleaned_meter_and_cabinet_df
        elif topology_processing_level == "cleaned_and_corrected":
            self._load_cleaned_and_corrected_meter_and_cabinet_data()
            meter_and_cabinet_df = self.cleaned_and_corrected_meter_and_cabinet_df
        else:
            err_msg = "topology_processing_level has to be 'raw', 'cleaned' or 'cleaned_and_corrected'."
            raise ValueError(err_msg)

        # Create list of all SMs with heatpump
        sm_ids_with_heatpump = list(
            meter_and_cabinet_df[meter_and_cabinet_df["has_heat_pump"]][self.meter_number_col].unique()
        )

        # Filter out all SM IDs without data
        self._get_timeseries_info(overwrite=overwrite_timeseries_info)
        sm_ids_with_heatpump = list(set(sm_ids_with_heatpump) & set(self.id_list_of_sms_with_data))

        # Load or extract and store SM datasets in batches as there are too many
        # to fit them all into memory at once
        chunks = [sm_ids_with_heatpump[i : i + chunk_size] for i in range(0, len(sm_ids_with_heatpump), chunk_size)]

        filepath = (
            f"{self.s3_base}/{PROCESSED_DATA_DIR}/sm_with_heatpump_datasets/"
            f"based_on_{topology_processing_level}_topology/{profile}"
        )
        for chunk in chunks:
            if profile == "cleaned_profiles":
                sm_data_list_batch = self.load_cleaned_dataset_for_multiple_sm(
                    meter_ids=chunk,
                    add_current=add_current,
                    add_unbalance=add_unbalance,
                    overwrite=False,
                )
            elif profile == "cleaned_and_phase_corrected_profiles":
                sm_data_list_batch = self.load_cleaned_and_phase_corrected_dataset_for_multiple_sm(
                    meter_ids=chunk,
                    add_current=add_current,
                    add_unbalance=add_unbalance,
                    overwrite=False,
                )
            else:
                sm_data_list_batch = self.load_raw_dataset_for_multiple_sm(
                    meter_ids=chunk,
                    add_current=add_current,
                    add_unbalance=add_unbalance,
                    overwrite=False,
                )

            # Save data of SMs with heatpump
            for idx, meter_id in enumerate(chunk):
                if sm_data_list_batch[idx] is not None:
                    self.s3_connector.write_parquet(
                        f"{filepath}/sm_{meter_id}.parquet",
                        sm_data_list_batch[idx],
                        engine="pyarrow",
                    )

        logger.info(f"Profile '{profile}': SM data of SMs with heatpump saved to directory {filepath}.")

    def extract_timeseries_data_for_sm_with_pv(
        self,
        topology_processing_level,
        add_current=False,
        add_unbalance=False,
        overwrite_timeseries_info=False,
        chunk_size=1000,
        profile="raw_profiles",
    ):
        if profile not in ALLOWED_PROFILES:
            raise ValueError(f"'profile' needs to be in {ALLOWED_PROFILES}")
        # Load the meter and cabinet df for the requested topology processing level
        if topology_processing_level == "raw":
            self._load_raw_meter_and_cabinet_data()
            meter_and_cabinet_df = self.raw_meter_and_cabinet_df
        elif topology_processing_level == "cleaned":
            self._load_cleaned_meter_and_cabinet_data()
            meter_and_cabinet_df = self.cleaned_meter_and_cabinet_df
        elif topology_processing_level == "cleaned_and_corrected":
            self._load_cleaned_and_corrected_meter_and_cabinet_data()
            meter_and_cabinet_df = self.cleaned_and_corrected_meter_and_cabinet_df
        else:
            err_msg = "topology_processing_level has to be 'raw', 'cleaned' or 'cleaned_and_corrected'."
            raise ValueError(err_msg)

        # Create list of all SMs with heatpump
        sm_ids_with_pv = list(
            meter_and_cabinet_df[meter_and_cabinet_df["has_solar_panel"]][self.meter_number_col].unique()
        )

        # Filter out all SM IDs without data
        self._get_timeseries_info(overwrite=overwrite_timeseries_info)
        sm_ids_with_pv = list(set(sm_ids_with_pv) & set(self.id_list_of_sms_with_data))

        # Load or extract and store SM datasets in batches as there are too many
        # to fit them all into memory at once
        chunks = [sm_ids_with_pv[i : i + chunk_size] for i in range(0, len(sm_ids_with_pv), chunk_size)]

        filepath = (
            f"{self.s3_base}/{PROCESSED_DATA_DIR}/sm_with_pv_datasets/"
            f"based_on_{topology_processing_level}_topology/{profile}"
        )
        for chunk in chunks:
            if profile == "cleaned_profiles":
                sm_data_list_batch = self.load_cleaned_dataset_for_multiple_sm(
                    meter_ids=chunk,
                    add_current=add_current,
                    add_unbalance=add_unbalance,
                    overwrite=False,
                )
            elif profile == "cleaned_and_phase_corrected_profiles":
                sm_data_list_batch = self.load_cleaned_and_phase_corrected_dataset_for_multiple_sm(
                    meter_ids=chunk,
                    add_current=add_current,
                    add_unbalance=add_unbalance,
                    overwrite=False,
                )
            else:
                sm_data_list_batch = self.load_raw_dataset_for_multiple_sm(
                    meter_ids=chunk,
                    add_current=add_current,
                    add_unbalance=add_unbalance,
                    overwrite=False,
                )

            # Save data of SMs with heatpump
            for idx, meter_id in enumerate(chunk):
                if sm_data_list_batch[idx] is not None:
                    self.s3_connector.write_parquet(
                        f"{filepath}/sm_{meter_id}.parquet",
                        sm_data_list_batch[idx],
                        engine="pyarrow",
                    )

        logger.info(f"Profile '{profile}': SM data of SMs with PV saved to directory {filepath}.")

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
        # Ensure substation_id is a valid string
        if pd.isna(zip_id):
            raise ValueError("Invalid zip_code.")
        zip_id = str(zip_id)

        if sm_ids_of_zip is None:
            # Load the topology df for the requested topology processing level
            if topology_processing_level == "raw":
                self._load_raw_topology_data()
                topology_df = self.raw_topology_df
            elif topology_processing_level == "cleaned":
                self._load_cleaned_topology_data()
                topology_df = self.cleaned_topology_df
            elif topology_processing_level == "cleaned_and_corrected":
                self._load_cleaned_and_corrected_topology_data()
                topology_df = self.cleaned_and_corrected_topology_df
            else:
                err_msg = "topology_processing_level has to be 'raw', 'cleaned' or 'cleaned_and_corrected'."
                raise ValueError(err_msg)

            # Get the secondary substation for the given ZIP code and use the
            # secondary substation method to load SM data
            substations = (
                topology_df.groupby("zip_code_secondary_substation")[self.sec_substation_col].unique().loc[zip_id]
            )
            substation_ids = [sub.split(".")[-1] for sub in substations]

            logger.info(f"Extracting SM data of all secondary substations belonging to zip code {zip_id}.")

            sm_data_list = []
            for substation_id in substation_ids:
                sm_data_list.append(
                    self.extract_raw_sm_dataset_for_secondary_substation(
                        substation_id=substation_id,
                        topology_processing_level=topology_processing_level,
                        use_existing_raw_sm_profiles=True,
                        add_current=add_current,
                        add_unbalance=add_unbalance,
                        save=False,
                        overwrite_topology_info=overwrite_topology_info,
                        overwrite_timeseries_info=overwrite_timeseries_info,
                    )
                )
        else:
            # Load datasets of all SMs associated with this zip
            if profile == "cleaned_profiles":
                sm_data_list = self.load_cleaned_dataset_for_multiple_sm(
                    meter_ids=sm_ids_of_zip,
                    add_current=add_current,
                    add_unbalance=add_unbalance,
                    overwrite=False,
                )
            elif profile == "cleaned_and_phase_corrected_profiles":
                sm_data_list = self.load_cleaned_and_phase_corrected_dataset_for_multiple_sm(
                    meter_ids=sm_ids_of_zip,
                    add_current=add_current,
                    add_unbalance=add_unbalance,
                    overwrite=False,
                )
            else:
                sm_data_list = self.load_raw_dataset_for_multiple_sm(
                    meter_ids=sm_ids_of_zip,
                    add_current=add_current,
                    add_unbalance=add_unbalance,
                    overwrite=False,
                )

        # Combine the data of all connected SMs
        zip_code_sm_data = pd.concat(sm_data_list, axis=1)

        if save:
            filepath = (
                f"{self.s3_base}/{PROCESSED_DATA_DIR}/individual_zip_code_datasets/"
                f"based_on_{topology_processing_level}_topology/{profile}/"
                f"zip_code_{zip_id}.parquet"
            )
            self.s3_connector.write_parquet(filepath, zip_code_sm_data, engine="pyarrow")
            logger.info(f"Profile '{profile}': SM data of zip code {zip_id} saved to directory {filepath}.")

        return zip_code_sm_data

    def _extract_csv_to_parquet(self, s3_path: str, file_pattern: str, csv_dir="/opt/airflow/data"):
        """
        Read CSV files from disk and convert to Parquet on S3.

        :param s3_path: Destination S3 path (e.g., 's3://my-bucket/folder/')
        :param file_pattern: CSV file pattern, e.g., 'phase_measurements_*.csv'
        :param csv_dir: Base directory where CSV files are stored
        """
        full_path = os.path.join(csv_dir, file_pattern)
        logger.info(f"Reading CSVs from {full_path}")

        # Optional: enforce dtypes if needed
        dtype = {"meter_number": "str"} if "phase" in file_pattern else None

        try:
            df = dd.read_csv(full_path, assume_missing=True, dtype=dtype)

            # Create timestamp column if needed
            if "timestamp_dst" in df.columns:
                df["timestamp"] = dd.to_datetime(df["timestamp_dst"], utc=True)
            elif "timestamp_utc" in df.columns:
                df["timestamp"] = dd.to_datetime(df["timestamp_utc"], utc=True)
            elif "timestamp" in df.columns:
                df["timestamp"] = dd.to_datetime(df["timestamp"], utc=True)
            else:
                raise ValueError("No valid timestamp column found in CSV")

            # Partition safely
            df["date"] = df["timestamp"].dt.strftime("%Y-%m-%d")

            # Repartition to avoid too many files per date
            df = df.repartition(partition_size="64MB")

            logger.info(f"Writing to Parquet at {s3_path}")
            df = df.reset_index(drop=True)
            df.to_parquet(
                s3_path,
                engine="pyarrow",
                storage_options=self.s3_connector.get_dask_storage_options(),
                partition_on=["date"],
                write_metadata_file=True,
                overwrite=True,
            )

            logger.info("Parquet files successfully written to S3")

        except Exception as e:
            logger.error(f"Failed to convert CSV to Parquet: {e}")
            raise


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
