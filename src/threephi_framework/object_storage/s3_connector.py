import json
import logging
import os
import tempfile
from concurrent.futures import as_completed
from concurrent.futures.thread import ThreadPoolExecutor

import dask.dataframe as dd
import pandas as pd
import pyarrow.parquet as pq
import s3fs

from threephi_framework.object_storage.base_connector import BaseConnector
from threephi_framework.util.util import v1_get_shard_for_meter_id

MAX_WORKERS = 12


class S3Connector(BaseConnector):
    def __init__(self, data_dir_path: str):
        self.endpoint_url = os.environ["S3_ENDPOINT_URL"]
        self.access_key = os.environ["S3_ACCESS_KEY"]
        self.secret_key = os.environ["S3_SECRET_KEY"]
        self.s3_base = "s3://3phi"
        self.dataset_root_path = f"{self.s3_base}/{data_dir_path}"

        self.fs = s3fs.S3FileSystem(
            key=self.access_key,
            secret=self.secret_key,
            client_kwargs={"endpoint_url": self.endpoint_url},
            config_kwargs={"s3": {"addressing_style": "path"}},
        )

    def exists(self, path):
        return self.fs.exists(path)

    def put_file(self, local_path, s3_path):
        self.fs.put_file(local_path, s3_path)

    def glob(self, path_pattern):
        return self.fs.glob(path_pattern)

    def discover_parquet_files(self, path):
        files = [p for p in self.fs.find(path) if p.endswith(".parquet")]
        return files

    def copy_file(self, src: str, dst: str):
        self.fs.copy(src, dst)

    def promote_staged_to_ready(self, staging_root: str, ready_root: str) -> list[str]:
        staged: list[str] = self.discover_parquet_files(staging_root)
        if not staged:
            logging.info("Found no files to promote under %s", staging_root)
            return []
        logging.info("Found %d file(s) to promote, e.g. %s", len(staged), staged[0])
        promoted_file_keys: list[str] = []
        rdy = ready_root.rstrip("/") + "/"
        # copy in multiple threads to minimize IO/Network Bottlenecks
        with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
            futures = {}
            for staged_key in staged:
                ready_key = f"{rdy}{'/'.join(staged_key.rstrip('/').split('/')[-3:])}"
                logging.info("Copying from %s to %s", staged_key, ready_key)
                futures[executor.submit(self.copy_file, staged_key, ready_key)] = ready_key
            for future in as_completed(futures):
                future.result()
                promoted_file_keys.append(futures[future])
        self.clean_up(staging_root)
        return promoted_file_keys

    def clean_up(self, prefix: str):
        # sorting ensures we first attempt to delete files, then dirs
        for file in sorted(self.fs.find(prefix), reverse=True):
            self.fs.rm(file)

    def read_csv(self, path, dtype=None, blocksize=None, **kwargs) -> dd.DataFrame:
        """
        Reads csv file using dask.
        :return: A dask dataframe
        """
        return dd.read_csv(
            path,
            storage_options=self.get_dask_storage_options(),
            dtype=dtype,
            blocksize=blocksize,
            **kwargs,
        )

    def read_small_csv(self, path, dtype=None, usecols=None, **kwargs) -> pd.DataFrame:
        """
        Reads csv file using pandas, use only for small csv files, no parallelization.
        :return: A pandas dataframe
        """
        with self.fs.open(path, "rb") as f:
            return pd.read_csv(f, dtype=dtype, usecols=usecols, **kwargs)

    def read_parquet(self, path) -> dd.DataFrame:
        return dd.read_parquet(path, storage_options=self.get_dask_storage_options())

    def write_parquet(self, path, df, **kwargs):
        df.to_parquet(path, storage_options=self.get_dask_storage_options(), **kwargs)

    def get_parquet_file_stats(self, path, dataframe, ts_col) -> dict:
        """
        Utility method to extract min & max timestamps, row count and file size stats.
        :param path: path to parquet file
        :param dataframe: the file as a (dask) dataframe
        :param ts_col: name of timestamp column
        :return: Dictionary containing file stats
        """
        with self.fs.open(path) as f:
            parquet_file = pq.ParquetFile(f)
            metadata = parquet_file.metadata
            tmin, tmax = None, None
            for i in range(metadata.num_row_groups):
                rg = metadata.row_group(i)
                for j in range(rg.num_columns):
                    col = rg.column(j)
                    # need timestamp col to determine min and max timestamp stats
                    if col.path_in_schema == ts_col and col.statistics:
                        s = col.statistics
                        if s.min is not None:
                            tmin = s.min if tmin is None else min(s.min, tmin)
                        if s.max is not None:
                            tmax = s.max if tmax is None else max(s.max, tmax)
            rows = metadata.num_rows
        size_bytes = self.fs.info(path)["Size"]

        return {"ts_min": tmin, "ts_max": tmax, "row_count": rows, "size_bytes": size_bytes}

    def get_meter_data(self, meter_ids: list[str], dataset_root_path: str | None = None) -> dd.DataFrame:
        """
        Function to get data for multiple meters
        :param meter_ids: Meter ID string, e.g.: "100025"
        :param dataset_root_path: Root path of the dataset; defaults to self.dataset_root_path if not provided
        :return: Dask Dataframe containing collected meter data
        """
        dataset_root_path = dataset_root_path or self.dataset_root_path
        logging.info(f"Dataset Root Path: {dataset_root_path}")

        meter_shards = map(v1_get_shard_for_meter_id, meter_ids)
        meter_shard_id_pairs = zip(meter_shards, meter_ids, strict=False)

        filter_clauses = [
            [("shard", "==", shard), ("meter_number", "==", meter_id)] for shard, meter_id in meter_shard_id_pairs
        ]

        ddf = dd.read_parquet(
            dataset_root_path,
            engine="pyarrow",  # pyarrow should deal well with hive style partitions
            filters=filter_clauses,
            filesystem=self.fs,
        )
        return ddf

    # === JSON support ===
    def write_json(self, path, data):
        with self.fs.open(path, "w") as f:
            json.dump(data, f)

    def read_json(self, path):
        try:
            with self.fs.open(path, "r") as f:
                return json.load(f)
        except FileNotFoundError:
            logging.warning("File not found: %s", path)
        except json.JSONDecodeError:
            logging.warning("Failed to decode JSON from: %s", path)
        except Exception as e:
            logging.error("Error reading %s: %s", path, e)
        return None

    # === Dask storage options ===
    def get_dask_storage_options(self):
        return {
            "key": self.access_key,
            "secret": self.secret_key,
            "client_kwargs": {"endpoint_url": self.endpoint_url},
        }

    # === Plotting ===
    def save_plot(self, path, fig, format="svg", transparent=False, dpi=300, overwrite=True, save_kwargs=None):
        """
        Function to save plots to the S3 bucket
        :param path: Path to save the plot, e.g.: "s3://3phi/plots/plot1.svg"
        :param fig: Matplotlib figure object to save
        :param format: Format to save the plot in, default is 'svg'. Options are 'svg', 'png'
        :param transparent: Whether the background of the plot should be transparent, default is False
        :param dpi: Resolution of the saved plot, default is 300
        :param overwrite: Whether to overwrite the plot if it already exists, default is True
        :param save_kwargs: Additional keyword arguments to pass to fig.savefig()
        """
        fmt = format.lower()
        if fmt not in {"svg", "png"}:
            raise ValueError(f"Unsupported format: {fmt}")

        if not isinstance(transparent, bool):
            raise ValueError("transparent must be a boolean value.")

        if not path.endswith(f".{fmt}"):
            raise ValueError(f"path must end with .{fmt}")

        if not overwrite and self.fs.exists(path):
            logging.info("Plot %s already exists and overwrite=False. Skipping.", path)
            return

        kwargs = {"format": fmt, "bbox_inches": "tight", "transparent": transparent}
        if fmt == "png":
            kwargs["dpi"] = dpi

        if save_kwargs is not None:
            if not isinstance(save_kwargs, dict):
                raise ValueError("save_kwargs must be a dictionary.")
            kwargs.update(save_kwargs)

        with tempfile.TemporaryDirectory() as tmpdir:
            local_path = os.path.join(tmpdir, os.path.basename(path))
            fig.savefig(local_path, **kwargs)
            self.put_file(local_path, path)

        logging.info("Plot saved to %s.", path)
