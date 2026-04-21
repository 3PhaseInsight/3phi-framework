import json
import logging
import os
import tempfile
from concurrent.futures import as_completed
from concurrent.futures.thread import ThreadPoolExecutor
from typing import Any

import dask.dataframe as dd
import pandas as pd
import pyarrow.parquet as pq
from adlfs import AzureBlobFileSystem

from threephi_framework.object_storage.base_connector import BaseConnector
from threephi_framework.util.util import v1_get_shard_for_meter_id

MAX_WORKERS = 12


class AzureBlobConnector(BaseConnector):
    """
    Azure Blob Storage implementation of BaseConnector.

    Reads credentials from environment variables:
        AZURE_STORAGE_ACCOUNT_NAME  (required)
        AZURE_STORAGE_CONTAINER_NAME (required)
        AZURE_STORAGE_ACCOUNT_KEY   (optional — falls back to DefaultAzureCredential)

    Paths follow the az://container/blob convention used by adlfs.
    """

    def __init__(self, data_dir_path: str):
        self.account_name = os.environ["AZURE_STORAGE_ACCOUNT_NAME"]
        self.account_key = os.environ.get("AZURE_STORAGE_ACCOUNT_KEY")
        self.container_name = os.environ["AZURE_STORAGE_CONTAINER_NAME"]

        self.az_base = f"az://{self.container_name}"
        self.dataset_root_path = f"{self.az_base}/{data_dir_path}"

        # account_key=None causes adlfs to use DefaultAzureCredential automatically,
        # which covers managed identity, service principal, and az CLI login.
        self.fs = AzureBlobFileSystem(
            account_name=self.account_name,
            account_key=self.account_key,
        )

    # === Existence / discovery ===

    def exists(self, path: str) -> bool:
        return self.fs.exists(path)

    def discover_parquet_files(self, path: str) -> list[str]:
        files = [p for p in self.fs.find(path) if p.endswith(".parquet")]
        if not files:
            raise RuntimeError(f"No parquet files found under {path}")
        return files

    # === File operations ===

    def put_file(self, local_path: str, az_path: str) -> None:
        self.fs.put_file(local_path, az_path)

    def glob(self, path_pattern: str) -> list[str]:
        return self.fs.glob(path_pattern)

    def copy_file(self, src: str, dst: str) -> None:
        self.fs.copy(src, dst)

    def promote_staged_to_ready(self, staging_root: str, ready_root: str) -> list[str]:
        staged = self.discover_parquet_files(staging_root)
        if not staged:
            return []
        logging.info("Found %d file(s) to promote, e.g. %s", len(staged), staged[0])
        promoted_file_keys: list[str] = []
        with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
            futures = {}
            for staged_key in staged:
                rdy = ready_root.rstrip("/") + "/"
                ready_key = f"{rdy}{'/'.join(staged_key.rstrip('/').split('/')[-3:])}"
                logging.info("Copying from %s to %s", staged_key, ready_key)
                futures[executor.submit(self.copy_file, staged_key, ready_key)] = ready_key
            for future in as_completed(futures):
                future.result()
                promoted_file_keys.append(futures[future])
        self.clean_up(staging_root)
        return promoted_file_keys

    def clean_up(self, prefix: str) -> None:
        # reverse sort ensures files are deleted before their virtual directory entries
        for file in sorted(self.fs.find(prefix), reverse=True):
            self.fs.rm(file)

    # === DataFrame reads / writes ===

    def read_csv(self, path: str, dtype: dict[str, Any] | None = None, blocksize: int | None = None, **kwargs: Any) -> dd.DataFrame:
        return dd.read_csv(
            path,
            storage_options=self.get_dask_storage_options(),
            dtype=dtype,
            blocksize=blocksize,
            **kwargs,
        )

    def read_small_csv(self, path: str, dtype: dict[str, Any] | None = None, usecols: list[str] | None = None, **kwargs: Any) -> pd.DataFrame:
        with self.fs.open(path, "rb") as f:
            return pd.read_csv(f, dtype=dtype, usecols=usecols, **kwargs)

    def read_parquet(self, path: str) -> dd.DataFrame:
        return dd.read_parquet(path, storage_options=self.get_dask_storage_options())

    def write_parquet(self, path: str, df: Any, **kwargs: Any) -> None:
        df.to_parquet(path, storage_options=self.get_dask_storage_options(), **kwargs)

    def get_parquet_file_stats(self, path: str, dataframe: Any, ts_col: str) -> dict[str, Any]:
        with self.fs.open(path) as f:
            parquet_file = pq.ParquetFile(f)
            metadata = parquet_file.metadata
            tmin, tmax = None, None
            for i in range(metadata.num_row_groups):
                rg = metadata.row_group(i)
                for j in range(rg.num_columns):
                    col = rg.column(j)
                    if col.path_in_schema == ts_col and col.statistics:
                        s = col.statistics
                        if s.min is not None:
                            tmin = s.min if tmin is None else min(s.min, tmin)
                        if s.max is not None:
                            tmax = s.max if tmax is None else max(s.max, tmax)
            rows = metadata.num_rows
        size_bytes = self.fs.info(path)["size"]  # adlfs uses lowercase "size"; s3fs uses "Size"
        return {"ts_min": tmin, "ts_max": tmax, "row_count": rows, "size_bytes": size_bytes}

    def get_meter_data(self, meter_ids: list[str], dataset_root_path: str | None = None) -> dd.DataFrame:
        dataset_root_path = dataset_root_path or self.dataset_root_path
        logging.info("Dataset Root Path: %s", dataset_root_path)

        meter_shards = map(v1_get_shard_for_meter_id, meter_ids)
        filter_clauses = [
            [("shard", "==", shard), ("meter_number", "==", meter_id)]
            for shard, meter_id in zip(meter_shards, meter_ids, strict=False)
        ]

        return dd.read_parquet(
            dataset_root_path,
            engine="pyarrow",
            filters=filter_clauses,
            filesystem=self.fs,
        )

    # === JSON ===

    def write_json(self, path: str, data: Any) -> None:
        with self.fs.open(path, "w") as f:
            json.dump(data, f)

    def read_json(self, path: str) -> Any:
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

    def get_dask_storage_options(self) -> dict[str, Any]:
        opts: dict[str, Any] = {"account_name": self.account_name}
        if self.account_key:
            opts["account_key"] = self.account_key
        return opts

    # === Plotting ===

    def save_plot(self, path: str, fig: Any, format: str = "svg", transparent: bool = False, dpi: int = 300, overwrite: bool = True, save_kwargs: dict | None = None) -> None:
        """
        Save a matplotlib figure to Azure Blob Storage.

        Args:
            path: Destination blob path, e.g. "az://container/plots/plot1.svg"
            fig: Matplotlib figure object.
            format: "svg" or "png".
            transparent: Whether the background should be transparent.
            dpi: Resolution for PNG output.
            overwrite: Skip saving if the blob already exists and overwrite=False.
            save_kwargs: Additional kwargs forwarded to fig.savefig().
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

        kwargs: dict[str, Any] = {"format": fmt, "bbox_inches": "tight", "transparent": transparent}
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
