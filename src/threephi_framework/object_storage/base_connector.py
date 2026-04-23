from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

import dask.dataframe as dd
import pandas as pd


class BaseConnector(ABC):
    @abstractmethod
    def exists(self, path: str) -> bool:
        """
        Check if an object exists in the storage.

        Args:
            path (str):
                Path to the object.

        Returns:
            bool:
                True if object exists, False otherwise.
        """
        pass

    @abstractmethod
    def discover_parquet_files(self, path: str) -> list[str]:
        """
        Discover parquet files under a given prefix/path.

        Args:
            path (str):
                Root path/prefix to search under.

        Returns:
            list[str]:
                List of fully-qualified parquet file paths.

        Raises:
            RuntimeError:
                If no parquet files are found under the given path.
        """
        pass

    @abstractmethod
    def copy_file(self, src: str, dst: str) -> None:
        """
        Copy a file from src to dst.

        Args:
            src (str):
                Source object path.
            dst (str):
                Destination object path.

        Returns:
            None

        Raises:
            FileNotFoundError:
                If either src or dst cannot be accessed during verification (implementation-dependent).
        """
        pass

    @abstractmethod
    def promote_staged_to_ready(self, staging_root: str, ready_root: str) -> list[str]:
        """
        Promote parquet files from a staging prefix to a ready prefix.

        Typical behavior:
        - Discover parquet files under staging_root
        - Copy them to a derived key under ready_root (often preserving a suffix of the path)
        - Clean up staging_root
        - Return the list of promoted destination keys

        Args:
            staging_root (str):
                Root/prefix where staged files exist.
            ready_root (str):
                Root/prefix to which staged files should be promoted.

        Returns:
            list[str]:
                List of promoted (destination) file keys/paths.
        """
        pass

    @abstractmethod
    def clean_up(self, prefix: str) -> None:
        """
        Remove all objects under a given prefix/path.

        Args:
            prefix (str):
                Root prefix/path to delete recursively.

        Returns:
            None
        """
        pass

    @abstractmethod
    def read_csv(
        self,
        path: str,
        dtype: dict[str, Any] | None = None,
        blocksize: int | None = None,
        **kwargs: Any,
    ) -> dd.DataFrame:
        """
        Read a CSV as a dask dataframe.

        Args:
            path (str):
                Path to the csv file(s).
            dtype (dict[str, Any] | None):
                Optional dtype mapping for columns.
            blocksize (int | None):
                Blocksize to use for dask read_csv.
            **kwargs (Any):
                Passed through to dask.dataframe.read_csv.

        Returns:
            dd.DataFrame:
                A dask dataframe.
        """
        pass

    @abstractmethod
    def read_small_csv(
        self,
        path: str,
        dtype: dict[str, Any] | None = None,
        usecols: list[str] | None = None,
        **kwargs: Any,
    ) -> pd.DataFrame:
        """
        Read a CSV as a pandas dataframe (intended for small files only).

        Args:
            path (str):
                Path to the csv file.
            dtype (dict[str, Any] | None):
                Optional dtype mapping for columns.
            usecols (list[str] | None):
                Columns to read.
            **kwargs (Any):
                Passed through to pandas.read_csv.

        Returns:
            pd.DataFrame:
                A pandas dataframe.
        """
        pass

    @abstractmethod
    def read_parquet(self, path: str) -> dd.DataFrame:
        """
        Read parquet dataset/file(s) as a dask dataframe.

        Args:
            path (str):
                Path to parquet dataset or file(s).

        Returns:
            dd.DataFrame:
                A dask dataframe.
        """
        pass

    @abstractmethod
    def write_parquet(self, path: str, df: Any, **kwargs: Any) -> None:
        """
        Write a dataframe to parquet.

        Args:
            path (str):
                Destination path.
            df (Any):
                Dataframe-like object supporting .to_parquet (dask or pandas).
            **kwargs (Any):
                Passed through to df.to_parquet.

        Returns:
            None
        """
        pass

    @abstractmethod
    def get_parquet_file_stats(self, path: str, dataframe: Any, ts_col: str) -> dict[str, Any]:
        """
        Extract min/max timestamps, row count, and file size for a parquet file.

        Args:
            path (str):
                Path to parquet file.
            dataframe (Any):
                The file as a (dask) dataframe (may be unused by some implementations).
            ts_col (str):
                Timestamp column name used to compute min/max from parquet statistics.

        Returns:
            dict[str, Any]:
                Dictionary containing file stats, e.g.:
                {"ts_min": ..., "ts_max": ..., "row_count": int, "size_bytes": int}
        """
        pass

    @abstractmethod
    def get_meter_data(self, meter_ids: list[str], dataset_root_path: str | None = None) -> dd.DataFrame:
        """
        Get data for a list of meters.

        Args:
            meter_ids (list[str]):
                Meter ID string, e.g. "100025".
            dataset_root_path (str | None):
                Root path of the dataset, e.g. "s3://bucket/prefix". If None, implementations
                may fall back to a path configured at construction time.

        Returns:
            dd.DataFrame:
                Filtered dask dataframe.
        """
        pass

    @abstractmethod
    def write_json(self, path: str, data: Any) -> None:
        """
        Write JSON-serializable data to the given path.

        Args:
            path (str):
                Destination path.
            data (Any):
                JSON-serializable object.

        Returns:
            None
        """
        pass

    @abstractmethod
    def read_json(self, path: str) -> Any:
        """
        Read JSON data from the given path.

        Args:
            path (str):
                Source path.

        Returns:
            Any:
                Parsed JSON object, or None if not found / unreadable (implementation-dependent).
        """
        pass

    @abstractmethod
    def get_dask_storage_options(self) -> dict[str, Any]:
        """
        Return storage options suitable for dask (e.g., s3fs credentials / endpoint config).

        Returns:
            dict[str, Any]:
                Storage options dict passed to dask APIs as `storage_options=...`.
        """
        pass
