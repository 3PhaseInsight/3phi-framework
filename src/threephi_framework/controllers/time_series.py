import logging

import dask.dataframe as dd
import pandas as pd

from threephi_framework.dtu.timeseries_cleaner import apply_corrections, apply_phase_map, apply_quality_flags
from threephi_framework.object_storage.base_connector import BaseConnector
from threephi_framework.processing_level import ProcessingLevel


class TimeSeriesController:
    """Controller for accessing smart meter timeseries data at any processing level.

    Storage layout
    --------------
    Timeseries data is stored in four co-located artifacts under a single base path,
    not three full copies. This keeps storage overhead minimal relative to the raw
    data volume:

    ``<base>/raw/``
        The canonical, unmodified parquet dataset. Partitioned by ``dt`` and ``shard``.
        Never overwritten. All processing levels derive from this base.

    ``<base>/flags/``
        A co-partitioned quality-flag dataset with the same ``dt``/``shard`` scheme as
        ``raw/``. Contains one ``int8`` flag column per measurement column, named
        ``<col>_flag``. Values are ``QualityFlag`` members (0 = OK, non-zero = bad).
        Serves as the audit trail — records *why* a value was replaced.
        Size is typically 3–5 % of the raw data volume. Produced by *TimeseriesCleaner*.

    ``<base>/corrections/``
        A co-partitioned sparse dataset with the same ``dt``/``shard`` scheme as ``raw/``.
        Contains the same measurement columns as raw, but only rows where at least one
        value was imputed. A ``NaN`` in a corrections column means no imputed value
        exists for that column in that row. Produced by *TimeseriesCleaner*.

    ``<base>/phase_map.parquet``
        A tiny static lookup table with one row per meter. Columns: ``meter_number``,
        ``l1_maps_to``, ``l2_maps_to``, ``l3_maps_to``. Records which physical phase
        each measurement column actually corresponds to for that meter. Loaded once and
        cached in-process. Produced by *PhaseCorrector*.

    Read-time assembly
    ------------------
    When a caller requests a non-RAW level, this controller assembles the result
    in-memory from the stored artifacts — no additional parquet files are written at
    query time:

    - ``RAW``: returns ``raw/`` directly.
    - ``CLEANED``: merges ``raw/`` + ``flags/`` (nulls bad values) then merges
      ``corrections/`` (fills those NaNs with model-imputed values). The result is a
      fully continuous signal with no NaNs where imputation succeeded. Output schema is
      identical to RAW.
    - ``CLEANED_AND_CORRECTED``: applies CLEANED read path, then rearranges measurement
      columns per meter according to the phase map to correct phase misassignment.

    Args:
        connector: Storage connector rooted at the timeseries base path
                   (e.g. ``phase_measurements``). Raw, flags, corrections, and the
                   phase map are all accessed as sub-paths via the connector's
                   built-in timeseries methods.
    """

    def __init__(self, connector: BaseConnector):
        self.connector = connector
        self._phase_map_cache: pd.DataFrame | None = None

    def get_time_series_data(
        self,
        meter_ids: list[str],
        processing_level: ProcessingLevel = ProcessingLevel.RAW,
    ) -> dd.DataFrame:
        """Retrieve timeseries data for the given meter IDs at the requested level.

        - ``RAW``: reads directly from the raw parquet dataset. No transformation.
        - ``CLEANED``: nulls flagged values via flags dataset, then fills those NaNs
          with model-imputed values from the corrections dataset. The result is a
          fully continuous signal wherever imputation succeeded.
        - ``CLEANED_AND_CORRECTED``: applies CLEANED, then corrects phase
          misassignment using the static phase map. The phase map is loaded once
          and cached.

        Args:
            meter_ids: List of meter ID strings.
            processing_level: Requested processing level. Defaults to RAW.

        Returns:
            dd.DataFrame: Uncomputed Dask DataFrame at the requested level.
        """
        logging.info(f"Retrieving timeseries for {len(meter_ids)} meters at level '{processing_level}'")
        raw_ddf = self.connector.get_raw_data(meter_ids=meter_ids)

        if processing_level == ProcessingLevel.RAW:
            return raw_ddf

        flags_ddf = self.connector.get_flags_data(meter_ids=meter_ids)
        corrections_ddf = self.connector.get_corrections_data(meter_ids=meter_ids)

        nulled_ddf = apply_quality_flags(raw_ddf, flags_ddf)
        cleaned_ddf = apply_corrections(nulled_ddf, corrections_ddf)

        if processing_level == ProcessingLevel.CLEANED:
            return cleaned_ddf

        return apply_phase_map(cleaned_ddf, self._load_phase_map())

    def write_flags(self, flags_ddf: dd.DataFrame) -> None:
        """Write a quality-flag DataFrame to the flags storage location.

        The DataFrame must contain ``shard`` and ``dt`` columns for correct Hive-style
        partitioning. Overwrites any existing flags at the destination.

        Args:
            flags_ddf: Flags Dask DataFrame produced by the TimeseriesCleaner data app.
        """
        self.connector.write_flags(
            flags_ddf,
            partition_on=["shard", "dt"],
            write_index=False,
            overwrite=True,
        )

    def write_corrections(self, corrections_ddf: dd.DataFrame) -> None:
        """Write a corrections DataFrame to the corrections storage location.

        The DataFrame must contain ``shard`` and ``dt`` columns for correct Hive-style
        partitioning. Only rows where at least one measurement column was imputed should
        be present. Overwrites any existing corrections at the destination.

        Args:
            corrections_ddf: Corrections Dask DataFrame produced by the TimeseriesCleaner
                             data app. Wide format: same measurement columns as raw, with
                             NaN where no correction exists for that column/row.
        """
        self.connector.write_corrections(
            corrections_ddf,
            partition_on=["shard", "dt"],
            write_index=False,
            overwrite=True,
        )

    def _load_phase_map(self) -> pd.DataFrame:
        if self._phase_map_cache is None:
            self._phase_map_cache = self.connector.read_parquet(self.connector.phase_map_path).compute()
        return self._phase_map_cache
