import logging

import dask.dataframe as dd
import pandas as pd

from threephi_framework.dtu.timeseries_cleaner import apply_phase_map, apply_quality_flags
from threephi_framework.object_storage.base_connector import BaseConnector
from threephi_framework.processing_level import ProcessingLevel


class TimeSeriesController:
    """Controller for accessing smart meter timeseries data at any processing level.

    Storage layout
    --------------
    Timeseries data is stored in three co-located S3 datasets, not three full copies.
    This keeps storage overhead minimal relative to the raw data volume:

    ``phase_measurements/raw/``
        The canonical, unmodified parquet dataset. Partitioned by ``dt`` and ``shard``.
        Never overwritten. All processing levels derive from this base.

    ``phase_measurements/flags/``
        A co-partitioned quality-flag dataset with the same ``dt``/``shard`` scheme as
        ``raw/``. Contains one ``int8`` flag column per measurement column, named
        ``<col>_flag``. Values are ``QualityFlag`` members (0 = OK, non-zero = bad).
        Because flags are a single byte per value and compress extremely well, the size
        of this dataset is typically 3–5 % of the raw data volume — negligible even at
        hundreds of GB. Produced by the *TimeseriesCleaner* data app (not yet
        implemented).

    ``phase_measurements/phase_map.parquet``
        A tiny static lookup table with one row per meter. Columns: ``meter_number``,
        ``l1_maps_to``, ``l2_maps_to``, ``l3_maps_to``. Records which physical phase
        each measurement column actually corresponds to for that meter. Loaded once and
        cached in-process. Produced by the *PhaseCorrector* data app (not yet
        implemented).

    Read-time assembly
    ------------------
    When a caller requests a non-RAW level, this controller assembles the result
    entirely in-memory from the stored artifacts — no additional parquet files are
    written at query time:

    - ``RAW``: returns ``raw/`` directly.
    - ``CLEANED``: merges ``raw/`` + ``flags/`` on ``(meter_number, timestamp)`` and
      sets flagged measurement values to ``NaN``. Output schema is identical to RAW.
    - ``CLEANED_AND_CORRECTED``: applies cleaning, then rearranges measurement columns
      per meter according to the phase map.

    Args:
        connector: Storage connector pointing at the raw timeseries dataset root
                   (``phase_measurements/raw``).
        flags_connector: Storage connector pointing at the quality-flag dataset root
                         (``phase_measurements/flags``). Required for CLEANED and
                         CLEANED_AND_CORRECTED levels.
        phase_map_path: Full S3 path to the static phase-assignment parquet
                        (``s3://3phi/phase_measurements/phase_map.parquet``).
                        Required for the CLEANED_AND_CORRECTED level.
    """

    def __init__(
        self,
        connector: BaseConnector,
        flags_connector: BaseConnector | None = None,
        phase_map_path: str | None = None,
    ):
        self.connector = connector
        self.flags_connector = flags_connector
        self.phase_map_path = phase_map_path
        self._phase_map_cache: pd.DataFrame | None = None

    def get_time_series_data(
        self,
        meter_ids: list[str],
        processing_level: ProcessingLevel = ProcessingLevel.RAW,
    ) -> dd.DataFrame:
        """Retrieve timeseries data for the given meter IDs at the requested level.

        RAW: reads directly from the raw parquet dataset. No transformation.
        CLEANED: reads raw data and applies quality flags (flagged values → NaN).
        CLEANED_AND_CORRECTED: applies cleaning, then corrects phase misassignment
            using the static phase map. The phase map is loaded once and cached.

        Args:
            meter_ids: List of meter ID strings.
            processing_level: Requested processing level. Defaults to RAW.

        Returns:
            dd.DataFrame: Uncomputed Dask DataFrame at the requested level.

        Raises:
            ValueError: If flags_connector or phase_map_path are missing for the
                        requested level.
        """
        logging.info(f"Retrieving timeseries for {len(meter_ids)} meters at level '{processing_level}'")
        raw_ddf = self.connector.get_meter_data(meter_ids=meter_ids)

        if processing_level == ProcessingLevel.RAW:
            return raw_ddf

        if self.flags_connector is None:
            raise ValueError("flags_connector is required for CLEANED and CLEANED_AND_CORRECTED levels.")

        flags_ddf = self.flags_connector.get_meter_data(meter_ids=meter_ids)
        cleaned_ddf = apply_quality_flags(raw_ddf, flags_ddf)

        if processing_level == ProcessingLevel.CLEANED:
            return cleaned_ddf

        if self.phase_map_path is None:
            raise ValueError("phase_map_path is required for the CLEANED_AND_CORRECTED level.")

        return apply_phase_map(cleaned_ddf, self._load_phase_map())

    def _load_phase_map(self) -> pd.DataFrame:
        if self._phase_map_cache is None:
            self._phase_map_cache = self.connector.read_parquet(self.phase_map_path).compute()
        return self._phase_map_cache
