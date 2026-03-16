import logging

import dask.dataframe as dd

from threephi_framework.object_storage.base_connector import BaseConnector


class TimeSeriesController:
    """
    Controller responsible for access to time series data for smart meters.

    The controller provides:
    - Methods to retrieve time series data for specific meters.

    Args:
        connector: A storage connector that implements timeseries data interactions.
    """

    def __init__(self, connector: BaseConnector):
        self.connector = connector

    def get_time_series_data(self, meter_ids: list[str]) -> dd.DataFrame:
        """
        Collect time series data for given meter ids.

        Args:
            meter_ids: The meter IDs to collect time series data for.

        Returns:
            dd.DataFrame: A Dask DataFrame containing the uncomputed task graph.
        """
        logging.info(f"Retrieving time series data for {len(meter_ids)} meters")
        data = self.connector.get_meter_data(meter_ids=meter_ids)
        logging.info(f"Collected {len(data)} records")
        return data
