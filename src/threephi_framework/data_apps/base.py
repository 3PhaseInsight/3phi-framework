import logging
from abc import abstractmethod
from functools import cached_property
from time import time

from dask.distributed import Client

import threephi_framework.db.db as threephi_db
from threephi_framework.controllers.ingestion import IngestionController
from threephi_framework.controllers.meta import MetaController
from threephi_framework.controllers.time_series import TimeSeriesController
from threephi_framework.controllers.topology import TopologyController
from threephi_framework.data_extractor.data_extractor import DataExtractor


def _set_up_logger():
    if not logging.getLogger().hasHandlers():
        logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


class BaseDataApp:
    """
    Data App Base Class. This class automatically sets up a connection to a Dask Cluster as a Context Manager.

    Usage::

        with DataApp(config=config) as app:
            app.run()

    This ensures that any Connection to the Dask Cluster is properly closed down again.
    """

    def __init__(self, config):
        self.config = config
        self.result_name = self.config.get("result_name", str(int(time())))
        self.dask_settings = self.config.get("dask", {"host": "dask-scheduler", "port": "8786"})
        self.data_extractor = DataExtractor()
        self.dask_client: Client

        _set_up_logger()

    @cached_property
    def topology_controller(self):
        return TopologyController(threephi_db.new_session)

    @cached_property
    def meta_controller(self):
        return MetaController(threephi_db.new_session)

    @cached_property
    def ingestion_controller(self):
        return IngestionController(threephi_db.new_session)

    @cached_property
    def time_series_controller(self):
        return TimeSeriesController(threephi_db.new_session)

    def __enter__(self):
        self.init_dask()
        return self

    def __exit__(self, exc_type, exc_value, exc_traceback):
        if exc_type is not None:
            logging.error(
                msg=f"Error during execution of {self.__class__.__name__}",
                exc_info=(exc_type, exc_value, exc_traceback),
            )
        self.close_dask()

    # Method to update config settings via the method arguments
    def _update_config(self, args):
        for arg_name, arg_value in args:
            if arg_name != "self" and arg_value is not None:
                setattr(self, arg_name, arg_value)

    @abstractmethod
    def run(self):
        pass

    def init_dask(self):
        if self.dask_settings.get("local", False):
            logging.info("Setting up local Dask Cluster")
            workers = self.dask_settings.get("n_workers", 2)
            self.dask_client = Client(n_workers=workers)
        else:
            logging.info(
                f"Setting up Dask with host: {self.dask_settings['host']} and port: {self.dask_settings['port']}"
            )
            self.dask_client = Client(f"tcp://{self.dask_settings['host']}:{self.dask_settings['port']}")

    def close_dask(self):
        self.dask_client.close()


if __name__ == "__main__":
    data_app = BaseDataApp({})
