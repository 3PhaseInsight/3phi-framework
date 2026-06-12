import hashlib
import json
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
from threephi_framework.object_storage.base_connector import BaseConnector
from threephi_framework.object_storage.factory import create_connector


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

    Object storage
    --------------
    Every data app works against a single :class:`BaseConnector`. It is resolved
    in this order:

    1. the ``connector`` constructor argument (dependency injection — pass any
       ``BaseConnector`` implementation to swap the storage backend in code)
    2. built via :func:`~threephi_framework.object_storage.factory.create_connector`
       from ``config["object_storage_backend"]`` ("s3" or "azure")
    3. the ``OBJECT_STORAGE_BACKEND`` environment variable, then the "s3" default

    The connector is rooted at ``config["data_dir_path"]`` (default
    ``phase_measurements/raw``) and shared with the DataExtractor and the
    TimeSeriesController.

    Workflow gating
    ---------------
    Data apps whose work is a whole-dataset step (ingestion, cleaning) record
    completion in ``meta.workflow_states`` so a re-run — e.g. a DAG chaining
    several apps — can skip work that already happened. Subclasses opt in by
    setting two class attributes:

    - ``WORKFLOW``: the base workflow name (e.g. ``"topology_ingestion"``)
    - ``IDENTITY_KEYS``: the config keys that affect the app's *outputs*
      (paths, thresholds — not dask or plotting settings)

    :meth:`workflow_name` appends a stable hash of the identity values, so the
    same app run with a different relevant config counts as a different
    workflow. ``config["override"] = True`` forces a re-run regardless.

    Apps that produce per-entity results (classifier output per meter, phase
    mapping per transformer) should derive completion from their result tables
    instead of this mechanism.
    """

    #: Base workflow name for completion gating; None disables the mechanism.
    WORKFLOW: str | None = None
    #: Config keys whose values define the workflow identity (affect outputs).
    IDENTITY_KEYS: tuple[str, ...] = ()

    def __init__(self, config, connector: BaseConnector | None = None):
        self.config = config
        self.result_name = self.config.get("result_name", str(int(time())))
        self.dask_settings = self.config.get("dask", {"host": "dask-scheduler", "port": "8786"})

        self.data_dir_path = self.config.get("data_dir_path", "phase_measurements/raw")
        self.connector = connector or create_connector(
            self.data_dir_path, backend=self.config.get("object_storage_backend")
        )
        self.data_extractor = DataExtractor(phase_measurements_dir=self.data_dir_path, connector=self.connector)
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
        # The level-aware controller is rooted one level above the raw dataset
        # (raw/, flags/, corrections/ and phase_map.parquet live underneath it)
        ts_base = self.data_dir_path.removesuffix("/raw")
        return TimeSeriesController(self.connector.with_data_dir(ts_base))

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

    # --- Workflow gating (see class docstring) --- #

    def workflow_identity(self) -> dict:
        """The subset of the config that defines this app's workflow identity."""
        return {key: self.config.get(key) for key in self.IDENTITY_KEYS}

    def workflow_name(self) -> str:
        """Config-scoped workflow name: ``<WORKFLOW>:<hash of identity>``.

        Apps without IDENTITY_KEYS keep the bare ``WORKFLOW`` name (and stay
        compatible with completions recorded before config scoping existed).
        """
        if self.WORKFLOW is None:
            raise NotImplementedError(f"{self.__class__.__name__} does not define a WORKFLOW name.")
        identity = self.workflow_identity()
        if not identity:
            return self.WORKFLOW
        digest = hashlib.sha256(json.dumps(identity, sort_keys=True, default=str).encode()).hexdigest()[:12]
        return f"{self.WORKFLOW}:{digest}"

    def workflow_completed(self) -> bool:
        """True if this workflow (for this config identity) already completed.

        Completions recorded under the bare ``WORKFLOW`` name (before config
        scoping) are honored as well, so existing deployments do not re-run
        non-idempotent steps after upgrading.
        """
        name = self.workflow_name()
        if self.meta_controller.is_workflow_completed(name):
            return True
        return name != self.WORKFLOW and self.meta_controller.is_workflow_completed(self.WORKFLOW)

    def mark_workflow_completed(self) -> None:
        """Record completion, storing the identity JSON as the description."""
        name = self.workflow_name()
        description = json.dumps(self.workflow_identity(), sort_keys=True, default=str)
        self.meta_controller.start_workflow(name, description=description)
        self.meta_controller.complete_workflow(name)

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
