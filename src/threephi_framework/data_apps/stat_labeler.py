import logging
from dataclasses import dataclass
from datetime import datetime

import numpy as np
from dask import compute, delayed

from threephi_framework.data_apps.base import BaseDataApp
from threephi_framework.data_apps.base_config import BaseConfig
from threephi_framework.dtu.stat_labeler import label_meters


class StatLabeler(BaseDataApp):
    """
    Statistical Labeling Data App.

    Evaluates each smart meter's time series data to produce a per-phase per-meter
    electric heating appliances labeling, using ABC and BIC-based model comparison, and writes the
    result to the meta DB via :class:`~threephi_framework.controllers.meta.MetaController`.
    *Disclaimer*: This data app is computationally heavy, and should be used to produce per-phase ground 
    truth labels which can be used for training a more efficient ML-based classifier (see: ./data_apps/electric_heating_identifier). 

    The per-phase per-meter evaluation is delegated to
    :func:`threephi_framework.dtu.stat_labeler.label_meters`.

    The following config parameters are supported:

    - ``dask`` (dict): Dask cluster settings (see :class:`BaseDataApp`).
    - ``data_dir_path`` (str): Root path of the raw timeseries dataset in
      object storage. Passed through to ``meter_evaluation``.
    - ``use_dask`` (bool): If True, distribute meter evaluation across Dask
      workers using ``dask.delayed``. Defaults to ``False``.
    - ``sm_ids`` (str | list[str]): ``"All"`` to classify every meter that has
      data, or an explicit list of meter ID strings.
    - ``save_plots`` (bool): Persist per-meter timeseries plots to object storage.
    - ``results_dir`` (str): Object storage path for plots. Defaults to
      ``s3://3phi/stat_labeler``.

    Usage::

        with StatLabeler(config) as app:
            app.run()
    """

    def __init__(self, config):
        super().__init__(config)
        self.use_dask = config.get('use_dask', False)
        self.sm_ids = config.get('sm_ids', None)
        self.save_plots = config.get('save_plots', True)
        self.temp_data_path = config.get('temp_data_path', "/data/temp_data.csv")
        self.process_only_sm_with_hp = config.get('process_only_sm_with_hp', False)
        self.thresholds = config.get('thresholds', {
            "max_bins": 12,
            "min_bins": 1,
            "confidence_threshold": 0.8,
        })
        self.include_mae = config.get('include_mae', True)
        self.add_meta_results = config.get('add_meta_results', True)
        self.overwrite_existing_results = config.get('overwrite_existing_results', False)

    def _validate_config(self) -> None:
        # Ensure that sm_ids are provided
        if not self.sm_ids:
            raise ValueError("sm_ids must be provided for feature extraction.")
        
        # If sm_ids is not a list, create a list with the single sm_id
        elif isinstance(self.sm_ids, int) or isinstance(self.sm_ids, str):
            logging.warning("sm_ids provided as a single value. Converting to a list for processing.")
            self.sm_ids = [self.sm_ids]
        
        # If sm_ids is a list of integers, convert them to strings
        if isinstance(self.sm_ids, list) and all(isinstance(sm_id, int) for sm_id in self.sm_ids):
            logging.warning("sm_ids provided as integers. Converting to strings for processing.")
            self.sm_ids = [str(sm_id) for sm_id in self.sm_ids]

    
    def _build_dtu_cfg(self) -> dict:
        """Build the per-task ``cfg`` dict consumed by ``dtu.meter_evaluation``."""
        cfg = dict(self.config)
        cfg.setdefault("results_dir", f"{self.data_extractor.s3_base}/stat_labeler")
        cfg.setdefault("data_dir_path", "phase_measurements/raw")
        cfg.setdefault("save_plots", self.save_plots)
        return cfg

    # Method to perform statistical labeling of heat pumps in smart meter data
    def stat_label_sm(self):

        # Put temperature file in S3 if not already there
        if not self.data_extractor.s3_connector.exists(self.data_extractor.s3_base + self.temp_data_path):
            self.data_extractor.s3_connector.put_file("/opt/airflow" + self.temp_data_path, self.data_extractor.s3_base + self.temp_data_path)
            logging.info(f"File uploaded to {self.data_extractor.s3_base + self.temp_data_path}")

        # Generate list of smart meters
        if self.sm_ids == "All":
            timeseries_dict = self.meta_controller.get_time_series_meta_info()
            self.sm_ids = timeseries_dict.get("id_list_of_sms_with_data", [])
        else:
            self.sm_ids = self.sm_ids

        # Validate that smart meter IDs are in the correct format and convert to strings if necessary
        self._validate_config()

        # Check in the meta.workflow_states if the result exists. If so, skip processing for that smart meter if overwrite_existing_results is False
        logging.info(f"Checking if results already exist for smart meters.")
        filtered_sm_ids = []
        for sm_id in self.sm_ids:
            workflow_completed = self.meta_controller.is_workflow_completed(f"stat_labeling_sm_{sm_id}")
            if workflow_completed and not self.overwrite_existing_results:
                logging.info(f"Smart meter {sm_id} already exist. Skipping processing.")
            else:
                filtered_sm_ids.append(sm_id)

        if not filtered_sm_ids:
            logging.info("All smart meters already processed. Skipping Dask computation.")
            return
        
        self.sm_ids = filtered_sm_ids
        
        logging.info(f"Smart meters to be processed after checking existing results: {self.sm_ids}")
        
        cfg = self._build_dtu_cfg()

        if self.use_dask:
            sm_with_hp = self.topology_controller.get_meters(has_heat_pump=True)
            n_workers = self.dask_settings.get("n_workers", 1) or 1
            sm_id_chunks = np.array_split(self.sm_ids, min(len(self.sm_ids), n_workers))
            logging.info(f"Processing {len(self.sm_ids)} smart meters in {len(sm_id_chunks)} chunks across {n_workers} workers.")
            delayed_tasks = [delayed(label_meters)(sm_ids_chunk, sm_with_hp, cfg) for sm_ids_chunk in sm_id_chunks]
            compute(*delayed_tasks)
        else:
            label_meters(self.sm_ids, self.topology_controller.get_meters(has_heat_pump=True), cfg)
        

    def run(self):
        self.stat_label_sm()


if __name__ == "__main__":
    config = {
        "use_dask": True,
        "dask": {"local": True, "n_workers": 4},
        "sm_ids": ['23405', '121184', '197256', '382729', '440937', '445107', '566340', '594794', '759234', '790516', '835841', '978146'], # Can either be a list of sm_ids or "All" to process all smart meters with data
        "overwrite_existing_results": False,
        "process_only_sm_with_hp": False,
        "save_plots": False,
        "save_meta_results": True,
        "data_dir_path": "phase_measurements/raw",
        "thresholds": {
            "max_bins": 12,
            "min_bins": 1,
            "confidence_threshold": 0.8,
        },
        "results_dir": "s3://3phi/stat_labeler",
        "temp_data_path": "/data/temp_data.csv",
        "add_meta_results": True,
        "include_mae": True,
    }

    with StatLabeler(config) as app:
        app.run()
