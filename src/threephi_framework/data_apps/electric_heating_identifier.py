import logging

import numpy as np
import pandas as pd
from dask import compute, delayed
from sklearn.preprocessing import StandardScaler

import threephi_framework.db.db as threephi_db
from threephi_framework.controllers.meta import MetaController
from threephi_framework.controllers.topology import TopologyController
from threephi_framework.data_apps.base import BaseDataApp
from threephi_framework.dtu.electric_heating_identifier import (
    feature_extraction,
    label_propagation,
)
from threephi_framework.processing_level import ProcessingLevel


class ElectricHeatingIdentifier(BaseDataApp):

    def __init__(self, config):
        super().__init__(config)
        self.use_dask = config.get('use_dask', False)
        self.unit_ids = config.get('unit_ids', None)
        self.sm_ids = config.get('sm_ids', None)
        self.save_plots = config.get('save_plots', True)
        self.data_dir_path = config.get('data_dir_path', None)
        self.temp_data_path = config.get('temp_data_path', "/data/temp_data.csv")
        self.spot_data_path = config.get('spot_data_path', "/data/spot_data.csv")
        self.ref_data_path = config.get('ref_data_path', "/data/ref_data.csv")
        self.ML_algorithm = config.get('ML_algorithm', "label_propagation")
        self.overwrite_existing_results = config.get('overwrite_existing_results', False)

    def _validate_config(self) -> None:

        # Ensure that either unit_ids or sm_ids are provided
        if not self.unit_ids and not self.sm_ids:
            raise ValueError("Either unit_ids or sm_ids must be provided for feature extraction.")
        elif self.unit_ids and self.sm_ids:
            logging.warning("Both unit_ids and sm_ids provided. Proceeding with sm_ids for feature extraction.")

    def _build_dtu_cfg(self) -> dict:
        """Build the per-task ``cfg`` dict consumed by ``dtu.meter_evaluation``."""
        cfg = dict(self.config)
        cfg.setdefault("results_dir", "electric_heating_identifier")
        cfg.setdefault("save_plots", self.save_plots)
        cfg.setdefault("sm_processing_level", ProcessingLevel.RAW)
        return cfg

    def identify_EH_applicances(self) -> None:
        """Evaluate the configured smart meters; results are persisted to the meta DB."""
        self._validate_config()
        topology_controller = TopologyController(threephi_db.new_session)
        meta_controller = MetaController(threephi_db.new_session)

        # Control that data files are present in S3, if not upload them from local path
        for file in [self.temp_data_path, self.spot_data_path, self.ref_data_path]:
            if not self.data_extractor.s3_connector.exists(self.data_extractor.s3_base + file):
                self.data_extractor.s3_connector.put_file(f"/opt/airflow/{file}", self.data_extractor.s3_base + file)
                logging.info(f"File uploaded to {self.data_extractor.s3_base + file}")


        # Generate list of smart meters based on config. If either sm_ids or unit_ids is 'all',
        # then process all smart meters with data
        if self.sm_ids == ["all"] or self.unit_ids == ["all"]:
            logging.info("Processing all smart meters with data in the dataset.")
            timeseries_dict = meta_controller.get_time_series_meta_info()
            self.sm_ids = timeseries_dict.get("id_list_of_sms_with_data", [])
        elif not self.sm_ids:
            self.sm_ids = []
            logging.info(f"Processing all smart meters in units {', '.join(self.unit_ids)}.")
            for unit_id in self.unit_ids:
                meters = topology_controller.get_meters_for_substation(int(unit_id))
                self.sm_ids.extend([m["id"] for m in meters])
        else:
            logging.info(f"Processing specified smart meters: {', '.join(self.sm_ids)}.")

        # Ensure all sm_ids are strings for consistent processing
        self.sm_ids = [str(sm_id) for sm_id in self.sm_ids]


        # Check if the ML results already exist for the smart meters
        logging.info(f"Checking if machine learning results already exist {len(self.sm_ids)} for smart meters.")
        self.sm_ids = [
            sm_id for sm_id in self.sm_ids
            if not self.meta_controller.is_workflow_completed(f"{self.ML_algorithm}_sm_{sm_id}")
            or self.overwrite_existing_results
        ]
        logging.info(
            f"Identifying electric heating appliances for {len(self.sm_ids)} "
            f"smart meters remaining after filtering."
        )

        if not self.sm_ids:
            logging.info("All smart meters already processed. Skipping electric heating identification.")
            return

        # Check if the feature results already exist for the smart meters
        logging.info(f"Checking if feature results already exist for {len(self.sm_ids)} smart meters.")
        feature_sm_ids = [
            sm_id for sm_id in self.sm_ids
            if not self.meta_controller.is_workflow_completed(f"feature_engineering_sm_{sm_id}")
            or self.overwrite_existing_results
        ]
        # Build config dict
        cfg = self._build_dtu_cfg()

        # Extract features if any smart meters remain after filtering
        if feature_sm_ids and self.use_dask:
            n_workers = self.dask_settings.get("n_workers", 1) or 1
            chunks = np.array_split(feature_sm_ids, min(len(feature_sm_ids), n_workers))
            tasks = [delayed(feature_extraction)(sm_ids=list(chunk), cfg=cfg) for chunk in chunks]
            compute(*tasks)
        elif feature_sm_ids:
            feature_extraction(sm_ids=feature_sm_ids, cfg=cfg)

        # Process ML algorithm if any smart meters remain after filtering
        if self.sm_ids:
            feature_results = {}
            for sm_id in self.sm_ids:
                for phase_str in ["L1", "L2", "L3"]:
                    meta_results = self.meta_controller.query_run_results(
                        source="Electric Heating Identifier",
                        meter_id=int(sm_id),
                        phase=phase_str,
                        label_type="Feature Engineering",
                    )
                    if meta_results:
                        features = meta_results[0].result
                        feature_results[(sm_id, phase_str)] = features

            # Make feature_results a pandas dataframe
            feature_results = pd.DataFrame.from_dict(feature_results, orient='index')

            # Scale features using StandardScaler
            scaler = StandardScaler()
            scaled_features_np = scaler.fit_transform(feature_results)
            feature_results = pd.DataFrame(
                scaled_features_np, index=feature_results.index, columns=feature_results.columns
            )

            # Run specific ML algorithm to identify electric heating appliances based on extracted features
            # Suggested: *Label propagation* due to unfinished ground truths
            if self.ML_algorithm == "label_propagation":
                label_propagation(feature_results=feature_results, sm_ids=self.sm_ids, cfg=cfg)
            # elif self.ML_algorithm == "hierarchical_clustering":
            #     hierarchical_clustering(sm_ids=self.sm_ids, cfg=cfg)
            # elif self.ML_algorithm == "logistic_regression":
            #     logistic_regression(sm_ids=self.sm_ids, cfg=cfg)



    def run(self):
        self.identify_EH_applicances()

if __name__ == "__main__":
    config = {
        "use_dask": True,
        "dask": {"local": True, "n_workers": 4},
        "data_dir_path": "phase_measurements/raw",
        "unit_ids": None,
        "sm_ids": [
            '23405',
            '121184',
            '197256',
            '382729',
            '440937',
            '445107',
            '566340',
            '594794',
            '759234',
            '790516',
            '835841',
            '978146',
        ],
        "save_plots": False,
        "overwrite_existing_results": False,
        "topology_processing_level": "raw",
        "save_results": False,
        "temp_data_path": "/data/temp_data.csv",
        "spot_data_path": "/data/spot_data.csv",
        "ref_data_path": "/data/ref_data.csv",
        "sm_processing_level": "raw",
        "results_dir": "electric_heating_identifier",
        "EH_threshold": 0.95,

        # ML_algorithm can be set to either "label_propagation", "logistic_regression", or "hierarchical_clustering"
        "ML_algorithm": "label_propagation",

        # Label propagation specific config
        # Kernal: Can be "knn" or "rbf"
        # Label_threshold: Threshold for label propagation step
        # Gamma parameter for label propagation
        # alpha: Alpha parameter for label spreading
        "label_propagation": {
            "kernel": "knn",
            "label_threshold": 0.5,
            "gamma": 10,
            "alpha": 0.2,}}

    with ElectricHeatingIdentifier(config) as app:
        app.run()
