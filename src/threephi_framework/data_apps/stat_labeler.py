import logging
from dataclasses import dataclass
from datetime import datetime

import numpy as np
from dask import compute, delayed

from threephi_framework.data_apps.base import BaseDataApp
from threephi_framework.data_apps.base_config import BaseConfig


@dataclass(frozen=True)
class StatLabelerConfig(BaseConfig):
    dask: dict
    sm_ids: list[str]
    data_dir_path: str
    process_only_sm_with_hp: bool
    overwrite_existing_results: bool
    label_summerhouse: bool
    filter_data: bool
    save_plots: bool
    use_ANOVA: bool
    thresholds: dict
    results_dir: str
    weather_file: str
    weather_file_local: str
    save_meta_results: bool
    # Optional backend selector for workers ("s3" / "azure"); None falls back to
    # the OBJECT_STORAGE_BACKEND env var, then "s3".
    object_storage_backend: str | None = None


class StatLabeler(BaseDataApp):
    # Initialization method which is automatically called when creating an instance of this class
    def __init__(self, config, connector=None):
        # Set up the config settings from the parent class
        # set up batch, profile_processing_level, result_name, dask client, logger, and data extractor
        super().__init__(config, connector=connector)

        self.config = StatLabelerConfig(**config)

    # Method to check for previous results in earlier results files
    def _check_previous_results(self, results_path, sm_id, heat_pump_returns):
        sm_str = str(sm_id)
        for prev_results in results_path:
            try:
                results_data = self.data_extractor.s3_connector.read_json(prev_results)
                if sm_str in results_data:
                    heat_pump_returns[sm_id] = results_data[sm_str]
                    logging.info(
                        f"Label results of {sm_id} already exists in earlier results. Loading existing results."
                    )
                    return heat_pump_returns, False
            except Exception as e:
                logging.warning(f"Could not read {prev_results}: {e}")
                continue
        return heat_pump_returns, True

    # Method to perform statistical labeling of heat pumps in smart meter data
    def stat_label_sm(self) -> dict:
        # Deferred so importing the package does not require scipy/sklearn/statsmodels
        from threephi_framework.dtu.stat_labeler import label_meters

        # Put weather_file to stat_labeler bucket
        if not self.data_extractor.s3_connector.exists(self.config.weather_file):
            self.data_extractor.s3_connector.put_file(self.config.weather_file_local, self.config.weather_file)
            logging.info(f"Weather file uploaded to {self.config.weather_file}")

        # initialize return dict
        heat_pump_returns = {}
        sm_to_process = []

        # Check for previous results to avoid re-processing
        if not self.config.overwrite_existing_results:
            earlier_results = self.data_extractor.s3_connector.glob(
                f"{self.config.results_dir}/heat_pump_results_*.json"
            )
            for sm_id in self.config.sm_ids:
                heat_pump_returns, proceed = self._check_previous_results(earlier_results, sm_id, heat_pump_returns)
                if proceed:
                    sm_to_process.append(sm_id)
                else:
                    logging.info(f"Skipping processing for SM {sm_id} as results already exist.")
        else:
            sm_to_process = self.config.sm_ids

        if sm_to_process:
            sm_with_hp = self.topology_controller.get_meters(has_heat_pump=True)

            sm_id_chunks = np.array_split(sm_to_process, min(len(sm_to_process), self.config.dask["n_workers"]))
            cfg = self.config.to_dict()
            delayed_tasks = [delayed(label_meters)(sm_ids_chunk, sm_with_hp, cfg) for sm_ids_chunk in sm_id_chunks]
            results = compute(*delayed_tasks)
            meta_results_list = [r[0] for r in results]
            heat_pump_list = [r[1] for r in results]
            heat_pump_returns_list = [r[2] for r in results]

            meta_results_merged = {k: v for d in meta_results_list for k, v in d.items()}
            heat_pump_merged = {k: v for d in heat_pump_list for k, v in d.items()}
            heat_pump_returns_merged = {k: v for d in heat_pump_returns_list for k, v in d.items()}

            # Update the heat_pump_returns with newly processed results
            heat_pump_returns.update(heat_pump_returns_merged)

            # Save results if desired
            # Add timestamp to the filename
            now = datetime.now().strftime("%Y-%m-%d_%H_%M_%S")
            if self.config.save_meta_results:
                self.data_extractor.s3_connector.write_json(
                    path=f"{self.config.results_dir}/meta_results_{now}.json", data=meta_results_merged
                )
            self.data_extractor.s3_connector.write_json(
                path=f"{self.config.results_dir}/heat_pump_results_{now}.json", data=heat_pump_merged
            )

        # Also covers the case where every meter already had results: return the
        # labels loaded from earlier result files instead of None.
        return {sm_id: heat_pump_returns.get(sm_id) for sm_id in self.config.sm_ids}

    def run(self):
        self.stat_label_sm()


if __name__ == "__main__":
    config = {
        "dask": {
            "local": True,
            "n_workers": 6,
        },
        "sm_ids": ["17", "999955", "999950", "999964"],
        "overwrite_existing_results": False,
        "filter_data": True,
        "process_only_sm_with_hp": False,
        "label_summerhouse": True,
        "use_ANOVA": True,
        "save_plots": False,
        "save_meta_results": True,
        "data_dir_path": "phase_measurements/raw",
        "thresholds": {
            "weekly_change": 0.01,
            "static_days": 0.4,
            "weekend_ratio": 0.45,
            "max_bins": 12,
            "min_bins": 1,
            "filter_temp": 3,
            "anova_pvalue": 0.01,
        },
        "results_dir": "s3://3phi/stat_labeler",
        "weather_file": "s3://3phi/stat_labeler/data/weather_data.csv",
        "weather_file_local": "/opt/airflow/data/weather_data.csv",
    }

    with StatLabeler(config) as app:
        app.run()
