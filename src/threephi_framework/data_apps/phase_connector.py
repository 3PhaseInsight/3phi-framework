import logging

from threephi_framework.data_apps.base import BaseDataApp
from threephi_framework.dtu.phase_connector import recommend_phase


class PhaseConnector(BaseDataApp):

    def __init__(self, config):
        super().__init__(config=config)
        self.use_dask = config.get("use_dask", False)
        self.sm_id = config.get("sm_id", None)
        self.save_plots = config.get("save_plots", False)
        self.plot_cfg = config.get("plot_cfg")

    def _build_dtu_cfg(self) -> dict:
        """Build the per-task ``cfg`` dict consumed by ``dtu.meter_evaluation``."""
        cfg = dict(self.config)
        cfg.setdefault("results_dir", f"{self.data_extractor.s3_base}/phase_connector")
        return cfg

    def identify_optimal_phase_connection(self) -> None:
        """Evaluate the configured smart meters; results are persisted to the meta DB."""

        cfg = self._build_dtu_cfg()

        results = recommend_phase(sm_id=self.sm_id, cfg=cfg)

        logging.info("Phase connection service run completed.")

        return results

    def run(self):
        self.identify_optimal_phase_connection()


if __name__ == "__main__":
    config = {
        "use_dask": True,
        "dask": {"local": True, "n_workers": 1},
        "data_dir_path": "phase_measurements/raw",
        "sm_id": '759234',
        "save_plots": False,
        "profile_processing_level": "raw",

        # appliance_type can be either hp, ev or pv
        "phase_scoring": {
            # Appliance to evaluate: hp | ev | pv
            "appliance_type": "hp",
            "label_columns": {
                "hp": "predicted_hp",
                "ev": "predicted_ev_phase",
                "pv": "predicted_pv"
            },
            # Scoring weights — must sum to 100
            "weights": {
                "C1_feeder_balance": 40,
                "C2_type_concentration": 30,
                "C3_household_balance": 30
            }
        },
        'HP_ML_algorithm': "label_propagation"
    }
    with PhaseConnector(config=config) as app:
        app.run()
