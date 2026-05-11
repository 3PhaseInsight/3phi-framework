import logging

import matplotlib
import numpy as np
from dask import compute, delayed

from threephi_framework.data_apps.base import BaseDataApp
from threephi_framework.dtu.sm_classifier import meter_evaluation

matplotlib.use("Agg")


class SMClassifier(BaseDataApp):
    """
    Smart Meter Classification Data App.

    Evaluates each smart meter's time series data to produce a per-meter
    characterization (data quality, statistics, connectivity) and writes the
    result to the ``data_quality``, ``data_statistics``, and ``connectivity``
    JSONB columns of ``meta.meter`` via
    :class:`~threephi_framework.controllers.meta.MetaController`.

    The per-meter evaluation is delegated to
    :func:`threephi_framework.dtu.sm_classifier.meter_evaluation`.

    The following config parameters are supported:

    - ``dask`` (dict): Dask cluster settings (see :class:`BaseDataApp`).
    - ``data_dir_path`` (str): Root path of the raw timeseries dataset in
      object storage. Passed through to ``meter_evaluation``.
    - ``use_dask`` (bool): If True, distribute meter evaluation across Dask
      workers using ``dask.delayed``. Defaults to ``False``.
    - ``sm_ids`` (str | list[str]): ``"All"`` to classify every meter that has
      data, or an explicit list of meter ID strings.
    - ``save_plots`` (bool): Persist per-meter timeseries plots to object storage.
    - ``plot_cfg`` (dict): Plot configuration with ``SM_selection``,
      ``Variable_selection``, ``Phase_selection``, and ``Plotting_format``.
      Required when ``save_plots=True``.
    - ``results_dir`` (str): Object storage path for plots. Defaults to
      ``s3://3phi/sm_classifier``.
    - Detection thresholds — see
      :func:`~threephi_framework.dtu.sm_classifier.meter_evaluation`:
      ``no_data_limit``, ``good_data_limit``, ``medium_data_limit``, ``v_lim``,
      ``offset_threshold``, ``cons_period_threshold``, ``frozen_range``.

    Usage::

        with SMClassifier(config) as app:
            app.run()
    """

    ALLOWED_VARIABLES = {"V", "P14", "P23", "Q12", "Q34"}
    ALLOWED_PHASES = {"L1", "L2", "L3"}

    def __init__(self, config):
        super().__init__(config=config)
        self.use_dask = config.get("use_dask", False)
        self.sm_ids = config.get("sm_ids", "All")
        self.save_plots = config.get("save_plots", False)
        self.plot_cfg = config.get("plot_cfg")

    def _validate_config(self) -> None:
        if not (
            isinstance(self.sm_ids, str)
            or (isinstance(self.sm_ids, list) and all(isinstance(i, str) for i in self.sm_ids))
        ):
            raise TypeError("sm_ids must be 'All' or a list of meter ID strings.")

        if self.save_plots:
            if self.plot_cfg is None:
                raise ValueError("plot_cfg must be provided when save_plots=True.")
            if sum(self.plot_cfg["SM_selection"].values()) == 0:
                raise ValueError("At least one 'SM_selection' option in plot_cfg must be True.")
            variables = {v.upper() for v in self.plot_cfg["Variable_selection"]}
            if not variables.issubset(self.ALLOWED_VARIABLES):
                raise ValueError(f"plot_cfg.Variable_selection must be a subset of {self.ALLOWED_VARIABLES}.")
            phases = {v.upper() for v in self.plot_cfg["Phase_selection"]}
            if not phases.issubset(self.ALLOWED_PHASES):
                raise ValueError(f"plot_cfg.Phase_selection must be a subset of {self.ALLOWED_PHASES}.")

    def _build_dtu_cfg(self) -> dict:
        """Build the per-task ``cfg`` dict consumed by ``dtu.meter_evaluation``."""
        cfg = dict(self.config)
        cfg.setdefault("phases", ["l1", "l2", "l3"])
        cfg.setdefault("variables", ["v", "p14", "p23", "q12", "q34"])
        cfg.setdefault("max_rec_period", None)
        cfg.setdefault("results_dir", f"{self.data_extractor.s3_base}/sm_classifier")
        cfg.setdefault("save_plots", self.save_plots)
        if self.plot_cfg is not None:
            cfg.setdefault(
                "selected_variables",
                [v.upper() for v in self.plot_cfg["Variable_selection"]],
            )
            cfg.setdefault(
                "selected_phases",
                [v.lower() for v in self.plot_cfg["Phase_selection"]],
            )
        return cfg

    def classify_smart_meters(self) -> None:
        """Evaluate the configured smart meters; results are persisted to the meta DB."""
        self._validate_config()

        if self.sm_ids == "All":
            sm_ids = self.meta_controller.get_time_series_meta_info()["id_list_of_sms_with_data"]
        else:
            sm_ids = self.sm_ids
        sm_ids = [str(sm_id) for sm_id in sm_ids]
        logging.info(f"Classifying {len(sm_ids)} smart meter(s).")

        cfg = self._build_dtu_cfg()

        if self.use_dask:
            n_workers = self.dask_settings.get("n_workers", 1) or 1
            chunks = np.array_split(sm_ids, min(len(sm_ids), n_workers))
            tasks = [delayed(meter_evaluation)(sm_ids=list(chunk), cfg=cfg) for chunk in chunks]
            compute(*tasks)
        else:
            meter_evaluation(sm_ids=sm_ids, cfg=cfg)

        logging.info("SM Classifier run completed.")

    def run(self):
        self.classify_smart_meters()


if __name__ == "__main__":
    config = {
        "use_dask": True,
        "dask": {"local": True, "n_workers": 4},
        "data_dir_path": "phase_measurements/raw",
        "sm_ids": "All",
        "save_plots": False,
        "no_data_limit": 0.025,
        "good_data_limit": 0.1,
        "medium_data_limit": 0.5,
        "v_lim": 207,
        "offset_threshold": 0.95,
        "cons_period_threshold": 192,
        "frozen_range": 12,
    }
    with SMClassifier(config=config) as app:
        app.run()
