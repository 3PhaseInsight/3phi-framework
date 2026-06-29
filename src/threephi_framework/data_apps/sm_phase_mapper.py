import logging

import matplotlib
import numpy as np
import pandas as pd
from dask import compute, delayed

from threephi_framework.data_apps.base import BaseDataApp

matplotlib.use("Agg")


class SMPhaseMapper(BaseDataApp):
    """
    Smart Meter Phase Mapper Data App.

    Identifies which physical transformer/feeder phase each smart meter phase is
    connected to by hierarchically clustering voltage time series per transformer,
    and writes one row per (meter, phase) to ``meta.sm_phase_mapping`` via
    :meth:`~threephi_framework.controllers.meta.MetaController.update_phase_mapping`.

    The per-transformer evaluation is delegated to
    :func:`threephi_framework.dtu.sm_phase_mapper.identify_sm_topology`.

    The following config parameters are supported:

    - ``dask`` (dict): Dask cluster settings (see :class:`BaseDataApp`).
    - ``use_dask`` (bool): If True, distribute transformer evaluation across Dask
      workers using ``dask.delayed``. Defaults to ``False``.
    - ``trafo_ids`` (str | list[str]): ``"All"`` to process every transformer in
      the current topology, or an explicit list of transformer ID strings.
    - ``data_dir_path`` (str): Root path of the raw timeseries dataset in object
      storage. Defaults to ``phase_measurements/raw``.
    - ``save_plots`` (bool): Persist per-transformer dendrogram plots to object
      storage. Defaults to ``True``.
    - ``results_dir`` (str): Object storage path for plots. Defaults to
      ``<storage_base>/sm_phase_mapper``.
    - Detection thresholds — see
      :func:`~threephi_framework.dtu.sm_phase_mapper.identify_sm_topology`:
      ``window_size``, ``lower_v_lim``, ``v_step_lim``, ``max_corruption_threshold``.

    Usage::

        with SMPhaseMapper(config) as app:
            app.run()
    """

    def __init__(self, config, connector=None):
        super().__init__(config=config, connector=connector)
        self.use_dask = config.get("use_dask", False)
        self.trafo_ids = config.get("trafo_ids", None)
        self.save_plots = config.get("save_plots", True)

    def _validate_config(self) -> None:
        if not (
            isinstance(self.trafo_ids, str)
            or (isinstance(self.trafo_ids, list) and all(isinstance(i, str) for i in self.trafo_ids))
        ):
            raise TypeError("trafo_ids must be 'All' or a list of transformer ID strings.")

    def _build_dtu_cfg(self) -> dict:
        """Build the per-task ``cfg`` dict consumed by ``dtu.sm_phase_mapper.identify_sm_topology``."""
        cfg = dict(self.config)
        cfg.setdefault("window_size", 30)
        cfg.setdefault("lower_v_lim", 207)
        cfg.setdefault("v_step_lim", 10)
        cfg.setdefault("max_corruption_threshold", 0.9)
        cfg.setdefault("data_dir_path", self.data_dir_path)
        cfg.setdefault("results_dir", f"{self.connector.storage_base}/sm_phase_mapper")
        cfg.setdefault("save_plots", self.save_plots)
        return cfg

    def map_smart_meter_phases(self) -> None:
        """Evaluate the configured transformers; results are persisted to the meta DB."""
        # Deferred so importing the package does not require scipy/scikit-learn
        from threephi_framework.dtu.sm_phase_mapper import identify_sm_topology

        self._validate_config()

        # Create list of transformer IDs to process
        if self.trafo_ids == "All":
            lv_topology = self.topology_controller.export_topology()
            trafos = lv_topology["transformer"].str.split(".").str[1].unique()
            self.trafo_ids = trafos.tolist()
        self.trafo_ids = [str(trafo_id) for trafo_id in self.trafo_ids]

        # Generate smart meter to topology mapping for all transformers to be processed.
        # This mapping is used for easy lookup of the topology chain for each smart meter
        # during phase mapping, which avoids repeated DB queries.
        logging.info("Building smart meter to topology mapping...")
        sm_topology_mapping: dict = {}
        for trafo_id in self.trafo_ids:
            # returns a dict with sm_id as key and topology chain as value
            mapping_chunk = self.topology_controller.get_topology_map_for_transformer(transformer_id=int(trafo_id))
            if not mapping_chunk:
                logging.warning(f"No topology mapping could be made for transformer {trafo_id}. Skipping.")
                continue
            sm_topology_mapping.update(mapping_chunk)

        sm_topology_mapping = pd.DataFrame.from_dict(sm_topology_mapping, orient="index")
        sm_topology_mapping.reset_index(inplace=True)
        sm_topology_mapping.rename(columns={"index": "Meter ID"}, inplace=True)
        logging.info("Finished building smart meter to topology mapping.")
        logging.info(f"Identifying phase mapping for {len(self.trafo_ids)} transformer(s): {', '.join(self.trafo_ids)}")

        cfg = self._build_dtu_cfg()

        if self.use_dask:
            n_workers = self.dask_settings.get("n_workers", 1) or 1
            chunks = np.array_split(self.trafo_ids, min(len(self.trafo_ids), n_workers))
            tasks = [
                delayed(identify_sm_topology)(trafo_ids=list(chunk), cfg=cfg, sm_topology_mapping=sm_topology_mapping)
                for chunk in chunks
            ]
            compute(*tasks)
        else:
            identify_sm_topology(trafo_ids=self.trafo_ids, cfg=cfg, sm_topology_mapping=sm_topology_mapping)

        logging.info("SM Phase Mapper run completed.")

    def run(self):
        self.map_smart_meter_phases()


if __name__ == "__main__":
    config = {
        "use_dask": True,
        "dask": {"local": True, "n_workers": 1},
        "data_dir_path": "phase_measurements/raw",
        "trafo_ids": ["494439"],
        "save_plots": True,
        "window_size": 30,
        "lower_v_lim": 207,
        "v_step_lim": 10,
        "max_corruption_threshold": 0.9,
    }
    with SMPhaseMapper(config=config) as app:
        app.run()
