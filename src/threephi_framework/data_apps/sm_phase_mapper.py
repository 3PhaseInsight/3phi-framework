import logging

import matplotlib
import numpy as np
from dask import compute, delayed

from threephi_framework.data_apps.base import BaseDataApp
from threephi_framework.dtu.sm_phase_mapper import identify_sm_topology
from threephi_framework.controllers.topology import TopologyController
import threephi_framework.db.db as threephi_db

import pandas as pd

matplotlib.use("Agg")


class SMPhaseMapper(BaseDataApp):

    def __init__(self, config):
        super().__init__(config=config)
        self.use_dask = config.get('use_dask', False)
        self.trafo_ids = config.get('trafo_ids', None)
        # self.sm_clf_file = config.get('sm_clf_file', None)
        self.save_plots = config.get('save_plots', True)
        self.data_dir_path = config.get('data_dir_path', None)

    
    def _validate_config(self) -> None:

        if not (
            isinstance(self.trafo_ids, str) 
            or (isinstance(self.trafo_ids, list) and all(isinstance(i, str) for i in self.trafo_ids))
        ):
            raise TypeError("trafo_ids must be a string or list of strings.")

    def _build_dtu_cfg(self) -> dict:
        """Build the per-task ``cfg`` dict consumed by ``dtu.meter_evaluation``."""
        cfg = dict(self.config)
        cfg.setdefault("window_size", 30)
        cfg.setdefault("lower_v_lim", 207)
        cfg.setdefault("v_step_lim", 10)
        cfg.setdefault("max_corruption_threshold", 0.9)
        cfg.setdefault("topology_processing_level", "raw")
        cfg.setdefault("save_missing_trafo_datasets", False)
        cfg.setdefault("max_rec_period", None)
        cfg.setdefault("results_dir", f"{self.data_extractor.s3_base}/sm_phase_mapper")
        cfg.setdefault("save_plots", self.save_plots)
        return cfg

    def map_smart_meter_phases(self) -> None:
        """Evaluate the configured smart meters; results are persisted to the meta DB."""
        self._validate_config()
        topology_controller = TopologyController(threephi_db.new_session)

        # Create list of transformer IDs to process
        if self.trafo_ids == "All":
            lv_topology = topology_controller.export_topology()
            trafos = lv_topology["transformer"].str.split(".").str[1].unique()
            self.trafo_ids = trafos.tolist()
        else:
            self.trafo_ids = self.trafo_ids
        self.trafo_ids = [str(trafo_id) for trafo_id in self.trafo_ids]
        print(self.trafo_ids)

        # Generate smart meter to topology mapping for all transformers to be processed
        # This mapping is used for easy lookup of the topology chain for each smart meter during phase mapping, which avoids repeated DB queries
        logging.info("Building smart meter to topology mapping...")
        SM_topology_mapping = {}
        for trafo_id in self.trafo_ids:
            SM_topology_mapping_chunk = topology_controller.get_topology_map_for_transformer(transformer_id=int(trafo_id)) # returns a dict with sm_id as key and topology chain as value
            if not SM_topology_mapping_chunk:
                logging.warning(f"No topology mapping could be made for transformer {trafo_id}. Skipping.")
                continue
            SM_topology_mapping.update(SM_topology_mapping_chunk)
            
        SM_topology_mapping = pd.DataFrame.from_dict(SM_topology_mapping, orient='index')
        SM_topology_mapping.reset_index(inplace=True)
        SM_topology_mapping.rename(columns={'index': 'Meter ID'}, inplace=True)
        logging.info("Finished building smart meter to topology mapping.")
        logging.info(f"Identifying phase mapping for transformer{'s' if len(self.trafo_ids) > 1 or self.trafo_ids == 'All' else ''} {', '.join(str(u) for u in self.trafo_ids)}..." if self.trafo_ids != "All" else ": Identifying phase mapping for all transformers...")

        cfg = self._build_dtu_cfg()

        if self.use_dask:
            n_workers = self.dask_settings.get("n_workers", 1) or 1
            chunks = np.array_split(self.trafo_ids, min(len(self.trafo_ids), n_workers))
            tasks = [delayed(identify_sm_topology)(trafo_ids=list(chunk), cfg=cfg, SM_topology_mapping=SM_topology_mapping) for chunk in chunks]
            compute(*tasks)
        else:
            identify_sm_topology(trafo_ids=self.trafo_ids, cfg=cfg, SM_topology_mapping=SM_topology_mapping)

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
        "topology_processing_level": "raw",
        "save_missing_trafo_datasets": False,
    }
    with SMPhaseMapper(config=config) as app:
        app.run()

