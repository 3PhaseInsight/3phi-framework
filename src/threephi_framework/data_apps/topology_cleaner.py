from threephi_framework.data_apps.base import BaseDataApp
from threephi_framework.dtu.topology_cleaner import clean_sm_cabinet, clean_topology


class TopologyCleaner(BaseDataApp):
    """
    Topology Cleaner Data App.

    Reads the current topology version from the database, applies data-quality
    cleaning (NaN imputation, duplicate edge removal), and ingests the result
    as a new topology version.

    The following config parameters are expected:

    - ``dask`` (dict): Dask cluster settings (see :class:`~threephi_framework.data_apps.base.BaseDataApp`).
    - ``override`` (bool): If ``True``, re-run cleaning and ingest a new version even if the workflow
      has already completed. Defaults to ``False``.

    Usage::

        with TopologyCleaner(config) as app:
            app.run()
    """

    _WORKFLOW = "topology_cleaning"

    # LV topology column names (second batch schema)
    # TODO: This is hardcoded to second batch, could be made more generic
    NODE1_COL = "node1"
    NODE2_COL = "node2"
    SEC_SUBSTATION_COL = "secondary_substation"
    CABLE_CAPACITY_COL = "cable_capacity"
    CABLE_LENGTH_COL = "cable_length"
    PHASE_SIZE_COL = "phase_size"
    RESISTANCE_COL = "resistance"
    REACTANCE_COL = "reactance"
    LV_FEEDER_FUSE_SIZE_COL = "lv_feeder_fuse_size"
    CABLE_TYPE_COL = "cable_type"
    PHASE_MATERIAL_COL = "phase_material"
    METER_NUMBER_COL = "meter_number"

    def __init__(self, config):
        super().__init__(config)
        self.override = self.config.get("override", False)

    def run(self):
        if not self.meta_controller.is_workflow_completed(self._WORKFLOW) or self.override:
            topology_ddf = self.topology_controller.export_topology(as_dask=True)
            sm_cab_ddf = self.topology_controller.export_sm_cabinet(as_dask=True)

            cleaned_topology_ddf = clean_topology(
                topology_ddf,
                node1_col=self.NODE1_COL,
                node2_col=self.NODE2_COL,
                sec_substation_col=self.SEC_SUBSTATION_COL,
                cable_capacity_col=self.CABLE_CAPACITY_COL,
                cable_length_col=self.CABLE_LENGTH_COL,
                phase_size_col=self.PHASE_SIZE_COL,
                resistance_col=self.RESISTANCE_COL,
                reactance_col=self.REACTANCE_COL,
                lv_feeder_fuse_size_col=self.LV_FEEDER_FUSE_SIZE_COL,
                cable_type_col=self.CABLE_TYPE_COL,
                phase_material_col=self.PHASE_MATERIAL_COL,
            )
            cleaned_sm_cab_ddf = clean_sm_cabinet(sm_cab_ddf, meter_number_col=self.METER_NUMBER_COL)

            self.topology_controller.ingest(cleaned_topology_ddf, cleaned_sm_cab_ddf)

        self.meta_controller.complete_workflow(self._WORKFLOW)


if __name__ == "__main__":
    config = {
        "override": False,
        "dask": {
            "local": True,
            "n_workers": 2,
        },
    }
    with TopologyCleaner(config) as app:
        app.run()
