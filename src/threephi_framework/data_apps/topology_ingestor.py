from threephi_framework.data_apps.base import BaseDataApp
from threephi_framework.processing_level import ProcessingLevel


class TopologyIngestor(BaseDataApp):
    """
    Topology Ingestion Data App.

    The following config parameters are expected to be present in the config:

    - result_name: This should describe the name any associated results will be stored under.
    - dask: This should be an object including "host" and "port" where the Dask Scheduler can be reached.
    - override: Boolean describing whether the workflow should be executed again, even if it already has been executed.
    - topology_source_path: Path where the topology source CSV file can be read.
    - sm_cab_source_path: Path where the meter-cabinet-connection source CSV file can be read.
    - parquet_destination_path: Path (in Blob Storage) where the parquet files should be stored.

    Usage::

        with TopologyIngestor(config) as app:
            app.run()
    """

    def __init__(self, config):
        super().__init__(config)
        self.override = self.config["override"]

        self.topology_source_path = self.data_extractor.s3_base + self.config["topology_source_path"]
        self.sm_cab_source_path = self.data_extractor.s3_base + self.config["sm_cab_source_path"]
        self.processing_level = self.config.get("processing_level", ProcessingLevel.RAW)
        self.use_dask = config.get('use_dask', False)

    def run(self):
        workflow = "topology_ingestion"
        completed = self.meta_controller.is_workflow_completed(workflow)
        if not completed or self.override:
            topology_ddf = self.topology_controller.read_topology(self.topology_source_path)
            sm_cab_ddf = self.topology_controller.read_sm_cab(self.sm_cab_source_path)
            self.topology_controller.ingest(topology_ddf, sm_cab_ddf, self.processing_level)
        self.meta_controller.complete_workflow(workflow)


if __name__ == "__main__":
    config = {
        "use_dask": True,
        "dask": {"local": True, "n_workers": 1},
        "override": True,
        "topology_source_path": "/data/lv_topology.csv",
        "sm_cab_source_path": "/data/meter_cabinet_connection.csv",
        "processing_level": "raw"}  # Example config
    with TopologyIngestor(config) as app:
        app.run()
