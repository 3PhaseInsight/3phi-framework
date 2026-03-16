import threephi_framework.db.db as threephi_db
from threephi_framework.controllers.topology import TopologyController
from threephi_framework.data_apps.base import BaseDataApp


class TopologyIngestor(BaseDataApp):
    """
    Topology Ingestion Data App.
    The following config parameters are expected to be present in the config:
        - result_name: This should describe the name any associated results will be stored under.
        - dask: This should be an object including "host" and "port" where the Dask Scheduler can be reached.
        - override: Boolean describing whether the workflow should be executed again,
          even if it already has been executed.
        - topology_source_path: Path where the topology source CSV file can be read.
        - sm_cab_source_path: Path where the meter-cabinet-connection source CSV file can be read.
        - parquet_destination_path: Path (in Blob Storage) where the parquet files should be stored.

    Usage:
    ```
    with TopologyIngestor(config) as app:
        app.run()
    ````
    """

    def __init__(self, config):
        super().__init__(config)
        self.override = self.config["override"]
        self.topology_source_path = self.config["topology_source_path"]
        self.sm_cab_source_path = self.config["sm_cab_source_path"]
        self.topology_controller = TopologyController(threephi_db.new_session)

    def run(self):
        workflow = "topology_ingestion"
        completed = self.data_extractor.db_connector.is_workflow_completed(workflow)
        if not completed or self.override:
            topology_ddf = self.topology_controller.read_topology(self.topology_source_path)
            sm_cab_ddf = self.topology_controller.read_sm_cab(self.sm_cab_source_path)
            self.topology_controller.ingest(topology_ddf, sm_cab_ddf)
        self.data_extractor.db_connector.complete_workflow(workflow)


if __name__ == "__main__":
    config = {}
    with TopologyIngestor(config) as app:
        app.run()
