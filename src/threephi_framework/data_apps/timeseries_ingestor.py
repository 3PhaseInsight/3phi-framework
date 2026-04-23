from threephi_framework.data_apps.base import BaseDataApp


class TimeseriesIngestor(BaseDataApp):
    """
    Timeseries Ingestion Data App.
    The following config parameters are expected to be present in the config:
        - result_name: This should describe the name any associated results will be stored under.
        - dask: This should be an object including "host" and "port" where the Dask Scheduler can be reached.
        - override: Boolean describing whether the workflow should be executed again,
          even if it already has been executed.
        - csv_source_path: Path where the source CSV files can be read.
        - csv_file_pattern: Filename pattern of the CSV files that should be ingested.
        - parquet_destination_path: Path (in Blob Storage) where the parquet files should be stored.

    Usage:
    ```
    with TimeseriesIngestor(config) as app:
        app.run()
    ````
    """

    def __init__(self, config):
        super().__init__(config)
        self.override = self.config["override"]
        self.csv_source_path = self.config["csv_source_path"]
        self.csv_file_pattern = self.config["csv_file_pattern"]
        self.parquet_destination_path = self.config["parquet_destination_path"]

    def run(self):
        workflow = "timeseries_csv_to_parquet_partitions"
        completed = self.meta_controller.is_workflow_completed(workflow)
        if not completed or self.override:
            self.data_extractor.v1_csv_to_parquet_partitions(
                csv_path="/opt/airflow/data",
                csv_file_pattern="phase_measurements_*.csv",
                bucket_dest_path=f"{self.data_extractor.s3_base}/{self.parquet_destination_path}",
            )
        self.meta_controller.complete_workflow(workflow)


if __name__ == "__main__":
    config = {}
    with TimeseriesIngestor(config) as app:
        app.run()
