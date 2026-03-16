import dask
import dask.dataframe as dd
import numpy as np
import pandas as pd


class TopologyCleaner:
    def __init__(self, path_to_topology_file, path_to_meter_cabinet_connection_file):
        # TODO: Implement connecting and reading to/from databases;
        # self.db_connector = DBConnector()
        # TODO: Should Dask Cluster be initiated here?
        self.topology_file_path = path_to_topology_file
        self.meter_cabinet_connection_file_path = path_to_meter_cabinet_connection_file

        # Define column names (in second batch).
        # TODO: This is hardcoded to second batch, could be made more generic
        self.meter_number_col = "meter_number"
        self.timestamp_dst_col = "timestamp_dst"
        self.cabinet_col = "cabinet"
        self.sec_substation_col = "secondary_substation"
        self.transformer_col = "transformer"
        self.lv_feeder_col = "lv_feeder"
        self.node_1_col = "node1"
        self.node_2_col = "node2"
        self.voltage_col = "voltage_l"
        self.current_col = "current_l"
        self.active_power_p14_col = "active_power_p14_l"
        self.active_power_p23_col = "active_power_p23_l"
        self.reactive_power_q12_col = "reactive_power_q12_l"
        self.reactive_power_q34_col = "reactive_power_q34_l"
        self.v_phase_unbalance_col = "v_unbalance_l"
        self.v_unbalance_col = "v_unbalance"
        self.i_phase_unbalance_col = "i_unbalance_l"
        self.i_unbalance_col = "i_unbalance"
        self.transformer_capacity_col = "transformer_capacity"
        self.lv_feeder_fuse_size_col = "lv_feeder_fuse_size"
        self.cable_id_col = "cable_id"
        self.cable_type_col = "cable_type"
        self.cable_length_col = "cable_length"
        self.phase_size_col = "phase_size"
        self.phase_material_col = "phase_material"
        self.cable_capacity_col = "cable_capacity"
        self.resistance_col = "resistance"
        self.reactance_col = "reactance"
        self.metercabinet_file = "meter_cabinet_connection.csv"
        self.cabinet_name = "Cabinet"

        # Define dtypes for meter and cabinet.
        # TODO: This is hardcoded to second batch, could be made more generic
        self.meter_and_cabinet_dtypes = {
            self.meter_number_col: "string",
            self.cabinet_col: "string",
            "delivery_point_id": "string",
            "lv_feeder": "string",
            "has_heat_pump": "boolean",
            "has_solar_panel": "boolean",
            "capacity_solar_panel": "Float32",
            "service_fuse_size": "Float32",
        }

        # Define dtypes for topology.
        # TODO: This is hardcoded to second batch, could be made more generic
        self.topology_dtypes = {
            self.sec_substation_col: "string",
            self.transformer_col: "string",
            self.transformer_capacity_col: "Float32",
            self.lv_feeder_col: "string",
            self.lv_feeder_fuse_size_col: "Float32",
            self.node_1_col: "string",
            self.node_2_col: "string",
            self.cable_id_col: "string",
            self.cable_type_col: "string",
            self.cable_length_col: "Float32",
            self.phase_size_col: "Float32",
            self.phase_material_col: "string",
            self.cable_capacity_col: "Float32",
            self.resistance_col: "Float32",
            self.reactance_col: "Float32",
            "zip_code_secondary_substation": "string",
        }

    def load_topology_csv_to_df(self) -> dd.DataFrame:
        return dd.read_csv(
            self.topology_file_path,
            dtype=self.topology_dtypes,
            assume_missing=True,
            na_values=["", "NA", "NaN", "null"],
            true_values=["True", "true", "1"],
            false_values=["False", "false", "0"],
            blocksize="64MB",
        )

    def load_mtr_cbnt_connection_csv_to_df(self) -> dd.DataFrame:
        return dd.read_csv(
            self.meter_cabinet_connection_file_path,
            dtype=self.meter_and_cabinet_dtypes,
            assume_missing=True,
            na_values=["", "NA", "NaN", "null"],
            true_values=["True", "true"],
            false_values=["False", "false"],
            blocksize="64MB",
        )

    def load_topology_csv_to_pandas(
        self,
    ) -> pd.DataFrame:  # TODO: Remove this method later (included for testing)...
        return pd.read_csv(
            self.topology_file_path,
            dtype=self.topology_dtypes,
            na_values=["", "NA", "NaN", "null"],
            true_values=["True", "true", "1"],
            false_values=["False", "false", "0"],
        )

    def load_mtr_cbnt_connection_csv_to_pandas(
        self,
    ) -> pd.DataFrame:  # TODO: Remove this method later (included for testing)...
        return pd.read_csv(
            self.meter_cabinet_connection_file_path,
            dtype=self.meter_and_cabinet_dtypes,
            na_values=["", "NA", "NaN", "null"],
            true_values=["True", "true"],
            false_values=["False", "false"],
        )

    def clean_topology_dask(self, topology: dd.DataFrame) -> dd.DataFrame:
        # CLEANING STEP 0: Adding helper columns with integer part of NODE1, NODE2
        # and Substation respectively (splitting strings at the dot)
        topology["NODE1_value"] = topology[self.node_1_col].str.split(".", n=1).str.get(1).astype("string")
        topology["NODE2_value"] = topology[self.node_2_col].str.split(".", n=1).str.get(1).astype("string")
        topology["Substation"] = topology[self.sec_substation_col].str.split(".", n=1).str.get(1).astype("string")

        # CLEANING STEP 1: Numerical columns: Fill nan with the mean of the
        # respective substation or the global mean if all values of substation are
        # nan.
        # TODO: Consider if we need to stick with Niels way of doing global mean
        # AFTER filling in per-substation mean
        # Make a list of numerical columns
        # TODO: This is hardcoded to second batch, could be made more generic
        cols: list[str] = [
            self.cable_capacity_col,
            self.cable_length_col,
            self.phase_size_col,
            self.resistance_col,
            self.reactance_col,
            self.lv_feeder_fuse_size_col,
        ]

        # 1a) Compute per-substation and global means
        valid: dd.DataFrame = topology.dropna(subset=["Substation"])
        mean_per_sub_dd: dd.DataFrame = valid.groupby("Substation")[cols].mean()

        # mean_per_sub_dd: dd.DataFrame = topology.groupby("Substation")[cols].mean()
        mean_global_dd: dd.Series = topology[cols].mean()
        mean_per_substation, mean_global = dask.compute(mean_per_sub_dd, mean_global_dd)

        mean_per_substation = mean_per_substation.round(
            0
        )  # TODO: Rounding to zero decimals is in line with Niels' code, but
        # could be reconsidered, especially for resistance and reactance, for
        # which all data points will be rounded to 0
        mean_global = mean_global.round(0)  # TODO: Rounding to zero decimals is in line with Niels' code, but
        # could be reconsidered, especially for resistance and reactance, for
        # which all data points will be rounded to 0

        for col in cols:
            fill_per_sub = topology["Substation"].map(mean_per_substation[col], meta=("Substation", "float32"))
            global_fill = np.float32(mean_global[col])
            topology[col] = (
                topology[col]
                .astype("float32")  # TODO: Use the self.topology_dtypes to determine dtype instead of
                # hardcoding float32
                .fillna(fill_per_sub)  # first fill with per-sub means
                .fillna(global_fill)  # fallback to global mean where all data in per-sub is NaN
            )

        final_defaults = {c: np.float32(mean_global[c]) for c in cols}
        topology[cols] = topology[cols].astype("float32").fillna(final_defaults)

        # CLEANING STEP 2: Categorical columns: Fill nan with the mean of the
        # respective substation or the global mean if all values of substation are
        # nan
        # TODO: 1b: Consider if we need to stick with Niels way of handling
        # columns with a tie in the count (choosing first alphabetically rather
        # than by first encounter as is done here)

        # Make a dict of categorical columns holding lists of categories as items.
        # TODO: This is hardcoded to second batch, could be made more generic
        categories = {
            self.cable_type_col: ["APB", "MAL", "PEX", "PEX ENLEDER", "PEXMAL", "PSP", "U"],
            self.phase_material_col: ["AL", "CU", "U"],
        }

        # Ensure columns are categorical with all the categories present (even if
        # not present in given partition)
        for col, cats in categories.items():
            topology[col] = topology[col].astype("category").cat.set_categories(cats)

        for col in categories:
            # 2a) Create new dataframe with counts of each category per substation
            counts: dd.DataFrame = (
                topology[["Substation", col]]
                .dropna(subset=[col])
                .groupby(["Substation", col], observed=True)
                .size()
                .compute()
                .reset_index(name="count")
            )

            # 2b) Create mapping of each substation to most common category
            mode_map = (
                counts.sort_values(["Substation", "count"], ascending=[True, False])
                .drop_duplicates("Substation")
                .set_index("Substation")[col]
                .to_dict()
            )

            # 2c) Fill NaN with most common category per substation, if substation
            # has all NaN in the column, fill with globally most common category
            # for given column
            topology[col] = topology[col].fillna(topology["Substation"].map(mode_map, meta=("Substation", "object")))
            global_mode = topology[col].value_counts().idxmax().compute()
            topology[col] = topology[col].fillna(global_mode)

        # CLEANING STEP 3: Remove lines with identical NODE1 and NODE2 values,
        # keep the line with the highest current carrying capacity. Also drop helper
        # columns to clean up the dataframe
        col = self.cable_capacity_col
        keys = ["NODE1_value", "NODE2_value"]

        # Per-group max capacity
        grp_max = topology.groupby(keys)[col].max().rename("max_cap").to_frame()

        # Keep only rows with max capacity per group
        topology = topology.merge(grp_max, left_on=keys, right_index=True, how="left")
        topology = topology[topology[col] == topology["max_cap"]]
        topology = topology.drop(columns=["max_cap"]).reset_index(drop=True)

        # If ties exist (multiple rows share the same max), pick one per group
        # (first encountered)
        topology = topology.drop_duplicates(subset=keys, keep="first")

        # Drop helper columns to maintain same data structure as raw files
        topology = topology.drop(columns=["NODE1_value", "NODE2_value", "Substation"])
        topology = topology.reset_index(drop=True)

        return topology

    def clean_topology_pandas(self, topology: pd.DataFrame) -> pd.DataFrame:
        # TODO: Remove this method later (included for testing)...
        # Add a numerical value for the node number and substation number
        # #TKJ: Adding helper columns for grouping
        topology["NODE1_value"] = topology[self.node_1_col].apply(lambda x: x.split(".")[1] if pd.notna(x) else pd.NA)
        topology["NODE2_value"] = topology[self.node_2_col].apply(lambda x: x.split(".")[1] if pd.notna(x) else pd.NA)
        topology["Substation"] = topology[self.sec_substation_col].apply(
            lambda x: x.split(".")[1] if pd.notna(x) else pd.NA
        )

        # Cleaning step 1) Fill nan with the mean of the respective substation or
        # the global mean if all values of substation are nan
        # Make a list of numerical columns, loop through them
        cols: list[str] = [
            self.cable_capacity_col,
            self.cable_length_col,
            self.phase_size_col,
            self.resistance_col,
            self.reactance_col,
            self.lv_feeder_fuse_size_col,
        ]

        topology[cols] = topology[cols].astype(
            "float32"
        )  # TODO: Use the self.topology_dtypes to determine dtype instead of
        # hardcoding float32
        per_sub_means: pd.Series = topology.groupby("Substation")[cols].transform("mean").round(0).astype("float32")
        mean_before_fills: pd.Series = (
            topology[cols].mean().round(0).astype("float32")
        )  # TODO: Rounding to zero decimals is in line with Niels' code, but
        # could be reconsidered, especially for resistance and reactance, for
        # which all data points will be rounded to 0

        for col in cols:
            topology[col] = topology[col].fillna(per_sub_means[col])
            topology[col] = topology[col].fillna(mean_before_fills[col])

        # Keep categorial columns as categorical, as opposed to casting to string
        # Make a dict of categorical columns holding lists of categories as
        # items. Loop through them
        categories = {
            self.cable_type_col: ["APB", "MAL", "PEX", "PEX ENLEDER", "PEXMAL", "PSP", "U"],
            self.phase_material_col: ["AL", "CU", "U"],
        }

        for col, cats in categories.items():
            topology[col] = topology[col].astype("category").astype(pd.CategoricalDtype(categories=cats))
            topology[col] = topology[col].fillna(
                topology.groupby("Substation")[col].transform(lambda x: x.mode()[0] if not x.mode().empty else pd.NA)
            )
            topology[col] = topology[col].fillna(topology[col].mode()[0])

        # Cleaning step 3) remove duplicated lines, take the one with the highest
        # current carrying capacity
        max_capacity_per_group = topology.groupby(["NODE1_value", "NODE2_value"])[self.cable_capacity_col].transform(
            "max"
        )
        topology = topology[topology[self.cable_capacity_col] == max_capacity_per_group]
        topology = topology.drop_duplicates(subset=["NODE1_value", "NODE2_value"], keep="first")
        topology = topology.reset_index(drop=True)

        # Drop helper columns to maintain same data structure as raw files
        topology.drop(columns=["NODE1_value", "NODE2_value", "Substation"], inplace=True)

        return topology

    def clean_meter_and_cabinet_connection_dask(self, mtr_cbnet_connection_df: dd.DataFrame) -> dd.DataFrame:
        # Remove meters and cabinets where the meter number is NaN (cabinet has
        # no meter attached)
        return mtr_cbnet_connection_df.dropna(subset=[self.meter_number_col])

    def clean_meter_and_cabinet_connection_pandas(self, mtr_cbnet_connection_df: pd.DataFrame) -> pd.DataFrame:
        # TODO: Remove this method later (included for testing)...
        # Remove meters and cabinets where the meter number is NaN (cabinet has
        # no meter attached)
        return mtr_cbnet_connection_df[~mtr_cbnet_connection_df[self.meter_number_col].isna()]

    def correct_topology(self, topology_df: dask.dataframe.DataFrame) -> dask.dataframe.DataFrame:
        pass


if __name__ == "__main__":
    import logging
    import time
    from contextlib import contextmanager
    from pathlib import Path

    from dask.distributed import Client, LocalCluster
    from pandas.testing import assert_frame_equal

    @contextmanager
    def timed(label: str):
        start = time.perf_counter()
        try:
            yield
        finally:
            print(f"{label}: {time.perf_counter() - start:.3f}s")

    logger = logging.getLogger("distributed.shuffle._scheduler_plugin")
    std_log_lvl = logger.level

    cluster = LocalCluster(n_workers=2, threads_per_worker=1, memory_limit="1GB")
    client = Client(cluster)
    print("Dask dashboard link:", client.dashboard_link)

    try:
        # suppress Dask warnings about shuffling of categorical dtypes during groupby operations
        logger.setLevel(logging.ERROR)

        # Example usage, local paths for Tian's machine
        rel_dir = Path(__file__).parent.parent.parent.parent / "LOCAL" / "data"
        topology_cleaner = TopologyCleaner(
            path_to_topology_file=rel_dir / "topology.csv",
            path_to_meter_cabinet_connection_file=rel_dir / "meter_cabinet_connection.csv",
        )

        # load raw data from csv files to Dask/Pandas DataFrames
        print("Loading CSV files to Dask DataFrames...", end="")
        mtr_cbnet_connection_df: dd.DataFrame = topology_cleaner.load_mtr_cbnt_connection_csv_to_df()
        topology_df: dd.DataFrame = topology_cleaner.load_topology_csv_to_df()
        print("done")
        print("Loading CSV files to Pandas DataFrames...", end="")
        mtr_cbnet_connection_pd: pd.DataFrame = (
            topology_cleaner.load_mtr_cbnt_connection_csv_to_pandas()
        )  # TODO: Remove this later (included for testing)...
        topology_pd: pd.DataFrame = (
            topology_cleaner.load_topology_csv_to_pandas()
        )  # TODO: Remove this method later (included for testing)...
        print("done")

        # clean loaded data
        print("Cleaning data using Dask and shuffling...", end="")
        with timed("Elapsed clock time"):
            mtr_cbnet_connection_cleaned_dask: dd.DataFrame = topology_cleaner.clean_meter_and_cabinet_connection_dask(
                mtr_cbnet_connection_df
            )
            topology_cleaned_dask: dd.DataFrame = topology_cleaner.clean_topology_dask(topology_df)
            print("done. ", end="")

        print("Cleaning data using Pandas (Niels' implementation)...", end="")
        with timed("Elapsed clock time"):
            mtr_cbnet_connection_cleaned_pandas: pd.DataFrame = (
                topology_cleaner.clean_meter_and_cabinet_connection_pandas(mtr_cbnet_connection_pd)
            )
            topology_cleaned_pandas: pd.DataFrame = topology_cleaner.clean_topology_pandas(topology_pd)
            print("done. ", end="")

        # Compare results from cleaning using Dask and Pandas
        print("Comparing results from Dask and Pandas cleaning...", end="")
        try:
            assert_frame_equal(
                mtr_cbnet_connection_cleaned_dask.compute().reset_index(drop=True),
                mtr_cbnet_connection_cleaned_pandas.reset_index(drop=True),
                check_dtype=False,
                check_like=True,
            )
            assert_frame_equal(
                topology_cleaned_dask.compute().reset_index(drop=True),
                topology_cleaned_pandas.reset_index(drop=True),
                check_dtype=False,
                check_like=True,
            )
        except AssertionError as e:
            print("done: FAILURE! Results differ between Dask and Pandas cleaning!")
            print(e)
            raise
        print("done: SUCCESS! Results are identical between Dask and Pandas cleaning.")

        # Write CSVs to desk to inspect results, make directory if it doesn't exist
        print("Writing cleaned data to CSV files...", end="")
        output_dir = rel_dir / "test_result_data"
        output_dir.mkdir(parents=True, exist_ok=True)

        # Compute Dask DataFrames to Pandas and save as single CSV files
        mtr_cbnet_connection_cleaned_dask.compute().to_csv(output_dir / "mtr_cbnet_connection_df.csv", index=False)
        topology_cleaned_dask.compute().to_csv(output_dir / "topology_df.csv", index=False)
        topology_cleaned_pandas.to_csv(output_dir / "topology_pd_df.csv", index=False)

        print(f"done. \nWrote CSVs to folder: {output_dir}")

    finally:  # Always close the client and cluster
        logger.setLevel(std_log_lvl)  # restore original logging level
        client.close()
        cluster.close()
