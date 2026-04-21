import logging
import os
import sys

import matplotlib
import numpy as np
import pandas as pd

from threephi_framework.data_apps.base import BaseDataApp
from threephi_framework.dtu.sm_classifier import meter_evaluation, save_sm_plot

matplotlib.use("Agg")
import json
from time import sleep, time

from dask import compute, delayed
from tqdm import tqdm

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../")))


class SMClassifier(BaseDataApp):
    """
    Smart Meter Classification Data App.

    Evaluates each smart meter's time series data to produce a per-meter
    characterization (data quality, statistics, connectivity) and a
    summary classification across the full meter population.

    The following config parameters are supported:

    - ``dataset_root_path`` (str): Root path of the time series dataset in object storage.
    - ``Data_batch`` (str): Batch identifier controlling column naming conventions.
      Use ``"first_batch"`` for datasets with uppercase column names (``METER_NUMBER``,
      ``VOLTAGE_``, etc.); any other value assumes lowercase names.
    - ``use_dask`` (bool): If ``True``, distribute meter evaluation across Dask workers.
    - ``sm_ids`` (str | list[str]): ``"All"`` to classify every meter with data, or an
      explicit list of meter ID strings.
    - ``run_name`` (str): Label for this run; used as the output directory name.
    - ``save_results`` (bool): Persist characterization and classification dicts to JSON.
    - ``save_plots`` (bool): Save per-meter time series plots to object storage.
    - ``plot_cfg`` (dict): Plot configuration including ``Variable_selection``,
      ``Phase_selection``, ``SM_selection``, and ``Plotting_format``.
    - ``no_data_limit`` (float): Fraction of max recording period below which a phase is
      considered unconnected. Defaults to ``0.025``.
    - ``good_data_limit`` (float): Corruption fraction threshold for "Good" quality.
      Defaults to ``0.1``.
    - ``medium_data_limit`` (float): Corruption fraction threshold for "Medium" quality.
      Defaults to ``0.5``.
    - ``v_lim`` (float): Voltage threshold (V) below which a measurement is considered
      implausible. Defaults to ``207``.
    - ``offset_threshold`` (float): Fraction of plausible measurements below ``v_lim``
      that triggers a connection error. Defaults to ``0.95``.
    - ``cons_period_threshold`` (int): Minimum consecutive time steps to count as a
      sustained on or off period when detecting switching. Defaults to ``192`` (2 days
      at 15-minute resolution).
    - ``frozen_range`` (int): Number of consecutive identical values that constitute a
      frozen signal. Defaults to ``12``.

    Usage::

        with SMClassifier(config) as app:
            app.run()
    """

    # Some variables for plotting
    DIR_DATA_APP = os.path.dirname(os.path.abspath(__file__))
    ALLOWED_VARIABLES = {"V", "P14", "P23", "Q12", "Q34"}
    ALLOWED_PHASES = {"L1", "L2", "L3"}

    def __init__(self, config):
        # Run BaseDataApp Init
        super().__init__(config=config)

        # Unpack the config
        self.dataset_root_path = config.get("dataset_root_path")
        self.batch = config.get("Data_batch")
        self.use_dask = config.get("use_dask")
        # self.n_workers = config["Cluster_settings"]["n_workers"]
        self.sm_ids = config.get("sm_ids", "All")
        self.topology_processing_level = config.get("topology_processing_level", None)
        self.overwrite_existing_raw_sm_datasets = config.get("overwrite_existing_raw_sm_datasets", False)
        self.overwrite_topology_info = config.get("overwrite_topology_info", False)
        self.overwrite_timeseries_info = config.get("overwrite_timeseries_info", False)
        self.run_name = config.get("run_name", str(int(time())))
        self.save_results = config.get("save_results", False)
        self.save_plots = config.get("save_plots", False)
        self.plot_cfg = config.get("plot_cfg", None)
        # Limit defining what counts as having "no data"
        # Fraction of total dataset of the longest recorded period of all SMs
        self.no_data_limit = config.get("no_data_limit", 0.025)
        # Limits defining what is "good", "medium", "bad"
        self.good_data_limit = config.get("good_data_limit", 0.1)
        # Limits defining what is "good", "medium", "bad"
        self.medium_data_limit = config.get("medium_data_limit", 0.5)
        # lower voltage limit for "plausible" measurements
        self.v_lim = config.get("v_lim", 207)
        # Fraction of total data which has to be below v_lim to be considered offset data
        self.offset_threshold = config.get("offset_threshold", 0.95)
        # Length of constant values from which on a period is considered constant period
        self.cons_period_threshold = config.get("cons_period_threshold", 4 * 24 * 2)
        # Range of consecutive values that have to be identical to consider it as frozen
        self.frozen_range = config.get("frozen_range", 3 * 4)
        # Plot Config# get plotting settings
        self.plot_format = config["plot_cfg"]["Plotting_format"]
        self.format = self.plot_format["plot_format"]
        self.dpi = self.plot_format["plot_dpi"]
        self.transparent = self.plot_format["plot_transparent"]
        self.overwrite_plots = self.plot_format["overwrite_plots"]

        # Placeholder for some class attributes that will be assigned later
        self.selected_variables = None
        self.selected_phases = None
        self.sm_selection = None
        self.max_rec_period = None
        self.SM_topology_mapping = None
        self.dir_of_current_run_results = None

        # Define correct file and column names depending on the data batch
        if self.batch == "first_batch":
            self.meter_number_col = "METER_NUMBER"
            self.voltage_col = "VOLTAGE_"
            self.active_power_p14_col = "ACTIVE_POWER_P14_"
            self.active_power_p23_col = "ACTIVE_POWER_P23_"
            self.reactive_power_q12_col = "REACTIVE_POWER_Q12_"
            self.reactive_power_q34_col = "REACTIVE_POWER_Q34_"
            self.metercabinet_file = "MeterAndCabinet.csv"
            self.cabinet_col = "CABINET"

        else:
            self.meter_number_col = "meter_number"
            self.voltage_col = "voltage_"
            self.active_power_p14_col = "active_power_p14_"
            self.active_power_p23_col = "active_power_p23_"
            self.reactive_power_q12_col = "reactive_power_q12_"
            self.reactive_power_q34_col = "reactive_power_q34_"
            self.metercabinet_file = "meter_cabinet_connection.csv"
            self.cabinet_col = "cabinet"

        # Define data type dicts for data loading
        self.meter_and_cabinet_dtypes = {
            self.meter_number_col: pd.StringDtype(),
            self.cabinet_col: pd.StringDtype(),
            **(
                {
                    "delivery_point_id": pd.StringDtype(),
                    "lv_feeder": pd.StringDtype(),
                    "has_heat_pump": pd.BooleanDtype(),
                    "has_solar_panel": pd.BooleanDtype(),
                    "capacity_solar_panel": pd.Float32Dtype(),
                    "service_fuse_size": pd.Float32Dtype(),
                }
                if self.batch == "second_batch" or self.batch == "third_batch"
                else {}
            ),
        }

    def _load_raw_data_of_single_sm(self, sm_id: str, save: bool) -> pd.DataFrame:
        data = self.time_series_controller.get_time_series_data([sm_id], self.dataset_root_path).compute()
        return data

    @staticmethod
    def _populate_result_summary(result_summary, sm_id, sm_id_results):
        # Append current sm_id to list of all SM IDs
        result_summary["All_SMs"].append(sm_id)

        # Append sm_id if has dataset and dataset contains data
        if (
            sm_id_results["Dataset Availability"]["Available"]
            and sm_id_results["Dataset Availability"]["Contains Data"]
        ):
            result_summary["SMs_with_dataset_containing_data"].append(sm_id)

        # Appends sm_id if has dataset but dataset contains no data
        if (
            sm_id_results["Dataset Availability"]["Available"]
            and not sm_id_results["Dataset Availability"]["Contains Data"]
        ):
            result_summary["SMs_with_dataset_containing_no_data"].append(sm_id)

        # Append if sm_id has no dataset
        if not sm_id_results["Dataset Availability"]["Available"]:
            result_summary["SMs_without_dataset"].append(sm_id)

        # Append if sm_id has incomplete topology info (e.g., no associated trafo id)
        if any(pd.isna(v) for v in sm_id_results["Topology"].values()):
            result_summary["SMs_with_incomplete_topology_info"].append(sm_id)

        # Check the data quality of all variables of current sm_id
        all_good = True
        any_medium_or_bad = False
        any_bad = False
        for _phase, phase_data in sm_id_results["Data Quality"].items():
            for _key, entry in phase_data.items():
                if isinstance(entry, dict) and "Summary" in entry:
                    summary_value = entry["Summary"]
                    if summary_value == "Bad":
                        any_bad = True
                        any_medium_or_bad = True
                    elif summary_value == "Medium":
                        any_medium_or_bad = True
                    elif summary_value != "Good":
                        all_good = False

        # Append if sm_id has only good data quality in all variables (V, ...)
        if all_good:
            result_summary["SMs_with_only_good_data_quality"].append(sm_id)

        # Append if sm_id has any variables (V, ...) with medium or bad data quality
        if any_medium_or_bad:
            result_summary["SMs_with_any_medium_or_bad_data_quality"].append(sm_id)

        # Append if sm_id has any variables (V, ...) with bad data quality
        if any_bad:
            result_summary["SMs_with_any_bad_data_quality"].append(sm_id)

        # Check how many phases are connected for current sm_id and append accordingly
        if isinstance(sm_id_results["Connectivity"]["Connected Phases"], list):
            if len(sm_id_results["Connectivity"]["Connected Phases"]) == 3:
                result_summary["SMs_with_3-phase_connection"].append(sm_id)
            if len(sm_id_results["Connectivity"]["Connected Phases"]) == 2:
                result_summary["SMs_with_2-phase_connection"].append(sm_id)
            if len(sm_id_results["Connectivity"]["Connected Phases"]) == 1:
                result_summary["SMs_with_1-phase_connection"].append(sm_id)

        # Check if any phase of sm_id has connection error (value constantly < v_lim) and then append
        if (
            isinstance(sm_id_results["Connectivity"]["Connection Error"], list)
            and len(sm_id_results["Connectivity"]["Connection Error"]) > 0
        ):
            result_summary["SMs_with_connection_error"].append(sm_id)

        # Check if any phase of sm_id has On/Off switch and then append
        if (
            isinstance(sm_id_results["Connectivity"]["Switching Phases"], list)
            and len(sm_id_results["Connectivity"]["Switching Phases"]) > 0
        ):
            result_summary["SMs_with_on_off_switch"].append(sm_id)

        return result_summary

    @staticmethod
    def progress_bar(iterable, total=None, prefix="", suffix="", length=50):
        total = total or len(iterable)
        for i, item in enumerate(iterable, start=1):
            percent = i / total
            filled_length = int(length * percent)
            bar = "█" * filled_length + "-" * (length - filled_length)
            print(f"\r{prefix} |{bar}| {i}/{total} {suffix}", end="", flush=True)
            yield item  # Yield the current item to the loop
        print()  # Newline after completion

    # Method to update config settings via the method arguments
    def _update_config(self, args):
        for arg_name, arg_value in args:
            if arg_name != "self" and arg_value is not None:
                setattr(self, arg_name, arg_value)

    def classify_smart_meters(
        self,
        sm_ids: str | list[str] = None,
        topology_processing_level: str = None,
        overwrite_existing_raw_sm_datasets: bool = None,
        overwrite_topology_info: bool = None,
        overwrite_timeseries_info: bool = None,
        run_name: str = None,
        save_results: bool = None,
        save_plots: bool = None,
        plot_cfg: dict = None,
    ) -> tuple:
        # Overwrite config settings with arguments if provided (allows to dynamically change data app run in pipeline)
        self._update_config(args=locals().items())

        # Check and store user settings
        if not (
            isinstance(self.sm_ids, str)
            or (isinstance(self.sm_ids, list) and all(isinstance(i, str) for i in self.sm_ids))
        ):
            raise TypeError("sm_ids must be set 'All' or user-specified SM IDs as string or list of strings.")

        if not isinstance(self.run_name, str):
            raise TypeError("run_name must be a string.")

        if sum(self.plot_cfg["SM_selection"].values()) == 0:
            raise ValueError("At least one 'SM_selection' option in the plot_cfg must be set to True.")
        if not set(var.upper() for var in self.plot_cfg["Variable_selection"]).issubset(self.ALLOWED_VARIABLES):
            raise ValueError(
                f"'Variable_selection' in the plot_cfg contains invalid entries. Allowed: {self.ALLOWED_VARIABLES}"
            )
        if not set(self.plot_cfg["Variable_selection"]):
            raise ValueError("At least one variable must be selected in 'Variable_selection' of the plot_cfg.")
        if not set(var.upper() for var in self.plot_cfg["Phase_selection"]).issubset(self.ALLOWED_PHASES):
            raise ValueError(
                f"'Phase_selection' in the plot_cfg contains invalid entries. Allowed entries: {self.ALLOWED_PHASES}"
            )
        if not set(self.plot_cfg["Phase_selection"]):
            raise ValueError("At least one phase must be selected in 'Phase_selection' of the plot_cfg.")

        # Check if the results file already exists
        self.dir_of_current_run_results = os.path.join(self.DIR_DATA_APP, "Results", f"{self.run_name}")

        if os.path.exists(self.dir_of_current_run_results):
            logging.info(f"SM classification run {self.run_name} exists. Loading and returning existing files...")

            with open(os.path.join(self.dir_of_current_run_results, f"{self.run_name}_SM_characterization.json")) as f:
                sm_characterization = json.load(f)

            with open(os.path.join(self.dir_of_current_run_results, f"{self.run_name}_SM_classification.json")) as f:
                sm_classification = json.load(f)

            return sm_characterization, sm_classification

        logging.info(f"Classifying Smart Meters based on {self.topology_processing_level} topology")

        # TODO: Add way to change between topology levels

        # Define list of SM IDs to be evaluated based on user settings
        SM_IDs = (
            self.meta_controller.get_time_series_meta_info()["id_list_of_sms_with_data"]
            if self.sm_ids == "All"
            else self.sm_ids
        )
        SM_IDs = [str(id) for id in SM_IDs]

        # Initialize plotting
        if self.plot_cfg is not None:
            self.selected_variables = [var.upper() for var in self.plot_cfg["Variable_selection"]]
            if self.batch == "first_batch":
                self.selected_phases = [var.upper() for var in self.plot_cfg["Phase_selection"]]
            else:
                self.selected_phases = [var.lower() for var in self.plot_cfg["Phase_selection"]]

        phases = ["L1", "L2", "L3"] if self.batch == "first_batch" else ["l1", "l2", "l3"]
        variables = (
            ["V", "P14", "P23", "Q12", "Q34"] if self.batch == "first_batch" else ["v", "p14", "p23", "q12", "q34"]
        )

        def _meter_evaluation(sm_ids):
            # Initialize detailed result dict
            sm_classification_chunk = {}

            # Initialize result summary dict
            result_summary_chunk = {
                "All_SMs": [],
                "SMs_with_dataset_containing_data": [],
                "SMs_with_dataset_containing_no_data": [],
                "SMs_without_dataset": [],
                "SMs_with_incomplete_topology_info": [],
                "SMs_with_only_good_data_quality": [],
                "SMs_with_any_medium_or_bad_data_quality": [],
                "SMs_with_any_bad_data_quality": [],
                "SMs_with_3-phase_connection": [],
                "SMs_with_2-phase_connection": [],
                "SMs_with_1-phase_connection": [],
                "SMs_with_connection_error": [],
                "SMs_with_on_off_switch": [],
            }

            # Loop over the list of SM IDs, create results of that SM,
            # and add it to the total detailed and summary result dict
            for sm_id in tqdm(
                sm_ids, desc="Classifying smart meters"
            ):  # self.progress_bar(sm_ids, prefix='Processing Smart Meters', suffix='Done', length=40):
                # Initialize results dict for current sm_id
                sm_id_results = {
                    "Topology": {
                        "Secondary Substation ID": None,
                        "Transformer ID": None,
                        "Feeder ID": None,
                        "Cabinet ID": None,
                    },
                    "Dataset Availability": {
                        "Available": False,
                        "Contains Data": None,
                        "Relative Length": None,
                        "Absolute Length": None,
                    },
                    "Data Quality": {
                        f"L{p}": {
                            "V": {
                                "Summary": None,
                                "Detailed": {
                                    "NaN frac": None,
                                    "Zero frac": None,
                                    "Below Vlim frac": None,
                                    "Frozen frac": None,
                                    "Total corruption frac": None,
                                },
                            },
                            "P14": {"Summary": None, "Detailed": {"NaN frac": None}},
                            "P23": {"Summary": None, "Detailed": {"NaN frac": None}},
                            "Q12": {"Summary": None, "Detailed": {"NaN frac": None}},
                            "Q34": {"Summary": None, "Detailed": {"NaN frac": None}},
                        }
                        for p in [1, 2, 3]
                    },
                    "Data Statistics": {
                        f"L{p}": {
                            "V": {"Min": None, "Max": None, "Mean": None, "Std": None},
                            "P14": {"Min": None, "Max": None, "Mean": None, "Std": None},
                            "P23": {"Min": None, "Max": None, "Mean": None, "Std": None},
                            "Q12": {"Min": None, "Max": None, "Mean": None, "Std": None},
                            "Q34": {"Min": None, "Max": None, "Mean": None, "Std": None},
                        }
                        for p in [1, 2, 3]
                    },
                    "Connectivity": {"Connected Phases": None, "Connection Error": None, "Switching Phases": None},
                }

                # Add the topology info to the results dict of current sm_id
                topology = self.SM_topology_mapping.get(
                    sm_id,
                    {
                        "Cabinet ID": np.nan,
                        "Feeder ID": np.nan,
                        "Transformer ID": np.nan,
                        "Secondary Substation ID": np.nan,
                    },
                )

                sm_id_results["Topology"] = {
                    "Secondary Substation ID": topology["Secondary Substation ID"],
                    "Transformer ID": topology["Transformer ID"],
                    "Feeder ID": topology["Feeder ID"],
                    "Cabinet ID": topology["Cabinet ID"],
                }

                sm_df = None
                # If there is a dataset for the current sm_id, evaluate it to get the remaining results
                if sm_id in SM_IDs:
                    # Load dataset for current sm_id and save it if it doesn't exist yet
                    sm_df = self.time_series_controller.get_time_series_data([sm_id], self.dataset_root_path)

                    # Get maximum recording period
                    # (sm_df all are of this length and just have nan in beginning and end)
                    self.max_rec_period = len(sm_df) if self.max_rec_period is None else self.max_rec_period

                    # Add the dataset availability information to the results dict of current sm_id
                    sm_id_results["Dataset Availability"]["Available"] = True

                    # Check if the dataset contains any meaningful data
                    if sm_df.empty or (sm_df.isna().all(axis=1)).all() or (sm_df.eq(0).all(axis=1)).all():
                        sm_id_results["Dataset Availability"]["Contains Data"] = False
                    else:
                        sm_id_results["Dataset Availability"]["Contains Data"] = True

                    # Add data length info to Dataset Availability
                    has_data_full_set = sm_df.notna() & (sm_df != 0)
                    has_data_full_set = has_data_full_set.any(axis=1)
                    last_valid_idx = sm_df.index.get_loc(has_data_full_set[::-1].idxmax())
                    first_valid_idx = sm_df.index.get_loc(has_data_full_set.idxmax())
                    length = last_valid_idx - first_valid_idx + 1
                    sm_id_results["Dataset Availability"]["Relative Length"] = length / self.max_rec_period
                    sm_id_results["Dataset Availability"]["Absolute Length"] = length

                    # Extract part of the dataset which contains data
                    # (required for data quality assessment further down)
                    sm_df_with_data = sm_df.iloc[first_valid_idx : last_valid_idx + 1]

                    # Calculate nan fractions for each of the variables of the current sm_id
                    # (required for data quality assess.)
                    nan_fractions = sm_df_with_data.isna().mean()

                    # Overwrite nan in Connectivity with empty list since data is available.
                    # Lists are populated further down
                    sm_id_results["Connectivity"]["Connected Phases"] = []
                    sm_id_results["Connectivity"]["Connection Error"] = []
                    sm_id_results["Connectivity"]["Switching Phases"] = []

                    for phase in phases:
                        # Check data quality for V
                        is_nan = sm_df_with_data[
                            f"{self.voltage_col}{phase}_{sm_id}"
                        ].isna()  # which entries are nan (True/False)
                        is_zero = (sm_df_with_data[f"{self.voltage_col}{phase}_{sm_id}"] == 0).fillna(
                            False
                        )  # which entries are 0 (True/False)
                        is_below_vlim = (sm_df_with_data[f"{self.voltage_col}{phase}_{sm_id}"] < self.v_lim).fillna(
                            False
                        )  # which entries are below v_lim
                        is_frozen = sm_df_with_data[f"{self.voltage_col}{phase}_{sm_id}"] == sm_df_with_data[
                            f"{self.voltage_col}{phase}_{sm_id}"
                        ].shift(1)
                        for i in range(
                            2, self.frozen_range
                        ):  # which entries are frozen for frozen_range consecutive entries
                            is_frozen &= sm_df_with_data[f"{self.voltage_col}{phase}_{sm_id}"] == sm_df_with_data[
                                f"{self.voltage_col}{phase}_{sm_id}"
                            ].shift(i)
                        is_frozen = is_frozen.fillna(False)
                        # which entries are nan,0,below lim or frozen
                        is_corrupted = is_nan | is_zero | is_below_vlim | is_frozen
                        # which entries are NOT nan and NOT 0 and NOT frozen
                        has_data = ~is_nan & ~is_zero & ~is_frozen
                        offset_data = (
                            is_below_vlim & has_data
                        )  # which entries are NOT nan and NOT 0 and NOT frozen and below lim

                        # Data quality assessment for V based on total corruption level
                        if is_corrupted.mean() < self.good_data_limit:
                            summary = "Good"
                        elif self.medium_data_limit > is_corrupted.mean() >= self.good_data_limit:
                            summary = "Medium"
                        else:
                            summary = "Bad"

                        # Overwrite summary if very little data,
                        # which is considered as "not connected" instead of bad quality
                        if has_data.sum() < self.no_data_limit * self.max_rec_period:
                            summary = "NaN"

                        # Add detailed and summarized data quality info to results dict of current sm_id
                        sm_id_results["Data Quality"][phase.upper()]["V"]["Detailed"]["NaN frac"] = float(is_nan.mean())
                        sm_id_results["Data Quality"][phase.upper()]["V"]["Detailed"]["Zero frac"] = float(
                            is_zero.mean()
                        )
                        sm_id_results["Data Quality"][phase.upper()]["V"]["Detailed"]["Below Vlim frac"] = float(
                            is_below_vlim.mean()
                        )
                        sm_id_results["Data Quality"][phase.upper()]["V"]["Detailed"]["Frozen frac"] = float(
                            is_frozen.mean()
                        )
                        sm_id_results["Data Quality"][phase.upper()]["V"]["Detailed"]["Total corruption frac"] = float(
                            is_corrupted.mean()
                        )
                        sm_id_results["Data Quality"][phase.upper()]["V"]["Summary"] = summary

                        # Check data quality for P14, P23, Q12, Q34 (which is only nan frac at the moment)
                        for variable in variables[1:]:
                            # Extract the associated column name of current variable
                            col_name = [col for col in sm_df_with_data.columns if (phase in col) and (variable in col)][
                                0
                            ]

                            # Get the nan fraction for current variable from nan_fractions calculated above
                            nan_frac = nan_fractions[col_name]

                            # Summarize the data quality of current variable based on nan fraction
                            if nan_frac < self.good_data_limit:
                                summary = "Good"
                            elif self.medium_data_limit > nan_frac >= self.good_data_limit:
                                summary = "Medium"
                            else:
                                summary = "Bad"

                            # Overwrite summary if very little data,
                            # which is considered as "not connected" instead of bad quality
                            if has_data.sum() < self.no_data_limit * self.max_rec_period:
                                summary = "NaN"

                            # Add detailed and summarized data quality info to result dict of current sm_id
                            sm_id_results["Data Quality"][phase.upper()][variable.upper()]["Detailed"]["NaN frac"] = (
                                float(nan_frac)
                            )
                            sm_id_results["Data Quality"][phase.upper()][variable.upper()]["Summary"] = summary

                        # Add the connectivity information to the results dict of current sm_id
                        if has_data.sum() > self.no_data_limit * self.max_rec_period:
                            # Check which phases are connected
                            sm_id_results["Connectivity"]["Connected Phases"].append(phase.upper())

                            # Check which phases are poorly connected and show constant voltage offset
                            if offset_data.mean() / has_data.mean() > self.offset_threshold:
                                sm_id_results["Connectivity"]["Connection Error"].append(phase.upper())

                            # Check which phases do switch On/Off
                            has_data_pd = pd.Series(has_data)
                            has_no_data_pd = pd.Series(is_zero)
                            consecutive_off = (
                                has_no_data_pd.astype(int)
                                .groupby((has_no_data_pd != has_no_data_pd.shift()).cumsum())
                                .transform("sum")
                            )
                            consecutive_on = (
                                has_data_pd.astype(int)
                                .groupby((has_data_pd != has_data_pd.shift()).cumsum())
                                .transform("sum")
                            )
                            if (consecutive_on >= self.cons_period_threshold).any() & (
                                consecutive_off >= self.cons_period_threshold
                            ).any():
                                sm_id_results["Connectivity"]["Switching Phases"].append(phase.upper())

                        # Add data statistics for P14, P23, Q12, Q34 and V
                        for variable in variables:
                            col_name = [col for col in sm_df_with_data.columns if (phase in col) and (variable in col)][
                                0
                            ]
                            sm_id_results["Data Statistics"][phase.upper()][variable.upper()]["Min"] = (
                                float(val) if pd.notna(val := sm_df_with_data[col_name].min(skipna=True)) else None
                            )
                            sm_id_results["Data Statistics"][phase.upper()][variable.upper()]["Max"] = (
                                float(val) if pd.notna(val := sm_df_with_data[col_name].max(skipna=True)) else None
                            )
                            sm_id_results["Data Statistics"][phase.upper()][variable.upper()]["Mean"] = (
                                float(val) if pd.notna(val := sm_df_with_data[col_name].mean(skipna=True)) else None
                            )
                            sm_id_results["Data Statistics"][phase.upper()][variable.upper()]["Std"] = (
                                float(val) if pd.notna(val := sm_df_with_data[col_name].std(skipna=True)) else None
                            )

                # Add results of current sm_id to the result summary dict
                result_summary_chunk = self._populate_result_summary(result_summary_chunk, sm_id, sm_id_results)  # TODO

                # Save plot on request (with option to create plots only for certain criteria like has connection error)
                if self.save_plots and sm_id in SM_IDs:
                    save_sm_plot(sm_id, sm_df, result_summary_chunk, self.config)

                # Add results of current sm_id to final detailed sm_classification results dict
                sm_classification_chunk[str(sm_id)] = sm_id_results

            return result_summary_chunk, sm_classification_chunk

        if self.use_dask:
            sm_id_chunks = np.array_split(SM_IDs, min(len(SM_IDs), self.config["dask"]["n_workers"]))
            delayed_tasks = [
                delayed(meter_evaluation)(sm_ids=sm_ids_chunk, cfg=self.config) for sm_ids_chunk in sm_id_chunks
            ]
            results_list = compute(*delayed_tasks)

            logging.info(results_list)
            result_summary_list = []
            sm_classification_list = []
            for res in results_list:
                if res:
                    result_summary_list.append(res[0])
                    sm_classification_list.append(res[1])

            sm_classification = {k: v for d in sm_classification_list for k, v in d.items()}

            result_summary = {}
            if result_summary_list:
                result_summary = {k: [] for k in result_summary_list[0]}
            for d in result_summary_list:
                for k in result_summary:
                    result_summary[k].extend(d[k])
            sleep(1)
        else:
            result_summary, sm_classification = _meter_evaluation(sm_ids=SM_IDs)

        # Save detailed and summarized SM classification results if selected
        if self.save_results:
            os.makedirs(self.dir_of_current_run_results, exist_ok=True)

            with open(
                os.path.join(self.dir_of_current_run_results, f"{self.run_name}_SM_characterization.json"), "w"
            ) as f:
                json.dump(sm_classification, f)

            with open(
                os.path.join(self.dir_of_current_run_results, f"{self.run_name}_SM_classification.json"), "w"
            ) as f:
                json.dump(result_summary, f)

        return sm_classification, result_summary

    def run(self):
        sm_characterization, sm_classification = self.classify_smart_meters(run_name=self.config["run_name"])
        logging.info("sm_characterization:")
        logging.info(sm_characterization)
        logging.info("sm_classification")
        logging.info(sm_classification)
        logging.info("SM Classifier run completed.")


if __name__ == "__main__":
    config = {
        "Name": "sm_classifier_config",
        "use_dask": True,
        "dask": {
            "local": True,
            "n_workers": 6,
        },
        "run_name": "Tester_17_02_26",
        "data_dir_path": "phase_measurements/raw",
        "sm_ids": "All",
        "topology_processing_level": "raw",
        "overwrite_existing_raw_sm_datasets": False,
        "overwrite_topology_info": False,
        "overwrite_timeseries_info": False,
        "save_results": True,
        "save_plots": True,
        "plot_cfg": {
            "SM_selection": {
                "All_(with_dataset_containing_data)": True,
                "With_only_good_data_quality": True,
                "With_any_medium_or_bad_data_quality": True,
                "With_1-phase_connection": True,
                "With_2-phase_connection": True,
                "With_connection_error": True,
                "With_on_off_switch": True,
            },
            "Variable_selection": ["V", "P14", "P23", "Q12", "Q34"],
            "Phase_selection": ["L1", "L2", "L3"],
            "voltage_col": "voltage_",
            "Plotting_format": {
                "plot_format": "svg",
                "plot_dpi": 300,
                "plot_transparent": False,
                "overwrite_plots": True,
            },
        },
        "no_data_limit": 0.025,
        "good_data_limit": 0.1,
        "medium_data_limit": 0.5,
        "v_lim": 207,
        "offset_threshold": 0.95,
        "cons_period_threshold": 192,
        "frozen_range": 12,
        "max_rec_period": 1000,
        "phases": ["l1", "l2", "l3"],
        "variables": ["v", "p14", "p23", "q12", "q34"],
    }

    config["selected_variables"] = (
        [var.upper() for var in config["plot_cfg"]["Variable_selection"]] if config["plot_cfg"] is not None else None,
    )
    config["selected_phases"] = (
        [var.lower() for var in config["plot_cfg"]["Phase_selection"]] if config["plot_cfg"] is not None else None,
    )
    config["voltage_col"] = (config["plot_cfg"]["voltage_col"] if config["plot_cfg"] is not None else None,)

    with SMClassifier(config=config) as app:
        app.run()
