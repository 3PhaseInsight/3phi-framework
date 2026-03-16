import matplotlib
import pandas as pd
from tqdm import tqdm

import threephi_framework.db.db as threephi_db
from threephi_framework import DataExtractor
from threephi_framework.controllers.meta import MetaController

matplotlib.use("Agg")
import os

from matplotlib import pyplot as plt

from threephi_framework import S3Connector


def save_sm_plot(sm_id, sm_df, sm_id_results, cfg):
    s3_connector = S3Connector()

    # Determine under which directories the plot has to be saved according to user settings and data characteristics
    dirs_to_save = []
    plot_selection = cfg["plot_cfg"]["SM_selection"]

    if (
        plot_selection["All_(with_dataset_containing_data)"]
        and not sm_df.empty
        and ((sm_df.notna() & (sm_df != 0)).any(axis=1)).any()
    ):
        dirs_to_save.append(os.path.join("Plots", "All_SMs_with_dataset_containing_data"))

    if plot_selection["With_only_good_data_quality"] or plot_selection["With_any_medium_or_bad_data_quality"]:
        # and sm_id in result_summary["SMs_with_only_good_data_quality"]:
        all_good = True
        any_medium_or_bad = False
        for _phase, phase_data in sm_id_results["Data Quality"].items():
            for _key, entry in phase_data.items():
                if isinstance(entry, dict) and "Summary" in entry:
                    summary_value = entry["Summary"]
                    if summary_value == "Bad" or summary_value == "Medium":
                        any_medium_or_bad = True
                    elif summary_value != "Good":
                        all_good = False

        if plot_selection["With_only_good_data_quality"] and all_good:
            dirs_to_save.append(os.path.join("Plots", "SMs_with_only_good_data_quality"))

        if plot_selection["With_any_medium_or_bad_data_quality"] and any_medium_or_bad:
            dirs_to_save.append(os.path.join("Plots", "SMs_with_any_medium_or_bad_data_quality"))

    if (
        plot_selection["With_1-phase_connection"]
        or plot_selection["With_2-phase_connection"]
        or plot_selection["With_3-phase_connection"]
    ):
        # Check how many phases are connected for current sm_id and append accordingly
        Two_phase = False
        One_phase = False
        if isinstance(sm_id_results["Connectivity"]["Connected Phases"], list):
            if len(sm_id_results["Connectivity"]["Connected Phases"]) == 2:
                Two_phase = True
            if len(sm_id_results["Connectivity"]["Connected Phases"]) == 1:
                One_phase = True

        if plot_selection["With_2-phase_connection"] and Two_phase:
            dirs_to_save.append(os.path.join("Plots", "SMs_with_2-phase_connection"))
        if plot_selection["With_1-phase_connection"] and One_phase:
            dirs_to_save.append(os.path.join("Plots", "SMs_with_1-phase_connection"))

    if plot_selection["With_connection_error"] and (
        isinstance(sm_id_results["Connectivity"]["Connection Error"], list)
        and len(sm_id_results["Connectivity"]["Connection Error"]) > 0
    ):
        dirs_to_save.append(os.path.join("Plots", "SMs_with_connection_error"))

    if plot_selection["With_on_off_switch"] and (
        isinstance(sm_id_results["Connectivity"]["Switching Phases"], list)
        and len(sm_id_results["Connectivity"]["Switching Phases"]) > 0
    ):
        dirs_to_save.append(os.path.join("Plots", "SMs_with_on_off_switch"))

    # If plot belongs in None of the categories, skip plotting
    if not dirs_to_save:
        return

    # Determine the rows needed for the selected variables and phases
    rows = []
    if "V" in cfg["selected_variables"]:
        rows.extend([f"voltage_{phase}" for phase in cfg["selected_phases"]])
    if any(var in cfg["selected_variables"] for var in ["P14", "P23"]):
        rows.extend([f"active_power_{phase}" for phase in cfg["selected_phases"]])
    if any(var in cfg["selected_variables"] for var in ["Q12", "Q34"]):
        rows.extend([f"reactive_power_{phase}" for phase in cfg["selected_phases"]])

    # Set up the figure with the correct number of subplots
    n_rows = len(rows)
    if n_rows > 0:
        fig, axes = plt.subplots(n_rows, 1, sharex=True, figsize=(10, 2 * n_rows))
        if n_rows == 1:
            axes = [axes]  # Ensure axes is iterable for a single row

        # Plot each variable for each selected phase
        for row_index, row in enumerate(rows):
            phase = row.split("_")[-1]

            # Set grid for each subplot
            axes[row_index].grid(True, which="both", axis="both", linestyle="--", linewidth=0.5)

            # Plot voltage if selected
            if row.startswith("voltage_") and "V" in cfg["selected_variables"]:
                col_name = f"voltage_{phase}"
                axes[row_index].plot(sm_df[col_name], label=col_name)
                axes[row_index].set_ylabel("V [V]")
                axes[row_index].set_title(f"Voltage - Phase {phase.upper()}")
                # Set x-axis limits to data range
                axes[row_index].set_xlim(sm_df.index.min(), sm_df.index.max())

            # Plot active power if selected (P14 and P23 together in the same subplot)
            elif row.startswith("active_power_") and any(var in cfg["selected_variables"] for var in ["P14", "P23"]):
                if "P23" in cfg["selected_variables"]:
                    col_name_p23 = f"active_power_p23_{phase}"
                    axes[row_index].plot(-sm_df[col_name_p23], label="Production")
                if "P14" in cfg["selected_variables"]:
                    col_name_p14 = f"active_power_p14_{phase}"
                    axes[row_index].plot(sm_df[col_name_p14], label="Consumption")
                axes[row_index].set_ylabel("P [W]")
                axes[row_index].set_title(f"Active Power - Phase {phase.upper()}")
                # Set legend only if both P14 and P23 are present
                if "P14" in cfg["selected_variables"] and "P23" in cfg["selected_variables"]:
                    axes[row_index].legend(loc="upper right")
                # Set x-axis limits to data range
                axes[row_index].set_xlim(sm_df.index.min(), sm_df.index.max())

            # Plot reactive power if selected (Q12 and Q34 together in the same subplot)
            elif row.startswith("reactive_power_") and any(var in cfg["selected_variables"] for var in ["Q12", "Q34"]):
                if "Q12" in cfg["selected_variables"]:
                    col_name_q12 = f"reactive_power_q12_{phase}"
                    axes[row_index].plot(sm_df[col_name_q12], label="Inductive")
                if "Q34" in cfg["selected_variables"]:
                    col_name_q34 = f"reactive_power_q34_{phase}"
                    axes[row_index].plot(-sm_df[col_name_q34], label="Capacitive")
                axes[row_index].set_ylabel("Q [Var]")
                axes[row_index].set_title(f"Reactive Power - Phase {phase.upper()}")
                # Set legend only if both Q12 and Q34 are present
                if "Q12" in cfg["selected_variables"] and "Q34" in cfg["selected_variables"]:
                    axes[row_index].legend(loc="upper right")
                # Set x-axis limits to data range
                axes[row_index].set_xlim(sm_df.index.min(), sm_df.index.max())

        # Set the title
        fig.text(0.5, 0.98, f"Smart Meter Data for {sm_id}", ha="center", fontsize=16, fontweight="bold")

        # Adjust layout to prevent overlap
        plt.tight_layout(rect=(0, 0, 1, 0.98))

        # get plotting settings
        plot_format = cfg["plot_cfg"]["Plotting_format"]
        format = plot_format["plot_format"]
        dpi = plot_format["plot_dpi"]
        transparent = plot_format["plot_transparent"]
        overwrite_plots = plot_format["overwrite_plots"]

        # Create filename
        filename = f"SM_{sm_id}_plot.{format}"

        for d in dirs_to_save:
            d_norm = str(d).replace("\\", "/").lstrip("/")
            s3_path = f"{cfg['results_dir']}/{d_norm}/{filename}"
            s3_connector.save_plot(
                s3_path, fig, format=format, transparent=transparent, dpi=dpi, overwrite=overwrite_plots
            )


def meter_evaluation(sm_ids, cfg):
    data_extractor = DataExtractor(phase_measurements_dir=cfg["data_dir_path"])
    meta_controller = MetaController(threephi_db.new_session)

    # Get timeseries data
    sm_with_data = meta_controller.get_time_series_meta_info()["id_list_of_sms_with_data"]
    sm_with_data = [str(sm_id) for sm_id in sm_with_data]

    # Loop over the list of SM IDs, create results of that SM and add it to the total detailed and summary result dict
    for sm_id in tqdm(sm_ids, desc="Classifying smart meters"):
        # Initialize results dict for current sm_id
        sm_id_results = {
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

        # Ensure to only process SMs which have data
        if sm_id in sm_with_data:
            # Load smart meter data
            # TODO: Add functionalities for cleaned and corrected?
            sm_df = data_extractor.v1_get_single_meter_data(sm_id)
            sm_df = sm_df.compute()
            sm_df.set_index("timestamp", inplace=True)
            sm_df.index = pd.to_datetime(sm_df.index).tz_convert("UTC")

            # Get maximum recording period (sm_df all are of this length and just have nan in beginning and end)
            cfg["max_rec_period"] = len(sm_df) if cfg["max_rec_period"] is None else cfg["max_rec_period"]

            has_data_full_set = (sm_df.notna() & (sm_df != 0)).any(axis=1)

            if not has_data_full_set.any():
                sm_df_with_data = sm_df.iloc[0:0]
            else:
                pos = has_data_full_set.to_numpy().nonzero()[0]  # integer positions
                first_valid_idx = int(pos[0])
                last_valid_idx = int(pos[-1])
                sm_df_with_data = sm_df.iloc[first_valid_idx : last_valid_idx + 1]

            # Calculate nan fractions for each of the variables of the current sm_id (required for data quality assess.)
            nan_fractions = sm_df_with_data.isna().mean()

            # Overwrite nan in Connectivity with empty list since data is available. Lists are populated further down
            sm_id_results["Connectivity"]["Connected Phases"] = []
            sm_id_results["Connectivity"]["Connection Error"] = []
            sm_id_results["Connectivity"]["Switching Phases"] = []

            for phase in cfg["phases"]:
                # Check data quality for V
                is_nan = sm_df_with_data[f"voltage_{phase}"].isna()  # which entries are nan (True/False)
                is_zero = (sm_df_with_data[f"voltage_{phase}"] == 0).fillna(False)  # which entries are 0 (True/False)
                is_below_vlim = (sm_df_with_data[f"voltage_{phase}"] < cfg["v_lim"]).fillna(
                    False
                )  # which entries are below v_lim
                is_frozen = sm_df_with_data[f"voltage_{phase}"] == sm_df_with_data[f"voltage_{phase}"].shift(1)
                for i in range(2, cfg["frozen_range"]):  # which entries are frozen for frozen_range consecutive entries
                    is_frozen &= sm_df_with_data[f"voltage_{phase}"] == sm_df_with_data[f"voltage_{phase}"].shift(i)
                is_frozen = is_frozen.fillna(False)
                is_corrupted = (
                    is_nan | is_zero | is_below_vlim | is_frozen
                )  # which entries are nan,0,below lim or frozen
                has_data = ~is_nan & ~is_zero & ~is_frozen  # which entries are NOT nan and NOT 0 and NOT frozen
                offset_data = (
                    is_below_vlim & has_data
                )  # which entries are NOT nan and NOT 0 and NOT frozen and below lim

                # Data quality assessment for V based on total corruption level
                if is_corrupted.mean() < cfg["good_data_limit"]:
                    summary = "Good"
                elif cfg["medium_data_limit"] > is_corrupted.mean() >= cfg["good_data_limit"]:
                    summary = "Medium"
                else:
                    summary = "Bad"

                # Overwrite summary if very little data, which is considered as "not connected" instead of bad quality
                if has_data.sum() < cfg["no_data_limit"] * cfg["max_rec_period"]:
                    summary = "NaN"

                # Add detailed and summarized data quality info to results dict of current sm_id
                sm_id_results["Data Quality"][phase.upper()]["V"]["Detailed"]["NaN frac"] = float(is_nan.mean())
                sm_id_results["Data Quality"][phase.upper()]["V"]["Detailed"]["Zero frac"] = float(is_zero.mean())
                sm_id_results["Data Quality"][phase.upper()]["V"]["Detailed"]["Below Vlim frac"] = float(
                    is_below_vlim.mean()
                )
                sm_id_results["Data Quality"][phase.upper()]["V"]["Detailed"]["Frozen frac"] = float(is_frozen.mean())
                sm_id_results["Data Quality"][phase.upper()]["V"]["Detailed"]["Total corruption frac"] = float(
                    is_corrupted.mean()
                )
                sm_id_results["Data Quality"][phase.upper()]["V"]["Summary"] = summary

                # Check data quality for P14, P23, Q12, Q34 (which is only nan frac at the moment)
                for variable in cfg["variables"][1:]:
                    # Extract the associated column name of current variable
                    col_name = [col for col in sm_df_with_data.columns if (phase in col) and (variable in col)][0]

                    # Get the nan fraction for current variable from nan_fractions calculated above
                    nan_frac = nan_fractions[col_name]

                    # Summarize the data quality of current variable based on nan fraction
                    if nan_frac < cfg["good_data_limit"]:
                        summary = "Good"
                    elif cfg["medium_data_limit"] > nan_frac >= cfg["good_data_limit"]:
                        summary = "Medium"
                    else:
                        summary = "Bad"

                    # Overwrite summary if very little data,
                    # which is considered as "not connected" instead of bad quality
                    if has_data.sum() < cfg["no_data_limit"] * cfg["max_rec_period"]:
                        summary = "NaN"

                    # Add detailed and summarized data quality info to result dict of current sm_id
                    sm_id_results["Data Quality"][phase.upper()][variable.upper()]["Detailed"]["NaN frac"] = float(
                        nan_frac
                    )
                    sm_id_results["Data Quality"][phase.upper()][variable.upper()]["Summary"] = summary

                # Add the connectivity information to the results dict of current sm_id
                if has_data.sum() > cfg["no_data_limit"] * cfg["max_rec_period"]:
                    # Check which phases are connected
                    sm_id_results["Connectivity"]["Connected Phases"].append(phase.upper())

                    # Check which phases are poorly connected and show constant voltage offset
                    if offset_data.mean() / has_data.mean() > cfg["offset_threshold"]:
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
                        has_data_pd.astype(int).groupby((has_data_pd != has_data_pd.shift()).cumsum()).transform("sum")
                    )
                    if (consecutive_on >= cfg["cons_period_threshold"]).any() & (
                        consecutive_off >= cfg["cons_period_threshold"]
                    ).any():
                        sm_id_results["Connectivity"]["Switching Phases"].append(phase.upper())

                # Add data statistics for P14, P23, Q12, Q34 and V
                for variable in cfg["variables"]:
                    col_name = [col for col in sm_df_with_data.columns if (phase in col) and (variable in col)][0]
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

            # Save plot on request (with option to create plots only for certain criteria like has connection error)
            if cfg["save_plots"]:
                save_sm_plot(sm_id, sm_df, sm_id_results, cfg)

        # Update the JSONB smart meter field with the results of the current sm_id
        meta_controller.update_sm_characterization(meter_id=sm_id, data=sm_id_results)
