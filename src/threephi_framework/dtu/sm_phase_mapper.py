import logging

import matplotlib
import numpy as np
import pandas as pd
from tqdm import tqdm

matplotlib.use("Agg", force=True)
from collections import Counter

import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import dendrogram, fcluster, linkage
from scipy.spatial.distance import pdist
from sklearn.preprocessing import StandardScaler

import threephi_framework.db.db as threephi_db
from threephi_framework.controllers.meta import MetaController
from threephi_framework.controllers.time_series import TimeSeriesController
from threephi_framework.object_storage.factory import create_connector


def _check_window_for_corrupted_data(start_end_tuple, data, lower_v_lim, v_step_lim) -> bool:
    # Extract the current window from dataset
    start, end = start_end_tuple
    window = data[start:end, :]

    # Condition 1: Check for any NaN values within voltage ts window
    if pd.isna(window).any():
        return True

    # Condition 2: Check if there is any value below 207 within voltage ts window
    if (window < lower_v_lim).any().any():
        return True

    # Condition 3: Check if any absolute differenced value is larger than threshold within voltage ts window
    differenced_abs_window = np.abs(np.diff(window, axis=0))
    return bool((differenced_abs_window > v_step_lim).any())


def _clean_data_of_single_unit(
    trafo_data_df,
    unconnected_phases,
    window_size: int = 30,
    lower_v_lim: float = 207,
    v_step_lim: float = 10,
    max_corruption_threshold: float = 0.9,
    trafo_id: str = None,
) -> tuple:
    # Remove rows in beginning and end of dataset where all column entries are nan
    first_non_nan_idx = trafo_data_df.index[trafo_data_df.notna().any(axis=1)].min()
    last_non_nan_idx = trafo_data_df.index[trafo_data_df.notna().any(axis=1)].max()

    trafo_data_df = trafo_data_df.loc[first_non_nan_idx:last_non_nan_idx]

    # Save a copy of the raw data in case that another cleaning iteration with different max_corruption_threshold is
    # required
    trafo_data_df_raw = trafo_data_df.copy()

    n_min = 7500

    # Repeats cleaning with decreasing tolerance for corrupted sm phases (max_corruption_threshold) until cleaned data
    # is sufficiently large
    while True:
        # Drop unconnected phases
        trafo_data_df = trafo_data_df.drop(
            columns=[phase for phase in unconnected_phases if phase in trafo_data_df.columns]
        )

        # Remove all columns which contain mostly NaNs
        nan_phases = trafo_data_df.columns[(trafo_data_df.isna().mean()) >= max_corruption_threshold].tolist()

        if nan_phases:
            logging.info(
                f"[Transformer {trafo_id}] The following phases have more than {max_corruption_threshold * 100}% "
                f"NaNs and are excluded from clustering: {', '.join(str(u[8:]) for u in nan_phases)}"
            )
            trafo_data_df = trafo_data_df.drop(columns=nan_phases)

        # Remove all columns which contain mostly values below 207
        constant_offset_phases = trafo_data_df.columns[
            (trafo_data_df < lower_v_lim).mean() > max_corruption_threshold
        ].tolist()

        if constant_offset_phases:
            logging.info(
                f"[Transformer {trafo_id}] The following phases have more than {max_corruption_threshold * 100}% "
                f"below {lower_v_lim}V and are excluded from clustering: "
                f"{', '.join(str(u[8:]) for u in constant_offset_phases)}"
            )
            trafo_data_df = trafo_data_df.drop(columns=constant_offset_phases)

        if trafo_data_df.shape[1] == 0:
            raise RuntimeError(
                f"[Transformer {trafo_id}] Remaining dataset after removing corrupted SM voltage phases is empty."
            )

        # Convert DataFrame to a NumPy array for faster operations
        unit_data_np = trafo_data_df.to_numpy(copy=True)

        # Initialize an empty list to store corruption flags (True or False) for each moving window of the dataset
        is_corrupted_list = []

        # Loop through each window and check if at least one of the sm phases in a window is corrupted
        for start, end in [(i, i + window_size) for i in range(len(unit_data_np) - window_size + 1)]:
            is_corrupted = _check_window_for_corrupted_data(
                start_end_tuple=(start, end), data=unit_data_np, lower_v_lim=lower_v_lim, v_step_lim=v_step_lim
            )
            is_corrupted_list.append(is_corrupted)

        # Initialize a mask for which rows of the dataset to remove due to potential corruption
        mask_of_corrupted_data = np.zeros_like(unit_data_np, dtype=bool)

        # Set mask entry for a window True if corrupted data was detected before
        for i, corrupted in enumerate(is_corrupted_list):
            if corrupted:
                mask_of_corrupted_data[i : i + window_size, :] = True

        # Apply the mask to set NaN values for all likely corrupted data
        unit_data_np[mask_of_corrupted_data] = np.nan

        # Convert back to DataFrame with all original columns
        trafo_data_df = pd.DataFrame(unit_data_np, columns=trafo_data_df.columns, index=trafo_data_df.index)

        # Check if we have at least n_min of data after the cleaning process
        if not len(trafo_data_df.dropna()) < n_min:
            break
        else:
            logging.info(
                f"[Transformer {trafo_id}] Cleaned data too little. Repeating with more conservative "
                f"max_corruption_threshold {max_corruption_threshold}."
            )
            trafo_data_df = trafo_data_df_raw.copy()
            max_corruption_threshold -= 0.05

            # Check if even very strong thresholds cannot ensure enough samples, then reduce required samples and retry
            if max_corruption_threshold < 0.099:
                n_min -= 200
                max_corruption_threshold = 0.9

    return trafo_data_df, nan_phases + constant_offset_phases


def _preprocess_data_of_single_unit(trafo_data_df_clean) -> pd.DataFrame:
    # Differentiate the data
    trafo_data_df_clean = trafo_data_df_clean.diff()

    # Drop all nan
    trafo_data_df_clean = trafo_data_df_clean.dropna()

    # Scale the data to mean=0 and variance of 1
    trafo_data_df_clean = pd.DataFrame(
        StandardScaler().fit_transform(trafo_data_df_clean), columns=trafo_data_df_clean.columns
    )

    return trafo_data_df_clean


def _add_cabinet_and_feeder_id_to_column_name(trafo_data_df, sm_topology_mapping):
    sm_topology_mapping = sm_topology_mapping.copy()
    sm_topology_mapping["Meter ID"] = sm_topology_mapping["Meter ID"].astype(str)

    new_columns = []

    for col in trafo_data_df.columns:
        parts = col.split("_")

        if len(parts) != 3 or not col.startswith("voltage_"):
            raise ValueError(f"Unexpected voltage column format: {col}. Expected format: voltage_l1_<meter_id>")

        sm_id = parts[2]

        row = sm_topology_mapping[sm_topology_mapping["Meter ID"] == sm_id]

        if row.empty:
            raise ValueError(f"No topology mapping found for meter_id={sm_id}. Column={col}.")

        cabinet_id, feeder_id = row[["Cabinet ID", "Feeder ID"]].iloc[0]

        if pd.isna(cabinet_id):
            cabinet_id = "nan"

        new_columns.append(f"{col}_{cabinet_id}_{feeder_id}")

    trafo_data_df.columns = new_columns

    return trafo_data_df


def _hierarchical_clustering(trafo_data_df):
    # The linkage matrix contains a row per cluster created in the hierarchical clustering. Each row has 4 entries.
    # The first two entries of each row provide the cluster indexes (not row index) of the two clusters which were
    # merged to form the cluster of this row
    # The cluster indexes start from 0 and go to number of df columns + number of formed clusters
    # The indexes from 0 to number of df columns represent individual columns so leaf clusters of single columns
    # So the row which represents a first cluster of more than two columns is row 0. However, the cluster index of
    # row 0 is not 0 but len of df columns, because it starts counting from after the individual column clusters

    # Create Spearman distance matrix (first need to rank so that its spearman not pearson correlation
    distance_matrix = pdist(trafo_data_df.rank(axis=0).T, metric="correlation")

    # Calculate the linkage matrix
    linkage_matrix = linkage(distance_matrix, method="complete")

    return linkage_matrix


def _get_feeder(col_name):
    return col_name.split("_")[-1]


def _get_phase(col_name):
    return col_name.split("_")[1]  # Assuming phase is the second part


def _check_majority_feeder(sm_phases_in_cluster, majority_feeder_threshold=0.5):
    feeder_ids = [_get_feeder(col) for col in sm_phases_in_cluster]
    feeder_ids_counts = pd.Series(feeder_ids).value_counts()
    majority_feeder_count = feeder_ids_counts.max()
    majority_feeder = (
        feeder_ids_counts.idxmax()
        if majority_feeder_count >= majority_feeder_threshold * len(sm_phases_in_cluster)
        else None
    )

    return majority_feeder


def _check_majority_phase(sm_phases_in_cluster, majority_phase_threshold=0.5):
    phases = [_get_phase(col) for col in sm_phases_in_cluster]
    phase_counts = pd.Series(phases).value_counts()
    majority_phase_count = phase_counts.max()

    if majority_phase_count >= majority_phase_threshold * len(sm_phases_in_cluster):
        return phase_counts.idxmax()
    else:
        return None


def _get_all_sm_phases_in_cluster(
    linkage_matrix, cluster_idx, trafo_data_df
):  # idx is the indices of the cluster which results from merging two clusters below
    if cluster_idx < len(
        trafo_data_df.columns
    ):  # It's a leaf (individual column) -> then there is no cluster below anymore that we could break down
        return [
            trafo_data_df.columns[cluster_idx]
        ]  # The cluster indexes start with the individual column clusters 0, 1, ... len(df.columns)
    else:
        # Recursive retrieval for merged clusters
        left_idx = int(
            linkage_matrix[cluster_idx - len(trafo_data_df.columns), 0]
        )  # The indexes of the two merged clusters are stored in the row of the parent cluster
        right_idx = int(
            linkage_matrix[cluster_idx - len(trafo_data_df.columns), 1]
        )  # However, cluster_idx != row_idx since the individual col clusters are not in Z, so they
        # need to be subtracted

        # It will recursively call this function again until all clusters are broken down into individual col clusters
        # And only then it returns the column name
        # Since the function calls itself on and on again, the final result is a list of all the individual column names
        # So we are not returning anything until we are on the individual col cluster level, and here all the
        # individual function results are summed
        return _get_all_sm_phases_in_cluster(linkage_matrix, left_idx, trafo_data_df) + _get_all_sm_phases_in_cluster(
            linkage_matrix, right_idx, trafo_data_df
        )


def _create_backup_phase_reference(trafo_data_df_cols, trafo_phase_clusters):
    # Create phase reference for small feeder phase clusters or feeder clusters with no clear phase majority
    # (Usually, the reference for phase twists is the phase majority of the feeder cluster a sm phase is in)
    # First, try getting phase reference from the phase majority of the same feeder in the same trafo phase
    feeder_IDS = set([col.split("_")[-1] for col in trafo_data_df_cols])

    # Create new trafo phase clusters per feeder
    trafo_phase_clusters_new_per_feeder = {"l1": [], "l2": [], "l3": []}

    for trafo_phase in trafo_phase_clusters:
        for feeder_id in feeder_IDS:
            feeder_phase_cols = [col for col in trafo_phase_clusters[trafo_phase] if col.endswith(feeder_id)]

            majority_phase = _check_majority_phase(feeder_phase_cols, majority_phase_threshold=0.6)

            if majority_phase is not None:
                trafo_phase_clusters_new_per_feeder[majority_phase].extend(feeder_phase_cols)
            else:
                trafo_phase_clusters_new_per_feeder[trafo_phase].extend(feeder_phase_cols)

    # Check if any phase key is empty
    if any(not value_list for value_list in trafo_phase_clusters_new_per_feeder.values()):
        raise ValueError(
            "One or more phase clusters are empty in backup phase reference. Check if trafo have low number of "
            "feeders or if there is a very dominant feeder phase majority in the trafo which leads to all phases "
            "being assigned to one trafo phase cluster and thus no clear phase majority on feeder level."
        )

    return {entry: key for key, value_list in trafo_phase_clusters_new_per_feeder.items() for entry in value_list}


def _determine_trafo_clusters(trafo_data_df_cols, linkage_matrix):
    # Allocate each sm phase to a trafo phase cluster (defined by fixing cluster number to 3)
    sm_phase_to_trafo_phase_allocation = fcluster(linkage_matrix, t=3, criterion="maxclust")

    # Create dict which stores the trafo phase clusters
    trafo_phase_clusters = {i: [] for i in range(1, 4)}
    for col, label in zip(trafo_data_df_cols, sm_phase_to_trafo_phase_allocation, strict=False):
        trafo_phase_clusters[label].append(col)

    # Key each cluster by its majority phase, guarding against ambiguous results: a
    # missing majority (None) or two clusters sharing the same majority would
    # otherwise surface later as a confusing KeyError / silently dropped cluster.
    keyed_clusters = {}
    for cols in trafo_phase_clusters.values():
        majority_phase = _check_majority_phase(cols, majority_phase_threshold=0.5)
        if majority_phase is None or majority_phase in keyed_clusters:
            raise ValueError(
                f"Could not assign distinct majority phases to the three trafo phase clusters "
                f"(got {majority_phase!r} for cluster of size {len(cols)}). The transformer's "
                f"phase labels are too mixed for reliable trafo-level clustering."
            )
        keyed_clusters[majority_phase] = cols

    return keyed_clusters


def _feeders_in_cluster_well_mixed(linkage_matrix, trafo_data_df, majority_feeder, cluster_idx):
    # Get idx of the two sub-clusters that form the current cluster
    left_cluster_idx = int(
        linkage_matrix[cluster_idx - len(trafo_data_df.columns), 0]
    )  # Subtract the df to get Z row idx from cluster idx
    right_cluster_idx = int(linkage_matrix[cluster_idx - len(trafo_data_df.columns), 1])

    # Get the sm phases in the two sub-clusters
    sm_phases_in_cluster_left = _get_all_sm_phases_in_cluster(linkage_matrix, left_cluster_idx, trafo_data_df)
    sm_phases_in_cluster_right = _get_all_sm_phases_in_cluster(linkage_matrix, right_cluster_idx, trafo_data_df)

    # Get the feeder IDs in the two sub-clusters
    feeder_ids_left = [_get_feeder(col) for col in sm_phases_in_cluster_left]
    feeder_ids_right = [_get_feeder(col) for col in sm_phases_in_cluster_right]

    # Check if the majority feeder is included in both sub-clusters
    is_well_mixed = majority_feeder in feeder_ids_left and majority_feeder in feeder_ids_right

    return is_well_mixed


def _minority_feeder_subcluster(majority_feeder, initial_wrong_feeder_labels, sm_phases_in_cluster):
    # Get list of sm phases in current cluster that do not belong to the majority feeder
    sm_phases_of_minority_feeder = [c for c in sm_phases_in_cluster if not c.endswith(majority_feeder)]

    # Check if all minority sm phases are declared wrong feeder labels in the pre clustering evaluation
    all_min_phases_wrong_feeder = all(
        sm_phase in initial_wrong_feeder_labels for sm_phase in sm_phases_of_minority_feeder
    )

    # If not all minority feeder sm phases are previously declared a wrong feeder label, we assume a sub-cluster
    has_minority_feeder_subcluster = not all_min_phases_wrong_feeder

    return has_minority_feeder_subcluster


def _check_for_twisted_sm_phases(sm_phases_in_cluster, majority_phase, backup_phase_reference):
    # Initialize dict for storing twisted phases of the current cluster
    twisted_phases_current_cluster = {}

    # Create list of phases in current cluster cluster_idx
    phases_in_cluster = [_get_phase(col) for col in sm_phases_in_cluster]

    # If a cluster has more than 2 columns and a 66% majority use the cluster phase majority as baseline
    if (phases_in_cluster.count(majority_phase) / len(sm_phases_in_cluster) > 0.66) and (len(sm_phases_in_cluster) > 2):
        for col in sm_phases_in_cluster:
            if _get_phase(col) != majority_phase:
                twisted_phases_current_cluster[col] = majority_phase

    # Otherwise use the trafo phase as baseline
    else:
        for col in sm_phases_in_cluster:
            if _get_phase(col) != backup_phase_reference[col]:
                twisted_phases_current_cluster[col] = backup_phase_reference[col]

    return twisted_phases_current_cluster


def _check_for_wrong_feeder_labels(sm_phase_in_cluster, majority_feeder, key):
    # Initialize dict for storing phases with wrong feeder labels in the current cluster
    wrong_feeder_label_current_cluster = {}

    # Identify sm phases with wrong feeder label and add them to the dict
    for sm_phase in sm_phase_in_cluster:
        if _get_feeder(sm_phase) != majority_feeder:
            wrong_feeder_label_current_cluster[sm_phase] = key

    return wrong_feeder_label_current_cluster


def _check_for_double_phase_connections(sm_phases_in_cluster, SMs_in_cluster):
    # Initialize list to store sm phases which are part of a double connection in the current cluster
    double_connections_current_cluster = []

    # Count how many phases of each sm are connected to the same feeder phase
    sm_phase_count = Counter(SMs_in_cluster)

    for sm_phase in sm_phases_in_cluster:
        if sm_phase_count[sm_phase.split("_")[2]] > 1:
            double_connections_current_cluster.append(sm_phase)

    return double_connections_current_cluster


def _check_for_wrong_trafo_sm(trafo_data_df, sm_phases_in_cluster, SMs_in_cluster, feeder_ids_in_parent_cluster):
    if (len(sm_phases_in_cluster) > 1) and len(set(SMs_in_cluster)) == 1 and len(feeder_ids_in_parent_cluster) > 1:
        # Get list of all phases of the wrong trafo SM
        sm_id = "_".join(sm_phases_in_cluster[0].split("_")[2:])
        all_phases_of_wrong_trafo_sm = [col for col in trafo_data_df.columns if col.endswith(sm_id)]

        return all_phases_of_wrong_trafo_sm
    else:
        return []


def _create_feeder_phase_cluster_key(majority_feeder, majority_phase, feeder_phase_clusters):
    key = f"{majority_feeder}_{majority_phase}"

    # If a certain feeder-phase cluster already exists, add a suffix
    if key in feeder_phase_clusters:
        count = 1
        new_key = f"{key}_{count}"
        while new_key in feeder_phase_clusters:
            count += 1
            new_key = f"{key}_{count}"
        key = new_key

    return key


def _check_for_feeder_phase_twists(trafo_data_df, trafo_id, trafo_phase_clusters, twisted_sm_phases, wrong_trafo_label):
    twisted_feeder_phases = []
    sm_to_trafo_phase_mapping = {}

    # Create a set of all feeder IDs of the current transformer
    feeder_IDs = set([col.split("_")[-1] for col in trafo_data_df.columns])

    # Create a trafo phase cluster dict where wrong trafo and phase labels are filtered out
    trafo_phase_clusters_wo_wrong_labels = {}
    for feeder_phase in trafo_phase_clusters:
        filtered_phases = [
            p for p in trafo_phase_clusters[feeder_phase] if p not in twisted_sm_phases and p not in wrong_trafo_label
        ]
        trafo_phase_clusters_wo_wrong_labels[feeder_phase] = filtered_phases

    # Create a list of all sm phases of the current transformer where wrong trafo and phase labels are removed
    sm_phases_filtered = [p for phases in trafo_phase_clusters_wo_wrong_labels.values() for p in phases]

    # Phase names match the canonical (lowercase) parquet column naming
    l1, l2, l3 = "l1", "l2", "l3"

    # Check feeder phase twists for each feeder of the current transformer
    for feeder_id in feeder_IDs:
        # Extract a list of l1, l2 and l3 sm phases for the current feeder
        feeder_phases = {
            f"{l1}_phases_of_feeder": [p for p in sm_phases_filtered if (p.endswith(feeder_id)) and (l1 in p)],
            f"{l2}_phases_of_feeder": [p for p in sm_phases_filtered if (p.endswith(feeder_id)) and (l2 in p)],
            f"{l3}_phases_of_feeder": [p for p in sm_phases_filtered if (p.endswith(feeder_id)) and (l3 in p)],
        }

        # Check if all L1, L2 or L3 sm phases of the current feeder are within a certain trafo phase cluster
        d = {}
        for f_p in [l1, l2, l3]:
            for t_p in [l1, l2, l3]:
                d[f"all_{f_p}_in_{t_p}"] = len(
                    [
                        col
                        for col in trafo_phase_clusters_wo_wrong_labels[t_p]
                        if (col.endswith(feeder_id)) and (f_p in col)
                    ]
                ) == len(feeder_phases[f"{f_p}_phases_of_feeder"])

        # Check for feeder to trafo phase patterns
        phase_patterns = {
            ((l1, l1), (l2, l2), (l3, l3)): sum([d["all_l1_in_l1"], d["all_l2_in_l2"], d["all_l3_in_l3"]]) >= 2,
            ((l1, l3), (l3, l1), (l2, l2)): d["all_l3_in_l1"] and d["all_l1_in_l3"] and d["all_l2_in_l2"],
            ((l1, l2), (l2, l1), (l3, l3)): d["all_l2_in_l1"] and d["all_l1_in_l2"] and d["all_l3_in_l3"],
            ((l2, l3), (l3, l2), (l1, l1)): d["all_l3_in_l2"] and d["all_l2_in_l3"] and d["all_l1_in_l1"],
            ((l1, l2), (l2, l3), (l3, l1)): d["all_l2_in_l1"] and d["all_l3_in_l2"] and d["all_l1_in_l3"],
            ((l1, l3), (l2, l1), (l3, l2)): d["all_l3_in_l1"] and d["all_l1_in_l2"] and d["all_l2_in_l3"],
        }

        # Store feeder to trafo phase pattern
        for pattern, pattern_is_true in phase_patterns.items():
            if pattern_is_true:
                for phase_connection in pattern:
                    if phase_connection[0] != phase_connection[1]:
                        twisted_feeder_phases.append(f"{feeder_id}_{phase_connection[1]}")
                    for col in feeder_phases[f"{phase_connection[0]}_phases_of_feeder"]:
                        sm_to_trafo_phase_mapping[col] = phase_connection[1]

    # Determine feeder phase twist for twisted SMs
    for twisted_sm_phase in twisted_sm_phases:
        # Get the actual sm phase and feeder id
        actual_sm_phase = twisted_sm_phases[twisted_sm_phase]

        feeder_id = twisted_sm_phase.split("_")[-1]

        # Check which trafo phase the cols of the actual phase have
        trafo_phases = list(
            {
                k: v
                for k, v in sm_to_trafo_phase_mapping.items()
                if k not in twisted_sm_phases
                and k not in wrong_trafo_label
                and actual_sm_phase in k
                and k.endswith(feeder_id)
            }.values()
        )

        # Add trafo phase info for the twisted SM phase based on majority count
        # sm_to_trafo_phase_mapping[twisted_sm_phase] = Counter(trafo_phases).most_common(1)[0][0]
        # TODO: Add this the summarization
        if len(trafo_phases) == 0:
            logging.info(
                f"[Trafo {trafo_id}] Feeder-phase unidentifiable for {twisted_sm_phase}. "
                f"Falling back to SM phase ({actual_sm_phase})."
            )
            sm_to_trafo_phase_mapping[twisted_sm_phase] = actual_sm_phase
        else:
            sm_to_trafo_phase_mapping[twisted_sm_phase] = Counter(trafo_phases).most_common(1)[0][0]

    return sm_to_trafo_phase_mapping, twisted_feeder_phases


def _create_wrong_cabinet_dict(linkage_matrix, wrong_feeder_labels, trafo_data_df, wrong_cabinet_labels):
    # Function to find all clusters that contain a given column index
    def find_clusters_containing_column(linkage_matrix, wrong_feeder_col, df_columns):
        target_index = trafo_data_df.columns.get_loc(wrong_feeder_col)

        # Initially, each column is its own cluster (use column names instead of indices)
        clusters = {i: [df_columns[i]] for i in range(len(df_columns))}

        # Store all clusters step by step
        cluster_history = []

        # For each merge operation (i.e., each row in the linkage matrix)
        for i, (c1, c2, _dist, _) in enumerate(linkage_matrix):
            c1, c2 = int(c1), int(c2)

            # Merge the two clusters into one
            new_cluster = clusters[c1] + clusters[c2]
            clusters[len(df_columns) + i] = new_cluster

            # Store the cluster at this step
            cluster_history.append(new_cluster)

            # Remove the old clusters (c1 and c2 are merged into one)
            del clusters[c1]
            del clusters[c2]

        # Find all clusters that contain the target column
        result_clusters = [cluster for cluster in cluster_history if df_columns[target_index] in cluster]

        return result_clusters

    def get_column_names_in_cluster(clusters_containing_column, wrong_feeder_labels):
        for cluster in clusters_containing_column:
            # Remove the target column from the cluster
            cluster_column_names = [col for col in cluster if col not in wrong_feeder_labels]

            if cluster_column_names:
                return cluster_column_names

    # Get the closest sm phases (EXCLUDING wrong feeder label phases) for each phase with wrong feeder label
    closest_phases = {}
    for wrong_feeder_col in wrong_feeder_labels:
        clusters_containing_column = find_clusters_containing_column(
            linkage_matrix, wrong_feeder_col, trafo_data_df.columns
        )
        cluster_column_names = get_column_names_in_cluster(clusters_containing_column, wrong_feeder_labels)
        closest_phases[wrong_feeder_col] = cluster_column_names

    # Combine the closest sm phases from all phases of the same meter
    combined_closest_phases = {}

    for wrong_feeder_col, closest_phases_cur_col in closest_phases.items():
        # Split the key and get the first number after '_LX_'
        sm_id = wrong_feeder_col.split("_")[2]

        # If the first number is not already a key in the new dictionary, add it
        if sm_id not in combined_closest_phases:
            combined_closest_phases[sm_id] = []

        # Add the current list to the corresponding entry in the new dictionary
        combined_closest_phases[sm_id].extend(closest_phases_cur_col)

    # Iterate through the combined_dict and find the most frequent second number after _LX_ in each list
    for sm_id, closest_phases_cur_sm in combined_closest_phases.items():
        # List to hold all the second numbers after '_LX_' from the current list
        cabinet_ids = []

        # Extract the second number from each item in the list
        for phase in closest_phases_cur_sm:
            cabinet_ids.append(phase.split("_")[3])

        # Count the frequency of each second number
        counter = Counter(cabinet_ids)

        # Find the most frequent second number
        most_frequent_close_cabinet = counter.most_common(1)[0][0]

        # Replace the list with the most frequent second number (store it as a single value)
        combined_closest_phases[sm_id] = most_frequent_close_cabinet

    # Print the resulting dictionary
    for wrong_feeder_col in wrong_feeder_labels:
        for sm_id, cabinet_id in combined_closest_phases.items():
            if sm_id in wrong_feeder_col:
                wrong_cabinet_labels[wrong_feeder_col] = cabinet_id

    return wrong_cabinet_labels


def _rule_based_clustering_evaluation(trafo_id, trafo_data_df, linkage_matrix, initial_wrong_feeder_labels):
    # Create Trafo clusters
    trafo_phase_clusters = _determine_trafo_clusters(trafo_data_df.columns, linkage_matrix)

    # Create phase reference used for detecting phase twists if feeder cluster is small or without clear phase majority
    backup_phase_reference = _create_backup_phase_reference(trafo_data_df.columns, trafo_phase_clusters)

    # Initialize dictionaries for storing results of cluster evaluation
    feeder_phase_clusters = {}  # Stores the final feeder phase clusters
    twisted_sm_phases = {}  # Stores sm phases which are twisted
    wrong_feeder_labels = {}  # Stores sm phases with wrong feeder label
    wrong_cabinet_labels = {}
    doubled_phase_connection = []  # Stores sm phases which are part of a multi-connection on same feeder phase
    cluster_key_by_col = {}
    cluster_idx_by_key = {}
    wrong_trafo_labels = []

    def recursive_cluster_evaluation(trafo_id, cluster_idx, sm_phases_in_parent_cluster=None):
        sm_phases_in_parent_cluster = sm_phases_in_parent_cluster or []
        # Create list of all SM phases that belong to the current cluster cluster_idx
        sm_phases_in_cluster = _get_all_sm_phases_in_cluster(linkage_matrix, cluster_idx, trafo_data_df)

        # Create list of SMs in current cluster cluster_idx
        SMs_in_cluster = [sm_phase.split("_")[2] for sm_phase in sm_phases_in_cluster]

        # Get the majority feeder and phase in the current cluster cluster_idx
        majority_feeder = _check_majority_feeder(sm_phases_in_cluster, majority_feeder_threshold=0.75)
        majority_phase = _check_majority_phase(sm_phases_in_cluster, majority_phase_threshold=0.58)

        # Create set of feeder IDs in the parent cluster
        feeder_ids_in_parent_cluster = set([col.split("_")[4] for col in sm_phases_in_parent_cluster])

        # Get number of feeders in cluster
        num_of_feeders_in_cluster = len(set([_get_feeder(col) for col in sm_phases_in_cluster]))

        # Check for wrong trafo SM (i.e. current cluster consists only of multiple phases of ONE SM)
        wrong_trafo_label_current_cluster = _check_for_wrong_trafo_sm(
            trafo_data_df, sm_phases_in_cluster, SMs_in_cluster, feeder_ids_in_parent_cluster
        )

        if wrong_trafo_label_current_cluster:
            # Add identified wrong trafo labels to the list
            wrong_trafo_labels.extend(wrong_trafo_label_current_cluster)

            # Create a separate feeder phase class for the SM from a different trafo
            feeder_phase_clusters[f"{majority_feeder}_LX"] = wrong_trafo_label_current_cluster

            return  # return to stop the recursive call of the function once we found a wrong trafo cluster

        # Check if feeders in cluster are well mixed (i.e., if a further split would not remove minority entirely)
        if num_of_feeders_in_cluster > 1:
            is_well_mixed = _feeders_in_cluster_well_mixed(linkage_matrix, trafo_data_df, majority_feeder, cluster_idx)
        else:
            is_well_mixed = True

        # Check if current cluster has minority feeder sub-cluster (i.e. not all minorities declared wrong feeder
        # label in pre-evaluation)
        if num_of_feeders_in_cluster > 1 and majority_feeder and initial_wrong_feeder_labels is not None:
            has_minority_feeder_subcluster = _minority_feeder_subcluster(
                majority_feeder, initial_wrong_feeder_labels, sm_phases_in_cluster
            )
        else:
            has_minority_feeder_subcluster = False

        # Check if current cluster is a feeder phase cluster and evaluate it if so. Otherwise, continue with
        # sub-clusters
        if (
            majority_feeder is not None
            and majority_phase is not None
            and is_well_mixed
            and not has_minority_feeder_subcluster
        ):
            # Create a key for the cluster
            feeder_phase_cluster_key = _create_feeder_phase_cluster_key(
                majority_feeder, majority_phase, feeder_phase_clusters
            )

            # Add phases in current cluster to the cluster dictionary
            feeder_phase_clusters[feeder_phase_cluster_key] = sm_phases_in_cluster

            # Store cluster idx by key
            cluster_idx_by_key[feeder_phase_cluster_key] = int(cluster_idx)

            # Map each column to its cluster key
            for col in sm_phases_in_cluster:
                cluster_key_by_col[col] = feeder_phase_cluster_key

            # Check for phase twists in current cluster
            wrong_phase_label_current_cluster = _check_for_twisted_sm_phases(
                sm_phases_in_cluster, majority_phase, backup_phase_reference
            )

            # Add identified phase twists to dict
            twisted_sm_phases.update(wrong_phase_label_current_cluster)

            # Check for wrong feeder labels in current cluster
            wrong_feeder_label_current_cluster = _check_for_wrong_feeder_labels(
                sm_phases_in_cluster, majority_feeder, feeder_phase_cluster_key
            )

            # Add identified wrong feeder labels to dict
            wrong_feeder_labels.update(wrong_feeder_label_current_cluster)

            # Check for double connections in current cluster
            doubled_phase_connection_current_cluster = _check_for_double_phase_connections(
                sm_phases_in_cluster, SMs_in_cluster
            )

            # Add identified double connections to list
            doubled_phase_connection.extend(doubled_phase_connection_current_cluster)

            return  # return to stop recursive call. Otherwise, infinite recursive calls.

        else:  # Split the current cluster further since it is not a feeder phase cluster
            left_cluster_idx = int(
                linkage_matrix[cluster_idx - len(trafo_data_df.columns), 0]
            )  # Subtract the df to get Z row idx from cluster idx
            right_cluster_idx = int(linkage_matrix[cluster_idx - len(trafo_data_df.columns), 1])

            # Recursively apply the rule to both clusters
            recursive_cluster_evaluation(trafo_id, left_cluster_idx, sm_phases_in_parent_cluster=sm_phases_in_cluster)
            recursive_cluster_evaluation(trafo_id, right_cluster_idx, sm_phases_in_parent_cluster=sm_phases_in_cluster)

    # Start recursive process for the root cluster (the last row in Z)
    recursive_cluster_evaluation(trafo_id, len(linkage_matrix) + len(trafo_data_df.columns) - 1)
    # Check for feeder phase twists and return dict which indicates for each SM phase to which trafo phase it is
    # connected
    sm_to_trafo_phase_mapping, twisted_feeder_phases = _check_for_feeder_phase_twists(
        trafo_data_df, trafo_id, trafo_phase_clusters, twisted_sm_phases, wrong_trafo_labels
    )

    # Remove wrong trafo phases from sm phase twists
    twisted_sm_phases = {
        sm_phase: feeder_phase
        for sm_phase, feeder_phase in twisted_sm_phases.items()
        if sm_phase not in wrong_trafo_labels
    }

    # Add wrong cabinet info only for wrong feeder IDs
    wrong_cabinet_labels = _create_wrong_cabinet_dict(
        linkage_matrix, wrong_feeder_labels, trafo_data_df, wrong_cabinet_labels
    )

    # Put results into a results dict
    cluster_evaluation_results = {
        "feeder_phase_clusters": feeder_phase_clusters,
        "cluster_idx_by_key": cluster_idx_by_key,
        "cluster_key_by_col": cluster_key_by_col,
        "twisted_sm_phases": twisted_sm_phases,
        "wrong_feeder_labels": wrong_feeder_labels,
        "wrong_cabinet_labels": wrong_cabinet_labels,
        "wrong_trafo_labels": wrong_trafo_labels,
        "doubled_phase_connection": doubled_phase_connection,
        "twisted_feeder_phases": twisted_feeder_phases,
        "sm_to_trafo_phase_mapping": sm_to_trafo_phase_mapping,
    }

    return cluster_evaluation_results


def _check_wrong_feeder_label_consistency(wrong_feeder_labels, corrupted_phases):
    # Create list of the feeder ids of the sm phases with potentially wrong feeder label
    wrong_feeder_ids = set(key.split("_")[0] for key in wrong_feeder_labels.values())

    # Initialize a counter that counts number of phases with wrong feeder label for several SMs
    phases_with_wrong_feeder_label_counter = Counter()

    for feeder_id in wrong_feeder_ids:
        # Extract only the entries belonging to the current feeder
        wrong_feeder_labels_current_feeder = {
            key: value for key, value in wrong_feeder_labels.items() if feeder_id in value
        }

        # Create a list of SMs with wrong feeder labels
        sm_ids_with_wrong_feeder_label = [sm_phase.split("_")[2] for sm_phase in wrong_feeder_labels_current_feeder]

        # Count how many phases of a smart meter have a potential wrong feeder label
        phases_with_wrong_feeder_label_counter += Counter(sm_ids_with_wrong_feeder_label)

    # Create a dict which counts how many phases of a SM are corrupted
    corrupted_phases_count_per_SM = {}

    for sm_phase in corrupted_phases:
        sm_id = sm_phase.split("_")[2]

        if sm_id in corrupted_phases_count_per_SM:
            corrupted_phases_count_per_SM[sm_id] += 1
        else:
            corrupted_phases_count_per_SM[sm_id] = 1

    # Check if in case of a potential wrong feeder label, the same wrong label appears on all uncorrupted phases of an
    # SM
    verified_wrong_feeder_labels = []

    for sm_phase in wrong_feeder_labels:
        sm_id = sm_phase.split("_")[2]

        # Get the number of uncorrupted phases of the current SM
        num_uncorrupted_phases = (
            3 - corrupted_phases_count_per_SM[sm_id] if sm_id in corrupted_phases_count_per_SM else 3
        )

        # If all available (uncorrupted) phases of an SM have the same wrong feeder label, add to list
        if (sm_id in phases_with_wrong_feeder_label_counter) and (
            phases_with_wrong_feeder_label_counter[sm_id] == num_uncorrupted_phases
        ):
            verified_wrong_feeder_labels.append(sm_phase)

    return verified_wrong_feeder_labels


def populate_result_df(trafo_results, cluster_evaluation, corrupted_phases, unconnected_phases):
    phase_l = "l"

    # Helper function for inserting results to the result df
    def _add_to_results(sm_list, result_col, alt_list=None, fixed_res=None, other_phase_info=None):
        for sm in sm_list:
            _, sm_phase, sm_id, cabinet_id, feeder_id = sm.split("_")
            sm_mask = (
                (trafo_results["Feeder ID"] == feeder_id)
                & (trafo_results["Cabinet ID"] == cabinet_id)
                & (trafo_results["SM ID"] == sm_id)
                & (trafo_results["SM Phase"] == sm_phase)
            )
            if alt_list:
                trafo_results.loc[sm_mask, result_col] = alt_list[sm]
            elif fixed_res:
                trafo_results.loc[sm_mask, result_col] = fixed_res
            elif other_phase_info:
                for phase in [f"{phase_l}1", f"{phase_l}2", f"{phase_l}3"]:
                    sm_mask_other_phase = (
                        (trafo_results["Feeder ID"] == feeder_id)
                        & (trafo_results["Cabinet ID"] == cabinet_id)
                        & (trafo_results["SM ID"] == sm_id)
                        & (trafo_results["SM Phase"] == phase)
                    )

                    other_phase_res = trafo_results.loc[sm_mask_other_phase, result_col]

                    if not (other_phase_res == "nan").all().all():
                        trafo_results.loc[sm_mask, result_col] = other_phase_res.values
                        break
            else:
                trafo_results.loc[sm_mask, result_col] = sm_list[sm]

    # Include twisted sm phases
    _add_to_results(cluster_evaluation["twisted_sm_phases"], "Feeder Phase")

    # Exclude feeder phase for wrong trafo sm phases
    _add_to_results(cluster_evaluation["wrong_trafo_labels"], "Feeder Phase", fixed_res="nan")

    # Include feeder phase twists
    _add_to_results(cluster_evaluation["sm_to_trafo_phase_mapping"], "Trafo Phase")

    # Exclude trafo phase for wrong trafo sm phases
    _add_to_results(cluster_evaluation["wrong_trafo_labels"], "Trafo Phase", fixed_res="nan")

    wrong_feeders_ids = {key: value.split("_")[0] for key, value in cluster_evaluation["wrong_feeder_labels"].items()}

    _add_to_results(cluster_evaluation["wrong_feeder_labels"], "True Feeder ID", alt_list=wrong_feeders_ids)

    # Exclude true feeder information for wrong trafo sm phases
    _add_to_results(cluster_evaluation["wrong_trafo_labels"], "True Feeder ID", fixed_res="nan")

    # Include true trafo information
    _add_to_results(cluster_evaluation["wrong_trafo_labels"], "True Trafo ID", fixed_res="other")

    # Include corrupted phase information
    _add_to_results(
        corrupted_phases, ["Feeder Phase", "Trafo Phase", "True Feeder ID", "True Trafo ID"], fixed_res="nan"
    )

    # Include unconnected phase information
    _add_to_results(
        unconnected_phases, ["Feeder Phase", "Trafo Phase", "True Feeder ID", "True Trafo ID"], fixed_res="nan"
    )
    _add_to_results(unconnected_phases, ["True Feeder ID", "True Trafo ID"], other_phase_info=True)

    # Include likely cabinet information
    _add_to_results(cluster_evaluation["wrong_cabinet_labels"], "Likely Cabinet ID")

    # Determine the status of each sm phase and add it to the trafo_results df
    for index, sm_phase in trafo_results.iterrows():
        # Reconstruct the phase name
        sm_phase_label = (
            f"voltage_{sm_phase['SM Phase']}_{sm_phase['SM ID']}_{sm_phase['Cabinet ID']}_{sm_phase['Feeder ID']}"
        )

        # Initialize the status string
        status = ""

        # Check if current sm phase is normal, i.e. Trafo ID, Feeder ID correct. No SM or feeder phase twist.
        is_trafo_id_correct = sm_phase["Trafo ID"] == sm_phase["True Trafo ID"]
        is_feeder_id_correct = sm_phase["Feeder ID"] == sm_phase["True Feeder ID"]
        is_phase_label_correct = sm_phase["SM Phase"] == sm_phase["Feeder Phase"]
        is_feeder_phase_correct = sm_phase["SM Phase"] == sm_phase["Trafo Phase"]

        if is_trafo_id_correct and is_feeder_id_correct and is_phase_label_correct and is_feeder_phase_correct:
            status = "Normal"

        # Check if phase is unconnected
        if sm_phase_label in unconnected_phases:
            status = status + "SM Phase Unconnected" if status == "" else status + " / SM Phase Unconnected"

        # Check if phase is corrupted
        if sm_phase_label in corrupted_phases:
            status = status + "SM Phase Insufficient Data" if status == "" else status + " / SM Phase Insufficient Data"

        # Check if SM phase twist
        if sm_phase_label in cluster_evaluation["twisted_sm_phases"]:
            status = status + "SM Phase Twist" if status == "" else status + " / SM Phase Twist"

        # Check if doubled SM connection
        if sm_phase_label in cluster_evaluation["doubled_phase_connection"]:
            status = status + "SM Phase Doubled" if status == "" else status + " / SM Phase Doubled"

        # Check if feeder phase twist
        if sm_phase["Feeder Phase"] != sm_phase["Trafo Phase"] and sm_phase["Trafo Phase"] != "nan":
            status = status + "Feeder Phase Twist" if status == "" else status + " / Feeder Phase Twist"

        # Check if unknown feeder phase
        if sm_phase["Feeder Phase"] != "nan" and sm_phase["Trafo Phase"] == "nan":
            status = status + "Unknown Feeder Phase" if status == "" else status + " / Unknown Feeder Phase"

        # Check if wrong feeder label
        if sm_phase_label in cluster_evaluation["wrong_feeder_labels"]:
            status = status + "Wrong Feeder" if status == "" else status + " / Wrong Feeder"

        # Check if wrong trafo
        if sm_phase_label in cluster_evaluation["wrong_trafo_labels"]:
            status = status + "Wrong Trafo" if status == "" else status + " / Wrong Trafo"

        trafo_results.loc[index, "Status"] = status

    return trafo_results


def _initialize_trafo_result_df(trafo_data_df_cols, trafo_id):
    sm_phase_rows = []
    for entry in trafo_data_df_cols:
        _, sm_phase, sm_id, cabinet_id, feeder_id = entry.split("_")
        sm_phase_rows.append(
            [trafo_id, feeder_id, cabinet_id, sm_id, sm_phase, sm_phase, "nan", feeder_id, trafo_id, None, "nan"]
        )
    result_df_cols = [
        "Trafo ID",
        "Feeder ID",
        "Cabinet ID",
        "SM ID",
        "SM Phase",
        "Feeder Phase",
        "Trafo Phase",
        "True Feeder ID",
        "True Trafo ID",
        "Likely Cabinet ID",
        "Status",
    ]

    trafo_results = pd.DataFrame(sm_phase_rows, columns=result_df_cols, dtype="object")

    return trafo_results


def _save_result_plot(trafo_id, trafo_data_df, linkage_matrix, cluster_evaluation, connector, cfg):
    # Unpack the cluster evaluation results
    feeder_phase_clusters = cluster_evaluation["feeder_phase_clusters"]
    twisted_sm_phases = cluster_evaluation["twisted_sm_phases"].keys()
    wrong_feeder_labels = cluster_evaluation["wrong_feeder_labels"]
    wrong_trafo_labels = cluster_evaluation["wrong_trafo_labels"]
    doubled_phase_connection = cluster_evaluation["doubled_phase_connection"]
    twisted_feeder_phases = cluster_evaluation["twisted_feeder_phases"]

    # Initialize figure
    fig = plt.figure(figsize=(65 * (len(trafo_data_df.columns) / 425), 7))

    # Set the title and axis label
    plt.title(f"Hierarchical clustering of transformer {trafo_id}")
    plt.ylabel("Spearman correlation-based distance")

    # Create a color palette for feeder phase clusters (Each feeder phase cluster gets a unique color)
    feeder_phase_clusters_colors = {label: f"C{idx % 10}" for idx, label in enumerate(feeder_phase_clusters.keys())}

    # Create an inverted version of the feeder phase cluster dict, where sm phases are keys and feeder phase the values
    feeder_phase_clusters_inverted = {
        sm_phase: feeder_phase for feeder_phase, phases in feeder_phase_clusters.items() for sm_phase in phases
    }

    # Create custom color dict for the dendrogram links based on the feeder phase cluster colors
    link_cols = {}
    for i, i12 in enumerate(linkage_matrix[:, :2].astype(int)):
        c1, c2 = (
            link_cols[x]
            if x > len(linkage_matrix)
            else feeder_phase_clusters_colors[feeder_phase_clusters_inverted[trafo_data_df.columns[x]]]
            for x in i12
        )
        link_cols[i + 1 + len(linkage_matrix)] = c1 if c1 == c2 else "#808080"

    # Create the dendrogram
    dendro = dendrogram(
        linkage_matrix,
        labels=trafo_data_df.columns,
        leaf_rotation=90,
        leaf_font_size=10,
        color_threshold=None,
        link_color_func=lambda x: link_cols[x],
    )

    # Determine positions of central sm phases in twisted feeder phase clusters to positioning of text
    central_sm_phases_in_twisted_feeder_phase_clusters = []

    for feeder_phase in twisted_feeder_phases:
        leaf_positions = [
            i
            for i, label in enumerate(dendro["ivl"])
            if label.endswith(feeder_phase.split("_")[0]) and feeder_phase.split("_")[1] in label
        ]

        if not leaf_positions:
            continue

        # Find the middle label in the cluster
        middle_idx = len(leaf_positions) // 2
        middle_label = dendro["ivl"][leaf_positions[middle_idx]]

        central_sm_phases_in_twisted_feeder_phase_clusters.append(middle_label)

    # Add explanation of the sm phase labels to the plot
    ax = plt.gca()
    ax.annotate(
        "Phase   SM   Cabinet   Feeder",
        xy=(0, 0),
        xytext=(-10, -155),
        textcoords="offset points",
        ha="center",
        rotation=90,
        fontsize=10,
        color="black",
    )

    # Loop over the sm phases, color them accordingly to their feeder phase cluster and add info for wrong phases etc.
    for sm_phase_label_obj in ax.get_xmajorticklabels():
        # Get sm phase name
        sm_phase = sm_phase_label_obj.get_text()

        # Set the color of the sm phase label in the plot based on the feeder phase cluster color palette
        sm_phase_label_obj.set_color(feeder_phase_clusters_colors[feeder_phase_clusters_inverted[sm_phase]])

        # Add "Feeder twist" indicator in the plot to centrally located sm phase in twisted feeder phases
        if sm_phase in central_sm_phases_in_twisted_feeder_phase_clusters:
            ax.annotate(
                "Feeder twist",
                xy=(sm_phase_label_obj.get_position()),
                xytext=(-3, -210),
                textcoords="offset points",
                ha="center",
                fontsize=8,
                color="red",
            )

        # Add indicator for sm phases with wrong phase label AND wrong feeder label
        if (sm_phase in twisted_sm_phases) and (sm_phase in wrong_feeder_labels):
            # Add red box around sm phase label
            sm_phase_label_obj.set_bbox(
                dict(facecolor="none", edgecolor="red", linewidth=1, boxstyle="round,pad=0.0001")
            )

            # Add text below sm phase label
            ax.annotate(
                "Twist &",
                xy=(sm_phase_label_obj.get_position()),
                xytext=(-3, -240),
                textcoords="offset points",
                ha="center",
                rotation=90,
                fontsize=8,
                color="red",
            )

            ax.annotate(
                "wrong feeder",
                xy=(sm_phase_label_obj.get_position()),
                xytext=(3, -255),
                textcoords="offset points",
                ha="center",
                rotation=90,
                fontsize=8,
                color="red",
            )

        # Add indicator for sm phases with wrong phase label AND doubled phase connection
        elif (sm_phase in twisted_sm_phases) and (sm_phase in doubled_phase_connection):
            # Add red box around sm phase label
            sm_phase_label_obj.set_bbox(
                dict(facecolor="none", edgecolor="red", linewidth=1, boxstyle="round,pad=0.0001")
            )

            # Add text below sm phase label
            ax.annotate(
                "Twist & double",
                xy=(sm_phase_label_obj.get_position()),
                xytext=(0, -258),
                textcoords="offset points",
                ha="center",
                rotation=90,
                fontsize=8,
                color="red",
            )

        # Add indicator for sm phases with doubled phase connection
        elif sm_phase in doubled_phase_connection:
            # Add red box around sm phase label
            sm_phase_label_obj.set_bbox(
                dict(facecolor="none", edgecolor="red", linewidth=1, boxstyle="round,pad=0.0001")
            )

            # Add text below sm phase label
            ax.annotate(
                "Double",
                xy=(sm_phase_label_obj.get_position()),
                xytext=(0, -230),
                textcoords="offset points",
                ha="center",
                rotation=90,
                fontsize=8,
                color="red",
            )

        # Add indicator for sm phases with wrong phase label
        elif sm_phase in twisted_sm_phases:
            # Add red box around sm phase label
            sm_phase_label_obj.set_bbox(
                dict(facecolor="none", edgecolor="red", linewidth=1, boxstyle="round,pad=0.0001")
            )

            # Add text below sm phase label
            ax.annotate(
                "Twist",
                xy=(sm_phase_label_obj.get_position()),
                xytext=(0, -222),
                textcoords="offset points",
                ha="center",
                rotation=90,
                fontsize=8,
                color="red",
            )

        # Add indicator for sm phases with wrong feeder label
        elif sm_phase in wrong_feeder_labels:
            # Add red box around sm phase label
            sm_phase_label_obj.set_bbox(
                dict(facecolor="none", edgecolor="red", linewidth=1, boxstyle="round,pad=0.0001")
            )

            # Add text below sm phase label
            ax.annotate(
                "Wrong feeder",
                xy=(sm_phase_label_obj.get_position()),
                xytext=(0, -253),
                textcoords="offset points",
                rotation=90,
                ha="center",
                fontsize=8,
                color="red",
            )

        # Add indicator for sm phases with wrong trafo label
        elif sm_phase in wrong_trafo_labels:
            # Add red box around sm phase label
            sm_phase_label_obj.set_bbox(
                dict(facecolor="none", edgecolor="red", linewidth=1, boxstyle="round,pad=0.0001")
            )

            # Add text below sm phase label
            ax.annotate(
                "Wrong trafo",
                xy=(sm_phase_label_obj.get_position()),
                xytext=(0, -249),
                textcoords="offset points",
                ha="center",
                rotation=90,
                fontsize=8,
                color="red",
            )

    # Save figure
    plt.tight_layout()

    filename = f"Trafo_{trafo_id}_plot.svg"

    plot_path = f"{cfg['results_dir']}/{filename}"
    connector.save_plot(plot_path, fig, format="svg", overwrite=True)
    plt.close(fig)


def _unconnected_phase_labels(sm_topology_info, sm_id) -> list[str]:
    """Build column-style labels (``voltage_l1_<sm>_<cabinet>_<feeder>``) for the
    phases the SM classifier marked as unconnected.

    Labels must exactly match the dataframe column naming produced by
    ``_add_cabinet_and_feeder_id_to_column_name``: lowercase phase names and
    ``"nan"`` for meters without a cabinet.
    """
    connectivity = sm_topology_info.get("Connectivity")
    if not connectivity or "Connected Phases" not in connectivity:
        return []

    connected_phases = connectivity["Connected Phases"]
    if not isinstance(connected_phases, list):
        return []

    cabinet_id = sm_topology_info["Topology"]["Cabinet ID"]
    cabinet_id = "nan" if cabinet_id is None or pd.isna(cabinet_id) else cabinet_id
    feeder_id = sm_topology_info["Topology"]["Feeder ID"]

    return [
        f"voltage_{phase.lower()}_{sm_id}_{cabinet_id}_{feeder_id}"
        for phase in ["L1", "L2", "L3"]
        if phase not in connected_phases
    ]


def identify_sm_topology(trafo_ids, cfg, sm_topology_mapping):
    """Identify the physical phase connection of every SM phase below the given transformers.

    Runs on Dask workers: controllers and the object-storage connector are
    constructed locally from ``cfg`` instead of being shipped through the task graph.
    """
    meta_controller = MetaController(threephi_db.new_session)
    # The level-aware controller is rooted one level above the raw dataset
    ts_base = cfg.get("data_dir_path", "phase_measurements/raw").removesuffix("/raw")
    connector = create_connector(ts_base, backend=cfg.get("object_storage_backend"))
    timeseries_controller = TimeSeriesController(connector)

    for trafo_id in tqdm(trafo_ids):
        try:
            logging.info(f"[Transformer {trafo_id}] Start smart meter topology identification...")

            # TODO: Re-verify whether TopologyController.get_meters_for_transformer can replace this
            # lookup (it was reported broken when the workaround via sm_topology_mapping was added;
            # the topology chain query has been rewritten since).
            sm_ids_from_trafo = sm_topology_mapping.loc[
                sm_topology_mapping["Transformer ID"] == int(trafo_id), "Meter ID"
            ].tolist()
            logging.info(f"SMs below trafo {trafo_id}: {sm_ids_from_trafo}")

            # Collect phases the SM classifier marked as unconnected; they are
            # excluded from clustering and flagged in the results. Ensuring the
            # classification exists is orchestration, not framework logic: the DAG
            # should chain the SMClassifier before the phase mapper for the same
            # meter scope. Meters without a classification are processed with all
            # phases assumed connected — warn so the gap is visible.
            unconnected_phases = []
            unclassified_sms = []
            for sm_id in sm_ids_from_trafo:
                sm_topology_info = meta_controller.get_sm_characterization(sm_id)
                if sm_topology_info.get("Data Quality") is None:
                    unclassified_sms.append(sm_id)
                unconnected_phases.extend(_unconnected_phase_labels(sm_topology_info, sm_id))

            if unclassified_sms:
                logging.warning(
                    f"[Transformer {trafo_id}] {len(unclassified_sms)} meter(s) have no SM classification yet; "
                    f"all of their phases are treated as connected: {unclassified_sms}. "
                    f"Run the SMClassifier for these meters first (chain it before the phase mapper in the DAG)."
                )

            # Load raw sm phase voltage time series of all SMs below current transformer trafo_id.
            # TODO: Support selecting a timeseries processing level (currently always RAW); pass a
            # ProcessingLevel to get_time_series_data once the mapper should run on cleaned data.
            sm_ids_str = [str(sm_id) for sm_id in sm_ids_from_trafo]
            trafo_data_df = timeseries_controller.get_time_series_data(meter_ids=sm_ids_str)
            trafo_data_df = trafo_data_df.compute()

            # Pivot to one voltage column per (phase, meter): voltage_{phase}_{sm_id}
            voltage_cols = ["voltage_l1", "voltage_l2", "voltage_l3"]
            trafo_data_df = trafo_data_df[["timestamp", "meter_number"] + voltage_cols]
            trafo_data_df = trafo_data_df.pivot(
                index="timestamp",
                columns="meter_number",
                values=voltage_cols,
            )
            trafo_data_df.columns = [f"{voltage_col}_{meter_id}" for voltage_col, meter_id in trafo_data_df.columns]

            trafo_data_df = _add_cabinet_and_feeder_id_to_column_name(trafo_data_df, sm_topology_mapping)

            # Initialize result dataframe. Each row stores information on an individual SM phase
            trafo_results = _initialize_trafo_result_df(trafo_data_df_cols=trafo_data_df.columns, trafo_id=trafo_id)

            # Clean raw sm phase voltage time series
            trafo_data_df, corrupted_phases = _clean_data_of_single_unit(
                trafo_data_df=trafo_data_df,
                unconnected_phases=unconnected_phases,
                window_size=cfg["window_size"],
                lower_v_lim=cfg["lower_v_lim"],
                v_step_lim=cfg["v_step_lim"],
                max_corruption_threshold=cfg["max_corruption_threshold"],
                trafo_id=trafo_id,
            )

            # Preprocess cleaned sm phase voltage time series
            trafo_data_df = _preprocess_data_of_single_unit(trafo_data_df_clean=trafo_data_df)

            # Hierarchical clustering of sm phase voltage time series
            linkage_matrix = _hierarchical_clustering(trafo_data_df)

            # Pre-evaluation of the clusters
            pre_cluster_evaluation = _rule_based_clustering_evaluation(
                trafo_id, trafo_data_df, linkage_matrix, initial_wrong_feeder_labels=None
            )

            # Check the wrong feeder labels for pattern match and redo cluster evaluation with this information
            initial_wrong_feeder_labels = _check_wrong_feeder_label_consistency(
                pre_cluster_evaluation["wrong_feeder_labels"], corrupted_phases + unconnected_phases
            )

            # Evaluate clusters
            cluster_evaluation = _rule_based_clustering_evaluation(
                trafo_id, trafo_data_df, linkage_matrix, initial_wrong_feeder_labels
            )

            # Save dendrogram with feeder clusters and SM topology identification results
            if cfg["save_plots"]:
                _save_result_plot(trafo_id, trafo_data_df, linkage_matrix, cluster_evaluation, connector, cfg)

            # Populate result dataframe based on cluster evaluation
            trafo_results = populate_result_df(trafo_results, cluster_evaluation, corrupted_phases, unconnected_phases)
            logging.info(f"[Transformer {trafo_id}] Phase mapping produced {len(trafo_results)} SM phase rows.")

            # Add the changes per trafo directly to the phase mapper schema
            meta_controller.update_phase_mapping(int(trafo_id), trafo_results)

        except Exception:
            logging.exception(f"Phase mapping failed for transformer {trafo_id}. Continuing with next transformer.")
            continue
