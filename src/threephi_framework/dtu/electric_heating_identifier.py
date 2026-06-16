import pandas as pd
import numpy as np
import logging
from tqdm import tqdm
import matplotlib
matplotlib.use("Agg", force=True)
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from collections import Counter
from scipy.spatial.distance import pdist
from scipy.cluster.hierarchy import linkage, fcluster, dendrogram

from threephi_framework.data_extractor.data_extractor import DataExtractor
import threephi_framework.db.db as threephi_db
from threephi_framework.controllers.meta import MetaController
from threephi_framework.object_storage.s3_connector import S3Connector
from threephi_framework.controllers.time_series import TimeSeriesController
from threephi_framework.controllers.topology import TopologyController


import os
import sys
import json
from sklearn.discriminant_analysis import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
import logging
from sklearn.cluster import AgglomerativeClustering
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import recall_score, silhouette_score
from scipy.cluster.hierarchy import linkage, dendrogram
from sklearn.semi_supervised import LabelPropagation
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
import pandas as pd
import numpy as np
from sklearn.metrics import recall_score, f1_score
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.colors import Normalize
from tqdm import tqdm
import warnings



sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../')))


def _preprocess_smart_meters(sm_df, sm_id):

    # Convert the "timestamp" column to index
    sm_df.index = pd.to_datetime(sm_df['timestamp'])  # Ensure the index is datetime
    sm_df.index.name = "timestamp"
    sm_df = sm_df.drop(columns=['timestamp'])

    # Keep only the active power columns and rename them to include the smart meter ID for uniqueness
    data = sm_df[[col for col in sm_df.columns if 'active_power_p14' in col]]
    data.columns = [col.replace('active_power_p14_', '') for col in data.columns]
    data.columns = [f"{col}_{sm_id}" for col in data.columns]
    
    # remove columns with zero or very low consumption
    min_std_threshold = 1e-8
    data = data.loc[:, data.std() > min_std_threshold]

    # Convert index to UTC timezone
    data.index = pd.to_datetime(data.index, utc=True)

    # Fill remaining NaNs with zeros
    data = data.fillna(0)

    # Remove periods with zero consumption across all phases (likely missing data)
    data = data[(data != 0).any(axis=1)]
    
    # Remove periods of sustained zero consumption longer than 1 weeks (assuming: 7 days * 4 recordings per hour * 24 = 672)
    zero_consumption_mask = (data == 0).all(axis=1)
    zero_consumption_groups = (zero_consumption_mask != zero_consumption_mask.shift()).cumsum()
    zero_consumption_durations = zero_consumption_mask.groupby(zero_consumption_groups).transform('sum')
    data = data[~((zero_consumption_mask) & (zero_consumption_durations >= 672))]



    return data

def _load_temp_spotprice_refload_data(cfg) -> pd.DataFrame:

    data_extractor = DataExtractor(phase_measurements_dir=cfg["data_dir_path"])

    # Create the path to the external data directory
    temp_data = data_extractor.s3_connector.read_small_csv(data_extractor.s3_base + cfg["temp_data_path"], dtype={"Temperature": float})
    spot_data = data_extractor.s3_connector.read_small_csv(data_extractor.s3_base + cfg["spot_data_path"], dtype={"SpotPrice": float})
    ref_data = data_extractor.s3_connector.read_small_csv(data_extractor.s3_base + cfg["ref_data_path"], dtype={"Reference_load": float})

    for df in [temp_data, spot_data, ref_data]:
        df.index = pd.to_datetime(df.iloc[:, 0], format="mixed", dayfirst=True, utc=True)
        df.index.name = "DateTime"
        df.drop(df.columns[0], axis=1, inplace=True)
        df.iloc[:, 0] = pd.to_numeric(df.iloc[:, 0], errors='coerce')
        df.interpolate(method="time", inplace=True)

    return temp_data, spot_data, ref_data


def _cluster_and_score(self, features_subset=None, n_clusters=2, true_labels=None, df=None):
    
    # Cluster and score a given feature subset
    X = df[features_subset].dropna()
    y = true_labels.loc[X.index]
    y = y['has_EH'].copy()
    if len(X) < n_clusters or {y.sum()} == 0:
        return None, None

    # Scale
    X_scaled = StandardScaler().fit_transform(X)

    # Cluster
    model = AgglomerativeClustering(n_clusters=n_clusters)
    labels = model.fit_predict(X_scaled)

    # If any cluster has less than 10 points, disregard this feature set
    unique, counts = np.unique(labels, return_counts=True)
    if np.any(counts < 10):
        return None, None

    # Only consider the smaller cluster for recall
    cluster_sizes = dict(zip(unique, counts))
    smallest_cluster_id = min(cluster_sizes, key=cluster_sizes.get)
    in_smallest_cluster = (labels == smallest_cluster_id)
    recall = y[in_smallest_cluster].sum() / y.sum() if y.sum() > 0 else 0
    
    # Find the smallest cluster
    unique, counts = np.unique(labels, return_counts=True)
    smallest_cluster_id = min(dict(zip(unique, counts)), key=lambda k: dict(zip(unique, counts))[k])

    # Get the meter numbers for samples in the smallest cluster
    meter_numbers_in_cluster = X.index[labels == smallest_cluster_id].map(lambda x: x.split("_")[1])
    unique_meters_in_cluster = set(meter_numbers_in_cluster)

    # Total unique smart meters in the dataset
    all_meter_numbers = set(X.index.map(lambda x: x.split("_")[1]))

    # Calculate coverage
    coverage = len(unique_meters_in_cluster) / len(all_meter_numbers) * 100
    relation_score = ((100 - coverage) + 100 * recall) / 2
    
    return relation_score, labels

def _forward_selection(self, df, true_labels, max_features=20, n_clusters=2):
    
    # Initialize variables
    all_features = df.columns.tolist()
    selected = []
    remaining = all_features.copy()
    best_score = -1
    history = []

    # Track last best selection for rollback
    while len(selected) < max_features and remaining:
        
        # Keep track of relation-scores for this iteration
        relation_scores = [] 

        for feature in remaining:
            current_features = selected + [feature]
            try:
                relation_score, _  = self._cluster_and_score(current_features, n_clusters=n_clusters, true_labels=true_labels, df=df)
                if relation_score is not None:
                    relation_scores.append((feature, relation_score))
            except:
                continue

        if not relation_scores:
            break

        best_feature, best_feature_score = max(relation_scores, key=lambda x: x[1])

        if best_feature_score > best_score:
            selected.append(best_feature)
            remaining.remove(best_feature)
            best_score = best_feature_score
            history.append((tuple(selected), best_score))
            count = 0
            last_best_selected = selected.copy()
            last_best_score = best_score

        else:
            count += 1
            selected.append(best_feature)
            remaining.remove(best_feature)

            if count >= 4:
                # Roll back to last best selection and stop
                selected = last_best_selected.copy()
                best_score = last_best_score
                break
        
    # Select the best feature set found
    feature_set, _ = max(history, key=lambda x: x[1])

    return list(feature_set), pd.DataFrame(history, columns=["feature_set", "best_score"])

def _backward_selection(self, df, true_labels, max_features=5, n_clusters=2):
    
    # Initialize variables
    selected = df.columns.tolist()
    best_score, _ = self._cluster_and_score(selected, n_clusters=n_clusters, true_labels=true_labels, df=df)
    
    if best_score is None:
        return pd.DataFrame(columns=["feature_set", "best_score"])

    history = [(tuple(selected), best_score)]

    while len(selected) > max_features:
        
        # Keep track of relation-scores for this iteration
        relation_scores = []

        for feature in selected:
            candidate = [f for f in selected if f != feature]
            try:
                relation_score, _ = self._cluster_and_score(candidate, n_clusters=n_clusters, true_labels=true_labels, df=df)
                if relation_score is not None:
                    relation_scores.append((feature, relation_score))
            except:
                continue

        if not relation_scores:
            break

        worst_feature, best_candidate_score = max(relation_scores, key=lambda x: x[1])

        if best_candidate_score >= best_score:
            selected.remove(worst_feature)
            best_score = best_candidate_score
            history.append((tuple(selected), best_score))
            count = 0  # Reset count if improvement
            last_best_selected = selected.copy()
            last_best_score = best_score

        else:
            count += 1
            selected.remove(worst_feature)
            
            if count >= 4:
            # Roll back to last best selection and stop
                selected = last_best_selected.copy()
                best_score = last_best_score
                break
    
    # Select the best feature set found
    feature_set, _ = max(history, key=lambda x: x[1])

    return list(feature_set), pd.DataFrame(history, columns=["feature_set", "best_score"])

def _feature_based_pruning_for_propagation(features, true_labels):

    """
    Prune features, and generate false labels for label propagation based on temperature ratio feature.
    Args:
        features (pd.DataFrame): DataFrame containing features for each phase.
        true_labels (pd.DataFrame): DataFrame containing true labels for each phase.
    """
    
    # Initialize threshold for propation feature pruning
    threshold = -0.1 

    # generates the negative class for label propagation
    keep_mask = (features["temp_ratio_low_to_total"] > threshold)

    while threshold < 0.2:
        keep_mask = features["temp_ratio_low_to_total"] > threshold
        if keep_mask.sum() < len(features):
            break
        threshold += 0.01

    if keep_mask.sum() == len(features):
        logging.warning(f"No phases meet the negative class criteria of a temperature ratio under {threshold} between high an low temperate periods.\n")
        logging.warning("Selecting random 5% non electric heating labeled phases to be false labels")

        # Randomly select 5% of the features to be false labels
        np.random.seed(42)
                
        candidates = features[~true_labels["has_EH"]].index

        if len(candidates) == 0:
            raise ValueError("No negative candidates available for sampling.")

        random_indices = np.random.choice(candidates, size=int(0.05 * len(features)), replace=False)
        remaining_labels = features[~features.index.isin(random_indices)]
        false_features = features.loc[random_indices]

        return remaining_labels, false_features
    
    else: 
        # Use the mask to split the data, but only remove rows where true_labels is False
        remaining_labels = features[keep_mask | true_labels["has_EH"]]

        # Removed features: not in keep_mask and not a true label
        false_features = features[~keep_mask & ~true_labels["has_EH"]]

        return remaining_labels, false_features


def feature_extraction(sm_ids, cfg) -> pd.DataFrame:
    
    """
    Extracts features per smart meter. Input can be either unit_ids (substations) or sm_ids (smart meters).
    
    Args:
        sm_ids (list): List of smart meter IDs to extract features for.
        cfg (dict): Configuration dictionary.
    
    Returns:
        pd.DataFrame: DataFrame containing extracted features for each smart meter.
    """

    s3_connector = S3Connector(data_dir_path="phase_measurements")
    timeseries_controller = TimeSeriesController(s3_connector)
    meta_controller = MetaController(threephi_db.new_session)

    # ----------- External data Loading -----------
    temp_data, spot_data, ref_data = _load_temp_spotprice_refload_data(cfg)
    
    # Remove duplicated indices
    temp_data = temp_data[~temp_data.index.duplicated(keep='first')]
    ref_data = ref_data[~ref_data.index.duplicated(keep='first')]
    spot_data = spot_data[~spot_data.index.duplicated(keep='first')]

    # Interpolate auxiliary data
    temp_data = temp_data.resample("15min").interpolate("linear")
    spot_data = spot_data[~spot_data.index.duplicated(keep="first")]
    spot_data = spot_data.resample("15min").interpolate("linear")


    # ----------- Feature dictonary -----------
    FEATURE_KEYS = [
        "max_consumption", "consumption_skewness", "consumption_std",
        "diff_mean", "std_diff", "diff_skewness", "active_diff_absolute",
        "max_jump", "max_jump_5", "high_low_diff_day", "nigth_fall",
        "morning_peak", "afternoon_peak", "night_peak", "weekend_consumption",
        "weekday_consumption", "weekend_weekday_ratio", "baseline_consumption",
        "thermal_consumption", "warm_temp_consumption", "low_temp_consumption",
        "temperature_correlation", "winter_temperature_correlation",
        "winter_daytime_temperature_correlation", "spot_correlation",
        "spot_correlation_daytime", "temperature_bins",
        "winter_temperature_correlation_top_consumption",
        "spot_correlation_top_consumption", "rmse", "mse", "correlation_ref",
        "autocorrelation_1", "autocorrelation_2", "autocorrelation_6",
        "autocorrelation_12", "autocorrelation_96", "autocorrelation_720",
        "temp_ratio_low_to_total", "temp_ratio_high_to_total",
        "temp_ratio_low_to_high", "temp_difference_low_high",
        "morning_peak_diff", "morning_rate", "midday_decrease",
        "midday_rate", "hdh_correlation", "hdh_slope", "hdh_mean_load"
    ]

    # Scaler
    scaler = StandardScaler()

    for sm_id in tqdm(sm_ids, desc="Feature extraction per smart meter"):

        meta_controller.start_workflow(workflow=f"feature_engineering_sm_{sm_id}")

        # ----------- SM Loading -----------

        sm_df = timeseries_controller.get_time_series_data(meter_ids = [str(sm_id)]) # processing_level=cfg["sm_processing_level"]
        sm_df = sm_df.compute()
        meter_data = _preprocess_smart_meters(sm_df, sm_id)
        if meter_data.empty:
            logging.warning(f"No valid meter data for {sm_id}. Skipping...")
            continue
    
        # ----------- Data Preprocessing -----------

        meter_data_scaled = pd.DataFrame(
            scaler.fit_transform(meter_data),
            index=meter_data.index,
            columns=meter_data.columns,
            )
        
        # ----------- Align indices ----------- #
        common_index = meter_data_scaled.index.intersection(temp_data.index)
        common_index_spot = meter_data_scaled.index.intersection(spot_data.index)
        common_ref_index = meter_data_scaled.index.intersection(ref_data.index)

        meter_common = meter_data_scaled.loc[common_index]
        temp_common = temp_data.loc[common_index]
        spot_common = spot_data.loc[common_index_spot]
        meter_spot_common = meter_data_scaled.loc[common_index_spot]
        meter_ref = meter_data_scaled.loc[common_ref_index]
        ref_common = ref_data.loc[common_ref_index]

        winter_months = [11, 12, 1, 2]
        winter_mask = meter_common.index.month.isin(winter_months)
        winter_mask_temp = temp_common.index.month.isin(winter_months)

        # --- Temperature Related Features ---

        # merged_data = meter_data_scaled.join(temp_data, how="inner")
        merge_input = meter_data_scaled.reset_index()
        merge_input["timestamp"] = merge_input["timestamp"].astype("datetime64[us, UTC]")

        temp_data_reset = temp_data.reset_index()
        temp_data_reset["DateTime"] = temp_data_reset["DateTime"].astype("datetime64[us, UTC]")

        merged_data = pd.merge_asof(
            merge_input.sort_values("timestamp"),
            temp_data_reset.sort_values("DateTime").rename(columns={"DateTime": "timestamp"}),
            on="timestamp",
            direction="nearest",
        )
        merged_data = merged_data.set_index("timestamp")
        merged_data = merged_data[[col for col in merged_data.columns if col.startswith('l')] + ["Temperature"]]

        print(f"DEBUG sm_id={sm_id}: meter_data_scaled shape={meter_data_scaled.shape}", flush=True)
        print(f"DEBUG sm_id={sm_id}: temp_data_reset shape={temp_data_reset.shape}", flush=True)
        print(f"DEBUG sm_id={sm_id}: merged_data shape={merged_data.shape}", flush=True)
        print(f"DEBUG sm_id={sm_id}: merged_data columns={list(merged_data.columns)}", flush=True)
        logging.info(f"DEBUG sm_id={sm_id}: meter_data_scaled shape={meter_data_scaled.shape}")
        logging.info(f"DEBUG sm_id={sm_id}: temp_data_reset shape={temp_data_reset.shape}")
        logging.info(f"DEBUG sm_id={sm_id}: merged_data shape={merged_data.shape}")
        logging.info(f"DEBUG sm_id={sm_id}: merged_data columns={list(merged_data.columns)}")


        nT = 12
        merged_data['Tbin'] = pd.qcut(merged_data['Temperature'], nT, labels=False, duplicates='drop')
        merged_data['Tbin'] = merged_data['Tbin'].astype(int)


        ## ----------- Feature Engineering ----------- ##


        phase_cols = [col for col in meter_data_scaled.columns if col.startswith('l')]
        for phase in phase_cols:

            # Create dictionary to hold features for this phase, initialize with NaN
            features_dict = {k: np.nan for k in FEATURE_KEYS}

            scaled_series = meter_data_scaled[phase]
            common_series = meter_common[phase]
            spot_series = meter_spot_common[phase]
            ref_series = meter_ref[phase]

            # --- Descriptive Statistics ---

            meter_diff = scaled_series.diff().dropna()

            features_dict["max_consumption"] = scaled_series.max()
            features_dict["consumption_skewness"] = scaled_series.skew()
            features_dict["consumption_std"] = scaled_series.std()
            features_dict["diff_mean"] = meter_diff.mean()
            features_dict["std_diff"] = meter_diff.std()
            features_dict["diff_skewness"] = meter_diff.skew()

            # --- Average Day Profiles ---

            avg_day = scaled_series.groupby(scaled_series.index.floor("15min").time).mean()
            avg_day.index = pd.to_datetime(avg_day.index.map(lambda t: t.strftime("%H:%M")), format="%H:%M")
            avg_day_diff = avg_day.diff().dropna()
            features_dict["active_diff_absolute"] = avg_day_diff.abs().mean()
            features_dict["max_jump"] = avg_day_diff.max()
            features_dict["max_jump_5"] = avg_day_diff.rolling(5).max().mean()
            features_dict["high_low_diff_day"] = avg_day_diff.max() - avg_day_diff.min()

            # --- Time of Use Features ---

            features_dict['nigth_fall'] = scaled_series.between_time('00:00', '03:00').mean()
            features_dict['morning_peak'] = scaled_series.between_time('06:00', '10:00').mean()
            features_dict['afternoon_peak'] = scaled_series.between_time('16:00', '20:00').mean()
            features_dict['night_peak'] = scaled_series.between_time('22:00', '03:00').mean()


            # -----weekend/weekday features-----

            weekend_mask = scaled_series.index.weekday >= 5
            features_dict['weekend_consumption'] = scaled_series[weekend_mask].mean()
            weekday_mask = scaled_series.index.weekday < 5
            features_dict['weekday_consumption'] = scaled_series[weekday_mask].mean()
            features_dict['weekend_weekday_ratio'] = features_dict['weekend_consumption'] / (features_dict['weekday_consumption'] + 0.1)

            features_dict['baseline_consumption'] = common_series[temp_common['Temperature'].between(14, 17)].mean()
            features_dict['thermal_consumption'] = common_series[temp_common['Temperature'] < 14].mean()
            features_dict['warm_temp_consumption'] = common_series[temp_common['Temperature'] > 17].mean()
            features_dict['low_temp_consumption'] = common_series[temp_common['Temperature'] < 5].mean()


            # --- Temperature correlations ---

            def correlation(a, b):
                # Suppress runtime warnings during scaling
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore", category=RuntimeWarning)
                    if len(a.dropna()) < 2 or len(b.dropna()) < 2:
                        return np.nan
                    return a.corr(b)

            features_dict["temperature_correlation"] = correlation(
                scaled_series, 
                temp_data["Temperature"]
            )
            features_dict["winter_temperature_correlation"] = correlation(
                common_series.loc[winter_mask],
                temp_common.loc[winter_mask_temp, "Temperature"]
            )

            winter_daytime_mask = (
                common_series.index.month.isin(winter_months) &
                (common_series.index.hour >= 5) & (common_series.index.hour <= 23)
            )

            features_dict["winter_daytime_temperature_correlation"] = correlation(
                common_series.loc[winter_daytime_mask],
                temp_common.loc[winter_daytime_mask, "Temperature"]
            )

            features_dict["spot_correlation"] = correlation(
                spot_series,
                spot_common["SpotPrice"]
            )

            spot_daytime_winter_mask = (
                spot_series.index.month.isin(winter_months) &
                (spot_series.index.hour >= 6) & (spot_series.index.hour <= 21)
            )
            features_dict["spot_correlation_daytime"] = correlation(
                spot_series.loc[spot_daytime_winter_mask],
                spot_common.loc[spot_daytime_winter_mask, "SpotPrice"]
            )


            # --- Temperature bins slope ---

            def polyfit(x, y):
                if len(x) < 2:
                    return np.nan
                try:
                    slope, _ = np.polyfit(x, y, 1)
                    return slope
                except Exception:
                    return np.nan
            
            bin_sums = merged_data.groupby("Tbin")[phase].median()
            bin_sums = bin_sums.iloc[:min(7, len(bin_sums))]
            features_dict["temperature_bins"] = polyfit(bin_sums.index, bin_sums.values)

            # Top consumption correlations
            top_idx = common_series.loc[common_series > 0].index
            meter_top = common_series.loc[top_idx]
            temp_top = temp_common.loc[top_idx, "Temperature"]
            spot_top = spot_common.loc[top_idx, "SpotPrice"]

            daytime_idx = (meter_top.index.hour >= 6) & (meter_top.index.hour <= 21)
            features_dict["winter_temperature_correlation_top_consumption"] = correlation(
                meter_top[daytime_idx],
                temp_top[daytime_idx]
            )
            features_dict["spot_correlation_top_consumption"] = correlation(
                meter_top[daytime_idx],
                spot_top[daytime_idx]
            )

            diff = ref_series - ref_common["Reference_load"]
            features_dict["rmse"] = np.sqrt(np.mean(diff**2))
            features_dict["mse"] = (np.mean(diff**2))
            features_dict["correlation_ref"] = correlation(
                ref_series,
                ref_common["Reference_load"]
            )

            # --- Autocorrelations ---

            def autocorrelation(series, lag):
                if len(series.dropna()) <= lag + 1:
                    return np.nan
                if series.nunique() <= 1:
                    return np.nan
                return series.autocorr(lag=lag)
            
            lags = [1, 2, 6, 12, 96, 720]
            for lag in lags:
                features_dict[f"autocorrelation_{lag}"] = autocorrelation(meter_data[phase], lag)
                

            # --- Temperature load ratios ---

            common_series_new = meter_data.loc[common_index, phase]
            temps = temp_data.loc[common_index, "Temperature"]

            low = common_series_new[temps < 2]
            high = common_series_new[temps > 6]
            mean_total = common_series_new.mean() + 0.1

            features_dict["temp_ratio_low_to_total"] = (low.mean() / mean_total)
            features_dict["temp_ratio_high_to_total"] = (high.mean() / mean_total)
            features_dict["temp_ratio_low_to_high"] = (low.mean() / (high.mean() + 0.1))
            features_dict["temp_difference_low_high"] = (low.mean() - high.mean())

            # --- Time-based daily features ---

            def time_stats(df, start, end, func):
                return getattr(df.between_time(start, end), func)()

            features_dict["morning_peak_diff"] = (
                time_stats(avg_day_diff, "06:00", "08:00", "max")
            )
            features_dict["morning_rate"] = (
                time_stats(avg_day_diff, "06:00", "08:00", "mean")
            )
            features_dict["midday_decrease"] = (
                time_stats(avg_day_diff, "10:00", "12:00", "min")
            )
            features_dict["midday_rate"] = (
                time_stats(avg_day_diff, "10:00", "12:00", "mean")
            )

            # --- Heating Degree Hours (HDH) analysis ---

            base_temp = 18.0
            series = meter_data[phase].copy()
            temps = temp_data["Temperature"].copy()

            common_idx = series.index.intersection(temps.index)
            load = series.loc[common_idx]
            temp = temps.loc[common_idx]
            hdh = (base_temp - temp).clip(lower=0)

            valid = pd.DataFrame({"load": load, "HDH": hdh}).dropna()

            if len(valid) >= 10:
                features_dict["hdh_correlation"] = correlation(
                    valid["load"],
                    valid["HDH"]
                )
                features_dict["hdh_slope"] = polyfit(valid["HDH"], valid["load"])
                features_dict["hdh_mean_load"] = valid.loc[valid["HDH"] > 0, "load"].mean()
            else:
                features_dict["hdh_correlation"] = (np.nan)
                features_dict["hdh_slope"] = (np.nan)
                features_dict["hdh_mean_load"] = (np.nan)
            
            phase_features = {k: None if (v is None or (isinstance(v, float) and (np.isnan(v) or np.isinf(v)))) else v 
                  for k, v in features_dict.items()}
            
            phase_str = phase.split("_")[0] # Extract phase (e.g., "l1", "l2", "l3")
            
            # Insert values into meta results per phase
            meta_controller.insert_run_result(
                    dag_id=cfg.get("dag_id", "default_dag"),
                    run_id=cfg.get("run_id", "default_run"),
                    meter_id=int(sm_id),
                    phase=str.capitalize(phase_str),
                    label_value="features",
                    confidence=0.0,
                    label_type="Feature Engineering",
                    topology_version=None,
                    result=phase_features,
                    source="Electric Heating Identifier",
                    node_id=None,
                    edge_id=None,
                    cable_id=None,
                )

        meta_controller.complete_workflow(workflow=f"feature_engineering_sm_{sm_id}")

    


def hierarchical_clustering(self, 
                            n_clusters: int = None,
                            linkage_method: str = None,
                            max_features: int = None,
                            sm_ids: list = None,
                            feature_selection: str = None,
                            plot_figures: bool = None,
                            save_plots: bool = None,
                            save_results: bool = None,
                            save_meta_results: bool = None,
                            overwrite_existing_results: bool = None,
                            plot_load_profiles: bool = None
                            ) -> pd.DataFrame:
    
    # Overwrite config settings with arguments if provided (allows to dynamically change data app run in pipeline)
    self._update_config(args=locals().items())

    # From fewest_meters_substation, unpack the dict (which have the structure like this: {"2600": {"SecondarySubstation.706039": {"Transformer.125763": {"LvFeeder.364204": {"Cabinet.86493": {"METER_NUMBER": ["319672", "749355"], "AVAILABLE_METERS": ["319672", "749355"], "MISSING_METERS": []}, "Cabinet.531278": {"METER_NUMBER": [], "AVAILABLE_METERS": [], "MISSING_METERS": []}, "Cabinet.459597": {"METER_NUMBER": ["606018", "97540", "531845", "621915"], "AVAILABLE_METERS": ["606018", "97540", "531845", "621915"], "MISSING_METERS": []}, "Cabinet.966954": {"METER_NUMBER": ["440503", "449868", "101382", "683210"], "AVAILABLE_METERS": ["440503", "449868", "101382", "683210"], "MISSING_METERS": []}, "Cabinet.426548": {"METER_NUMBER": ["769621", "620793", "363757"], "AVAILABLE_METERS": ["76...) (and mulitiple zip codes not just 2600), and find the  SecondarySubstation with fewest available meters, and extract that substations name
    fewest_meters_substation = os.path.join(self.DataExtractor.processed_data_dir, "topology", "raw", "Raw_Topology_SM_mapping.json")
    
    # Unpact the dict
    with open(fewest_meters_substation, 'r') as f:
        topology_dict = json.load(f)
    
    def count_available_meters(substation):
        total = 0
        for transformer in substation.values():
            for feeder in transformer.values():
                for cabinet in feeder.values():
                    total += len(cabinet.get("AVAILABLE_METERS", []))
        return total

    min_count = float("inf")
    min_substation = None

    for zip_code, substations in topology_dict.items():
        for substation_name, substation_data in substations.items():
            count = count_available_meters(substation_data)
            if 30 < count < min_count:
                min_count = count
                min_substation = substation_name

    # Extract just the numeric ID
    substation_few = min_substation.split(".")[-1] if min_substation else None
    logging.info(f"Substation with fewest available meters: {substation_few} ({min_count} meters)")
    
    substation_few = int(substation_few)

    # Check if results already exist
    hierarchical_result_path = os.path.join(self.DIR_DATA_APP, "Results", self.result_name, 'hierarchical_clustering', 'hierarchical_clustering_predictions.csv')
    if os.path.exists(hierarchical_result_path) and not self.overwrite_existing_results:
        logging.info(f"Hierarchical run of {self.result_name} already exists. Loading and returning results...")
        
        # Extract results
        results = pd.read_csv(hierarchical_result_path)
        return results

    # Create results directory for saving plots, meta data, etc.
    results_dir = os.path.join(self.DIR_DATA_APP, "Results", self.result_name, 'hierarchical_clustering')
    os.makedirs(results_dir, exist_ok=True)

    # Set up meta data dict
    if self.save_meta_results:      
        meta_results = {"Feature Selection": self.feature_selection,
                        "Classifying Threshold": "N/A",
                        "Selected Features": None,
                        "Total SMs": None,
                        "Total Phase Profiles": None,
                        "Labeled EH Phase Profiles": None,
                        "Identified 1-Phase Meters": None,
                        "Identified 2-Phase Meters": None,
                        "Identified 3-Phase Meters": None,
                        "Coverage of SMs with EH": None,
                        "Recall of SM Phases with EH": None,
                        "Relation-Score": None,
                        "Silhouette Score": None,
                        "Percentage of Phases Clustered as EH": None
                        }


    # Load Feature Extraction
    logging.info("Loading features for hierarchical clustering...")
    features = self.feature_extraction(save_results = True, unit_ids=[f"{substation_few}"])
    features = features.dropna(how="all") # drop un-recorded phases
    features = features.fillna(0) # fill remaining NaNs with 0 for clustering

    # Create list of smart meter ids if not provided
    if not self.sm_ids:
        # Get smart meter ids from features index
        self.sm_ids = list(set([idx.split('_', 1)[1] for idx in features.index]))
    
    true_labels = None
    if self.use_labels:
        # Load True Labels only if label-dependent evaluation is enabled
        logging.info("Loading true labels for hierarchical clustering...")
        df_labels = self._load_EH_labels(sm_ids=self.sm_ids)
        df_labels_list = df_labels.stack().reset_index()
        df_labels_list.columns = ['meter_number', 'phase', 'has_EH']
        df_labels_list['meter_number'] = df_labels_list['phase'] + '_' + df_labels_list['meter_number'].astype(str)
        df_labels_list['meter_number'] = df_labels_list['meter_number'].str.replace('l', 'l', regex=False)
        df_labels_list.set_index('meter_number', inplace=True)
        df_labels_list = df_labels_list[['has_EH']]
        df_labels_list['has_EH'] = df_labels_list['has_EH'] > self.EH_threshold # If the confidence is over the threshold, consider it true
        true_labels = df_labels_list
    
    # Preprocessing for clustering
    logging.info("Performing hierarchical clustering...")
    agg = AgglomerativeClustering(n_clusters=self.n_clusters, linkage=self.linkage_method)
    
    logging.info(f"Selecting features via {self.feature_selection} selection..." if self.feature_selection else "No feature selection applied.")
    if self.feature_selection == "forward" and self.use_labels:
        best_features, _ = self._forward_selection(features, true_labels, max_features=self.max_features, n_clusters=self.n_clusters)
        X = features[best_features]
        logging.info(f"Selected features: {best_features}")
    elif self.feature_selection == "backward" and self.use_labels:
        best_features, _ = self._backward_selection(features, true_labels, max_features=self.max_features, n_clusters=self.n_clusters)
        X = features[best_features]
        logging.info(f"Selected features: {best_features}")
    elif self.feature_selection in ["forward", "backward"] and not self.use_labels:
        logging.warning("Feature selection requires labels. Using all features because use_labels=False.")
        best_features = features.columns.tolist()
        X = features
    else:
        logging.info("All features used.")
        best_features = features.columns.tolist()
        X = features

    logging.info(f"Features shape: {features.shape}, index sample: {features.index[:5].tolist()}")
    

    if self.save_meta_results:
        meta_results['Selected Features'] = best_features
    
    # Fit labels based on selected features
    labels = agg.fit_predict(X)
    results = pd.DataFrame({'cluster': labels}, index=X.index)
    logging.info("Hierarchical clustering completed.")

    # Silhouette Score
    if self.save_meta_results:
        if self.n_clusters > 1:
            sil_score = silhouette_score(X, labels)
        else:
            sil_score = "N/A"
        meta_results['Silhouette Score'] = sil_score
    
    if self.plot_figures:

        # # Set seaborn darkgrid style for all plots
        # sns.set_style('darkgrid')  
        # sns.set_palette("muted")
        # colors_tab = ['blue', 'tab:orange']
        # cmap_custom = mcolors.ListedColormap(colors_tab)
        # unique_clusters = np.unique(labels)
        # min_cluster = np.min(unique_clusters)
        # max_cluster = np.max(unique_clusters)
        # norm = Normalize(vmin=min_cluster, vmax=max_cluster)
        # EH_indices = true_labels[true_labels['has_EH'] == 1].index if self.use_labels else pd.Index([])
        

        # Plot dendrogram
        Z = linkage(X, method=self.linkage_method)
        plt.figure(figsize=(12, 6))
        dendrogram(Z, no_labels=True)
        plt.title('Hierarchical Clustering Dendrogram', fontsize=16, fontweight='bold')
        plt.xlabel('Smart meter phases', fontsize=14)
        plt.ylabel('Distance', fontsize=14)
        
        if self.save_plots:
            plt.savefig(os.path.join(results_dir, 'hierarchical_clustering_dendrogram.png'))
        else:
            plt.show()


        # Plot PCA projection with True Heat Pumps highlighted
        pca = PCA(n_components=2)
        pca_features = pca.fit_transform(X)
        plt.figure(figsize=(10, 8))
        scatter = plt.scatter(
            pca_features[:, 0], pca_features[:, 1],
            c=labels, cmap = cmap_custom, norm=norm, alpha=0.6, edgecolors='none'
        )

        if self.use_labels:
            # Plot true electric heating as circles with black edge
            plt.scatter(
                pca_features[results.index.isin(EH_indices), 0],
                pca_features[results.index.isin(EH_indices), 1],
                facecolors='none', edgecolors='black', linewidths=1.5,
                label='True Electric Heating', s=80, marker='o'
            )

        legend_elements = [plt.Line2D([0], [0], marker='o', color='w', label=f'Cluster {cluster_id}',
                                    markerfacecolor=scatter.cmap(norm(cluster_id)), markersize=8)
                        for cluster_id in unique_clusters]
        if self.use_labels:
            legend_elements.append(plt.Line2D([0], [0], marker='o', color='black', label='Electric Heating Index',
                                            markerfacecolor='none', markersize=8, linestyle='None'))
        
        legend = plt.legend(handles=legend_elements, title='Clusters and Labels', fontsize=14, title_fontsize=16)
        legend.get_frame().set_edgecolor('black')
        legend.get_frame().set_linewidth(0.8)
        plt.xlabel('Principal Component 1', fontsize=14)
        plt.ylabel('Principal Component 2', fontsize=14)
        plt.title(f'Agglomerative Hierarchical Results (PCA Projection)', fontsize=16, fontweight='bold')
        plt.grid(True)
        
        if self.save_plots:
            plt.savefig(os.path.join(results_dir, 'hierarchical_clustering_pca.png'))
        else:
            plt.show()

    cluster_counts = results['cluster'].value_counts()
    smallest_cluster_id = cluster_counts.idxmin()
    has_electric_heating = None
    if self.use_labels:
        results_aligned, true_labels_aligned = results.align(true_labels, join="inner", axis=0)
        total_true_labels = true_labels_aligned['has_EH'].sum()

        # Metrics Calculation
        ## Percentage: Calculate percentage of true labels in the smallest cluster only
        mask_smallest = results_aligned['cluster'] == smallest_cluster_id
        true_in_smallest = true_labels_aligned.loc[results_aligned.index[mask_smallest], 'has_EH'].sum()
        percentage_smallest = (true_in_smallest / total_true_labels) * 100 if total_true_labels > 0 else 0

        ## Coverage: Calculate the percentage of smart meters with at least one phase in the smallest cluster
        meter_numbers_in_smallest_cluster = results_aligned.index[results_aligned['cluster'] == smallest_cluster_id].map(lambda x: x.split("_")[1])
        unique_meters_in_smallest_cluster = set(meter_numbers_in_smallest_cluster)
        all_meter_numbers = set(true_labels_aligned.index.map(lambda x: x.split("_")[1]))
        coverage_smallest = (len(unique_meters_in_smallest_cluster) / len(all_meter_numbers)) * 100

        ## Cluster-Recall: Calculate recall of true labels in the smallest cluster
        has_electric_heating = true_labels_aligned['has_EH']
        found_profiles = results_aligned.index[(has_electric_heating == 1) & (results_aligned['cluster'] == smallest_cluster_id)]
        recall_smallest = has_electric_heating[found_profiles].sum() / has_electric_heating.sum() if has_electric_heating.sum() > 0 else 0

        ## Relation-Score: Final score as the average of coverage and recall
        relation_score = (coverage_smallest + (recall_smallest * 100)) / 2

    # Extract information on 1-, 2-, and 3-phase meters from predicted HP labels
    EH_indices = results[results['cluster'] == smallest_cluster_id].index
    meter_numbers = EH_indices.map(lambda x: x.split("_")[1])
    duplicate_meter_counts = meter_numbers.value_counts() # Count occurrences of each meter number
    one_phase = duplicate_meter_counts[duplicate_meter_counts == 1] # Extract 1-phase meters
    two_phase = duplicate_meter_counts[duplicate_meter_counts == 2] # Extract 2-phase meters
    three_phase = duplicate_meter_counts[duplicate_meter_counts == 3] # Extract 3-phase meters


    # Save metrics to meta results
    if self.save_meta_results:
        meta_results["Total SMs"] = f"{len(self.sm_ids)}"
        meta_results["Total Phase Profiles"] = f"{len(X)}"
        meta_results["Labeled EH Phase Profiles"] = f"{len(EH_indices)}"
        meta_results["Identified 1-Phase Meters"] = f"{len(one_phase)}"
        meta_results["Identified 2-Phase Meters"] = f"{len(two_phase)}"
        meta_results["Identified 3-Phase Meters"] = f"{len(three_phase)}"
        if self.use_labels:
            meta_results["Coverage of SMs with EH"] = f"{coverage_smallest:.2f}%"
            meta_results["Recall of SM Phases with EH"] = f"{recall_smallest:.2f}"
            meta_results['Relation-Score'] = f"{relation_score:.2f}"
            meta_results['Percentage of Phases Clustered as EH'] = f"{percentage_smallest:.2f}%"
        else:
            meta_results["Coverage of SMs with EH"] = "N/A"
            meta_results["Recall of SM Phases with EH"] = "N/A"
            meta_results['Relation-Score'] = "N/A"
            meta_results['Percentage of Phases Clustered as EH'] = "N/A"

        # Save meta results as json
        with open(os.path.join(results_dir, 'hierarchical_clustering_meta_results.json'), 'w') as f:
            json.dump(meta_results, f, indent=4)

    # Convert cluster to boolean: smallest cluster = True, others = False
    results['cluster'] = results['cluster'] == smallest_cluster_id
    results = results.rename(columns={'cluster': 'predicted_EH'})

    # Save predictions from label propagation
    if self.save_results:
        results.to_csv(os.path.join(results_dir, 'hierarchical_clustering_predictions.csv'))
        logging.info(f"Hierarchical clustering results saved to {hierarchical_result_path}")

    if self.plot_load_profiles and self.use_labels:
        self._plot_load_profiles(results, results_dir, has_electric_heating, method_name='Hierarchical Clustering')
        logging.info(f'Created load profile for hierarchical clustering')
    elif self.plot_load_profiles and not self.use_labels:
        logging.warning("plot_load_profiles requires labels. Skipping because use_labels=False.")

    return results

def label_propagation(feature_results, sm_ids, cfg) -> pd.DataFrame:
    
    # Overwrite config settings with arguments if provided (allows to dynamically change data app run in pipeline)


    # TODO: Add meta results to meta.run_results as JSON
    # # Set up meta data dict
    # if self.save_meta_results:
    #     meta_results = {"Feature Selection": "N/A",
    #                     "Classifying Threshold": self.label_threshold,
    #                     "Selected Features": None,
    #                     "Total SMs": None,
    #                     "Total Phase Profiles": None,
    #                     "Labeled EH Phase Profiles": None,
    #                     "Identified 1-Phase Meters": None,
    #                     "Identified 2-Phase Meters": None,
    #                     "Identified 3-Phase Meters": None,
    #                     "Coverage of SMs with EH": None,
    #                     "Recall of SM Phases with EH": None,
    #                     "Relation-Score": None,
    #                     "Kernel": self.kernel,
    #                     "Gamma Value": self.label_gamma,
    #                     }


    # Load Feature Extraction
    s3_connector = S3Connector(data_dir_path="phase_measurements")
    meta_controller = MetaController(threephi_db.new_session)

    features = feature_results.dropna(how="all") # drop un-recorded phases
    features = features.fillna(0) # fill remaining NaNs with 0 for clustering
    
    # TODO: Add integration with StatLabeler
    logging.info(f"Checking if results already exist for smart meters.")
    lack_workflow = []
    for sm_id in sm_ids:
        workflow_completed = meta_controller.is_workflow_completed(f"stat_labeling_sm_{sm_id}")
        if not workflow_completed:
            lack_workflow.append(sm_id)
    if lack_workflow:
        raise ValueError(f"StatLabeling workflow has not been completed for the following smart meters: {lack_workflow}. Please run the StatLabeler workflow for these meters before running label propagation.")

    # Load the per-phase ground thruth labels from StatLabeler and store them in a dataframe
    # Save the label as a boolean based on whether the confidence is above the EH threshold defined in the config
    rows = []
    for sm_id in sm_ids:
        meta_results = meta_controller.query_run_results(source="StatLabeler", meter_id=int(sm_id))
        for res in meta_results:
            rows.append({
                "meter_id": int(res.meter_id),
                "phase": res.phase.lower(),
                "confidence": res.confidence,
                "has_EH": res.confidence > cfg["EH_threshold"]
            })
    df_labels = pd.DataFrame(rows)
    df_labels["meter_phase"] = df_labels["phase"] + "_" + df_labels["meter_id"].astype(str)
    df_labels = df_labels.set_index("meter_phase")
    true_labels = df_labels[["has_EH"]]
    true_labels = true_labels.reindex(features.index).fillna(False)



    # Load True Labels
    # logging.info("Loading true labels for label propagation...")
    # df_labels = self._load_EH_labels(sm_ids=self.sm_ids)
    # df_labels_list = df_labels.stack().reset_index()
    # df_labels_list.columns = ['meter_number', 'phase', 'has_EH']
    # df_labels_list['meter_number'] = df_labels_list['phase'] + '_' + df_labels_list['meter_number'].astype(str)
    # df_labels_list['meter_number'] = df_labels_list['meter_number'].str.replace('l', 'l', regex=False)
    # df_labels_list.set_index('meter_number', inplace=True)
    # df_labels_list = df_labels_list[['has_EH']]
    # df_labels_list['has_EH'] = df_labels_list['has_EH'] > self.EH_threshold # If the confidence is over the threshold, consider it true
    # true_labels = df_labels_list

    
    # Initilize preprocessing for label propagation

    ## Preprocessing for label propagation
    remaining_labels, false_features = _feature_based_pruning_for_propagation(features, true_labels)
    print(f"Remaining labels after pruning: {len(remaining_labels)}")
    print(f"False features after pruning: {len(false_features)}")

    ## Check for overlap only in the index (row labels)
    overlap = set(remaining_labels.index) & set(false_features.index)
    if overlap:
        logging.info(f"Overlap found in index: {overlap}")
        logging.info(f"Number of overlapping indices: {len(overlap)}")
        
        ## Remove overlapping indices from false_features
        false_features = false_features.loc[~false_features.index.isin(overlap)]
    
    ## Split true and false labels
    false_labels = true_labels.loc[false_features.index]

    ## Build input X for label propagation
    X = features

    positive_idx = true_labels.index[true_labels["has_EH"]].intersection(X.index)
    negative_idx = false_labels.index[false_labels["has_EH"] == False].intersection(X.index)
    print(f"Positive indices: {len(positive_idx)}")
    print(f"Negative indices: {len(negative_idx)}")

    
    if len(positive_idx) == 0:
        logging.warning("No positive labels found. Selecting highest-confidence candidate as seed.")

        valid_conf = df_labels["confidence"].reindex(X.index).dropna()

        if len(valid_conf) > 0:
            fallback = valid_conf.idxmax()
            positive_idx = pd.Index([fallback])
        else:
            fallback = features["temp_ratio_low_to_total"].idxmax()
            positive_idx = pd.Index([fallback])

        negative_idx = negative_idx.difference(positive_idx)


    y = pd.Series(-1, index=X.index, dtype=int, name="has_EH")
    y.loc[positive_idx] = 1
    y.loc[negative_idx] = 0
    y = y.to_frame()

    # Initilize and fit label propagation model
    logging.info("Performing label propagation...")

    model_lp = LabelPropagation(kernel=cfg["label_propagation"]["kernel"], gamma=cfg["label_propagation"]["gamma"])
    model_lp.fit(X.values, y['has_EH'].values)

    ## Get predictions
    predicted_labels_lp = model_lp.transduction_
    propagated_probabilities_lp = model_lp.label_distributions_[:, 1]  # Get probabilities for the positive class (EH)

    logging.info("Label propagation completed.")

    if cfg["save_plots"]:

        unlabeled_mask = (y['has_EH'] == -1).to_numpy()
        unlabeled_probabilities_lp = propagated_probabilities_lp[unlabeled_mask]

        # Histogram of predicted probabilities
        fig = plt.figure(figsize=(10, 6))
        plt.hist(propagated_probabilities_lp, bins=50, color='skyblue', edgecolor='black', alpha=0.7)
        plt.axvline(x=cfg["label_propagation"]["label_threshold"], color='red', linestyle='--', label=f'Confidence Threshold ({cfg["label_propagation"]["label_threshold"]})')
        plt.xlabel('Predicted Probability of Electric Heating')
        plt.ylabel('Frequency')
        plt.title('Distribution of Predicted Electric Heating Probabilities (Label Propagation)')
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        s3_path = f"{cfg['results_dir']}/label_propagation_histogram.png"
        s3_connector.save_plot(s3_path, fig, format="png", overwrite=True)
        plt.close(fig)

        # Histogram of predicted probabilities for unlabeled samples only
        fig = plt.figure(figsize=(10, 6))
        plt.hist(unlabeled_probabilities_lp, bins=50, color='skyblue', edgecolor='black', alpha=0.7)
        plt.axvline(x=cfg["label_propagation"]["label_threshold"], color='red', linestyle='--', label=f'Confidence Threshold ({cfg["label_propagation"]["label_threshold"]})')
        plt.xlabel('Predicted Probability of Electric Heating')
        plt.ylabel('Frequency')
        plt.title('Distribution of Predicted Electric Heating Probabilities (Unlabeled Samples Only)')
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        s3_path = f"{cfg['results_dir']}/label_propagation_histogram_unlabeled_only.png"
        s3_connector.save_plot(s3_path, fig, format="png", overwrite=True)
        plt.close(fig)

        # PCA plot
        pca = PCA(n_components=2) # Create PCA in 2 dimensions
        X_pca = pca.fit_transform(X)

        ## Map predicted labels to colors: 1 (HP) = orange, 0 (not HP) = blue
        colors_tab = ['blue','tab:orange']
        cmap_custom = mcolors.ListedColormap(colors_tab)

        fig = plt.figure(figsize=(10, 8))
        scatter = plt.scatter(
            X_pca[:, 0], X_pca[:, 1],
            c=predicted_labels_lp,
            cmap=cmap_custom,
            alpha=0.6,
            edgecolors='none'
        )
        ## Overlay EH-classified labels from true_labels with black edge
        true_EH_indices = true_labels[true_labels['has_EH'] == True].index
        mask_true_EH = np.array([idx in true_EH_indices for idx in X.index])
        plt.scatter(
            X_pca[mask_true_EH, 0],
            X_pca[mask_true_EH, 1],
            facecolors='none', edgecolors='black', linewidths=1.5,
            label='Index Electric Heating', s=80, marker='o'
        )
        legend_elements = [
            plt.Line2D([0], [0], marker='o', color='w', label='Classified not EH', markerfacecolor=colors_tab[0], markersize=8),
            plt.Line2D([0], [0], marker='o', color='w', label='Classified EH', markerfacecolor=colors_tab[1], markersize=8),
            plt.Line2D([0], [0], marker='o', color='black', label='Electric Heating Index', markerfacecolor='none', markersize=8, linestyle='None')
        ]
        legend = plt.legend(handles=legend_elements, title='Predicted Labels', fontsize=14, title_fontsize=16)
        legend.get_frame().set_edgecolor('black')
        legend.get_frame().set_linewidth(0.8)
        plt.title("Label Propagation Results (PCA Projection)", fontsize=16, fontweight='bold')
        plt.xlabel("PCA Component 1", fontsize=14)
        plt.ylabel("PCA Component 2", fontsize=14)
        plt.grid(True)
        plt.tight_layout()
        s3_path = f"{cfg['results_dir']}/label_propagation_pca.png"
        s3_connector.save_plot(s3_path, fig, format="png", overwrite=True)
        plt.close(fig)


    # Extract information on 1-, 2-, and 3-phase meters from predicted EH labels

    EH_indices = y[predicted_labels_lp == 1].index
    # meter_numbers = EH_indices.map(lambda x: x.split("_")[1])
    meter_numbers = EH_indices.map(lambda x: x[0])
    duplicate_meter_counts = meter_numbers.value_counts() # Count occurrences of each meter number
    one_phase = duplicate_meter_counts[duplicate_meter_counts == 1] # Extract 1-phase meters
    two_phase = duplicate_meter_counts[duplicate_meter_counts == 2] # Extract 2-phase meters
    three_phase = duplicate_meter_counts[duplicate_meter_counts == 3] # Extract 3-phase meters


    # Calculate performance metrics
    ## Due to the nature of label propagation, we cannot directly calculate recall or relation-score on the predicted labels since we used the true labels for training. if wished, we can evaluate how well the model guessed the hidden test labels that were not used during training.
    ## TODO: make a recall calculation, NOTE: would result in running the moddel twice, or redusing the quality of the result


    ### Calculate coverage_hp
    coverage = len(duplicate_meter_counts) / len(sm_ids) * 100

    
    # Add meta results
    ## Add information on labels to meta results
    # if cfg["save_meta_results"]:
    #     meta_results["Total SMs"] = f"{len(sm_ids)}"
    #     meta_results["Total Phase Profiles"] = f"{len(X)}"
    #     meta_results["Labeled EH Phase Profiles"] = f"{predicted_labels_lp.sum()}"
    #     meta_results["Identified 1-Phase Meters"] = f"{len(one_phase)}"
    #     meta_results["Identified 2-Phase Meters"] = f"{len(two_phase)}"
    #     meta_results["Identified 3-Phase Meters"] = f"{len(three_phase)}"
    #     meta_results["Coverage of SMs with EH"] = f"{coverage:.2f}%"

    df_results_lp = pd.DataFrame(predicted_labels_lp == 1, index=y.index, columns=['predicted_EH'])
    df_results_lp['confidence'] = model_lp.label_distributions_[:, 1]
    print("df_results_lp")
    print(df_results_lp)

    current_sm_id = None
    for (sm_id, phase), row in df_results_lp.iterrows():
        if sm_id != current_sm_id:
            if current_sm_id is not None:
                meta_controller.complete_workflow(workflow=f"label_propagation_sm_{current_sm_id}")
            meta_controller.start_workflow(workflow=f"label_propagation_sm_{sm_id}")
            current_sm_id = sm_id

        meta_controller.insert_run_result(
            dag_id=cfg.get("dag_id", "default_dag"),
            run_id=cfg.get("run_id", "default_run"),
            meter_id=int(sm_id),
            phase=phase,
            label_type="Electric Heating Label Propagation",
            label_value=str(row["predicted_EH"]),
            confidence=float(row["confidence"]),
            source="Electric Heating Identifier",
            result=None,
            topology_version=None,
            node_id=None,
            edge_id=None,
            cable_id=None,
        )

    # Complete the last meter
    if current_sm_id is not None:
        meta_controller.complete_workflow(workflow=f"label_propagation_sm_{current_sm_id}")
    

def logistic_regression(sm_ids: list = None,
                        max_features: int = None,
                        n_clusters: int = None,
                        feature_selection: str = None,
                        logistic_threshold: float = None,
                        plot_figures: bool = None,
                        save_plots: bool = None,
                        save_results: bool = None,
                        save_meta_results: bool = None,
                        overwrite_existing_results: bool = None,):
    
    # Check if results already exist
    logistic_result_path = os.path.join(self.DIR_DATA_APP, "Results", self.result_name, 'logistic_regression', 'logistic_regression_predictions.csv')
    if os.path.exists(logistic_result_path) and not self.overwrite_existing_results:
        logging.info(f"Logistic regression run of {self.result_name} already exists. Loading and returning results...")
        
        # Extract results
        results = pd.read_csv(logistic_result_path)
        return results

    # Create results directory for saving plots, meta data, etc.
    results_dir = os.path.join(self.DIR_DATA_APP, "Results", self.result_name, 'logistic_regression')
    os.makedirs(results_dir, exist_ok=True)

    # Set up meta data dict
    if self.save_meta_results:
        meta_results = {"Feature Selection": self.feature_selection,
                        "Classifying Threshold": self.logistic_threshold,
                        "Selected Features": None,
                        "Total SMs": None,
                        "Total Phase Profiles": None,
                        "Labeled EH Phase Profiles": None,
                        "Identified 1-Phase Meters": None,
                        "Identified 2-Phase Meters": None,
                        "Identified 3-Phase Meters": None,
                        "Coverage of SMs with EH": None,
                        "Recall of SM Phases with EH": None,
                        "Relation-Score": None
                        }


    # Load Feature Extraction
    logging.info("Loading features for logistic regression...")
    features = self.feature_extraction(save_results = True)
    features = features.dropna(how="all") # drop un-recorded phases
    features = features.fillna(0) # fill remaining NaNs with 0 for clustering

    # Create list of smart meter ids if not provided
    if not self.sm_ids:
        # Get smart meter ids from features index
        self.sm_ids = list(set([idx.split('_', 1)[1] for idx in features.index]))
    

    # Load True Labels
    logging.info("Loading true labels for logistic regression...")
    df_labels = self._load_EH_labels(sm_ids=self.sm_ids)
    df_labels_list = df_labels.stack().reset_index()
    df_labels_list.columns = ['meter_number', 'phase', 'has_EH']
    df_labels_list['meter_number'] = df_labels_list['phase'] + '_' + df_labels_list['meter_number'].astype(str)
    df_labels_list['meter_number'] = df_labels_list['meter_number'].str.replace('l', 'l', regex=False)
    df_labels_list.set_index('meter_number', inplace=True)
    df_labels_list = df_labels_list[['has_EH']]
    df_labels_list['has_EH'] = df_labels_list['has_EH'] > self.EH_threshold # If the confidence is over the threshold, consider it true
    true_labels = df_labels_list

    # Prepare X and y for logistic regression        
    ## Remove all meters from true_labels that are not in features
    true_labels = true_labels.loc[true_labels.index.intersection(features.index)]

    ## Add all meter numbers from features that are not already in true_labels, and set them to False
    missing_indices_features = features.index.difference(true_labels.index)
    if not missing_indices_features.empty:
        missing_df = pd.DataFrame(False, index=missing_indices_features, columns=['has_EH'])
        true_labels = pd.concat([true_labels, missing_df])
    
    logging.info(f"Selecting features via {self.feature_selection} selection..." if self.feature_selection else "No feature selection applied.")
    if self.feature_selection == "forward":
        best_features, _ = self._forward_selection(features, true_labels, max_features=self.max_features, n_clusters=self.n_clusters)
        X = features[best_features]
        logging.info(f"Selected features: {best_features}")
    elif self.feature_selection == "backward":
        best_features, _ = self._backward_selection(features, true_labels, max_features=self.max_features, n_clusters=self.n_clusters)
        X = features[best_features]
        logging.info(f"Selected features: {best_features}")
    else:
        logging.info("All features used.")
        best_features = features.columns.tolist()
        X = features

    if self.save_meta_results:
        meta_results['Selected Features'] = best_features

    ## Create y
    y = true_labels
    
    # Custom train-test split to ensure no data leakage between phases of the same meter
    ## Create training and test sets based on unique meter numbers
    xmeters = X.index.map(lambda x: x.split("_")[1])
    
    ## Select random 10% of xmeters for training
    x_test_meters = np.random.choice(xmeters.unique(), size=int(len(xmeters) * self.train_split), replace=False)
    
    ## Split X and y based on selected test meters
    X_test = X[xmeters.isin(x_test_meters)]
    X_train = X[~xmeters.isin(x_test_meters)]

    y_train = y.loc[X_train.index]
    y_test = y.loc[X_test.index]

    # Generate logistic regression model 
    log_model = make_pipeline(
        StandardScaler(),
        LogisticRegression(max_iter=1000, random_state=42, class_weight='balanced', solver='liblinear', C=1)
    )

    logging.info("Performing logistic regression...")
    ## Fit model
    log_model.fit(X_train, y_train.values.ravel())

    
    # Prepare results on entire dataset
    probs = log_model.predict_proba(X)[:, 1]
    mask = probs >= self.logistic_threshold
    results = pd.DataFrame({'predicted_EH': mask, 'logistic_probability': probs}, index=X.index)
    logging.info("Logistic regression completed.")
    
    # Generate predicition metrics
    ## Get predictions on test set
    probs_test = log_model.predict_proba(X_test)[:, 1]
    mask_test = probs_test >= self.logistic_threshold
    preds = mask_test.astype(int)

    # Calculate percentage of meters with at least one phase classified as true
    ## Get indices of predicted positives in test set
    X_test_pred = X_test[mask_test].index
    sm_ids_pred = X_test_pred.map(lambda x: x.split("_")[1])
    sm_ids_pred = set(sm_ids_pred)

    ## Get all unique sm_ids from the test set
    sm_ids_test = set(X_test.index.map(lambda x: x.split("_")[1]))

    # Generate percentage and recall metrics
    coverage = len(sm_ids_pred) / len(sm_ids_test) * 100
    recall = recall_score(y_test, preds)

    # Generate relation-score
    ## Final score: average of (100 - coverage) and (100 * recall_score)
    relation_score = ((100 - coverage) + 100 * recall) / 2

    # Extract information on 1-, 2-, and 3-phase meters from predicted EH labels
    EH_indices = results[results['predicted_EH'] == 1].index
    meter_numbers = EH_indices.map(lambda x: x.split("_")[1])
    duplicate_meter_counts = meter_numbers.value_counts() # Count occurrences of each meter number
    one_phase = duplicate_meter_counts[duplicate_meter_counts == 1] # Extract 1-phase meters
    two_phase = duplicate_meter_counts[duplicate_meter_counts == 2] # Extract 2-phase meters
    three_phase = duplicate_meter_counts[duplicate_meter_counts == 3] # Extract 3-phase meters

    # Add meta results
    if self.save_meta_results:
        meta_results["Total SMs"] = f"{len(self.sm_ids)}"
        meta_results["Total Phase Profiles"] = f"{len(X)}"
        meta_results["Labeled EH Phase Profiles"] = f"{len(EH_indices)}"
        meta_results["Identified 1-Phase Meters"] = f"{len(one_phase)}"
        meta_results["Identified 2-Phase Meters"] = f"{len(two_phase)}"
        meta_results["Identified 3-Phase Meters"] = f"{len(three_phase)}"
        meta_results["Coverage of SMs with EH"] = f"{coverage:.2f}%"
        meta_results["Recall of SM Phases with EH"] = f"{recall:.2f}"
        meta_results['Relation-Score'] = f"{relation_score:.2f}"

        # Save meta results as json
        with open(os.path.join(results_dir, 'label_propagation_meta_results.json'), 'w') as f:
            json.dump(meta_results, f, indent=4)

    if self.plot_figures:

        # Histogram of predicted probabilities
        plt.figure(figsize=(10, 6))
        plt.hist(results['logistic_probability'], bins=50, color='skyblue', edgecolor='black', alpha= 0.7)
        plt.axvline(x=self.logistic_threshold, color='red', linestyle='--', label=f'Confidence Threshold ({self.logistic_threshold})')
        plt.xlabel('Predicted Probability of Electric Heating')
        plt.ylabel('Frequency')
        plt.title('Distribution of Predicted Electric Heating Probabilities (Logistic Regression)')
        plt.legend()
        plt.grid(True)
        plt.tight_layout()

        if self.save_plots:
            plt.savefig(os.path.join(results_dir, 'logistic_regression_histogram.png'))
        plt.show()


    if self.save_results:
        results.to_csv(os.path.join(results_dir, 'logistic_regression_predictions.csv'))
        logging.info(f"Logistic regression results saved to {logistic_result_path}")

    return results



