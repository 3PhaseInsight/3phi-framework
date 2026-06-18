import logging
import time
import numpy as np
import pandas as pd

import threephi_framework.db.db as threephi_db
from threephi_framework import TopologyController
from threephi_framework import TimeSeriesController
from threephi_framework import MetaController
from threephi_framework.object_storage.s3_connector import S3Connector


def _load_feeder_data(sm_id, cfg):
    """Load feeder time-series data. Label loading is handled separately via _load_labels()."""

    topology_controller = TopologyController(threephi_db.new_session)
    s3_connector = S3Connector(data_dir_path="phase_measurements")
    timeseries_controller = TimeSeriesController(s3_connector)

    chain = topology_controller.get_topology_chain_for_meter(int(sm_id))
    feeder_id = chain[0].feeder_id if chain else None

    if feeder_id is None:
        raise ValueError(f"Could not determine feeder for SM {sm_id}. Check topology mapping.")
    current_feeder_id  = feeder_id

    meters_dict= topology_controller.get_meters_for_node(node_id=feeder_id, node_type='lv_feeder')
    
    meters_for_feeder = [str(m['id']) for m in meters_dict]
    sm_ids_of_feeder = meters_for_feeder   # cache for label loaders

    feeder_df = timeseries_controller.get_time_series_data(meter_ids=meters_for_feeder, processing_level=cfg["profile_processing_level"])
    feeder_df = feeder_df.compute()  # bring into memory for processing

    active_power_cols = ["active_power_p14_l1", "active_power_p14_l2", "active_power_p14_l3"]
    feeder_df = feeder_df[["timestamp", "meter_number"] + active_power_cols]
    feeder_df = feeder_df.pivot(index="timestamp", columns="meter_number", values=active_power_cols,)
    feeder_df.columns = [f"{active_power_col}_{meter_id}" for active_power_col, meter_id in feeder_df.columns]
    feeder_df.index = pd.to_datetime(feeder_df.index, utc=True)

    return feeder_df, current_feeder_id, sm_ids_of_feeder

# ── Label loaders — one per source ───────────────────────────────────────

def _load_hp_labels(sm_ids_of_feeder, cfg) -> pd.DataFrame:

    meta_controller = MetaController(threephi_db.new_session)
    
    ALL_ALGORITHMS = ["label_propagation", "logistic_regression", "hierarchical_clustering"]
    preferred = cfg['HP_ML_algorithm']
    fallback_algorithms = [a for a in ALL_ALGORITHMS if a != preferred]

    # Track algorithm per SM
    sm_algo_map = {}

    # First try preferred
    for sm_id in sm_ids_of_feeder:
        if meta_controller.is_workflow_completed(f"{preferred}_sm_{sm_id}"):
            sm_algo_map[sm_id] = preferred
        time.sleep(0.05)

    missing_labels = set(sm_ids_of_feeder) - set(sm_algo_map.keys())

    # Try fallbacks
    if missing_labels:
        print(f"Missing HP labels for {len(missing_labels)} SMs using {preferred}. Trying fallback algorithms...")

        for algorithm in fallback_algorithms:
            still_missing = []

            for sm_id in missing_labels:
                if meta_controller.is_workflow_completed(f"{algorithm}_sm_{sm_id}"):
                    sm_algo_map[sm_id] = algorithm
                    time.sleep(0.05)
                else:
                    still_missing.append(sm_id)

            missing_labels = set(still_missing)
            if not missing_labels:
                break

    # Fail if still missing
    if missing_labels:
        raise ValueError(
            f"No labels found for SM IDs: {', '.join(str(s) for s in missing_labels)} "
            "using any algorithm. Run 'electric_heating_identifier' first."
        )

    # Load label results
    ALGO_LABEL_MAP = {
        "label_propagation": "Electric Heating Label Propagation",
        "logistic_regression": "Electric Heating Logistic Regression",
        "hierarchical_clustering": "Electric Heating Hierarchical Clustering",
    }
        
    label_data = pd.DataFrame()
    for sm_id, algo in sm_algo_map.items():
        label_type = ALGO_LABEL_MAP[algo]

        meta_results = meta_controller.query_run_results(
            source="Electric Heating Identifier",
            meter_id=int(sm_id),
            phase=None,
            label_type=label_type
        )

        if not meta_results:
            continue

        rows = []
        for res in meta_results:
            rows.append({
                "sm_id": str(sm_id),
                "phase": res.phase,
                "label": res.label_value,
                "confidence": res.confidence,
                "label_type": res.label_type
            })

        df = pd.DataFrame(rows)
        label_data = pd.concat([label_data, df], ignore_index=True)

    if not label_data.empty:
        label_data['sm_id'] = label_data['sm_id'].astype(str)

        # Extract base id (same logic as before)
        label_data['base_sm_id'] = label_data['sm_id'].str.rsplit('_', n=1).str[-1]

        sm_ids_str = set(map(str, sm_ids_of_feeder))

        labels_for_feeder = label_data[
            label_data['base_sm_id'].isin(sm_ids_str) |
            label_data['sm_id'].isin(sm_ids_str)
        ].copy()
    else:
        labels_for_feeder = pd.DataFrame()

    return labels_for_feeder


    # label_path = os.path.join(
    #     self.DataExtractor.processed_data_dir,
    #     'label_results',
    #     'label_propagation_predictions.csv',
    # )

    # if not os.path.isfile(label_path):
    #     logging.warning(f"HP label file not found: {label_path}")
    #     return pd.DataFrame()

    # label_data = pd.read_csv(label_path)
    # if 'sm_id' not in label_data.columns:
    #     label_data = label_data.rename(columns={label_data.columns[0]: 'sm_id'})

    # label_data['sm_id']      = label_data['sm_id'].astype(str)
    # label_data['base_sm_id'] = label_data['sm_id'].str.rsplit('_', n=1).str[-1]

    # labels_for_feeder = label_data[
    #     label_data['base_sm_id'].isin(self._sm_ids_of_feeder) |
    #     label_data['sm_id'].isin(self._sm_ids_of_feeder)
    # ].copy()

    # self.feeder_labels_by_feeder = self.feeder_labels_by_feeder or {}
    # self.feeder_labels_by_feeder[self.current_feeder_id] = labels_for_feeder
    # self.labels_for_used_sm_ids = labels_for_feeder.copy()

    # return labels_for_feeder

def _load_ev_labels(sm_ids_of_feeder, cfg) -> pd.DataFrame:
    """
    Load EV labels from anna_evs_detect/EV_detection_reduced_60000_SM.csv.
    Reshapes the source into one row per phase per SM in the form:
        sm_id (e.g. 'L1_12345'), predicted_ev_phase (bool), confidence_score
    filtered to the current feeder. Returns empty DataFrame on failure.
    # """

    pass

    # TODO: fix this later
    # ev_label_path = os.path.join(
    #     os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    #     'anna_evs_detect',
    #     'EV_detection_reduced_60000_SM.csv',
    # )

    # if not os.path.isfile(ev_label_path):
    #     logging.warning(f"EV label file not found: {ev_label_path}")
    #     return pd.DataFrame()

    # try:
    #     ev_data = pd.read_csv(ev_label_path)
    # except Exception as e:
    #     logging.error(f"Error loading EV labels: {e}")
    #     return pd.DataFrame()

    # if 'EV_detected' not in ev_data.columns:
    #     raise KeyError(f"'EV_detected' column not found in {ev_label_path}")

    # # Normalise SM ID to plain numeric string matching sm_ids_of_feeder
    # ev_data['base_sm_id'] = (
    #     ev_data['SM_ID']
    #     .astype(str)
    #     .str.replace('.parquet', '', regex=False)
    #     .str.replace('sm_', '', regex=False)
    #     .str.strip()
    # )

    # ev_data = ev_data[ev_data['base_sm_id'].isin(self._sm_ids_of_feeder)].copy()

    # if ev_data.empty:
    #     logging.warning(f"No EV labels found for feeder {self.current_feeder_id}")
    #     return pd.DataFrame()

    # logging.info(f"Loaded {len(ev_data)} EV label rows for feeder {self.current_feeder_id}")

    # ev_data = ev_data.rename(columns={'EV_detected': 'predicted_ev_phase'})
    # ev_data['predicted_ev_phase'] = ev_data['predicted_ev_phase'].str.strip().str.lower() == 'yes'
    # base_sm_suffix = ev_data['base_sm_id'].str.extract(r'(\d+)$')[0]

    # phase_frames = []
    # for phase in ['L1', 'L2', 'L3']:
    #     phase_df           = ev_data[['base_sm_id', 'predicted_ev_phase', 'active_phases', 'confidence_score']].copy()
    #     phase_df['sm_id']  = phase + '_' + base_sm_suffix
    #     phase_key          = phase.lower()
    #     phase_df['predicted_ev_phase'] = phase_df.apply(
    #         lambda row: bool(row['predicted_ev_phase']) and (
    #             pd.isna(row['active_phases']) or           # nan → phase info unknown, assume active
    #             phase_key in str(row['active_phases']).lower()
    #         ),
    #         axis=1,
    #     )
    #     phase_frames.append(phase_df[['sm_id', 'predicted_ev_phase', 'confidence_score']])

    # ev_labels = pd.concat(phase_frames, ignore_index=True)
    # ev_labels = ev_labels.dropna(subset=['sm_id'])
    # ev_labels['_sort_key'] = ev_labels['sm_id'].str.extract(r'_(\d+)$')[0].astype(int)
    # ev_labels = (
    #     ev_labels
    #     .sort_values('_sort_key')
    #     .drop('_sort_key', axis=1)
    #     .reset_index(drop=True)
    # )
    # return ev_labels

# ── Shared computation — single source of truth ───────────────────────────

def _compute_distributions(feeder_df, labels_df, sm_id):
    """
    Computes the three distributions used for both pie charts and phase scoring.

    Returns
    -------
    phase_consumption : pd.Series  feeder-wide energy per phase (L1/L2/L3)
    sm_id_consumption : pd.Series  household energy per phase
    phase_counts      : pd.Series  same-type appliance count per phase
    power_cols        : list       columns that matched power_col_filter
    """
    numeric_df = feeder_df.select_dtypes(include=[np.number])

    power_cols = [c for c in numeric_df.columns]

    def phase_labels_for(cols):
        return (
            pd.Series(cols, index=cols)
            .str.lower()
            .str.extract(r'(?:^|_)l([123])(?:_|$)', expand=False)
            .map({'1': 'L1', '2': 'L2', '3': 'L3'})
        )

    # C1 — feeder-wide energy per phase
    feeder_col_energy = numeric_df[power_cols].abs().sum()
    phase_consumption = (
        feeder_col_energy
        .groupby(phase_labels_for(power_cols)).sum()
        .reindex(['L1', 'L2', 'L3'], fill_value=0)
    )

    # C3 — household energy per phase for this SM
    sm_power_cols = [c for c in power_cols if sm_id.lower() in c.lower()]
    if sm_power_cols:
        sm_col_energy     = numeric_df[sm_power_cols].abs().sum()
        sm_id_consumption = (
            sm_col_energy
            .groupby(phase_labels_for(sm_power_cols)).sum()
            .reindex(['L1', 'L2', 'L3'], fill_value=0)
        )
    else:
        print(f"No power columns found for SM ID '{sm_id}' in feeder data. Check column names and SM ID formatting.")
        sm_id_consumption = pd.Series({'L1': 1.0, 'L2': 1.0, 'L3': 1.0})

    # C2 — same-type appliance count per phase
    # pred_col = label_columns.get(cfg["phase_scoring"]["appliance_type"], 'predicted_hp')
        
    if not labels_df.empty:
        typed = labels_df[labels_df['label'] == True].copy()

        phase_counts = (
            typed['phase']
            .value_counts()
            .reindex(['L1', 'L2', 'L3'], fill_value=0)
        )
    else:
        logging.warning("No labels available — C2 set to equal counts.")
        phase_counts = pd.Series({'L1': 1, 'L2': 1, 'L3': 1})


    return phase_consumption, sm_id_consumption, phase_counts, power_cols

# ── Pie charts ────────────────────────────────────────────────────────────

def plot_phase_distributions(self):
    """
    Plots three pie charts:
        1. Feeder-wide phase energy
        2. Household SM phase energy
        3. Same-type appliance count per phase (routed by appliance_type)
    """
    import matplotlib.pyplot as plt

    logging.info("Plotting phase distributions")
    feeder_df = self._load_feeder_data()
    labels_df = self._load_labels(self.appliance_type)
    phase_consumption, sm_id_consumption, phase_counts, _ = \
        self._compute_distributions(feeder_df, labels_df)

    sm_id = str(self.sm_ids).strip()

    for data, title in [
        (phase_consumption[phase_consumption > 0],
            f"Phase-wise Power Consumption — Feeder {self.current_feeder_id}"),
        (sm_id_consumption[sm_id_consumption > 0],
            f"Phase-wise Power Consumption — SM {sm_id}"),
        (phase_counts[phase_counts > 0],
            f"{self.appliance_type.upper()} Distribution Across Phases — Feeder {self.current_feeder_id}"),
    ]:
        if data.empty:
            logging.warning(f"No data to plot for: {title}")
            continue
        plt.figure(figsize=(8, 8))
        plt.pie(data, labels=data.index, autopct='%1.1f%%', startangle=140)
        plt.title(title)
        plt.axis('equal')
        plt.show()

# ── Phase recommendation ──────────────────────────────────────────────────

def recommend_phase(sm_id, cfg):
    """
    Recommend which phase (L1/L2/L3) to connect a new appliance to.

    Uses _compute_distributions() — the exact same values shown in the
    pie charts — to score each phase on:
        C1 (config weight) — feeder phase energy share
        C2 (config weight) — same-type appliance count share
        C3 (config weight) — household phase energy share

    All inputs converted to share of total (ideal = 0.333 per phase).
    HP/EV: lower share → higher score (avoid loaded phases).
    PV:    C1 and C3 inverted (prefer loaded phases — generation offsets load).

    Saves result as JSON and returns the dict.
    """

    app   = cfg["appliance_type"]
    is_pv = app == 'pv'
    IDEAL = 1 / 3
    W = cfg["weights"]
    feeder_df, current_feeder_id, sm_ids_of_feeder = _load_feeder_data(sm_id, cfg)
    if cfg["appliance_type"] == "ev":
        labels_df = _load_ev_labels()
    else:        
        labels_df = _load_hp_labels(sm_ids_of_feeder, cfg)

    phase_consumption, sm_id_consumption, phase_counts, _ = \
        _compute_distributions(feeder_df, labels_df, sm_id)
    

    def to_share(series):
        total = series.sum()
        return series / total if total > 0 else pd.Series({'L1': IDEAL, 'L2': IDEAL, 'L3': IDEAL})

    feeder_share = to_share(phase_consumption)
    count_share  = to_share(phase_counts)
    hh_share     = to_share(sm_id_consumption)

    phases = ['L1', 'L2', 'L3']
    scores = {}
    for ph in phases:
        c1 = (feeder_share[ph] / IDEAL) * W['C1_feeder_balance'] if is_pv \
                else max(0.0, 1 - feeder_share[ph] / IDEAL) * W['C1_feeder_balance']

        c2 = max(0.0, 1 - count_share[ph] / IDEAL) * W['C2_type_concentration']

        c3 = (hh_share[ph] / IDEAL) * W['C3_household_balance'] if is_pv \
                else max(0.0, 1 - hh_share[ph] / IDEAL) * W['C3_household_balance']

        scores[ph] = {
            'C1_feeder_balance':     float(round(c1, 1)),
            'C2_type_concentration': float(round(c2, 1)),
            'C3_household_balance':  float(round(c3, 1)),
            'total':                 float(round(c1 + c2 + c3, 1)),
        }

    ranking = sorted(
        phases,
        key=lambda p: (scores[p]['total'], scores[p]['C2_type_concentration'], scores[p]['C1_feeder_balance']),
        reverse=True,
    )

    result = {
        'sm_id':             str(sm_id),
        'feeder_id':         str(current_feeder_id),
        'appliance_type':    app,
        'recommended_phase': ranking[0],
        'ranking':           ranking,
        'scores':            scores,
        'input_shares': {
            ph: {
                'feeder_share': float(round(feeder_share[ph], 3)),
                'count_share':  float(round(count_share[ph],  3)),
                'hh_share':     float(round(hh_share[ph],     3)),
            }
            for ph in phases
        },
        'phase_consumption': {ph: float(round(phase_consumption[ph], 3)) for ph in phases},
        'sm_id_consumption': {ph: float(round(sm_id_consumption[ph], 3)) for ph in phases},
        'phase_counts':      {ph: int(phase_counts[ph]) for ph in phases},
    }

    meta_controller = MetaController(threephi_db.new_session)
    APPLICATION_LABEL_MAP = {
        "ev": "EV implementation",
        "hp": "Heat Pump implementation",
        "pv": "PV implementation",
    }

    print(f"DEBUG: app={app!r}, type={type(app)}", flush=True)
    logging.info(f"DEBUG: app={app!r}, type={type(app)}")

    meta_controller.insert_run_result(
    dag_id=cfg.get("dag_id", "default_dag"),
    run_id=cfg.get("run_id", "default_run"),
    meter_id=int(sm_id),
    phase=ranking[0],
    label_type="Phase Connector Recommendation",
    label_value=APPLICATION_LABEL_MAP[app],
    confidence=0.0,
    source="Phase Connector",
    result=result,
    topology_version=None,
    node_id=None,
    edge_id=None,
    cable_id=None,
    )

    return result
    