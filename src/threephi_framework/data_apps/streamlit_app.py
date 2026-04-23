import json
import logging
import os
import random
import re
import sys
from datetime import datetime
from pathlib import Path

import altair as alt
import pandas as pd
import streamlit as st
from sqlalchemy import func, select, text

try:
    from streamlit.runtime.scriptrunner_utils.exceptions import ScriptControlException
except Exception:  # pragma: no cover - compatibility with older/newer Streamlit layouts
    ScriptControlException = None

# Ensure package imports work when running via:
# streamlit run src/threephi_framework/data_apps/streamlit_app.py
CURRENT_FILE = Path(__file__).resolve()
SRC_DIR = CURRENT_FILE.parents[2]
APP_ROOT = CURRENT_FILE.parents[3]
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

import threephi_framework.db.db as threephi_db  # noqa: E402
from threephi_framework.controllers.topology import TopologyController  # noqa: E402
from threephi_framework.data_apps.sm_classifier import SMClassifier  # noqa: E402
from threephi_framework.data_apps.stat_labeler import StatLabeler  # noqa: E402
from threephi_framework.data_apps.timeseries_ingestor import TimeseriesIngestor  # noqa: E402
from threephi_framework.data_apps.topology_ingestor import TopologyIngestor  # noqa: E402
from threephi_framework.data_extractor.data_extractor import DataExtractor  # noqa: E402
from threephi_framework.models.meta.meter import MetaMeterModel  # noqa: E402
from threephi_framework.models.topology.assets.feeder import FeederModel  # noqa: E402
from threephi_framework.models.topology.assets.transformer import TransformerModel  # noqa: E402
from threephi_framework.models.topology.graph.cable import CableModel  # noqa: E402
from threephi_framework.models.topology.graph.node import NodeModel  # noqa: E402
from threephi_framework.models.topology.graph.topology_version import TopologyVersionModel  # noqa: E402
from threephi_framework.object_storage.s3_connector import S3Connector  # noqa: E402

## Labels and helper functions for SM classifier results processing and presentation for the UI
SM_CLASSIFIER_CATEGORY_LABELS = {
    "All_SMs": "All meters",
    "SMs_with_dataset_containing_data": "Datasets with usable data",
    "SMs_with_dataset_containing_no_data": "Datasets without usable data",
    "SMs_without_dataset": "Missing dataset",
    "SMs_with_incomplete_topology_info": "Incomplete topology",
    "SMs_with_only_good_data_quality": "Only good data quality",
    "SMs_with_any_medium_or_bad_data_quality": "Any medium or bad data quality",
    "SMs_with_any_bad_data_quality": "Any bad data quality",
    "SMs_with_3-phase_connection": "Three-phase connection",
    "SMs_with_2-phase_connection": "Two-phase connection",
    "SMs_with_1-phase_connection": "One-phase connection",
    "SMs_with_connection_error": "Connection error",
    "SMs_with_on_off_switch": "On/off switching",
}

SM_CLASSIFIER_CATEGORY_ORDER = [
    "All_SMs",
    "SMs_with_dataset_containing_data",
    "SMs_with_dataset_containing_no_data",
    "SMs_without_dataset",
    "SMs_with_incomplete_topology_info",
    "SMs_with_only_good_data_quality",
    "SMs_with_any_medium_or_bad_data_quality",
    "SMs_with_any_bad_data_quality",
    "SMs_with_3-phase_connection",
    "SMs_with_2-phase_connection",
    "SMs_with_1-phase_connection",
    "SMs_with_connection_error",
    "SMs_with_on_off_switch",
]

VOLTAGE_QUALITY_METRIC_LABELS = {
    "NaN frac": "Missing readings",
    "Zero frac": "Zero readings",
    "Below Vlim frac": "Below voltage limit",
    "Frozen frac": "Frozen readings",
    "Total corruption frac": "Total corrupted readings",
    "summary": "Quality summary",
}

SM_CLASSIFIER_VARIABLE_LABELS = {
    "V": "Voltage",
    "P14": "Active power import",
    "P23": "Active power export",
    "Q12": "Reactive power inductive",
    "Q34": "Reactive power capacitive",
}

SM_CLASSIFIER_STATISTIC_LABELS = {
    "Min": "Minimum",
    "Max": "Maximum",
    "Mean": "Mean",
    "Std": "Standard deviation",
}

SM_CLASSIFIER_PLOT_FILTERS = {
    "All with dataset containing data": "SMs_with_dataset_containing_data",
    "With only good data quality": "SMs_with_only_good_data_quality",
    "With any medium or bad data quality": "SMs_with_any_medium_or_bad_data_quality",
    "With 1-phase connection": "SMs_with_1-phase_connection",
    "With 2-phase connection": "SMs_with_2-phase_connection",
    "With connection error": "SMs_with_connection_error",
    "With on off switch": "SMs_with_on_off_switch",
}


class StreamlitLogHandler(logging.Handler):
    def __init__(self, callback):
        super().__init__()
        self.callback = callback

    def emit(self, record):
        try:
            self.callback(self.format(record))
        except BaseException as exc:
            if ScriptControlException and isinstance(exc, ScriptControlException):
                return
            # Do not break the app execution due to UI log rendering issues.
            pass


@st.cache_resource
def get_extractor(data_dir_path: str) -> DataExtractor:
    """Initialize a shared DataExtractor and bind it to the selected dataset root."""
    extractor = DataExtractor()
    extractor.s3_connector = S3Connector(data_dir_path=data_dir_path)
    return extractor


class StreamlitOrchestrator:
    """Thin wrapper that orchestrates existing data apps with validated runtime config."""

    def __init__(self, data_dir_path: str):
        if not data_dir_path or not str(data_dir_path).strip():
            raise ValueError("data_dir_path is required.")
        self.data_dir_path = str(data_dir_path).strip()

    @staticmethod
    def _validate_workers(n_workers: int) -> int:
        workers = int(n_workers)
        if workers < 1:
            raise ValueError("n_workers must be >= 1.")
        return workers

    @staticmethod
    def _normalize_sm_ids(sm_ids: list[str]) -> list[str]:
        normalized = [str(sm_id).strip() for sm_id in sm_ids if str(sm_id).strip()]
        if not normalized:
            raise ValueError("At least one smart meter id is required.")
        return normalized

    # a method for timeseries ingestor. Inherits the logic from TimeseriesIngestor,
    # but allows to specify some of the config parameters from the UI.
    def run_timeseries_ingestor(
        self,
        csv_source_path: str,
        csv_file_pattern: str,
        parquet_destination_path: str,
        override: bool,
        n_workers: int,
    ) -> str:
        """ A Streamlit specific method to run the timeseries ingestor with config parameters from the UI."""
        workers = self._validate_workers(n_workers)
        cfg = {
            "dask": {"local": True, "n_workers": workers},
            "csv_source_path": csv_source_path,
            "csv_file_pattern": csv_file_pattern,
            "parquet_destination_path": parquet_destination_path,
            "override": override,
        }
        with TimeseriesIngestor(cfg) as app:
            app.run()
        return "Timeseries ingestion completed."

    # a method for topology ingestor. Inherits the logic from TopologyIngestor,
    # but allows to specify some of the config parameters from the UI.
    def run_topology_ingestor(
        self,
        topology_source_path: str,
        sm_cab_source_path: str,
        override: bool,
        n_workers: int,
    ) -> str:
        """ A Streamlit specific method to run the topology ingestor with config parameters from the UI."""
        workers = self._validate_workers(n_workers)
        cfg = {
            "dask": {"local": True, "n_workers": workers},
            "topology_source_path": topology_source_path,
            "sm_cab_source_path": sm_cab_source_path,
            "override": override,
        }
        with TopologyIngestor(cfg) as app:
            app.run()
        return "Topology ingestion completed."

    # a method for SM classifier. Inherits the logic from SMClassifier,
    # but allows to specify some of the config parameters from the UI.
    def run_sm_classifier(
        self,
        run_name: str,
        sm_ids: list[str],
        overwrite_existing_results: bool,
        n_workers: int,
    ) -> dict:
        """ A Streamlit specific method to run the SM classifier with config parameters from the UI."""
        workers = self._validate_workers(n_workers)
        normalized_sm_ids = self._normalize_sm_ids(sm_ids)
        final_run_name = run_name
        if overwrite_existing_results:
            final_run_name = f"{run_name}_{datetime.utcnow().strftime('%Y%m%d%H%M%S')}"

        cfg = {
            "Name": "sm_classifier_config_streamlit",
            "use_dask": True,
            "dask": {"local": True, "n_workers": workers},
            "run_name": final_run_name,
            "data_dir_path": self.data_dir_path,
            "sm_ids": normalized_sm_ids,
            "topology_processing_level": "raw",
            "overwrite_existing_raw_sm_datasets": False,
            "overwrite_topology_info": False,
            "overwrite_timeseries_info": False,
            "save_results": True,
            "save_plots": False,
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
                    "overwrite_plots": False,
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
        with SMClassifier(config=cfg) as app:
            sm_characterization, result_summary = app.classify_smart_meters(run_name=final_run_name)
        return {
            "run_name": final_run_name,
            "classified_meters": len(sm_characterization or {}),
            "summary_keys": sorted((result_summary or {}).keys()),
        }

    # a method for stat labeler. Inherits the logic from StatLabeler,
    # but allows to specify some of the config parameters from the UI.
    def run_stat_labeler(
        self,
        sm_ids: list[str], # reads from a given list fetched from DB based on data quality
        n_workers: int,
        overwrite_existing_results: bool,
        thresholds: dict,
        process_only_sm_with_hp: bool,
        filter_data: bool,
        label_summerhouse: bool,
        use_anova: bool,
        save_meta_results: bool,
        weather_file_local: str,
        progress_callback=None,
    ) -> dict:
        """ A Streamlit specific method to run the stat labeler with config parameters from the UI."""
        workers = self._validate_workers(n_workers)
        normalized_sm_ids = self._normalize_sm_ids(sm_ids)
        cfg = {
            "dask": {"local": True, "n_workers": workers},
            "sm_ids": normalized_sm_ids,
            "overwrite_existing_results": overwrite_existing_results,
            "filter_data": filter_data,
            "process_only_sm_with_hp": process_only_sm_with_hp,
            "label_summerhouse": label_summerhouse,
            "use_ANOVA": use_anova,
            "save_plots": False,
            "save_meta_results": save_meta_results,
            "data_dir_path": self.data_dir_path,
            "thresholds": thresholds,
            "results_dir": "s3://3phi/stat_labeler",
            "weather_file": "s3://3phi/stat_labeler/data/weather_data.csv",
            "weather_file_local": weather_file_local,
        }
        with StatLabeler(cfg) as app:
            if callable(progress_callback):
                app.progress_callback = progress_callback
            return app.stat_label_sm() or {}


def create_orchestrator(data_dir_path: str) -> StreamlitOrchestrator:
    """Create a short-lived orchestrator to avoid leaking app runtime state across reruns."""
    return StreamlitOrchestrator(data_dir_path=data_dir_path)


@st.cache_resource
def load_runtime_env() -> list[str]:
    """Load .env and report missing required variables."""
    env_file = APP_ROOT / ".env"
    if env_file.exists():
        for line in env_file.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, value = line.split("=", 1)
            key = key.strip()
            value = value.strip()
            # Preserve explicitly exported variables from the shell.
            os.environ.setdefault(key, value)

    required_vars = [
        "S3_ENDPOINT_URL",
        "S3_ACCESS_KEY",
        "S3_SECRET_KEY",
        "DB_USER",
        "DB_PASSWORD",
        "DB_HOST",
        "DB_PORT",
        "DB_NAME",
    ]
    return [name for name in required_vars if not os.environ.get(name)]


def _get_feature_name(col: str) -> str:
    if "_" not in col:
        return col
    return col.rsplit("_", 1)[0]


def _safe_entity_id(entity_id: str, label: str) -> str:
    value = entity_id.strip()
    if not value:
        raise ValueError(f"{label} is required.")
    return value


def _default_data_platform_data_dir() -> str:
    return str(APP_ROOT / "data-platform" / "data")


def _list_sm_classifier_runs() -> list[str]:
    results_root = CURRENT_FILE.parent / "Results"
    if not results_root.exists():
        return []
    return sorted([p.name for p in results_root.iterdir() if p.is_dir()], reverse=True)


def _safe_numeric_from_stat(text: str):
    if not isinstance(text, str):
        return None
    match = re.search(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?", text)
    if not match:
        return None
    try:
        return float(match.group(0))
    except ValueError:
        return None


def _days_since(value) -> int | None:
    if value in (None, "-", ""):
        return None
    try:
        ts = pd.to_datetime(value, utc=True, errors="coerce")
        if pd.isna(ts):
            return None
        return int((pd.Timestamp.utcnow().normalize() - ts.normalize()).days)
    except Exception:
        return None


@st.cache_data(show_spinner=False, ttl=300)
def _get_meter_ids_with_data_quality() -> list[str]:
    """Return meter IDs where data_quality is available."""
    session = threephi_db.new_session()
    try:
        stmt = (
            select(MetaMeterModel.id)
            .where(MetaMeterModel.data_quality.is_not(None))
            .order_by(MetaMeterModel.id.asc())
        )
        return [str(meter_id) for meter_id in session.execute(stmt).scalars().all()]
    finally:
        session.close()


@st.cache_data(show_spinner=False, ttl=300)
def _get_meter_ids_with_timeseries() -> list[str]:
    """Return meter IDs that have timeseries rows available."""
    session = threephi_db.new_session()
    try:
        stmt = select(MetaMeterModel.id).where(MetaMeterModel.total_rows > 0).order_by(MetaMeterModel.id.asc())
        return [str(meter_id) for meter_id in session.execute(stmt).scalars().all()]
    finally:
        session.close()


@st.cache_data(show_spinner=False, ttl=300)
def _get_sm_classifier_db_snapshot(limit: int = 5000) -> pd.DataFrame:
    """Load existing SM classifier results persisted to meter metadata."""
    session = threephi_db.new_session()
    try:
        stmt = (
            select(
                MetaMeterModel.id.label("meter_id"),
                MetaMeterModel.total_rows.label("total_rows"),
                MetaMeterModel.updated_at.label("updated_at"),
                MetaMeterModel.data_quality.label("data_quality"),
                MetaMeterModel.data_statistics.label("data_statistics"),
                MetaMeterModel.connectivity.label("connectivity"),
            )
            .where(
                MetaMeterModel.data_quality.is_not(None)
                | MetaMeterModel.data_statistics.is_not(None)
                | MetaMeterModel.connectivity.is_not(None)
            )
            .order_by(MetaMeterModel.updated_at.desc(), MetaMeterModel.id.asc())
            .limit(limit)
        )
        rows = session.execute(stmt).all()
        if not rows:
            return pd.DataFrame(
                columns=["meter_id", "total_rows", "updated_at", "data_quality", "data_statistics", "connectivity"]
            )
        return pd.DataFrame(rows)
    finally:
        session.close()


def _summarize_sm_classifier_db(snapshot_df: pd.DataFrame) -> tuple[dict, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    if snapshot_df is None or snapshot_df.empty:
        empty = pd.DataFrame()
        return {}, empty, empty, empty

    metrics = {
        "classified_meters": int(snapshot_df["meter_id"].nunique()),
        "meters_with_data_quality": int(snapshot_df["data_quality"].notna().sum()),
        "meters_with_connectivity": int(snapshot_df["connectivity"].notna().sum()),
        "meters_with_statistics": int(snapshot_df["data_statistics"].notna().sum()),
        "meters_with_connection_error": 0,
        "meters_with_on_off_switch": 0,
    }

    connected_phase_rows = []
    quality_rows = []
    preview_rows = []
    for row in snapshot_df.itertuples(index=False):
        connectivity = row.connectivity if isinstance(row.connectivity, dict) else {}
        data_quality = row.data_quality if isinstance(row.data_quality, dict) else {}

        connected_phases = connectivity.get("Connected Phases") if isinstance(connectivity, dict) else None
        connection_error = connectivity.get("Connection Error") if isinstance(connectivity, dict) else None
        switching_phases = connectivity.get("Switching Phases") if isinstance(connectivity, dict) else None

        if isinstance(connection_error, list) and connection_error:
            metrics["meters_with_connection_error"] += 1
        if isinstance(switching_phases, list) and switching_phases:
            metrics["meters_with_on_off_switch"] += 1

        connected_phase_rows.append(
            {
                "label": (
                    "3-phase" if isinstance(connected_phases, list) and len(connected_phases) == 3
                    else "2-phase" if isinstance(connected_phases, list) and len(connected_phases) == 2
                    else "1-phase" if isinstance(connected_phases, list) and len(connected_phases) == 1
                    else "Unknown"
                )
            }
        )

        for phase in ["L1", "L2", "L3"]:
            phase_quality = data_quality.get(phase, {}) if isinstance(data_quality, dict) else {}
            voltage_quality = phase_quality.get("V", {}) if isinstance(phase_quality, dict) else {}
            summary = voltage_quality.get("Summary") if isinstance(voltage_quality, dict) else None
            if summary:
                quality_rows.append({"phase": phase, "summary": summary})

        preview_rows.append(
            {
                "meter_id": row.meter_id,
                "connected_phases": ", ".join(connected_phases) if isinstance(connected_phases, list) else "-",
                "connection_error": ", ".join(connection_error) if isinstance(connection_error, list) else "-",
                "switching_phases": ", ".join(switching_phases) if isinstance(switching_phases, list) else "-",
                "updated_at": row.updated_at,
            }
        )

    connected_phase_df = pd.DataFrame(connected_phase_rows)
    if not connected_phase_df.empty:
        connected_phase_df = (
            connected_phase_df["label"].value_counts().rename_axis("label").reset_index(name="meters")
        )

    quality_df = pd.DataFrame(quality_rows)
    if not quality_df.empty:
        quality_df = (
            quality_df.groupby(["phase", "summary"], as_index=False).size().rename(columns={"size": "meters"})
        )

    preview_df = pd.DataFrame(preview_rows)
    return metrics, connected_phase_df, quality_df, preview_df


def _refresh_sm_classifier_state(run_name: str) -> None:
    _load_sm_classifier_run.clear()
    _get_sm_classifier_db_snapshot.clear()
    st.session_state["smc_existing_run_pending"] = run_name
    st.session_state["results_selected_run_pending"] = run_name


def _apply_pending_selection(widget_key: str, options: list[str]) -> None:
    if not options:
        return

    pending_key = f"{widget_key}_pending"
    pending_value = st.session_state.pop(pending_key, None)
    current_value = st.session_state.get(widget_key)

    if pending_value in options:
        st.session_state[widget_key] = pending_value
    elif current_value not in options:
        st.session_state[widget_key] = options[0]


def _summarize_sm_classifier_run(characterization: dict, classification: dict) -> tuple[dict[str, int], bool]:
    category_members, derived_summary_used = _get_sm_classifier_category_members(characterization, classification)
    category_counts = {key: len(value) for key, value in category_members.items()}
    return category_counts, derived_summary_used


def _get_sm_classifier_category_members(
    characterization: dict, classification: dict
) -> tuple[dict[str, list[str]], bool]:
    category_members = {
        key: sorted(str(meter_id) for meter_id in value)
        for key, value in classification.items()
        if isinstance(value, list)
    }
    if "All_SMs" not in category_members:
        category_members["All_SMs"] = sorted(str(meter_id) for meter_id in characterization)

    expected_keys = [
        "All_SMs",
        "SMs_with_dataset_containing_data",
        "SMs_with_dataset_containing_no_data",
        "SMs_without_dataset",
        "SMs_with_incomplete_topology_info",
        "SMs_with_only_good_data_quality",
        "SMs_with_any_medium_or_bad_data_quality",
        "SMs_with_any_bad_data_quality",
        "SMs_with_3-phase_connection",
        "SMs_with_2-phase_connection",
        "SMs_with_1-phase_connection",
        "SMs_with_connection_error",
        "SMs_with_on_off_switch",
    ]
    missing_keys = [key for key in expected_keys if key not in classification]
    if not missing_keys:
        return category_members, False

    derived_members = {key: [] for key in expected_keys}
    for meter_id, meter in characterization.items():
        if not isinstance(meter, dict):
            continue

        normalized_meter_id = str(meter_id)
        data_quality = meter.get("Data Quality", {})
        connectivity = meter.get("Connectivity", {})
        topology = meter.get("Topology", {})
        availability = meter.get("Dataset Availability", {})
        summaries = []

        derived_members["All_SMs"].append(normalized_meter_id)

        if isinstance(availability, dict):
            available = availability.get("Available")
            contains_data = availability.get("Contains Data")
            if available is True and contains_data is True:
                derived_members["SMs_with_dataset_containing_data"].append(normalized_meter_id)
            elif available is True and contains_data is False:
                derived_members["SMs_with_dataset_containing_no_data"].append(normalized_meter_id)
            elif available is False:
                derived_members["SMs_without_dataset"].append(normalized_meter_id)

        if isinstance(topology, dict) and topology:
            missing_topology = [value for value in topology.values() if value in (None, "") or pd.isna(value)]
            if missing_topology:
                derived_members["SMs_with_incomplete_topology_info"].append(normalized_meter_id)

        for phase_data in data_quality.values():
            if not isinstance(phase_data, dict):
                continue
            for entry in phase_data.values():
                if isinstance(entry, dict) and "Summary" in entry and entry["Summary"] is not None:
                    summaries.append(entry["Summary"])

        if summaries and all(summary == "Good" for summary in summaries):
            derived_members["SMs_with_only_good_data_quality"].append(normalized_meter_id)
        if any(summary in {"Medium", "Bad"} for summary in summaries):
            derived_members["SMs_with_any_medium_or_bad_data_quality"].append(normalized_meter_id)
        if any(summary == "Bad" for summary in summaries):
            derived_members["SMs_with_any_bad_data_quality"].append(normalized_meter_id)

        connected_phases = connectivity.get("Connected Phases")
        if isinstance(connected_phases, list):
            if len(connected_phases) == 3:
                derived_members["SMs_with_3-phase_connection"].append(normalized_meter_id)
            elif len(connected_phases) == 2:
                derived_members["SMs_with_2-phase_connection"].append(normalized_meter_id)
            elif len(connected_phases) == 1:
                derived_members["SMs_with_1-phase_connection"].append(normalized_meter_id)

        connection_error = connectivity.get("Connection Error")
        if isinstance(connection_error, list) and connection_error:
            derived_members["SMs_with_connection_error"].append(normalized_meter_id)

        switching_phases = connectivity.get("Switching Phases")
        if isinstance(switching_phases, list) and switching_phases:
            derived_members["SMs_with_on_off_switch"].append(normalized_meter_id)

    for key in missing_keys:
        category_members[key] = derived_members[key]

    return category_members, True


def _build_sm_classifier_voltage_quality_aggregate(characterization: dict) -> tuple[pd.DataFrame, pd.DataFrame]:
    summary_rows = []
    detail_rows = []

    for meter in characterization.values():
        if not isinstance(meter, dict):
            continue

        data_quality = meter.get("Data Quality", {})
        for phase in ["L1", "L2", "L3"]:
            phase_quality = data_quality.get(phase, {}) if isinstance(data_quality, dict) else {}
            voltage_quality = phase_quality.get("V", {}) if isinstance(phase_quality, dict) else {}
            summary = voltage_quality.get("Summary") if isinstance(voltage_quality, dict) else None
            detailed = voltage_quality.get("Detailed") if isinstance(voltage_quality, dict) else None

            if summary:
                summary_rows.append({"phase": phase, "summary": summary})

            if isinstance(detailed, dict):
                detail_rows.append(
                    {
                        "phase": phase,
                        "NaN frac": detailed.get("NaN frac"),
                        "Zero frac": detailed.get("Zero frac"),
                        "Below Vlim frac": detailed.get("Below Vlim frac"),
                        "Frozen frac": detailed.get("Frozen frac"),
                        "Total corruption frac": detailed.get("Total corruption frac"),
                    }
                )

    summary_df = pd.DataFrame(summary_rows)
    if not summary_df.empty:
        summary_df = summary_df.groupby(["phase", "summary"], as_index=False).size().rename(columns={"size": "meters"})

    detail_df = pd.DataFrame(detail_rows)
    if not detail_df.empty:
        metric_columns = [
            "NaN frac",
            "Zero frac",
            "Below Vlim frac",
            "Frozen frac",
            "Total corruption frac",
        ]
        detail_df[metric_columns] = detail_df[metric_columns].apply(pd.to_numeric, errors="coerce")
        detail_df = detail_df.groupby("phase", as_index=False)[metric_columns].mean().fillna(0.0)

    return summary_df, detail_df


def _format_sm_classifier_category_counts(category_counts: dict[str, int]) -> pd.DataFrame:
    ordered_keys = [key for key in SM_CLASSIFIER_CATEGORY_ORDER if key in category_counts]
    ordered_keys.extend(key for key in category_counts if key not in ordered_keys)

    rows = [
        {
            "category": SM_CLASSIFIER_CATEGORY_LABELS.get(key, key.replace("_", " ")),
            "count": category_counts[key],
        }
        for key in ordered_keys
    ]
    return pd.DataFrame(rows).set_index("category")


def _format_voltage_quality_detail_df(detail_df: pd.DataFrame) -> pd.DataFrame:
    if detail_df.empty:
        return detail_df
    rename_map = {key: value for key, value in VOLTAGE_QUALITY_METRIC_LABELS.items() if key in detail_df.columns}
    return detail_df.rename(columns=rename_map)


def _format_voltage_quality_meter_df(detail_df: pd.DataFrame) -> pd.DataFrame:
    if detail_df.empty:
        return detail_df
    rename_map = {key: value for key, value in VOLTAGE_QUALITY_METRIC_LABELS.items() if key in detail_df.columns}
    return detail_df.rename(columns=rename_map)


def _format_sm_classifier_category_label(category_key: str) -> str:
    return SM_CLASSIFIER_CATEGORY_LABELS.get(category_key, category_key.replace("_", " "))


def _flatten_sm_classifier_topology(characterization: dict) -> pd.DataFrame:
    rows = []
    for meter_id, meter in characterization.items():
        topology = meter.get("Topology", {}) if isinstance(meter, dict) else {}
        if not isinstance(topology, dict) or not topology:
            continue
        rows.append(
            {
                "meter_id": str(meter_id),
                "secondary_substation_id": topology.get("Secondary Substation ID"),
                "transformer_id": topology.get("Transformer ID"),
                "feeder_id": topology.get("Feeder ID"),
                "cabinet_id": topology.get("Cabinet ID"),
            }
        )
    return pd.DataFrame(rows)


def _flatten_sm_classifier_availability(characterization: dict) -> pd.DataFrame:
    rows = []
    for meter_id, meter in characterization.items():
        availability = meter.get("Dataset Availability", {}) if isinstance(meter, dict) else {}
        if not isinstance(availability, dict) or not availability:
            continue
        rows.append(
            {
                "meter_id": str(meter_id),
                "available": availability.get("Available"),
                "contains_data": availability.get("Contains Data"),
                "relative_length": availability.get("Relative Length"),
                "absolute_length": availability.get("Absolute Length"),
            }
        )
    return pd.DataFrame(rows)


def _flatten_sm_classifier_quality(characterization: dict) -> pd.DataFrame:
    rows = []
    for meter_id, meter in characterization.items():
        data_quality = meter.get("Data Quality", {}) if isinstance(meter, dict) else {}
        if not isinstance(data_quality, dict):
            continue
        for phase, phase_payload in data_quality.items():
            if not isinstance(phase_payload, dict):
                continue
            for variable, variable_payload in phase_payload.items():
                if not isinstance(variable_payload, dict):
                    continue
                detail = variable_payload.get("Detailed") if isinstance(variable_payload.get("Detailed"), dict) else {}
                rows.append(
                    {
                        "meter_id": str(meter_id),
                        "phase": phase,
                        "variable": variable,
                        "summary": variable_payload.get("Summary"),
                        "NaN frac": detail.get("NaN frac"),
                        "Zero frac": detail.get("Zero frac"),
                        "Below Vlim frac": detail.get("Below Vlim frac"),
                        "Frozen frac": detail.get("Frozen frac"),
                        "Total corruption frac": detail.get("Total corruption frac"),
                    }
                )
    return pd.DataFrame(rows)


def _flatten_sm_classifier_statistics(characterization: dict) -> pd.DataFrame:
    rows = []
    for meter_id, meter in characterization.items():
        data_statistics = meter.get("Data Statistics", {}) if isinstance(meter, dict) else {}
        if not isinstance(data_statistics, dict):
            continue
        for phase, phase_payload in data_statistics.items():
            if not isinstance(phase_payload, dict):
                continue
            for variable, variable_payload in phase_payload.items():
                if not isinstance(variable_payload, dict):
                    continue
                rows.append(
                    {
                        "meter_id": str(meter_id),
                        "phase": phase,
                        "variable": variable,
                        "Min": variable_payload.get("Min"),
                        "Max": variable_payload.get("Max"),
                        "Mean": variable_payload.get("Mean"),
                        "Std": variable_payload.get("Std"),
                    }
                )
    return pd.DataFrame(rows)


def _flatten_sm_classifier_connectivity(characterization: dict) -> pd.DataFrame:
    rows = []
    for meter_id, meter in characterization.items():
        connectivity = meter.get("Connectivity", {}) if isinstance(meter, dict) else {}
        if not isinstance(connectivity, dict) or not connectivity:
            continue
        connected_phases = connectivity.get("Connected Phases")
        connection_error = connectivity.get("Connection Error")
        switching_phases = connectivity.get("Switching Phases")
        rows.append(
            {
                "meter_id": str(meter_id),
                "connected_phases": ", ".join(connected_phases) if isinstance(connected_phases, list) else "-",
                "connected_phase_count": len(connected_phases) if isinstance(connected_phases, list) else 0,
                "connection_error_phases": ", ".join(connection_error) if isinstance(connection_error, list) else "-",
                "connection_error_count": len(connection_error) if isinstance(connection_error, list) else 0,
                "switching_phases": ", ".join(switching_phases) if isinstance(switching_phases, list) else "-",
                "switching_phase_count": len(switching_phases) if isinstance(switching_phases, list) else 0,
            }
        )
    return pd.DataFrame(rows)


def _select_sm_classifier_plot_columns(
    plot_df: pd.DataFrame, phases: list[str], variable_groups: list[str]
) -> list[str]:
    feature_name_map = {
        "Voltage": {"voltage"},
        "Active power": {"active_power_p14", "active_power_p23"},
        "Reactive power": {"reactive_power_q12", "reactive_power_q34"},
    }
    selected_features = set()
    for variable_group in variable_groups:
        selected_features.update(feature_name_map.get(variable_group, set()))

    selected_phases = {phase.lower() for phase in phases}
    numeric_columns = plot_df.select_dtypes(include=["number"]).columns.tolist()
    return [
        column
        for column in numeric_columns
        if _get_feature_name(column) in selected_features
        and any(column.lower().endswith(phase) for phase in selected_phases)
    ]


def _list_sm_classifier_plot_artifacts(run_name: str) -> list[Path]:
    run_dir = CURRENT_FILE.parent / "Results" / run_name / "Plots"
    if not run_dir.exists():
        return []
    return sorted(run_dir.rglob("*.svg"))


def _render_sm_classifier_classification_tab(
    category_members: dict[str, list[str]],
    category_counts: dict[str, int],
    widget_prefix: str,
    derived_summary_used: bool,
) -> None:
    if derived_summary_used:
        st.caption(
            "Some saved summary categories were missing for this run, so category members and counts "
            "were backfilled from the per-meter characterization data."
        )

    if category_counts:
        counts_df = _format_sm_classifier_category_counts(category_counts)
        st.markdown("**Classification category counts**")
        st.bar_chart(counts_df, use_container_width=True)

    available_categories = [key for key in SM_CLASSIFIER_CATEGORY_ORDER if key in category_members]
    available_categories.extend(key for key in category_members if key not in available_categories)
    if not available_categories:
        st.info("No classification categories are available for this run.")
        return

    selected_category = st.selectbox(
        "Classification category",
        options=available_categories,
        format_func=_format_sm_classifier_category_label,
        key=f"{widget_prefix}_classification_category",
    )
    member_ids = category_members.get(selected_category, [])
    c1, c2 = st.columns(2)
    c1.metric("Meters in category", len(member_ids))
    c2.metric(
        "Share of run",
        f"{(len(member_ids) / max(category_counts.get('All_SMs', 0), 1)):.0%}",
    )
    st.dataframe(pd.DataFrame({"meter_id": member_ids}), use_container_width=True)


def _render_sm_classifier_characterization_tab(
    characterization: dict,
    category_members: dict[str, list[str]],
    widget_prefix: str,
) -> None:
    scope_options = ["__all__"] + [
        key for key in SM_CLASSIFIER_CATEGORY_ORDER if key in category_members and key != "All_SMs"
    ]
    selected_scope = st.selectbox(
        "Meter scope",
        options=scope_options,
        format_func=lambda value: "All meters" if value == "__all__" else _format_sm_classifier_category_label(value),
        key=f"{widget_prefix}_characterization_scope",
    )
    scoped_meter_ids = (
        set(str(meter_id) for meter_id in characterization)
        if selected_scope == "__all__"
        else set(category_members.get(selected_scope, []))
    )
    scoped_characterization = {
        str(meter_id): payload
        for meter_id, payload in characterization.items()
        if str(meter_id) in scoped_meter_ids
    }

    st.caption(f"Showing characterization for {len(scoped_characterization)} meters in the selected scope.")

    block = st.selectbox(
        "Characterization section",
        options=["Topology", "Data availability", "Data quality", "Data statistics", "Connectivity"],
        key=f"{widget_prefix}_characterization_block",
    )

    if block == "Topology":
        topology_df = _flatten_sm_classifier_topology(scoped_characterization)
        if topology_df.empty:
            st.info("No topology connections are stored for this run.")
        else:
            st.dataframe(topology_df, use_container_width=True)
        return

    if block == "Data availability":
        availability_df = _flatten_sm_classifier_availability(scoped_characterization)
        if availability_df.empty:
            st.info("No data availability fields are stored for this run.")
        else:
            st.dataframe(availability_df, use_container_width=True)
            if "relative_length" in availability_df.columns:
                chart_df = availability_df[["meter_id", "relative_length"]].dropna().set_index("meter_id")
                if not chart_df.empty:
                    st.markdown("**Relative data length by meter**")
                    st.bar_chart(chart_df, use_container_width=True)
        return

    if block == "Data quality":
        quality_df = _flatten_sm_classifier_quality(scoped_characterization)
        if quality_df.empty:
            st.info("No data quality details are stored for this run.")
            return

        phase_options = sorted(quality_df["phase"].dropna().unique().tolist())
        variable_options = sorted(quality_df["variable"].dropna().unique().tolist())
        selected_phases = st.multiselect(
            "Phases",
            options=phase_options,
            default=phase_options,
            key=f"{widget_prefix}_dq_phases",
        )
        selected_variables = st.multiselect(
            "Variables",
            options=variable_options,
            default=variable_options,
            format_func=lambda value: SM_CLASSIFIER_VARIABLE_LABELS.get(value, value),
            key=f"{widget_prefix}_dq_variables",
        )
        filtered_df = quality_df[
            quality_df["phase"].isin(selected_phases) & quality_df["variable"].isin(selected_variables)
        ].copy()
        if filtered_df.empty:
            st.info("No data quality rows match the selected filters.")
            return

        summary_df = (
            filtered_df.groupby(["phase", "variable", "summary"], as_index=False)
            .size()
            .rename(columns={"size": "meters"})
        )
        summary_df["phase_variable"] = summary_df["phase"] + " | " + summary_df["variable"].map(
            lambda value: SM_CLASSIFIER_VARIABLE_LABELS.get(value, value)
        )
        summary_pivot_df = summary_df.pivot(index="phase_variable", columns="summary", values="meters").fillna(0)
        st.markdown("**Quality summary distribution**")
        st.bar_chart(summary_pivot_df, use_container_width=True)
        st.dataframe(filtered_df, use_container_width=True)
        return

    if block == "Data statistics":
        stats_df = _flatten_sm_classifier_statistics(scoped_characterization)
        if stats_df.empty:
            st.info("No data statistics are stored for this run.")
            return

        phase_options = sorted(stats_df["phase"].dropna().unique().tolist())
        variable_options = sorted(stats_df["variable"].dropna().unique().tolist())
        statistic_options = ["Min", "Max", "Mean", "Std"]
        selected_phases = st.multiselect(
            "Phases",
            options=phase_options,
            default=phase_options,
            key=f"{widget_prefix}_stats_phases",
        )
        selected_variables = st.multiselect(
            "Variables",
            options=variable_options,
            default=variable_options,
            format_func=lambda value: SM_CLASSIFIER_VARIABLE_LABELS.get(value, value),
            key=f"{widget_prefix}_stats_variables",
        )
        selected_statistics = st.multiselect(
            "Statistics",
            options=statistic_options,
            default=["Mean", "Std"],
            format_func=lambda value: SM_CLASSIFIER_STATISTIC_LABELS.get(value, value),
            key=f"{widget_prefix}_stats_metrics",
        )
        filtered_df = stats_df[
            stats_df["phase"].isin(selected_phases) & stats_df["variable"].isin(selected_variables)
        ].copy()
        if filtered_df.empty:
            st.info("No data statistics rows match the selected filters.")
            return

        aggregate_df = filtered_df.groupby(["phase", "variable"], as_index=False)[selected_statistics].mean().fillna(0)
        aggregate_df["phase_variable"] = aggregate_df["phase"] + " | " + aggregate_df["variable"].map(
            lambda value: SM_CLASSIFIER_VARIABLE_LABELS.get(value, value)
        )
        st.markdown("**Average statistics across selected meters**")
        st.bar_chart(aggregate_df.set_index("phase_variable")[selected_statistics], use_container_width=True)
        st.dataframe(filtered_df, use_container_width=True)
        return

    connectivity_df = _flatten_sm_classifier_connectivity(scoped_characterization)
    if connectivity_df.empty:
        st.info("No connectivity results are stored for this run.")
        return

    summary_df = (
        connectivity_df["connected_phase_count"]
        .value_counts()
        .rename_axis("connected_phase_count")
        .reset_index(name="meters")
        .sort_values("connected_phase_count")
    )
    c1, c2, c3 = st.columns(3)
    c1.metric("Meters with connection error", int((connectivity_df["connection_error_count"] > 0).sum()))
    c2.metric("Meters with on/off switching", int((connectivity_df["switching_phase_count"] > 0).sum()))
    c3.metric("Meters with any connected phase", int((connectivity_df["connected_phase_count"] > 0).sum()))
    st.markdown("**Connected phase counts**")
    st.bar_chart(summary_df.set_index("connected_phase_count")[["meters"]], use_container_width=True)
    st.dataframe(connectivity_df, use_container_width=True)


def _render_sm_classifier_plot_tab(
    run_name: str,
    category_members: dict[str, list[str]],
    data_dir_path: str,
    widget_prefix: str,
) -> None:
    plot_filter_options = list(SM_CLASSIFIER_PLOT_FILTERS.keys())
    selected_filter = st.selectbox(
        "Plot filter",
        options=plot_filter_options,
        key=f"{widget_prefix}_plot_filter",
    )
    category_key = SM_CLASSIFIER_PLOT_FILTERS[selected_filter]
    filtered_meter_ids = category_members.get(category_key, [])

    c1, c2 = st.columns(2)
    c1.metric("Meters matching filter", len(filtered_meter_ids))
    c2.metric("Selected filter", selected_filter)

    plot_artifacts = _list_sm_classifier_plot_artifacts(run_name)
    if plot_artifacts:
        artifact_options = [
            str(path.relative_to(CURRENT_FILE.parent / "Results" / run_name)) for path in plot_artifacts
        ]
        selected_artifact = st.selectbox(
            "Saved plot artifact",
            options=artifact_options,
            key=f"{widget_prefix}_plot_artifact",
        )
        artifact_path = CURRENT_FILE.parent / "Results" / run_name / selected_artifact
        st.image(str(artifact_path), use_container_width=True)
    else:
        st.caption(
            "No saved SM classifier SVG plot artifacts were found for this run in the local Results directory. "
            "The selector below still lets you inspect the meters that match each plot filter "
            "and preview their timeseries."
        )

    st.dataframe(pd.DataFrame({"meter_id": filtered_meter_ids}), use_container_width=True)
    if not filtered_meter_ids:
        return

    selected_meter = st.selectbox(
        "Meter plot preview",
        options=filtered_meter_ids,
        key=f"{widget_prefix}_plot_meter",
    )
    selected_phases = st.multiselect(
        "Plot phases",
        options=["L1", "L2", "L3"],
        default=["L1", "L2", "L3"],
        key=f"{widget_prefix}_plot_phases",
    )
    selected_variable_groups = st.multiselect(
        "Variable selection",
        options=["Voltage", "Active power", "Reactive power"],
        default=["Voltage"],
        key=f"{widget_prefix}_plot_variables",
    )

    if not selected_phases or not selected_variable_groups:
        st.info("Select at least one phase and one variable group to preview meter plots.")
        return

    try:
        with st.spinner("Loading meter data for plot preview..."):
            plot_df = load_plot_df(
                data_dir_path=data_dir_path,
                entity="meter",
                entity_id=selected_meter,
                topology_level="raw",
                profile="raw_profiles",
                add_current=False,
                add_unbalance=False,
                load_existing_only=True,
            )
        if plot_df is None or plot_df.empty:
            st.warning("No raw timeseries data is available for the selected meter.")
            return

        if not isinstance(plot_df.index, pd.DatetimeIndex):
            for candidate in ["timestamp", "timestamp_dst"]:
                if candidate in plot_df.columns:
                    plot_df[candidate] = pd.to_datetime(plot_df[candidate], utc=True, errors="coerce")
                    plot_df = plot_df.set_index(candidate)
                    break

        selected_columns = _select_sm_classifier_plot_columns(plot_df, selected_phases, selected_variable_groups)
        if not selected_columns:
            st.info("No numeric timeseries columns matched the selected phase and variable filters.")
            return

        st.line_chart(plot_df[selected_columns], use_container_width=True)
        st.dataframe(plot_df[selected_columns].head(200), use_container_width=True)
    except Exception as exc:
        st.warning(f"Could not load plot preview data for the selected meter: {exc}")


def _render_sm_classifier_results_browser(
    run_name: str,
    characterization: dict,
    classification: dict,
    data_dir_path: str,
    widget_prefix: str,
) -> None:
    category_members, derived_summary_used = _get_sm_classifier_category_members(characterization, classification)
    category_counts = {key: len(value) for key, value in category_members.items()}

    top_c1, top_c2, top_c3, top_c4 = st.columns(4)
    top_c1.metric("Meters in selected run", len(characterization))
    top_c2.metric("Only good data quality", category_counts.get("SMs_with_only_good_data_quality", 0))
    top_c3.metric("Connection errors", category_counts.get("SMs_with_connection_error", 0))
    top_c4.metric("On/off switch", category_counts.get("SMs_with_on_off_switch", 0))

    browser_tabs = st.tabs(["Classification", "Characterization", "Smart meter plots"])
    with browser_tabs[0]:
        _render_sm_classifier_classification_tab(category_members, category_counts, widget_prefix, derived_summary_used)
    with browser_tabs[1]:
        _render_sm_classifier_characterization_tab(characterization, category_members, widget_prefix)
    with browser_tabs[2]:
        _render_sm_classifier_plot_tab(run_name, category_members, data_dir_path, widget_prefix)


def _render_sm_classifier_existing_results() -> None:
    st.markdown("### Existing SM Classifier Results")

    run_names = _list_sm_classifier_runs()
    db_snapshot_df = _get_sm_classifier_db_snapshot()

    if run_names:
        _apply_pending_selection("smc_existing_run", run_names)
        selected_run = st.selectbox("Classifier run", options=run_names, key="smc_existing_run")
        try:
            sm_characterization, sm_classification = _load_sm_classifier_run(selected_run)
            _render_sm_classifier_results_browser(
                run_name=selected_run,
                characterization=sm_characterization,
                classification=sm_classification,
                data_dir_path=st.session_state.get(
                    "shared_data_dir", os.getenv("S3_DATA_DIR_PATH", "phase_measurements/raw")
                ),
                widget_prefix="smc_existing",
            )
        except Exception as exc:
            st.warning(f"Could not load saved SM classifier run: {exc}")

    elif not db_snapshot_df.empty:
        metrics, connected_phase_df, quality_df, preview_df = _summarize_sm_classifier_db(db_snapshot_df)
        st.caption(
            "No saved SM classifier run directory found. "
            "Showing classifier metadata currently stored in PostgreSQL."
        )

        top_c1, top_c2, top_c3, top_c4 = st.columns(4)
        top_c1.metric("Classified meters", metrics.get("classified_meters", 0))
        top_c2.metric("With data quality", metrics.get("meters_with_data_quality", 0))
        top_c3.metric("With connectivity", metrics.get("meters_with_connectivity", 0))
        top_c4.metric("With statistics", metrics.get("meters_with_statistics", 0))

        top_c5, top_c6 = st.columns(2)
        top_c5.metric("Connection errors", metrics.get("meters_with_connection_error", 0))
        top_c6.metric("On/off switch", metrics.get("meters_with_on_off_switch", 0))

        plot_c1, plot_c2 = st.columns(2)
        with plot_c1:
            if not connected_phase_df.empty:
                st.markdown("**Connected Phase Distribution**")
                st.bar_chart(connected_phase_df.set_index("label")[["meters"]], use_container_width=True)
        with plot_c2:
            if not quality_df.empty:
                st.markdown("**Voltage Quality Summary By Phase**")
                quality_pivot_df = quality_df.pivot(index="phase", columns="summary", values="meters").fillna(0)
                st.bar_chart(quality_pivot_df, use_container_width=True)

        if not preview_df.empty:
            st.markdown("**Recent Classified Meters From PostgreSQL**")
            st.dataframe(preview_df.head(200), use_container_width=True)

    else:
        st.info("No SM classifier results found yet. Run the SM classifier to generate and store results.")


@st.cache_data(show_spinner=False)
def _get_meter_inventory_overview() -> dict:
    """Aggregate inventory stats from meter metadata table."""
    session = threephi_db.new_session()
    try:
        total_meters = session.execute(select(func.count()).select_from(MetaMeterModel)).scalar_one()
        meters_with_rows = session.execute(
            select(func.count()).select_from(MetaMeterModel).where(MetaMeterModel.total_rows > 0)
        ).scalar_one()
        meters_with_data_quality = session.execute(
            select(func.count()).select_from(MetaMeterModel).where(MetaMeterModel.data_quality.is_not(None))
        ).scalar_one()
        meters_with_data_statistics = session.execute(
            select(func.count()).select_from(MetaMeterModel).where(MetaMeterModel.data_statistics.is_not(None))
        ).scalar_one()
        meters_with_connectivity = session.execute(
            select(func.count()).select_from(MetaMeterModel).where(MetaMeterModel.connectivity.is_not(None))
        ).scalar_one()

        min_first_seen, max_last_seen = session.execute(
            select(func.min(MetaMeterModel.first_seen), func.max(MetaMeterModel.last_seen)).where(
                MetaMeterModel.total_rows > 0
            )
        ).one()
        latest_db_update = session.execute(select(func.max(MetaMeterModel.updated_at))).scalar_one()

        return {
            "total_meters": int(total_meters or 0),
            "meters_with_rows": int(meters_with_rows or 0),
            "meters_with_data_quality": int(meters_with_data_quality or 0),
            "meters_with_data_statistics": int(meters_with_data_statistics or 0),
            "meters_with_connectivity": int(meters_with_connectivity or 0),
            "min_first_seen": str(min_first_seen) if min_first_seen else "-",
            "max_last_seen": str(max_last_seen) if max_last_seen else "-",
            "latest_db_update": str(latest_db_update) if latest_db_update else "-",
        }
    finally:
        session.close()


@st.cache_data(show_spinner=False)
def _get_meter_inventory_preview(limit: int = 2000) -> pd.DataFrame:
    """Load a compact meter inventory preview from database."""
    session = threephi_db.new_session()
    try:
        stmt = (
            select(
                MetaMeterModel.id.label("meter_id"),
                MetaMeterModel.total_rows.label("total_rows"),
                MetaMeterModel.first_seen.label("first_seen"),
                MetaMeterModel.last_seen.label("last_seen"),
                MetaMeterModel.data_quality.is_not(None).label("has_data_quality"),
                MetaMeterModel.data_statistics.is_not(None).label("has_data_statistics"),
                MetaMeterModel.connectivity.is_not(None).label("has_connectivity"),
            )
            .order_by(MetaMeterModel.total_rows.desc(), MetaMeterModel.id.asc())
            .limit(limit)
        )
        rows = session.execute(stmt).all()
        if not rows:
            return pd.DataFrame(
                columns=[
                    "meter_id",
                    "total_rows",
                    "first_seen",
                    "last_seen",
                    "has_data_quality",
                    "has_data_statistics",
                    "has_connectivity",
                ]
            )
        return pd.DataFrame(rows)
    finally:
        session.close()


@st.cache_data(show_spinner=False)
def _get_raw_parquet_inventory(data_dir_path: str) -> tuple[dict, pd.DataFrame, pd.DataFrame]:
    """Summarize raw parquet inventory under the configured S3 data directory."""
    extractor = get_extractor(data_dir_path=data_dir_path)
    base = f"s3://3phi/{data_dir_path.strip('/')}"
    parquet_paths = sorted(extractor.s3_connector.glob(f"{base}/dt=*/shard=*/*.parquet"))

    by_date: dict[str, int] = {}
    by_shard: dict[str, int] = {}
    date_pattern = re.compile(r"/dt=(\d{4}-\d{2}-\d{2})/")
    shard_pattern = re.compile(r"/shard=(\d+)/")

    for path in parquet_paths:
        date_match = date_pattern.search(path)
        shard_match = shard_pattern.search(path)
        if date_match:
            date_key = date_match.group(1)
            by_date[date_key] = by_date.get(date_key, 0) + 1
        if shard_match:
            shard_key = shard_match.group(1)
            by_shard[shard_key] = by_shard.get(shard_key, 0) + 1

    date_df = pd.DataFrame(
        [{"date": date_key, "parquet_files": count} for date_key, count in sorted(by_date.items())]
    )
    shard_df = pd.DataFrame(
        [{"shard": shard_key, "parquet_files": count} for shard_key, count in sorted(by_shard.items())]
    )

    overview = {
        "parquet_files": len(parquet_paths),
        "covered_dates": len(by_date),
        "covered_shards": len(by_shard),
        "min_date": min(by_date.keys()) if by_date else "-",
        "max_date": max(by_date.keys()) if by_date else "-",
        "sample_paths": parquet_paths[:10],
    }
    return overview, date_df, shard_df


@st.cache_data(show_spinner=False)
def _get_latest_timeseries_update() -> str:
    """Return latest committed ingestion timestamp from file index."""
    session = threephi_db.new_session()
    try:
        latest = session.execute(text("SELECT max(committed_at) FROM file_index WHERE status = 'ready'"))
        latest_value = latest.scalar_one_or_none()
        return str(latest_value) if latest_value else "-"
    finally:
        session.close()


@st.cache_data(show_spinner=False)
def _get_topology_versions() -> list[dict]:
    """Return topology versions ordered by newest first."""
    session = threephi_db.new_session()
    try:
        rows = session.execute(
            select(
                TopologyVersionModel.version,
                TopologyVersionModel.ingested_at,
                TopologyVersionModel.is_current,
            ).order_by(TopologyVersionModel.version.desc())
        ).all()
        return [
            {
                "version": int(row.version),
                "ingested_at": str(row.ingested_at) if row.ingested_at else "-",
                "is_current": bool(row.is_current),
            }
            for row in rows
        ]
    finally:
        session.close()


@st.cache_data(show_spinner=False)
def _get_topology_inventory_overview(version: int | None = None) -> dict:
    """Aggregate topology inventory for a selected topology version."""
    session = threephi_db.new_session()
    try:
        version_row = None
        if version is None:
            version_row = session.execute(
                select(TopologyVersionModel.version, TopologyVersionModel.ingested_at, TopologyVersionModel.is_current)
                .where(TopologyVersionModel.is_current.is_(True))
                .limit(1)
            ).one_or_none()
        else:
            version_row = session.execute(
                select(TopologyVersionModel.version, TopologyVersionModel.ingested_at, TopologyVersionModel.is_current)
                .where(TopologyVersionModel.version == int(version))
                .limit(1)
            ).one_or_none()

        if version_row is None:
            return {
                "topology_version": None,
                "topology_ingested_at": "-",
                "is_current": False,
                "substations": 0,
                "transformers": 0,
                "feeders": 0,
                "cabinets": 0,
                "delivery_points": 0,
                "cables": 0,
            }

        selected_version = int(version_row.version)

        feeder_ids_subq = (
            select(NodeModel.feeder_id)
            .where(
                NodeModel.version == selected_version,
                NodeModel.node_type == "LvFeeder",
                NodeModel.feeder_id.is_not(None),
            )
            .distinct()
            .subquery()
        )
        cabinet_ids_subq = (
            select(NodeModel.cabinet_id)
            .where(
                NodeModel.version == selected_version,
                NodeModel.node_type == "Cabinet",
                NodeModel.cabinet_id.is_not(None),
            )
            .distinct()
            .subquery()
        )
        delivery_point_ids_subq = (
            select(NodeModel.delivery_point_id)
            .where(
                NodeModel.version == selected_version,
                NodeModel.node_type == "DeliveryPoint",
                NodeModel.delivery_point_id.is_not(None),
            )
            .distinct()
            .subquery()
        )

        substations = int(
            session.execute(
                select(func.count(func.distinct(TransformerModel.substation_id)))
                .select_from(FeederModel)
                .join(TransformerModel, FeederModel.transformer_id == TransformerModel.id)
                .join(feeder_ids_subq, FeederModel.id == feeder_ids_subq.c.feeder_id)
            ).scalar_one()
            or 0
        )
        transformers = int(
            session.execute(
                select(func.count(func.distinct(FeederModel.transformer_id)))
                .select_from(FeederModel)
                .join(feeder_ids_subq, FeederModel.id == feeder_ids_subq.c.feeder_id)
            ).scalar_one()
            or 0
        )
        feeders = int(
            session.execute(select(func.count()).select_from(feeder_ids_subq)).scalar_one() or 0
        )
        cabinets = int(
            session.execute(select(func.count()).select_from(cabinet_ids_subq)).scalar_one() or 0
        )
        delivery_points = int(
            session.execute(select(func.count()).select_from(delivery_point_ids_subq)).scalar_one() or 0
        )
        cables = int(
            session.execute(
                select(func.count(func.distinct(CableModel.cable_id))).where(CableModel.version == selected_version)
            ).scalar_one()
            or 0
        )

        return {
            "topology_version": selected_version,
            "topology_ingested_at": str(version_row.ingested_at) if version_row.ingested_at else "-",
            "is_current": bool(version_row.is_current),
            "substations": substations,
            "transformers": transformers,
            "feeders": feeders,
            "cabinets": cabinets,
            "delivery_points": delivery_points,
            "cables": cables,
        }
    finally:
        session.close()


@st.cache_data(show_spinner=False)
def _get_topology_connection_view() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Export current topology and derive connection summaries for plotting."""
    controller = TopologyController(threephi_db.new_session)
    topo_pdf = controller.export_topology(as_dask=False)

    if topo_pdf is None or topo_pdf.empty:
        empty = pd.DataFrame()
        return empty, empty, empty

    component_counts_df, connection_type_df = _summarize_topology_connections(topo_pdf)
    return topo_pdf, component_counts_df, connection_type_df


@st.cache_data(show_spinner=False)
def _get_topology_meter_overlay() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Build meter availability summaries for feeder and cabinet nodes."""
    controller = TopologyController(threephi_db.new_session)
    sm_cab_pdf = controller.export_sm_cabinet(as_dask=False)
    if sm_cab_pdf is None or sm_cab_pdf.empty:
        empty = pd.DataFrame()
        return empty, empty

    overlay_df = sm_cab_pdf[["meter_number", "cabinet", "lv_feeder"]].copy()
    overlay_df["meter_id"] = pd.to_numeric(overlay_df["meter_number"], errors="coerce").astype("Int64")
    overlay_df = overlay_df.dropna(subset=["meter_id"])
    overlay_df["meter_id"] = overlay_df["meter_id"].astype("int64")

    session = threephi_db.new_session()
    try:
        meter_meta_rows = session.execute(
            select(
                MetaMeterModel.id.label("meter_id"),
                MetaMeterModel.total_rows.label("total_rows"),
                MetaMeterModel.data_quality.is_not(None).label("has_data_quality"),
            )
        ).all()
    finally:
        session.close()

    meter_meta_df = pd.DataFrame(meter_meta_rows)
    if meter_meta_df.empty:
        overlay_df["has_timeseries"] = False
        overlay_df["has_data_quality"] = False
    else:
        overlay_df = overlay_df.merge(meter_meta_df, on="meter_id", how="left")
        overlay_df["has_timeseries"] = overlay_df["total_rows"].fillna(0).astype(float) > 0
        overlay_df["has_data_quality"] = overlay_df["has_data_quality"].fillna(False).astype(bool)

    def _summarize_by_node(frame: pd.DataFrame, node_col: str) -> pd.DataFrame:
        if frame.empty or node_col not in frame.columns:
            return pd.DataFrame(
                columns=["node_label", "total_meters", "meters_with_timeseries", "meters_with_data_quality"]
            )
        summary_df = (
            frame.dropna(subset=[node_col])
            .groupby(node_col, as_index=False)
            .agg(
                total_meters=("meter_id", "nunique"),
                meters_with_timeseries=("has_timeseries", "sum"),
                meters_with_data_quality=("has_data_quality", "sum"),
            )
            .rename(columns={node_col: "node_label"})
        )
        return summary_df

    feeder_overlay_df = _summarize_by_node(overlay_df, "lv_feeder")
    cabinet_overlay_df = _summarize_by_node(overlay_df, "cabinet")
    return feeder_overlay_df, cabinet_overlay_df


def _summarize_topology_connections(topo_pdf: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Derive component and connection summaries from exported topology."""
    if topo_pdf is None or topo_pdf.empty:
        empty = pd.DataFrame()
        return empty, empty

    edges_df = topo_pdf[["secondary_substation", "transformer", "lv_feeder", "node1", "node2", "cable_id"]].copy()

    def _node_type(node_label: str) -> str:
        if not isinstance(node_label, str) or "." not in node_label:
            return "Unknown"
        return node_label.split(".", 1)[0]

    type_rows = []
    for column in ["secondary_substation", "transformer", "lv_feeder", "node1", "node2"]:
        values = edges_df[column].dropna().astype(str)
        if column in {"secondary_substation", "transformer", "lv_feeder"}:
            type_name = {
                "secondary_substation": "SecondarySubstation",
                "transformer": "Transformer",
                "lv_feeder": "LvFeeder",
            }[column]
            type_rows.append({"component_type": type_name, "count": int(values.nunique())})
        else:
            for type_name, count in values.map(_node_type).value_counts().items():
                type_rows.append({"component_type": type_name, "count": int(count)})

    component_counts_df = pd.DataFrame(type_rows)
    if not component_counts_df.empty:
        component_counts_df = (
            component_counts_df.groupby("component_type", as_index=False)["count"]
            .sum()
            .sort_values("count", ascending=False)
        )

    connection_type_df = edges_df[["node1", "node2"]].dropna().copy()
    if not connection_type_df.empty:
        connection_type_df["source_type"] = connection_type_df["node1"].map(_node_type)
        connection_type_df["target_type"] = connection_type_df["node2"].map(_node_type)
        connection_type_df = (
            connection_type_df.groupby(["source_type", "target_type"], as_index=False)
            .size()
            .rename(columns={"size": "connections"})
            .sort_values("connections", ascending=False)
        )

    return component_counts_df, connection_type_df


def _format_connection_types_for_plot(connection_type_df: pd.DataFrame) -> pd.DataFrame:
    """Format connection type summary for chart-friendly display."""
    if connection_type_df is None or connection_type_df.empty:
        return pd.DataFrame(columns=["connection_type", "connections"])

    plot_df = connection_type_df.copy()
    plot_df["connection_type"] = (
        plot_df["source_type"].astype(str) + " -> " + plot_df["target_type"].astype(str)
    )
    return plot_df[["connection_type", "connections"]].sort_values("connections", ascending=False)


def _build_connected_vs_unconnected_df(component_df: pd.DataFrame, topology_inventory: dict) -> pd.DataFrame:
    """Compare connected components in export against inventory totals by category."""
    if component_df is None or component_df.empty or not topology_inventory:
        return pd.DataFrame(columns=["category", "Connected", "Not connected"])

    component_map = {
        "SecondarySubstation": "substations",
        "Transformer": "transformers",
        "LvFeeder": "feeders",
        "Cabinet": "cabinets",
        "DeliveryPoint": "delivery_points",
    }
    display_map = {
        "SecondarySubstation": "Substations",
        "Transformer": "Transformers",
        "LvFeeder": "Feeders",
        "Cabinet": "Cabinets",
        "DeliveryPoint": "Delivery points",
    }

    connected_counts = {
        row.component_type: int(row.count)
        for row in component_df.itertuples(index=False)
        if getattr(row, "component_type", None) in component_map
    }

    rows: list[dict] = []
    for component_type, inventory_key in component_map.items():
        total_count = int(topology_inventory.get(inventory_key, 0) or 0)
        connected_count = int(connected_counts.get(component_type, 0) or 0)
        rows.append(
            {
                "category": display_map[component_type],
                "Connected": min(connected_count, total_count),
                "Not connected": max(total_count - connected_count, 0),
            }
        )

    return pd.DataFrame(rows)


def _format_overlay_preview_for_display(overlay_preview_df: pd.DataFrame) -> pd.DataFrame:
    """Rename and format overlay columns for readability."""
    if overlay_preview_df is None or overlay_preview_df.empty:
        return pd.DataFrame(
            columns=[
                "Node type",
                "Node label",
                "Total meters",
                "Meters with time-series",
                "Meters with data quality",
            ]
        )

    display_df = overlay_preview_df.rename(
        columns={
            "node_type": "Node type",
            "node_label": "Node label",
            "total_meters": "Total meters",
            "meters_with_timeseries": "Meters with time-series",
            "meters_with_data_quality": "Meters with data quality",
        }
    ).copy()
    return display_df[
        [
            "Node type",
            "Node label",
            "Total meters",
            "Meters with time-series",
            "Meters with data quality",
        ]
    ]


def _format_meter_preview_for_display(meter_preview_df: pd.DataFrame) -> pd.DataFrame:
    """Format meter inventory preview table with human-readable column names."""
    if meter_preview_df is None or meter_preview_df.empty:
        return pd.DataFrame()

    display_df = meter_preview_df.rename(
        columns={
            "meter_id": "Meter ID",
            "total_rows": "Total rows",
            "first_seen": "First seen",
            "last_seen": "Last seen",
            "has_data_quality": "Has data quality",
            "has_data_statistics": "Has data statistics",
            "has_connectivity": "Has connectivity",
        }
    ).copy()

    for bool_col in ["Has data quality", "Has data statistics", "Has connectivity"]:
        if bool_col in display_df.columns:
            display_df[bool_col] = display_df[bool_col].map(lambda value: "Yes" if bool(value) else "No")

    ordered_columns = [
        "Meter ID",
        "Total rows",
        "First seen",
        "Last seen",
        "Has data quality",
        "Has data statistics",
        "Has connectivity",
    ]
    return display_df[ordered_columns]


def _drilldown_no_data_guidance(entity: str, profile: str, topology_level: str) -> list[str]:
    """Return practical next steps when drilldown selection yields no rows."""
    guidance: list[str] = []

    if entity == "meter" and profile == "raw_profiles":
        guidance.append("Disable 'Load existing only (meter/raw)' to extract raw profiles on demand.")
        guidance.append("Try another meter ID that appears in 'Meter inventory preview'.")
    elif entity == "meter":
        guidance.append("Try profile 'raw_profiles' first, then compare with cleaned profiles.")
        guidance.append("Verify the meter has data rows and timestamp coverage in the preview table.")
    else:
        guidance.append("Try a broader entity first (e.g., feeder or transformer) to validate data availability.")
        guidance.append("Switch topology level between raw/cleaned/cleaned_and_corrected.")
        if profile != "raw_profiles":
            guidance.append("If cleaned profiles are empty, retry with profile 'raw_profiles'.")

    if topology_level == "cleaned_and_corrected":
        guidance.append("If no rows are returned, retry with topology level 'cleaned' or 'raw'.")

    return guidance


def _filter_topology_by_scope(topology_df: pd.DataFrame, scope_type: str, scope_value: str) -> pd.DataFrame:
    """Filter exported topology by selected scope."""
    if topology_df is None or topology_df.empty or scope_type == "All" or scope_value == "All":
        return topology_df

    scope_column_map = {
        "Substation": "secondary_substation",
        "Transformer": "transformer",
        "Feeder": "lv_feeder",
    }
    scope_column = scope_column_map.get(scope_type)
    if not scope_column or scope_column not in topology_df.columns:
        return topology_df

    return topology_df[topology_df[scope_column].astype(str) == str(scope_value)].copy()


def _filter_overlay_by_scope(
    overlay_df: pd.DataFrame,
    topology_df: pd.DataFrame,
    node_type: str,
) -> pd.DataFrame:
    """Filter feeder/cabinet overlay rows to the nodes present in the scoped topology slice."""
    if overlay_df is None or overlay_df.empty or topology_df is None or topology_df.empty:
        return pd.DataFrame(columns=getattr(overlay_df, "columns", []))

    if node_type == "feeder":
        valid_nodes = set(topology_df["lv_feeder"].dropna().astype(str).tolist())
    else:
        valid_nodes = set()
        for column in ["node1", "node2"]:
            if column in topology_df.columns:
                valid_nodes.update(
                    value
                    for value in topology_df[column].dropna().astype(str).tolist()
                    if value.startswith("Cabinet.")
                )

    if not valid_nodes:
        return overlay_df.iloc[0:0].copy()
    return overlay_df[overlay_df["node_label"].astype(str).isin(valid_nodes)].copy()


def _build_topology_graphviz(
    topology_df: pd.DataFrame,
    feeder_overlay_df: pd.DataFrame | None = None,
    cabinet_overlay_df: pd.DataFrame | None = None,
    max_edges: int = 80,
) -> str:
    """Build a Graphviz graph for a subset of current topology edges."""
    if topology_df is None or topology_df.empty:
        return "digraph G { label=\"No topology data available\"; }"

    graph_df = topology_df[["secondary_substation", "transformer", "lv_feeder", "node1", "node2"]].dropna().copy()
    graph_df = graph_df.drop_duplicates().head(max_edges)

    color_map = {
        "LvFeeder": "#2E7D32",
        "Cabinet": "#1565C0",
        "DeliveryPoint": "#EF6C00",
        "SecondarySubstation": "#6A1B9A",
        "Transformer": "#8D6E63",
    }

    def _node_type(node_label: str) -> str:
        if not isinstance(node_label, str) or "." not in node_label:
            return "Unknown"
        return node_label.split(".", 1)[0]

    feeder_overlay_map = {}
    cabinet_overlay_map = {}
    if feeder_overlay_df is not None and not feeder_overlay_df.empty:
        feeder_overlay_map = feeder_overlay_df.set_index("node_label").to_dict(orient="index")
    if cabinet_overlay_df is not None and not cabinet_overlay_df.empty:
        cabinet_overlay_map = cabinet_overlay_df.set_index("node_label").to_dict(orient="index")

    def _availability_ratio(overlay: dict | None) -> float | None:
        if not overlay:
            return None
        total = int(overlay.get("total_meters", 0) or 0)
        if total <= 0:
            return None
        with_ts = int(overlay.get("meters_with_timeseries", 0) or 0)
        return max(0.0, min(1.0, with_ts / total))

    def _availability_color(node_type: str, overlay: dict | None) -> str:
        # Only override colors for feeder/cabinet where availability context exists.
        if node_type not in {"LvFeeder", "Cabinet"}:
            return color_map.get(node_type, "#90A4AE")

        ratio = _availability_ratio(overlay)
        if ratio is None:
            return "#90A4AE"  # unknown availability
        if ratio < 0.5:
            return "#C62828"  # low availability
        if ratio < 0.8:
            return "#F9A825"  # medium availability
        return "#2E7D32"  # high availability

    def _format_node_label(node_label: str) -> str:
        node_type = _node_type(node_label)
        overlay = None
        if node_type == "LvFeeder":
            overlay = feeder_overlay_map.get(node_label)
        elif node_type == "Cabinet":
            overlay = cabinet_overlay_map.get(node_label)

        if not overlay:
            return node_label

        return (
            f"{node_label}\\n"
            f"meters: {int(overlay['total_meters'])} | "
            f"ts: {int(overlay['meters_with_timeseries'])} | "
            f"dq: {int(overlay['meters_with_data_quality'])} | "
            f"avail: {int((_availability_ratio(overlay) or 0) * 100)}%"
        )

    node_defs: dict[str, str] = {}
    edge_defs: list[str] = []

    for column in ["secondary_substation", "transformer", "lv_feeder"]:
        for label in graph_df[column].dropna().astype(str).unique().tolist():
            label_type = _node_type(label)
            overlay = feeder_overlay_map.get(label) if label_type == "LvFeeder" else None
            label_color = _availability_color(label_type, overlay)
            pretty_label = _format_node_label(label)
            node_defs[label] = (
                f'"{label}" [label="{pretty_label}", style=filled, fillcolor="{label_color}", fontcolor="white"];'
            )

    for row in graph_df.itertuples(index=False):
        substation = str(row.secondary_substation)
        transformer = str(row.transformer)
        feeder = str(row.lv_feeder)
        node1 = str(row.node1)
        node2 = str(row.node2)
        for node in [node1, node2]:
            node_type = _node_type(node)
            overlay = None
            if node_type == "LvFeeder":
                overlay = feeder_overlay_map.get(node)
            elif node_type == "Cabinet":
                overlay = cabinet_overlay_map.get(node)
            fill = _availability_color(node_type, overlay)
            pretty_label = _format_node_label(node)
            node_defs.setdefault(
                node,
                f'"{node}" [label="{pretty_label}", style=filled, fillcolor="{fill}", fontcolor="white"];',
            )
        edge_defs.append(f'"{substation}" -> "{transformer}" [style=dashed, color="#CE93D8"];')
        edge_defs.append(f'"{transformer}" -> "{feeder}" [style=dashed, color="#BCAAA4"];')
        edge_defs.append(f'"{node1}" -> "{node2}" [color="#607D8B"];')
        if feeder != node1:
            edge_defs.append(f'"{feeder}" -> "{node1}" [style=dashed, color="#A5D6A7"];')

    dot_lines = [
        "digraph Topology {",
        "rankdir=LR;",
        'graph [bgcolor="white"];',
        'node [shape=box, fontsize=10, fontname="Helvetica"];',
        'edge [arrowsize=0.7];',
        *node_defs.values(),
        *edge_defs,
        "}",
    ]
    return "\n".join(dot_lines)


def _extract_stat_labeler_result_stamp(path: str, prefix: str) -> str:
    filename = str(path).rsplit("/", 1)[-1]
    if filename.startswith(prefix) and filename.endswith(".json"):
        return filename[len(prefix):-5]
    return filename


def _format_stat_labeler_result_option(path: str) -> str:
    stamp = _extract_stat_labeler_result_stamp(path, "heat_pump_results_")
    return stamp.replace("_", " ")


@st.cache_data(show_spinner=False)
def _list_stat_labeler_result_files(data_dir_path: str) -> tuple[list[str], dict[str, str]]:
    extractor = get_extractor(data_dir_path=data_dir_path)
    hp_paths = sorted(extractor.s3_connector.glob("s3://3phi/stat_labeler/heat_pump_results_*.json"))
    meta_paths = sorted(extractor.s3_connector.glob("s3://3phi/stat_labeler/meta_results_*.json"))

    hp_paths = list(reversed(hp_paths))
    meta_by_stamp = {
        _extract_stat_labeler_result_stamp(meta_path, "meta_results_"): meta_path for meta_path in meta_paths
    }
    return hp_paths, meta_by_stamp


@st.cache_data(show_spinner=False)
def _load_stat_labeler_results(
    data_dir_path: str,
    hp_path: str | None = None,
    meta_path: str | None = None,
) -> tuple[str | None, dict, str | None, dict]:
    extractor = get_extractor(data_dir_path=data_dir_path)
    hp_paths, meta_by_stamp = _list_stat_labeler_result_files(data_dir_path)

    if hp_path is None and hp_paths:
        hp_path = hp_paths[0]
    if meta_path is None and hp_path:
        stamp = _extract_stat_labeler_result_stamp(hp_path, "heat_pump_results_")
        meta_path = meta_by_stamp.get(stamp)

    hp_payload = extractor.s3_connector.read_json(hp_path) if hp_path else {}
    meta_payload = extractor.s3_connector.read_json(meta_path) if meta_path else {}

    return hp_path, hp_payload or {}, meta_path, meta_payload or {}


def _refresh_stat_labeler_result_state(data_dir_path: str) -> None:
    _list_stat_labeler_result_files.clear()
    _load_stat_labeler_results.clear()

    hp_paths, _ = _list_stat_labeler_result_files(data_dir_path)
    if not hp_paths:
        return

    latest_hp_path = hp_paths[0]
    st.session_state["sl_latest_results_selected_file_pending"] = latest_hp_path
    st.session_state["results_stat_labeler_selected_file_pending"] = latest_hp_path


def _to_altair_number_format(value_format: str) -> str:
    if not value_format:
        return ""
    if value_format.startswith("{:") and value_format.endswith("}"):
        return value_format[2:-1]
    return value_format


def _render_compact_phase_bar_chart(
    chart_df: pd.DataFrame,
    phase_col: str,
    value_col: str,
    label: str,
    color: str,
    value_format: str = "{:.0f}",
    chart_height: int = 140,
    title_caption: str | None = None,
) -> None:
    if chart_df.empty or phase_col not in chart_df.columns or value_col not in chart_df.columns:
        return

    phase_order = ["L1", "L2", "L3"]
    plot_df = chart_df[[phase_col, value_col]].copy()
    plot_df[phase_col] = plot_df[phase_col].astype(str).str.upper()
    plot_df["phase_sort"] = plot_df[phase_col].map({phase: index for index, phase in enumerate(phase_order)})
    plot_df = plot_df.sort_values("phase_sort")
    plot_df[value_col] = pd.to_numeric(plot_df[value_col], errors="coerce").fillna(0)
    altair_format = _to_altair_number_format(value_format)

    st.markdown(f"**{label}**")
    if title_caption:
        st.caption(title_caption)

    try:
        chart = alt.Chart(plot_df).mark_bar(size=20, cornerRadiusEnd=4).encode(
            y=alt.Y(f"{phase_col}:N", sort=phase_order, title=None),
            x=alt.X(f"{value_col}:Q", title=None),
            color=alt.value(color),
            tooltip=[
                alt.Tooltip(f"{phase_col}:N", title="Phase"),
                alt.Tooltip(f"{value_col}:Q", title=label, format=altair_format),
            ],
        )
        text = alt.Chart(plot_df).mark_text(align="left", baseline="middle", dx=6).encode(
            y=alt.Y(f"{phase_col}:N", sort=phase_order),
            x=alt.X(f"{value_col}:Q"),
            text=alt.Text(f"{value_col}:Q", format=altair_format),
        )

        st.altair_chart(
            (chart + text).properties(height=chart_height).configure_view(strokeWidth=0),
            use_container_width=True,
            theme="streamlit",
        )
    except Exception:
        fallback_df = plot_df[[phase_col, value_col]].set_index(phase_col)
        st.bar_chart(fallback_df, use_container_width=True)


def _render_phase_codetection_heatmap(meter_df: pd.DataFrame, phase_cols: list[str]) -> None:
    if meter_df.empty or not all(phase in meter_df.columns for phase in phase_cols):
        return

    rows = []
    total_meters = max(len(meter_df), 1)
    for phase_x in phase_cols:
        for phase_y in phase_cols:
            codetected = int((meter_df[phase_x] & meter_df[phase_y]).sum())
            rows.append(
                {
                    "phase_x": phase_x,
                    "phase_y": phase_y,
                    "meters": codetected,
                    "share": codetected / total_meters,
                }
            )

    heatmap_df = pd.DataFrame(rows)
    st.markdown("**Phase-To-Phase Co-Detection Heatmap**")
    st.caption(
        "Each cell shows how many meters were flagged on both phases in the selected result set. "
        "Diagonal cells are per-phase detections. Off-diagonal cells show phase-to-phase co-detections."
    )

    try:
        cell_step = 72
        base = alt.Chart(heatmap_df).encode(
            x=alt.X(
                "phase_x:N",
                sort=phase_cols,
                title=None,
                axis=alt.Axis(
                    labelAngle=0,
                    labelPadding=6,
                    labelFontSize=12,
                    domain=False,
                    ticks=False,
                    grid=False,
                ),
                scale=alt.Scale(paddingInner=0, paddingOuter=0),
            ),
            y=alt.Y(
                "phase_y:N",
                sort=phase_cols,
                title=None,
                axis=alt.Axis(
                    labelPadding=8,
                    labelFontSize=12,
                    domain=False,
                    ticks=False,
                    grid=False,
                ),
                scale=alt.Scale(paddingInner=0, paddingOuter=0),
            ),
            tooltip=[
                alt.Tooltip("phase_x:N", title="Phase X"),
                alt.Tooltip("phase_y:N", title="Phase Y"),
                alt.Tooltip("meters:Q", title="Meters"),
                alt.Tooltip("share:Q", title="Share", format=".1%"),
            ],
        )
        heatmap = base.mark_rect(strokeWidth=0).encode(
            color=alt.Color(
                "meters:Q",
                title="Meters",
                scale=alt.Scale(scheme="teals"),
                legend=alt.Legend(
                    orient="right",
                    direction="vertical",
                    gradientLength=cell_step * len(phase_cols),
                    gradientThickness=16,
                    titleOrient="right",
                ),
            )
        )
        text = base.mark_text(fontSize=12, fontWeight="bold").encode(
            text=alt.Text("meters:Q", format=".0f"),
            color=alt.condition(
                alt.datum.meters > heatmap_df["meters"].max() / 2,
                alt.value("white"),
                alt.value("#16324F"),
            ),
        )
        heatmap_chart = (heatmap + text).properties(
            width={"step": cell_step},
            height={"step": cell_step},
        ).configure_view(stroke=None)
        st.altair_chart(heatmap_chart, use_container_width=False, theme="streamlit")
        # st.caption("Diagonal cells are per-phase detections. Off-diagonal cells show phase-to-phase co-detections.")
    except Exception:
        fallback_df = (
            heatmap_df.pivot(index="phase_y", columns="phase_x", values="meters")
            .reindex(index=phase_cols, columns=phase_cols)
        )
        st.dataframe(fallback_df, use_container_width=True)


def _render_stat_labeler_results(results_payload: dict, meta_payload: dict) -> None:
    if not results_payload:
        st.info("No stat_labeler results available yet.")
        return

    meter_rows = []
    for meter_id, phase_map in results_payload.items():
        phase_map = phase_map or {}
        l1 = bool(phase_map.get("l1", False))
        l2 = bool(phase_map.get("l2", False))
        l3 = bool(phase_map.get("l3", False))
        meter_rows.append(
            {
                "meter_id": str(meter_id),
                "L1": l1,
                "L2": l2,
                "L3": l3,
                "phase_hits": int(l1) + int(l2) + int(l3),
            }
        )

    meter_df = pd.DataFrame(meter_rows).sort_values(["phase_hits", "meter_id"], ascending=[False, True])

    counts = {
        "L1": int(meter_df["L1"].sum()),
        "L2": int(meter_df["L2"].sum()),
        "L3": int(meter_df["L3"].sum()),
    }
    counts_df = pd.DataFrame({"phase": list(counts.keys()), "detected_heat_pump": list(counts.values())})

    phase_combo_lookup = {0: "No phase", 1: "Single-phase", 2: "Two-phase", 3: "Three-phase"}
    phase_hits_df = (
        meter_df["phase_hits"]
        .value_counts()
        .rename_axis("phase_hits")
        .reset_index(name="meters")
        .sort_values("phase_hits")
    )
    phase_hits_df["label"] = phase_hits_df["phase_hits"].map(phase_combo_lookup).fillna("Unknown")

    combo_df = meter_df.assign(
        phase_combo=meter_df.apply(
            lambda row: "+".join([phase for phase in ["L1", "L2", "L3"] if bool(row[phase])]) or "None",
            axis=1,
        )
    )
    combo_counts_df = (
        combo_df["phase_combo"]
        .value_counts()
        .rename_axis("phase_combo")
        .reset_index(name="meters")
        .sort_values(["meters", "phase_combo"], ascending=[False, True])
    )

    c1, c2, c3 = st.columns(3)
    c1.metric("Meters evaluated", len(meter_df))
    c2.metric("Number of HP-positive meters", int((meter_df["phase_hits"] > 0).sum()))
    c3.metric("Total HP-positive phases", int(meter_df["phase_hits"].sum()))

    slope_avg_df = pd.DataFrame()
    slope_margin_df = pd.DataFrame()
    if not meta_payload:
        _render_compact_phase_bar_chart(
            counts_df,
            phase_col="phase",
            value_col="detected_heat_pump",
            label="Heat Pump Detections By Phase",
            color="#2E86AB",
        )
        summary_col1, summary_col2 = st.columns(2)
        with summary_col1:
            st.markdown("**Meters By Number Of Positive Phases**")
            st.bar_chart(phase_hits_df.set_index("label")[["meters"]], use_container_width=True)
        with summary_col2:
            st.markdown("**Most Common Phase Detection Combinations**")
            st.bar_chart(combo_counts_df.set_index("phase_combo")[["meters"]], use_container_width=True)

        _render_phase_codetection_heatmap(meter_df, ["L1", "L2", "L3"])

        st.markdown("**Per-meter phase labels**")
        st.dataframe(meter_df, use_container_width=True)
        return

    slope_rows = []
    threshold_rows = []
    maer_rows = []
    for meter_id, meta in meta_payload.items():
        if not isinstance(meta, dict):
            continue
        for stat in meta.get("Heat_pump_stats", []):
            if not isinstance(stat, str) or not stat.startswith("slope"):
                continue
            parts = stat.split(":", 1)
            if len(parts) != 2:
                continue
            phase = parts[0].split()[-1].strip().lower()
            slope_value = _safe_numeric_from_stat(parts[1])
            if slope_value is not None:
                slope_rows.append({"meter_id": str(meter_id), "phase": phase, "slope": slope_value})
            threshold_match = re.search(r"Threshold:\s*([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)", parts[1])
            if threshold_match:
                threshold_rows.append(
                    {
                        "meter_id": str(meter_id),
                        "phase": phase,
                        "threshold": float(threshold_match.group(1)),
                    }
                )

        for stat in meta.get("MAEr", []):
            if not isinstance(stat, str):
                continue
            parts = stat.split(":", 1)
            if len(parts) != 2:
                continue
            phase = parts[0].strip().lower()
            maer_value = _safe_numeric_from_stat(parts[1])
            if maer_value is not None:
                maer_rows.append({"meter_id": str(meter_id), "phase": phase, "maer_percent": maer_value})

    slope_df = pd.DataFrame()
    if slope_rows:
        slope_df = pd.DataFrame(slope_rows)
        slope_avg_df = slope_df.groupby("phase", as_index=False)["slope"].mean()
        slope_avg_df["phase"] = slope_avg_df["phase"].str.upper()
        if threshold_rows:
            threshold_df = pd.DataFrame(threshold_rows)
            slope_margin_df = slope_df.merge(threshold_df, on=["meter_id", "phase"], how="inner")
            slope_margin_df["phase"] = slope_margin_df["phase"].str.upper()
            slope_margin_df["margin_to_threshold"] = (
                slope_margin_df["threshold"] - slope_margin_df["slope"]
            )

    chart_col1, chart_col2 = st.columns(2)
    with chart_col1:
        _render_compact_phase_bar_chart(
            counts_df,
            phase_col="phase",
            value_col="detected_heat_pump",
            label="Heat Pump Detections By Phase",
            color="#2E86AB",
        )
    with chart_col2:
        if not slope_avg_df.empty:
            _render_compact_phase_bar_chart(
                slope_avg_df,
                phase_col="phase",
                value_col="slope",
                label="Average Slope By Phase",
                color="#E07A5F",
                value_format="{:.3f}",
            )

    summary_col1, summary_col2 = st.columns(2)
    with summary_col1:
        st.markdown("**Meters By Number Of Positive Phases**")
        st.bar_chart(phase_hits_df.set_index("label")[["meters"]], use_container_width=True)
    with summary_col2:
        st.markdown("**Most Common Phase Detection Combinations**")
        st.bar_chart(combo_counts_df.set_index("phase_combo")[["meters"]], use_container_width=True)

    if not slope_df.empty:
        slope_plot_df = slope_df.copy()
        slope_plot_df["phase"] = slope_plot_df["phase"].str.upper()
        slope_plot_df["meter_sort"] = pd.to_numeric(slope_plot_df["meter_id"], errors="coerce")
        slope_plot_df["meter_sort"] = slope_plot_df["meter_sort"].fillna(float("inf"))
        slope_plot_df = slope_plot_df.sort_values(["meter_sort", "meter_id", "phase"])
        slope_line_df = (
            slope_plot_df.pivot_table(index="meter_id", columns="phase", values="slope", aggfunc="mean")
            .reindex(columns=[phase for phase in ["L1", "L2", "L3"] if phase in slope_plot_df["phase"].unique()])
        )
        if not slope_line_df.empty:
            st.markdown("**Slope Lines By Phase Across Meters**")
            st.caption("Each line shows slope values across meters for one phase, ordered by meter ID.")
            st.line_chart(slope_line_df, use_container_width=True)

        st.caption(
            "Hint: slope here means the fitted change in estimated thermal load per 1 degree C change in outdoor "
            "temperature, computed phase-by-phase on cold days. More negative slopes indicate load rises as "
            "temperature falls, which is one of the heat-pump indicators used by stat_labeler."
        )

    if not slope_margin_df.empty:
        margin_avg_df = (
            slope_margin_df.groupby("phase", as_index=False)["margin_to_threshold"].mean()
            .rename(columns={"margin_to_threshold": "avg_margin_to_threshold"})
        )
        insight_col1, insight_col2 = st.columns(2)
        with insight_col1:
            _render_phase_codetection_heatmap(meter_df, ["L1", "L2", "L3"])
        with insight_col2:
            _render_compact_phase_bar_chart(
                margin_avg_df,
                phase_col="phase",
                value_col="avg_margin_to_threshold",
                label="Average Margin To Slope Threshold By Phase",
                color="#5B8E7D",
                value_format="{:.3f}",
                chart_height=220,
                title_caption=(
                    "Positive margin means the fitted slope sits below the phase threshold and therefore supports a "
                    "heat-pump detection more strongly."
                ),
            )
    else:
        _render_phase_codetection_heatmap(meter_df, ["L1", "L2", "L3"])

    if maer_rows:
        maer_df = pd.DataFrame(maer_rows)
        maer_avg_df = maer_df.groupby("phase", as_index=True)["maer_percent"].mean().to_frame("avg_MAEr_%")
        st.markdown("**Average MAEr by phase (from meta results)**")
        st.caption(
            "MAEr is the mean absolute error divided by the mean observed phase load. Lower values indicate the "
            "reconstructed load matches the measured phase load more closely."
        )
        st.line_chart(maer_avg_df)

    st.markdown("**Per-meter phase labels**")
    st.dataframe(meter_df, use_container_width=True)


@st.cache_data(show_spinner=False)
def _load_sm_classifier_run(run_name: str) -> tuple[dict, dict]:
    """ Load characterization and classification results for a given SM classifier run from Results directory."""
    CURRENT_FILE = Path(__file__)
    run_name = run_name.strip()
    run_dir = CURRENT_FILE.parent / "Results" / run_name
    characterization_path = run_dir / f"{run_name}_SM_characterization.json"
    classification_path = run_dir / f"{run_name}_SM_classification.json"
    if not characterization_path.exists() or not classification_path.exists():
        raise FileNotFoundError("Missing classifier result files for selected run.")
    with characterization_path.open("r", encoding="utf-8") as f:
        characterization = json.load(f)
    with classification_path.open("r", encoding="utf-8") as f:
        classification = json.load(f)
    return characterization, classification


def _load_meter_df(
    extractor: DataExtractor,
    meter_id: str,
    profile: str,
    add_current: bool,
    add_unbalance: bool,
    load_existing_only: bool,
) -> pd.DataFrame:
    if profile == "raw_profiles":
        if load_existing_only:
            df = extractor.load_raw_dataset_for_sm(
                meter_id=meter_id,
                add_current=add_current,
                add_unbalance=add_unbalance,
            )
            if df is None:
                raise FileNotFoundError(
                    "No existing raw profile found for this meter. "
                    "Disable 'Load existing only' to extract from raw data."
                )
            return df

        return extractor.extract_raw_dataset_for_sm(
            meter_id=meter_id,
            add_current=add_current,
            add_unbalance=add_unbalance,
            save=False,
        )

    if profile == "cleaned_profiles":
        return extractor.load_cleaned_dataset_for_sm(
            meter_id=meter_id,
            add_current=add_current,
            add_unbalance=add_unbalance,
        )

    return extractor.load_cleaned_and_phase_corrected_dataset_for_sm(
        meter_id=meter_id,
        add_current=add_current,
        add_unbalance=add_unbalance,
    )


def _load_group_df(
    extractor: DataExtractor,
    entity: str,
    entity_id: str,
    topology_level: str,
    profile: str,
    add_current: bool,
    add_unbalance: bool,
) -> pd.DataFrame:
    """ Load or extract dataset for a non-meter entity based on selected profile and topology scope."""
    if entity == "cabinet":
        if profile == "raw_profiles":
            return extractor.extract_raw_sm_dataset_for_cabinet(
                cabinet_id=entity_id,
                topology_processing_level=topology_level,
                use_existing_raw_sm_profiles=True,
                add_current=add_current,
                add_unbalance=add_unbalance,
                save=False,
            )
        if profile == "cleaned_profiles":
            return extractor.extract_cleaned_sm_dataset_for_cabinet(
                cabinet_id=entity_id,
                topology_processing_level=topology_level,
                add_current=add_current,
                add_unbalance=add_unbalance,
                save=False,
            )
        return extractor.extract_cleaned_and_phase_corrected_sm_dataset_for_cabinet(
            cabinet_id=entity_id,
            topology_processing_level=topology_level,
            add_current=add_current,
            add_unbalance=add_unbalance,
            save=False,
        )

    if entity == "feeder":
        if profile == "raw_profiles":
            return extractor.extract_raw_sm_dataset_for_feeder(
                feeder_id=entity_id,
                topology_processing_level=topology_level,
                use_existing_raw_sm_profiles=True,
                add_current=add_current,
                add_unbalance=add_unbalance,
                save=False,
            )
        if profile == "cleaned_profiles":
            return extractor.extract_cleaned_sm_dataset_for_feeder(
                feeder_id=entity_id,
                topology_processing_level=topology_level,
                add_current=add_current,
                add_unbalance=add_unbalance,
                save=False,
            )
        return extractor.extract_cleaned_and_phase_corrected_sm_dataset_for_feeder(
            feeder_id=entity_id,
            topology_processing_level=topology_level,
            add_current=add_current,
            add_unbalance=add_unbalance,
            save=False,
        )

    if entity == "transformer":
        if profile == "raw_profiles":
            return extractor.extract_raw_sm_dataset_for_transformer(
                transformer_id=entity_id,
                topology_processing_level=topology_level,
                use_existing_raw_sm_profiles=True,
                add_current=add_current,
                add_unbalance=add_unbalance,
                save=False,
            )
        if profile == "cleaned_profiles":
            return extractor.extract_cleaned_sm_dataset_for_transformer(
                transformer_id=entity_id,
                topology_processing_level=topology_level,
                add_current=add_current,
                add_unbalance=add_unbalance,
                save=False,
            )
        return extractor.extract_cleaned_and_phase_corrected_sm_dataset_for_transformer(
            transformer_id=entity_id,
            topology_processing_level=topology_level,
            add_current=add_current,
            add_unbalance=add_unbalance,
            save=False,
        )

    if entity == "secondary_substation":
        if profile == "raw_profiles":
            return extractor.extract_raw_sm_dataset_for_secondary_substation(
                substation_id=entity_id,
                topology_processing_level=topology_level,
                use_existing_raw_sm_profiles=True,
                add_current=add_current,
                add_unbalance=add_unbalance,
                save=False,
            )
        if profile == "cleaned_profiles":
            return extractor.extract_cleaned_sm_dataset_for_secondary_substation(
                substation_id=entity_id,
                topology_processing_level=topology_level,
                add_current=add_current,
                add_unbalance=add_unbalance,
                save=False,
            )
        return extractor.extract_cleaned_and_phase_corrected_sm_dataset_for_secondary_substation(
            substation_id=entity_id,
            topology_processing_level=topology_level,
            add_current=add_current,
            add_unbalance=add_unbalance,
            save=False,
        )

    # zip
    if profile == "raw_profiles":
        return extractor.extract_raw_sm_dataset_for_zip(
            zip_id=entity_id,
            topology_processing_level=topology_level,
            add_current=add_current,
            add_unbalance=add_unbalance,
            save=False,
        )
    if profile == "cleaned_profiles":
        return extractor.extract_cleaned_sm_dataset_for_zip(
            zip_id=entity_id,
            topology_processing_level=topology_level,
            add_current=add_current,
            add_unbalance=add_unbalance,
            save=False,
        )
    return extractor.extract_cleaned_and_phase_corrected_sm_dataset_for_zip(
        zip_id=entity_id,
        topology_processing_level=topology_level,
        add_current=add_current,
        add_unbalance=add_unbalance,
        save=False,
    )


@st.cache_data(show_spinner=False)
def load_plot_df(
    data_dir_path: str,
    entity: str,
    entity_id: str,
    topology_level: str,
    profile: str,
    add_current: bool,
    add_unbalance: bool,
    load_existing_only: bool,
) -> pd.DataFrame:
    extractor = get_extractor(data_dir_path=data_dir_path)
    if entity == "meter":
        return _load_meter_df(
            extractor=extractor,
            meter_id=entity_id,
            profile=profile,
            add_current=add_current,
            add_unbalance=add_unbalance,
            load_existing_only=load_existing_only,
        )
    return _load_group_df(
        extractor=extractor,
        entity=entity,
        entity_id=entity_id,
        topology_level=topology_level,
        profile=profile,
        add_current=add_current,
        add_unbalance=add_unbalance,
    )





### ---------- Data Explorer UI logic -----------------  ###
# Main rendering function for the data explorer section of the Streamlit app.
def _render_data_explorer(data_dir_path: str) -> None:
    st.subheader("Data Explorer")
    st.caption(
        "Overview of what data is available in the framework: metadata loaded in the database "
        "and raw parquet available in object storage."
    )

    with st.sidebar:
        st.markdown("### Data Explorer Config")
        if st.button("Refresh Availability", key="explorer_refresh"):
            st.cache_data.clear()
            st.success("Availability cache refreshed.")

    topology_versions = _get_topology_versions()
    current_topology_version = next((row["version"] for row in topology_versions if row.get("is_current")), None)
    selected_topology_version = None
    if topology_versions:
        version_options = [row["version"] for row in topology_versions]
        selected_index = 0
        if current_topology_version in version_options:
            selected_index = version_options.index(current_topology_version)
        selected_topology_version = st.selectbox(
            "Topology version",
            options=version_options,
            index=selected_index,
            format_func=lambda value: f"{value} " if value == current_topology_version else str(value),
            key="explorer_topology_version",
        )

    try:
        with st.spinner("Loading dataset inventory from database and raw parquet..."):
            db_overview = _get_meter_inventory_overview()
            topology_overview = _get_topology_inventory_overview(selected_topology_version)
            parquet_overview, parquet_by_date_df, _ = _get_raw_parquet_inventory(data_dir_path)
            latest_timeseries_update = _get_latest_timeseries_update()
            topology_plot_df, _, _ = _get_topology_connection_view()
            feeder_overlay_df, cabinet_overlay_df = _get_topology_meter_overlay()
            current_topology_inventory = _get_topology_inventory_overview(current_topology_version)

        latest_db_update_days = _days_since(db_overview["latest_db_update"])
        latest_timeseries_days = _days_since(latest_timeseries_update)
        latest_topology_days = _days_since(topology_overview["topology_ingested_at"])

        overview_cols = st.columns(8)
        overview_cols[0].metric("Meters in database", db_overview["total_meters"])
        overview_cols[1].metric("Meters with rows", db_overview["meters_with_rows"])
        overview_cols[2].metric("Meters with data_quality", db_overview["meters_with_data_quality"])
        overview_cols[3].metric("Meters with data_statistics", db_overview["meters_with_data_statistics"])
        overview_cols[4].metric("Meters with connectivity", db_overview["meters_with_connectivity"])
        overview_cols[5].metric("Raw parquet files", parquet_overview["parquet_files"])
        overview_cols[6].metric(
            "Days since latest database update",
            latest_db_update_days if latest_db_update_days is not None else "-",
        )
        overview_cols[7].metric(
            "Days since latest timeseries update",
            latest_timeseries_days if latest_timeseries_days is not None else "-",
        )



        # c10, c11 = st.columns(2)

        # c10.metric(
        #     "Days since latest timeseries update",
        #     latest_timeseries_days if latest_timeseries_days is not None else "-",
        # )
        # c11.metric(
        #     "Days since current topology ingest",
        #     latest_topology_days if latest_topology_days is not None else "-",
        # )

        st.markdown("**Phase measurements time coverage**")
        st.write(
            {
                "First measurement": db_overview["min_first_seen"],
                "Last measurement": db_overview["max_last_seen"],
                "Latest database update": db_overview["latest_db_update"],
                "Latest timeseries update": latest_timeseries_update,
                "Raw parquet min date": parquet_overview["min_date"],
                "Raw parquet max date": parquet_overview["max_date"],
                "Selected topology version": topology_overview["topology_version"],
                "Selected topology ingest": topology_overview["topology_ingested_at"],
            }
        )

        st.markdown("**LV topology inventory**")
        topo_cols = st.columns(7)
        topo_cols[0].metric("Topology version", topology_overview["topology_version"] or "-")
        topo_cols[1].metric("Substations", topology_overview["substations"])
        topo_cols[2].metric("Transformers", topology_overview["transformers"])
        topo_cols[3].metric("Feeders", topology_overview["feeders"])
        topo_cols[4].metric("Cabinets", topology_overview["cabinets"])
        topo_cols[5].metric("Delivery points", topology_overview["delivery_points"])
        topo_cols[6].metric("Cables", topology_overview["cables"])
        st.caption(
            "Selected topology ingest age: "
            + (f"{latest_topology_days} days" if latest_topology_days is not None else "-")
        )

        st.markdown("**Topology connection overview**")
        if (
            selected_topology_version is not None
            and current_topology_version is not None
            and int(selected_topology_version) != int(current_topology_version)
        ):
            st.info(
                f"Connection charts below use exported topology for current version {current_topology_version}. "
                f"Inventory metrics above are shown for selected version {selected_topology_version}."
            )
        if topology_plot_df.empty:
            st.info("No current topology export available for connection plots.")
        else:
            scope_type = st.selectbox(
                "Topology scope",
                options=["All", "Substation", "Transformer", "Feeder"],
                index=0,
                key="explorer_topology_scope_type",
            )
            scope_column_map = {
                "Substation": "secondary_substation",
                "Transformer": "transformer",
                "Feeder": "lv_feeder",
            }
            if scope_type == "All":
                scope_value = "All"
            else:
                scope_column = scope_column_map[scope_type]
                scope_options = sorted(topology_plot_df[scope_column].dropna().astype(str).unique().tolist())
                scope_value = st.selectbox(
                    f"{scope_type} selection",
                    options=scope_options,
                    index=0,
                    key="explorer_topology_scope_value",
                )

            scoped_topology_df = _filter_topology_by_scope(topology_plot_df, scope_type, scope_value)
            scoped_component_df, scoped_connection_df = _summarize_topology_connections(scoped_topology_df)
            scoped_connection_plot_df = _format_connection_types_for_plot(scoped_connection_df)
            scoped_feeder_overlay_df = _filter_overlay_by_scope(feeder_overlay_df, scoped_topology_df, "feeder")
            scoped_cabinet_overlay_df = _filter_overlay_by_scope(cabinet_overlay_df, scoped_topology_df, "cabinet")

            top_c1, top_c2 = st.columns(2)
            with top_c1:
                if not scoped_component_df.empty:
                    st.markdown("**Components represented in exported topology**")
                    st.bar_chart(scoped_component_df.set_index("component_type"))
            with top_c2:
                if not scoped_connection_plot_df.empty:
                    st.markdown("**Connection types between nodes**")
                    st.bar_chart(scoped_connection_plot_df.set_index("connection_type"))

            connected_balance_df = _build_connected_vs_unconnected_df(
                scoped_component_df,
                current_topology_inventory,
            )
            if not connected_balance_df.empty:
                st.markdown("**Not connected components**")
                st.bar_chart(connected_balance_df.set_index("category")[["Not connected"]])

            st.caption(
                f"Graph scope: {scope_type}" + ("" if scope_value == "All" else f" | selection: {scope_value}")
            )
            overlay_preview_df = pd.concat(
                [
                    scoped_feeder_overlay_df.assign(node_type="LvFeeder"),
                    scoped_cabinet_overlay_df.assign(node_type="Cabinet"),
                ],
                ignore_index=True,
            )
            if not overlay_preview_df.empty:
                st.markdown("**Meter availability overlay for scoped nodes**")
                st.dataframe(_format_overlay_preview_for_display(overlay_preview_df), use_container_width=True)
            graph_max_edges = st.slider(
                "Max topology graph edges",
                min_value=10,
                max_value=200,
                value=80,
                step=10,
                key="explorer_topology_graph_edges",
            )
            with st.expander("Topology plot legend and notes", expanded=False):
                st.caption("Open for extra context about charts and graph rendering.")
                st.write(
                    {
                        "Scope behavior": "All topology plots and graph follow the selected "
                        "scope and scope selection.",
                        "Max topology graph edges": "Caps rendered edges for readability; "
                        "lower values show a smaller subgraph.",
                        "Base node colors": "LvFeeder=Green, Cabinet=Blue, DeliveryPoint=Orange, "
                        "SecondarySubstation=Purple, Transformer=Brown, Other=Gray.",
                        "Availability color override": "LvFeeder/Cabinet are recolored by time-series "
                        "availability when overlay exists: Red<50%, Amber 50-79%, Green>=80%, Gray=unknown.",
                        "Edge styles": "Dashed arrows show hierarchy links (Substation->Transformer, "
                        "Transformer->Feeder, and feeder attachment to node1). Solid arrows show topology "
                        "link node1->node2.",
                        "Dotted edges": "Not used in current renderer.",
                    }
                )
            dot_graph = _build_topology_graphviz(
                scoped_topology_df,
                feeder_overlay_df=scoped_feeder_overlay_df,
                cabinet_overlay_df=scoped_cabinet_overlay_df,
                max_edges=int(graph_max_edges),
            )
            try:
                st.graphviz_chart(dot_graph, use_container_width=True)
            except Exception as exc:
                st.warning(f"Topology graph rendering is not available in this environment: {exc}")
                st.code(dot_graph, language="dot")

        st.markdown("**Raw parquet inventory trend by date**")
        if parquet_by_date_df.empty:
            st.info("No parquet files found in the selected data directory.")
        else:
            parquet_date_plot_df = parquet_by_date_df.copy()
            parquet_date_plot_df["date"] = pd.to_datetime(parquet_date_plot_df["date"], errors="coerce")
            parquet_date_plot_df = parquet_date_plot_df.dropna(subset=["date"]).sort_values("date")
            st.line_chart(parquet_date_plot_df.set_index("date")["parquet_files"])

    except Exception as exc:
        st.error(f"Failed to load dataset availability overview: {exc}")

    with st.expander("Timeseries drilldown plot", expanded=False):
        st.caption("Use this for detailed plotting of a specific entity after reviewing availability above.")

        entity = st.selectbox(
            "Entity",
            ["meter", "cabinet", "feeder", "transformer", "secondary_substation", "zip"],
            index=0,
            key="explorer_entity",
        )
        entity_id = st.text_input("Entity ID", value="87", key="explorer_entity_id")
        profile = st.selectbox(
            "Profile",
            ["raw_profiles", "cleaned_profiles", "cleaned_and_phase_corrected_profiles"],
            index=0,
            key="explorer_profile",
        )
        profile_label_map = {
            "raw_profiles": "Raw",
            "cleaned_profiles": "Cleaned",
            "cleaned_and_phase_corrected_profiles": "Cleaned + phase corrected",
        }
        topology_level = st.selectbox(
            "Topology level",
            ["raw", "cleaned", "cleaned_and_corrected"],
            index=0,
            key="explorer_topology",
            disabled=(entity == "meter"),
        )
        if entity == "meter":
            st.caption(
                "Topology level applies to aggregated entities only "
                "(cabinet/feeder/transformer/substation/zip)."
            )
        st.caption(
            f"Selected profile: {profile_label_map.get(profile, profile)}"
            + (" (meter-specific loading)" if entity == "meter" else " (used with selected topology level)")
        )
        add_current = st.checkbox("Add current", value=False, key="explorer_add_current")
        add_unbalance = st.checkbox("Add unbalance", value=False, key="explorer_add_unbalance")
        load_existing_only = st.checkbox(
            "Load existing only (meter/raw)",
            value=True,
            key="explorer_load_existing_only",
            disabled=not (entity == "meter" and profile == "raw_profiles"),
        )

        if st.button("Load Drilldown Plot", type="primary", key="explorer_load_button"):
            try:
                normalized_id = _safe_entity_id(entity_id, "Entity ID")
                df = load_plot_df(
                    data_dir_path=data_dir_path,
                    entity=entity,
                    entity_id=normalized_id,
                    topology_level=topology_level,
                    profile=profile,
                    add_current=add_current,
                    add_unbalance=add_unbalance,
                    load_existing_only=load_existing_only,
                )

                if df is None or df.empty:
                    st.warning("No data returned for this selection.")
                    for hint in _drilldown_no_data_guidance(entity, profile, topology_level):
                        st.caption(f"Tip: {hint}")
                else:
                    plot_df = df.copy()
                    if not isinstance(plot_df.index, pd.DatetimeIndex):
                        for candidate in ["timestamp", "timestamp_dst"]:
                            if candidate in plot_df.columns:
                                plot_df[candidate] = pd.to_datetime(plot_df[candidate], utc=True, errors="coerce")
                                plot_df = plot_df.set_index(candidate)
                                break

                    if isinstance(plot_df.index, pd.DatetimeIndex):
                        plot_df = plot_df.sort_index()

                    st.success(f"Loaded {plot_df.shape[0]} rows and {plot_df.shape[1]} columns.")
                    numeric_cols = plot_df.select_dtypes(include=["number"]).columns.tolist()
                    if not numeric_cols:
                        st.warning("No numeric columns available to plot.")
                        st.dataframe(plot_df.head(50))
                    else:
                        feature_options = sorted({_get_feature_name(c) for c in numeric_cols})
                        selected_features = st.multiselect(
                            "Features to plot",
                            options=feature_options,
                            default=feature_options[: min(3, len(feature_options))],
                            key="explorer_features",
                        )
                        selected_cols = (
                            [c for c in numeric_cols if _get_feature_name(c) in selected_features]
                            if selected_features
                            else numeric_cols[: min(6, len(numeric_cols))]
                        )
                        st.line_chart(plot_df[selected_cols])
                        st.dataframe(plot_df[selected_cols].head(200), use_container_width=True)

                        csv_bytes = plot_df[selected_cols].to_csv().encode("utf-8")
                        st.download_button(
                            label="Download plotted data as CSV",
                            data=csv_bytes,
                            file_name=f"3phi_{entity}_{normalized_id}_{profile}.csv",
                            mime="text/csv",
                            key="explorer_download",
                        )
            except Exception as exc:
                st.error(f"Failed to load or plot data: {exc}")

# Rendering function for the timeseries ingestor section of the Streamlit app,
# allowing users to trigger ingestion with custom settings.
def _render_timeseries_ingestor(orchestrator: StreamlitOrchestrator) -> None:
    st.subheader("Timeseries Ingestor")
    st.caption("Load raw CSV timeseries into parquet datasets.")

    with st.sidebar:
        st.markdown("### Timeseries Ingestor Config")
        csv_source_path = st.text_input(
            "CSV source folder", value=_default_data_platform_data_dir(), key="tsi_csv_source"
        )
        csv_file_pattern = st.text_input("CSV file pattern", value="phase_measurements_*.csv", key="tsi_pattern")
        parquet_destination_path = st.text_input("Parquet destination", value="phase_measurements/raw", key="tsi_dest")
        override = st.checkbox("Override existing outputs", value=False, key="tsi_override")
        n_workers = st.number_input("Workers", min_value=1, max_value=16, value=4, key="tsi_workers")
        run_clicked = st.button("Run Timeseries Ingestor", type="primary", key="tsi_run")

    if run_clicked:
        with st.spinner("Running timeseries ingestor..."):
            msg = orchestrator.run_timeseries_ingestor(
                csv_source_path=csv_source_path,
                csv_file_pattern=csv_file_pattern,
                parquet_destination_path=parquet_destination_path,
                override=override,
                n_workers=int(n_workers),
            )
        st.success(msg)
    else:
        st.info("Configure settings in the sidebar and run the app.")


def _render_topology_ingestor(orchestrator: StreamlitOrchestrator) -> None:
    st.subheader("Topology Ingestor")
    st.caption("Load topology and meter-cabinet mappings.")

    with st.sidebar:
        st.markdown("### Topology Ingestor Config")
        topology_source_path = st.text_input(
            "Topology CSV path",
            value=str(Path(_default_data_platform_data_dir()) / "lv_topology.csv"),
            key="topology_csv",
        )
        sm_cab_source_path = st.text_input(
            "Meter-cabinet CSV path",
            value=str(Path(_default_data_platform_data_dir()) / "meter_cabinet_connection.csv"),
            key="topology_sm_cab_csv",
        )
        override = st.checkbox("Override existing outputs", value=False, key="topology_override")
        n_workers = st.number_input("Workers", min_value=1, max_value=16, value=4, key="topology_workers")
        run_clicked = st.button("Run Topology Ingestor", type="primary", key="topology_run")

    if run_clicked:
        with st.spinner("Running topology ingestor..."):
            msg = orchestrator.run_topology_ingestor(
                topology_source_path=topology_source_path,
                sm_cab_source_path=sm_cab_source_path,
                override=override,
                n_workers=int(n_workers),
            )
        st.success(msg)
    else:
        st.info("Configure settings in the sidebar and run the app.")


def _render_sm_classifier(orchestrator: StreamlitOrchestrator, data_dir_path: str) -> None:
    st.subheader("SM Classifier")
    st.caption(
        "Review existing classifier outputs first, then run additional classification for a selected meter sample "
        "when needed."
    )

    run_success_message = st.session_state.pop("smc_run_success_message", None)
    if run_success_message:
        st.success(run_success_message)

    with st.sidebar:
        st.markdown("### SM Classifier Config")
        classifier_run_name = st.text_input("Run name", value="streamlit_sm_classifier", key="smc_run_name")
        classifier_meter_count = st.number_input(
            "Number of smart meters to classify",
            min_value=1,
            max_value=1000000,
            value=100,
            step=1,
            key="smc_meter_count",
        )
        classifier_seed = st.number_input(
            "Random selection seed",
            min_value=0,
            max_value=2147483647,
            value=42,
            step=1,
            key="smc_seed",
        )
        classifier_overwrite = st.checkbox("Override existing results", value=False, key="smc_overwrite")
        n_workers = st.number_input("Workers", min_value=1, max_value=16, value=4, key="smc_workers")
        run_clicked = st.button("Run SM Classifier", type="primary", key="smc_run")

    _render_sm_classifier_existing_results()

    if run_clicked:
        available_sm_ids = _get_meter_ids_with_timeseries()
        requested_count = int(classifier_meter_count)
        rng = random.Random(int(classifier_seed))
        if requested_count >= len(available_sm_ids):
            sm_ids = list(available_sm_ids)
        else:
            sm_ids = rng.sample(available_sm_ids, k=requested_count)

        if not available_sm_ids:
            st.error("No smart meters with timeseries rows are available. Run the timeseries ingestor first.")
        elif not sm_ids:
            st.error("Requested meter count resolved to zero smart meters.")
        else:
            st.caption(
                f"Classifying {len(sm_ids)} smart meters from the inventory "
                f"(requested {requested_count}, available {len(available_sm_ids)}, seed {int(classifier_seed)})."
            )
            with st.spinner("Running SM classifier..."):
                result = orchestrator.run_sm_classifier(
                    run_name=classifier_run_name,
                    sm_ids=sm_ids,
                    overwrite_existing_results=classifier_overwrite,
                    n_workers=int(n_workers),
                )
            _refresh_sm_classifier_state(result["run_name"])
            st.session_state["smc_run_success_message"] = (
                f"SM classifier run ready: {result['run_name']} "
                f"({result['classified_meters']} meters currently stored in that run)."
            )
            st.rerun()
    else:
        st.caption("Use the sidebar to run a new classification sample if you need fresh outputs.")


def _render_stat_labeler(orchestrator: StreamlitOrchestrator, data_dir_path: str) -> None:
    st.subheader("Stat Labeler")
    st.caption("Run heat-pump statistical labeling with progress and live logs.")

    with st.sidebar:
        st.markdown("### Stat Labeler Config")
        stat_sm_count = st.number_input(
            "Number of smart meters to process",
            min_value=1,
            max_value=1000000,
            value=100,
            step=1,
            key="sl_sm_count",
        )
        stat_sm_seed = st.number_input(
            "Random selection seed",
            min_value=0,
            max_value=2147483647,
            value=42,
            step=1,
            key="sl_sm_seed",
        )
        n_workers = st.number_input("Workers", min_value=1, max_value=16, value=4, key="sl_workers")
        stat_overwrite = st.checkbox("Overwrite existing results", value=False, key="sl_overwrite")
        stat_only_with_hp = st.checkbox("Only meters marked heat-pump", value=False, key="sl_only_hp")
        stat_filter_data = st.checkbox("Filter timeseries", value=True, key="sl_filter")
        stat_label_summerhouse = st.checkbox("Label summerhouse", value=True, key="sl_summerhouse")
        stat_use_anova = st.checkbox("Use ANOVA", value=True, key="sl_anova")
        stat_save_meta = st.checkbox("Save meta results", value=True, key="sl_save_meta")
        stat_weather_file_local = st.text_input(
            "Weather file local path",
            value=str(APP_ROOT / "data-platform" / "data" / "weather_data.csv"),
            key="sl_weather_file",
        )

        with st.expander("Thresholds", expanded=False):
            weekly_change = st.number_input("Weekly change", value=0.01, step=0.01, format="%.4f", key="sl_thr_weekly")
            static_days = st.number_input("Static days", value=0.4, step=0.05, format="%.4f", key="sl_thr_static")
            weekend_ratio = st.number_input("Weekend ratio", value=0.45, step=0.05, format="%.4f", key="sl_thr_weekend")
            min_bins = st.number_input("Min bins", min_value=1, max_value=50, value=1, step=1, key="sl_thr_min_bins")
            max_bins = st.number_input("Max bins", min_value=1, max_value=100, value=12, step=1, key="sl_thr_max_bins")
            filter_temp = st.number_input("Filter temp", value=3.0, step=0.5, format="%.2f", key="sl_thr_filter_temp")
            anova_pvalue = st.number_input("ANOVA p-value", value=0.01, step=0.005, format="%.4f", key="sl_thr_anova")

        run_clicked = st.button("Run Stat Labeler", type="primary", key="sl_run")

    if not run_clicked:
        st.info("Configure settings in the sidebar and run the app.")
    else:
        available_sm_ids = _get_meter_ids_with_data_quality()
        requested_count = int(stat_sm_count)
        rng = random.Random(int(stat_sm_seed))
        if requested_count >= len(available_sm_ids):
            sm_ids = list(available_sm_ids)
        else:
            sm_ids = rng.sample(available_sm_ids, k=requested_count)

        if not available_sm_ids:
            st.error("No smart meters found with non-empty data_quality in the meter table.")
        elif not sm_ids:
            st.error("Requested meter count resolved to zero smart meters.")
        else:
            st.caption(
                f"Using {len(sm_ids)} randomly selected smart meters from database list "
                f"(requested {requested_count}, available {len(available_sm_ids)}, seed {int(stat_sm_seed)})."
            )
            progress_placeholder = st.progress(0, text="Preparing stat_labeler")
            logs_box = st.empty()
            log_lines = []
            ui_updates_enabled = True

            def push_log(line: str) -> None:
                nonlocal ui_updates_enabled
                if not ui_updates_enabled:
                    return
                log_lines.append(line)
                try:
                    logs_box.code("\n".join(log_lines[-120:]), language="text")
                except BaseException as exc:
                    if ScriptControlException and isinstance(exc, ScriptControlException):
                        ui_updates_enabled = False
                        return
                    raise

            def push_progress(value: float, text: str) -> None:
                nonlocal ui_updates_enabled
                if not ui_updates_enabled:
                    return
                try:
                    progress_placeholder.progress(int(max(0.0, min(1.0, value)) * 100), text=text)
                except BaseException as exc:
                    if ScriptControlException and isinstance(exc, ScriptControlException):
                        ui_updates_enabled = False
                        return
                    raise

            thresholds = {
                "weekly_change": float(weekly_change),
                "static_days": float(static_days),
                "weekend_ratio": float(weekend_ratio),
                "max_bins": int(max_bins),
                "min_bins": int(min_bins),
                "filter_temp": float(filter_temp),
                "anova_pvalue": float(anova_pvalue),
            }

            root_logger = logging.getLogger()
            streamlit_log_handler = StreamlitLogHandler(push_log)
            streamlit_log_handler.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))
            root_logger.addHandler(streamlit_log_handler)
            try:
                push_progress(0.02, "Launching stat_labeler")
                labels = orchestrator.run_stat_labeler(
                    sm_ids=sm_ids,
                    n_workers=int(n_workers),
                    overwrite_existing_results=stat_overwrite,
                    thresholds=thresholds,
                    process_only_sm_with_hp=stat_only_with_hp,
                    filter_data=stat_filter_data,
                    label_summerhouse=stat_label_summerhouse,
                    use_anova=stat_use_anova,
                    save_meta_results=stat_save_meta,
                    weather_file_local=stat_weather_file_local,
                    progress_callback=push_progress,
                )
                push_progress(1.0, "Stat labeler finished")
                st.session_state["stat_labeler_latest_labels"] = labels
                st.session_state["stat_labeler_thresholds"] = thresholds
                _refresh_stat_labeler_result_state(data_dir_path)
                if ui_updates_enabled:
                    st.success("Stat labeler run completed.")
            except BaseException as exc:
                if ScriptControlException and isinstance(exc, ScriptControlException):
                    return
                if not isinstance(exc, Exception):
                    raise
                push_progress(1.0, "Stat labeler failed")
                st.error(f"Stat labeler run failed: {exc}")
            finally:
                root_logger.removeHandler(streamlit_log_handler)

    st.markdown("### Latest Stat Labeler Results")
    if st.session_state.get("stat_labeler_thresholds"):
        st.caption(f"Latest run thresholds: {st.session_state['stat_labeler_thresholds']}")

    labels_payload = st.session_state.get("stat_labeler_latest_labels", {})
    try:
        hp_paths, _ = _list_stat_labeler_result_files(data_dir_path)
        selected_hp_path = None
        if hp_paths:
            _apply_pending_selection("sl_latest_results_selected_file", hp_paths)
            selected_hp_path = st.selectbox(
                "Select stat labeler result file",
                options=hp_paths,
                format_func=_format_stat_labeler_result_option,
                key="sl_latest_results_selected_file",
            )

        hp_path, hp_payload, meta_path, meta_payload = _load_stat_labeler_results(
            data_dir_path,
            hp_path=selected_hp_path,
        )
        if hp_payload:
            labels_payload = hp_payload
        if hp_path:
            st.caption(f"Selected heat_pump_results file: {hp_path}")
        if meta_path:
            st.caption(f"Matched meta_results file: {meta_path}")
        _render_stat_labeler_results(labels_payload, meta_payload)
    except Exception as exc:
        st.warning(f"Could not load stat_labeler results from object storage: {exc}")
        _render_stat_labeler_results(labels_payload, {})


def _render_results_viewer(data_dir_path: str) -> None:
    st.subheader("Results Viewer")
    st.caption("Inspect generated outputs without re-running apps.")

    extractor = get_extractor(data_dir_path=data_dir_path)

    with st.expander("Dataset info", expanded=False):
        if st.button("Show timeseries info", key="results_show_info"):
            try:
                info = extractor.v1_get_timeseries_info()
                st.json(info)
            except Exception as exc:
                st.error(f"Could not load timeseries info: {exc}")

    with st.expander("SM Classifier Results", expanded=True):
        run_names = _list_sm_classifier_runs()
        if not run_names:
            st.info("No SM classifier runs found in data_apps/Results yet.")
        else:
            _apply_pending_selection("results_selected_run", run_names)
            selected_run = st.selectbox("SM classifier run", options=run_names, key="results_selected_run")
            try:
                sm_characterization, sm_classification = _load_sm_classifier_run(selected_run)
                _render_sm_classifier_results_browser(
                    run_name=selected_run,
                    characterization=sm_characterization,
                    classification=sm_classification,
                    data_dir_path=data_dir_path,
                    widget_prefix="results_smc",
                )
            except Exception as exc:
                st.error(f"Could not load generated results: {exc}")

    with st.expander("Stat Labeler Results", expanded=True):
        labels_payload = st.session_state.get("stat_labeler_latest_labels", {})
        try:
            hp_paths, _ = _list_stat_labeler_result_files(data_dir_path)
            selected_hp_path = None
            if hp_paths:
                _apply_pending_selection("results_stat_labeler_selected_file", hp_paths)
                selected_hp_path = st.selectbox(
                    "Stat labeler result file",
                    options=hp_paths,
                    format_func=_format_stat_labeler_result_option,
                    key="results_stat_labeler_selected_file",
                )

            hp_path, hp_payload, meta_path, meta_payload = _load_stat_labeler_results(
                data_dir_path,
                hp_path=selected_hp_path,
            )
            if hp_payload:
                labels_payload = hp_payload
            if hp_path:
                st.caption(f"Selected heat_pump_results file: {hp_path}")
            if meta_path:
                st.caption(f"Matched meta_results file: {meta_path}")
            _render_stat_labeler_results(labels_payload, meta_payload)
        except Exception as exc:
            st.warning(f"Could not load stat_labeler results from object storage: {exc}")
            _render_stat_labeler_results(labels_payload, {})


def main() -> None:
    st.set_page_config(page_title="3PHI Data Apps", layout="wide")
    st.title("3PhaseInsights Data Apps Showcase")
    st.caption("Choose an app from the sidebar and configure only what is essential.")

    missing_env_vars = load_runtime_env()
    if missing_env_vars:
        st.error(
            "Missing required environment variables: "
            + ", ".join(missing_env_vars)
            + ". Set them in shell or in .env at project root."
        )
        return

    with st.sidebar:
        st.header("Navigation")
        app_choice = st.selectbox(
            "Choose app",
            [
                "Data Explorer",
                "Timeseries Ingestor",
                "Topology Ingestor",
                "SM Classifier",
                "Stat Labeler",
                "Results Viewer",
            ],
            index=0,
            key="app_choice",
        )

        st.header("Shared")
        default_data_dir = os.getenv("S3_DATA_DIR_PATH", "phase_measurements/raw")
        data_dir_path = st.text_input("Data dir path", value=default_data_dir, key="shared_data_dir")

    if app_choice in {"Timeseries Ingestor", "Topology Ingestor", "SM Classifier", "Stat Labeler"}:
        orchestrator = create_orchestrator(data_dir_path=data_dir_path)
    else:
        orchestrator = None

    if app_choice == "Data Explorer":
        _render_data_explorer(data_dir_path=data_dir_path)
    elif app_choice == "Timeseries Ingestor":
        _render_timeseries_ingestor(orchestrator=orchestrator)
    elif app_choice == "Topology Ingestor":
        _render_topology_ingestor(orchestrator=orchestrator)
    elif app_choice == "SM Classifier":
        _render_sm_classifier(orchestrator=orchestrator, data_dir_path=data_dir_path)
    elif app_choice == "Stat Labeler":
        _render_stat_labeler(orchestrator=orchestrator, data_dir_path=data_dir_path)
    else:
        _render_results_viewer(data_dir_path=data_dir_path)


if __name__ == "__main__":
    main()
