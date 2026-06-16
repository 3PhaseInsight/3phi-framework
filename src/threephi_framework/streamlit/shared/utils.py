import logging
import os
import re
from pathlib import Path

import pandas as pd
import streamlit as st

try:
    from streamlit.runtime.scriptrunner_utils.exceptions import ScriptControlException
except Exception:  # pragma: no cover - compatibility with older/newer Streamlit layouts
    ScriptControlException = None

_UTILS_FILE = Path(__file__).resolve()
# parents: shared/ -> streamlit/ -> threephi_framework/ -> src/ -> project root
APP_ROOT = _UTILS_FILE.parents[4]
# Results dir stays at data_apps/Results/ (unchanged from original location)
SM_CLASSIFIER_RESULTS_DIR = _UTILS_FILE.parents[2] / "data_apps" / "Results"

from threephi_framework.data_extractor.data_extractor import DataExtractor  # noqa: E402


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
    """Initialize a shared DataExtractor rooted at the selected dataset path.

    The object-storage backend is resolved by the connector factory
    (OBJECT_STORAGE_BACKEND env var, "s3" by default), so the app works against
    any supported backend without code changes.
    """
    return DataExtractor(phase_measurements_dir=data_dir_path)


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
        "DB_USER",
        "DB_PASSWORD",
        "DB_HOST",
        "DB_PORT",
        "DB_NAME",
    ]
    # Storage credentials depend on the selected object-storage backend
    backend = (os.environ.get("OBJECT_STORAGE_BACKEND") or "s3").strip().lower()
    if backend in {"azure", "az", "azure_blob"}:
        required_vars += ["AZURE_STORAGE_ACCOUNT_NAME", "AZURE_STORAGE_CONTAINER_NAME"]
    else:
        required_vars += ["S3_ENDPOINT_URL", "S3_ACCESS_KEY", "S3_SECRET_KEY"]
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


def _validate_workers(n_workers: int) -> int:
    workers = int(n_workers)
    if workers < 1:
        raise ValueError("n_workers must be >= 1.")
    return workers


def _normalize_sm_ids(sm_ids: list[str]) -> list[str]:
    normalized = [str(sm_id).strip() for sm_id in sm_ids if str(sm_id).strip()]
    if not normalized:
        raise ValueError("At least one smart meter id is required.")
    return normalized


def _load_meter_df(
    extractor: DataExtractor,
    meter_id: str,
    add_current: bool,
    add_unbalance: bool,
    load_existing_only: bool,
) -> pd.DataFrame:
    # The DataExtractor assembles meter frames on demand from the canonical
    # parquet store; there is no per-SM cache anymore, so load and extract
    # return the same data and `load_existing_only` makes no difference.
    try:
        return extractor.extract_raw_dataset_for_sm(
            meter_id=meter_id,
            add_current=add_current,
            add_unbalance=add_unbalance,
            save=False,
        )
    except ValueError as exc:
        raise ValueError(
            f"No timeseries data found for meter {meter_id}. "
            "Run the Timeseries Ingestor first to populate the canonical parquet store."
        ) from exc


def _load_group_df(
    extractor: DataExtractor,
    entity: str,
    entity_id: str,
    topology_version: int | None,
    add_current: bool,
    add_unbalance: bool,
) -> pd.DataFrame:
    """Load or extract raw dataset for a non-meter entity based on selected topology version."""
    # DataExtractor group APIs currently accept processing level, not explicit topology version.
    # Streamlit uses raw topology processing while exposing version selection in UI.
    topology_processing_level = "raw"

    if entity == "cabinet":
        return extractor.extract_raw_sm_dataset_for_cabinet(
            cabinet_id=entity_id,
            topology_processing_level=topology_processing_level,
            use_existing_raw_sm_profiles=True,
            add_current=add_current,
            add_unbalance=add_unbalance,
            save=False,
        )

    if entity == "feeder":
        return extractor.extract_raw_sm_dataset_for_feeder(
            feeder_id=entity_id,
            topology_processing_level=topology_processing_level,
            use_existing_raw_sm_profiles=True,
            add_current=add_current,
            add_unbalance=add_unbalance,
            save=False,
        )

    if entity == "transformer":
        return extractor.extract_raw_sm_dataset_for_transformer(
            transformer_id=entity_id,
            topology_processing_level=topology_processing_level,
            use_existing_raw_sm_profiles=True,
            add_current=add_current,
            add_unbalance=add_unbalance,
            save=False,
        )

    if entity == "secondary_substation":
        return extractor.extract_raw_sm_dataset_for_secondary_substation(
            substation_id=entity_id,
            topology_processing_level=topology_processing_level,
            use_existing_raw_sm_profiles=True,
            add_current=add_current,
            add_unbalance=add_unbalance,
            save=False,
        )

    # zip
    return extractor.extract_raw_sm_dataset_for_zip(
        zip_id=entity_id,
        topology_processing_level=topology_processing_level,
        add_current=add_current,
        add_unbalance=add_unbalance,
        save=False,
    )


@st.cache_data(show_spinner=False)
def load_plot_df(
    data_dir_path: str,
    entity: str,
    entity_id: str,
    topology_version: int | None,
    add_current: bool,
    add_unbalance: bool,
    load_existing_only: bool,
) -> pd.DataFrame:
    extractor = get_extractor(data_dir_path=data_dir_path)
    if entity == "meter":
        return _load_meter_df(
            extractor=extractor,
            meter_id=entity_id,
            add_current=add_current,
            add_unbalance=add_unbalance,
            load_existing_only=load_existing_only,
        )
    return _load_group_df(
        extractor=extractor,
        entity=entity,
        entity_id=entity_id,
        topology_version=topology_version,
        add_current=add_current,
        add_unbalance=add_unbalance,
    )
