import logging
import os
import re
import sys
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

import threephi_framework.db.db as threephi_db  # noqa: E402
from threephi_framework.data_extractor.data_extractor import DataExtractor  # noqa: E402
from threephi_framework.object_storage.s3_connector import S3Connector  # noqa: E402


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
    """Load or extract dataset for a non-meter entity based on selected profile and topology scope."""
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
