import os
import sys
from pathlib import Path

import streamlit as st

# Ensure package imports work when running via:
# streamlit run src/threephi_framework/streamlit/app.py
_CURRENT_FILE = Path(__file__).resolve()
_SRC_DIR = _CURRENT_FILE.parents[2]  # streamlit/ -> threephi_framework/ -> src/
if str(_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(_SRC_DIR))

from threephi_framework.streamlit.data_explorer.views import _render_data_explorer  # noqa: E402
from threephi_framework.streamlit.ingestors.views import (  # noqa: E402
    _render_timeseries_ingestor,
    _render_topology_ingestor,
)
from threephi_framework.streamlit.shared.utils import _apply_pending_selection, load_runtime_env  # noqa: E402
from threephi_framework.streamlit.sm_classifier.views import (  # noqa: E402
    _list_sm_classifier_runs,
    _load_sm_classifier_run,
    _render_sm_classifier,
    _render_sm_classifier_results_browser,
)
from threephi_framework.streamlit.stat_labeler.views import (  # noqa: E402
    _format_stat_labeler_result_option,
    _list_stat_labeler_result_files,
    _load_stat_labeler_results,
    _render_stat_labeler,
    _render_stat_labeler_results,
)


def _render_results_viewer(data_dir_path: str) -> None:
    from threephi_framework.streamlit.shared.utils import get_extractor

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

    if app_choice == "Data Explorer":
        _render_data_explorer(data_dir_path=data_dir_path)
    elif app_choice == "Timeseries Ingestor":
        _render_timeseries_ingestor()
    elif app_choice == "Topology Ingestor":
        _render_topology_ingestor()
    elif app_choice == "SM Classifier":
        _render_sm_classifier(data_dir_path=data_dir_path)
    elif app_choice == "Stat Labeler":
        _render_stat_labeler(data_dir_path=data_dir_path)
    else:
        _render_results_viewer(data_dir_path=data_dir_path)


if __name__ == "__main__":
    main()
