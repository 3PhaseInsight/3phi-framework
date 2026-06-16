from pathlib import Path

import streamlit as st

from threephi_framework.data_apps.timeseries_ingestor import TimeseriesIngestor
from threephi_framework.data_apps.topology_ingestor import TopologyIngestor
from threephi_framework.streamlit.shared.utils import _default_data_platform_data_dir, _validate_workers


def run_timeseries_ingestor(
    csv_source_path: str,
    csv_file_pattern: str,
    parquet_destination_path: str,
    override: bool,
    n_workers: int,
) -> str:
    workers = _validate_workers(n_workers)
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


def run_topology_ingestor(
    topology_source_path: str,
    sm_cab_source_path: str,
    override: bool,
    n_workers: int,
) -> str:
    workers = _validate_workers(n_workers)
    cfg = {
        "dask": {"local": True, "n_workers": workers},
        "topology_source_path": topology_source_path,
        "sm_cab_source_path": sm_cab_source_path,
        "override": override,
    }
    with TopologyIngestor(cfg) as app:
        app.run()
    return "Topology ingestion completed."


def _render_timeseries_ingestor() -> None:
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
            msg = run_timeseries_ingestor(
                csv_source_path=csv_source_path,
                csv_file_pattern=csv_file_pattern,
                parquet_destination_path=parquet_destination_path,
                override=override,
                n_workers=int(n_workers),
            )
        st.success(msg)
    else:
        st.info("Configure settings in the sidebar and run the app.")


def _render_topology_ingestor() -> None:
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
            msg = run_topology_ingestor(
                topology_source_path=topology_source_path,
                sm_cab_source_path=sm_cab_source_path,
                override=override,
                n_workers=int(n_workers),
            )
        st.success(msg)
    else:
        st.info("Configure settings in the sidebar and run the app.")
