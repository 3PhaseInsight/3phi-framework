import logging
import re

import altair as alt
import pandas as pd
import streamlit as st
from sqlalchemy import select

import threephi_framework.db.db as threephi_db
from threephi_framework.data_apps.stat_labeler import StatLabeler
from threephi_framework.models.meta.meter import MetaMeterModel
from threephi_framework.streamlit.shared.utils import (
    APP_ROOT,
    ScriptControlException,
    StreamlitLogHandler,
    _apply_pending_selection,
    _normalize_sm_ids,
    _safe_numeric_from_stat,
    _validate_workers,
    get_extractor,
)


@st.cache_data(show_spinner=False, ttl=300)
def _get_meter_ids_with_data_quality() -> list[str]:
    """Return meter IDs where data_quality is available."""
    session = threephi_db.new_session()
    try:
        stmt = (
            select(MetaMeterModel.id).where(MetaMeterModel.data_quality.is_not(None)).order_by(MetaMeterModel.id.asc())
        )
        return [str(meter_id) for meter_id in session.execute(stmt).scalars().all()]
    finally:
        session.close()


def _extract_stat_labeler_result_stamp(path: str, prefix: str) -> str:
    filename = str(path).rsplit("/", 1)[-1]
    if filename.startswith(prefix) and filename.endswith(".json"):
        return filename[len(prefix) : -5]
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
        chart = (
            alt.Chart(plot_df)
            .mark_bar(size=20, cornerRadiusEnd=4)
            .encode(
                y=alt.Y(f"{phase_col}:N", sort=phase_order, title=None),
                x=alt.X(f"{value_col}:Q", title=None),
                color=alt.value(color),
                tooltip=[
                    alt.Tooltip(f"{phase_col}:N", title="Phase"),
                    alt.Tooltip(f"{value_col}:Q", title=label, format=altair_format),
                ],
            )
        )
        text = (
            alt.Chart(plot_df)
            .mark_text(align="left", baseline="middle", dx=6)
            .encode(
                y=alt.Y(f"{phase_col}:N", sort=phase_order),
                x=alt.X(f"{value_col}:Q"),
                text=alt.Text(f"{value_col}:Q", format=altair_format),
            )
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
        heatmap_chart = (
            (heatmap + text)
            .properties(
                width={"step": cell_step},
                height={"step": cell_step},
            )
            .configure_view(stroke=None)
        )
        st.altair_chart(heatmap_chart, use_container_width=False, theme="streamlit")
    except Exception:
        fallback_df = heatmap_df.pivot(index="phase_y", columns="phase_x", values="meters").reindex(
            index=phase_cols, columns=phase_cols
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
            slope_margin_df["margin_to_threshold"] = slope_margin_df["threshold"] - slope_margin_df["slope"]

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
        slope_line_df = slope_plot_df.pivot_table(
            index="meter_id", columns="phase", values="slope", aggfunc="mean"
        ).reindex(columns=[phase for phase in ["L1", "L2", "L3"] if phase in slope_plot_df["phase"].unique()])
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
            slope_margin_df.groupby("phase", as_index=False)["margin_to_threshold"]
            .mean()
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


def run_stat_labeler(
    data_dir_path: str,
    sm_ids: list[str],
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
    workers = _validate_workers(n_workers)
    normalized_sm_ids = _normalize_sm_ids(sm_ids)
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
        "data_dir_path": data_dir_path,
        "thresholds": thresholds,
        "results_dir": "s3://3phi/stat_labeler",
        "weather_file": "s3://3phi/stat_labeler/data/weather_data.csv",
        "weather_file_local": weather_file_local,
    }
    with StatLabeler(cfg) as app:
        if callable(progress_callback):
            app.progress_callback = progress_callback
        return app.stat_label_sm() or {}


def _render_stat_labeler(data_dir_path: str) -> None:
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

    import random

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
                labels = run_stat_labeler(
                    data_dir_path=data_dir_path,
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
