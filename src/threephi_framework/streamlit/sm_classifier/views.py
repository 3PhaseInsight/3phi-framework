import json
import os
import random
from datetime import datetime
from pathlib import Path

import pandas as pd
import streamlit as st
from sqlalchemy import select

import threephi_framework.db.db as threephi_db
from threephi_framework.data_apps.sm_classifier import SMClassifier
from threephi_framework.models.meta.meter import MetaMeterModel
from threephi_framework.streamlit.shared.utils import (
    SM_CLASSIFIER_RESULTS_DIR,
    _apply_pending_selection,
    _get_feature_name,
    _normalize_sm_ids,
    _validate_workers,
    load_plot_df,
)
from threephi_framework.streamlit.sm_classifier.constants import (
    SM_CLASSIFIER_CATEGORY_LABELS,
    SM_CLASSIFIER_CATEGORY_ORDER,
    SM_CLASSIFIER_PLOT_FILTERS,
    SM_CLASSIFIER_STATISTIC_LABELS,
    SM_CLASSIFIER_VARIABLE_LABELS,
    VOLTAGE_QUALITY_METRIC_LABELS,
)


def _list_sm_classifier_runs() -> list[str]:
    if not SM_CLASSIFIER_RESULTS_DIR.exists():
        return []
    return sorted([p.name for p in SM_CLASSIFIER_RESULTS_DIR.iterdir() if p.is_dir()], reverse=True)


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
                    "3-phase"
                    if isinstance(connected_phases, list) and len(connected_phases) == 3
                    else "2-phase"
                    if isinstance(connected_phases, list) and len(connected_phases) == 2
                    else "1-phase"
                    if isinstance(connected_phases, list) and len(connected_phases) == 1
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
        connected_phase_df = connected_phase_df["label"].value_counts().rename_axis("label").reset_index(name="meters")

    quality_df = pd.DataFrame(quality_rows)
    if not quality_df.empty:
        quality_df = quality_df.groupby(["phase", "summary"], as_index=False).size().rename(columns={"size": "meters"})

    preview_df = pd.DataFrame(preview_rows)
    return metrics, connected_phase_df, quality_df, preview_df


def _refresh_sm_classifier_state(run_name: str) -> None:
    _load_sm_classifier_run.clear()
    _get_sm_classifier_db_snapshot.clear()
    st.session_state["smc_existing_run_pending"] = run_name
    st.session_state["results_selected_run_pending"] = run_name


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
    run_dir = SM_CLASSIFIER_RESULTS_DIR / run_name / "Plots"
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
        str(meter_id): payload for meter_id, payload in characterization.items() if str(meter_id) in scoped_meter_ids
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
        summary_df["phase_variable"] = (
            summary_df["phase"]
            + " | "
            + summary_df["variable"].map(lambda value: SM_CLASSIFIER_VARIABLE_LABELS.get(value, value))
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
        aggregate_df["phase_variable"] = (
            aggregate_df["phase"]
            + " | "
            + aggregate_df["variable"].map(lambda value: SM_CLASSIFIER_VARIABLE_LABELS.get(value, value))
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
        artifact_options = [str(path.relative_to(SM_CLASSIFIER_RESULTS_DIR / run_name)) for path in plot_artifacts]
        selected_artifact = st.selectbox(
            "Saved plot artifact",
            options=artifact_options,
            key=f"{widget_prefix}_plot_artifact",
        )
        artifact_path = SM_CLASSIFIER_RESULTS_DIR / run_name / selected_artifact
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
            "No saved SM classifier run directory found. Showing classifier metadata currently stored in PostgreSQL."
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
def _load_sm_classifier_run(run_name: str) -> tuple[dict, dict]:
    """Load characterization and classification results for a given SM classifier run from Results directory."""
    run_name = run_name.strip()
    run_dir = SM_CLASSIFIER_RESULTS_DIR / run_name
    characterization_path = run_dir / f"{run_name}_SM_characterization.json"
    classification_path = run_dir / f"{run_name}_SM_classification.json"
    if not characterization_path.exists() or not classification_path.exists():
        raise FileNotFoundError("Missing classifier result files for selected run.")
    with characterization_path.open("r", encoding="utf-8") as f:
        characterization = json.load(f)
    with classification_path.open("r", encoding="utf-8") as f:
        classification = json.load(f)
    return characterization, classification


def run_sm_classifier(
    data_dir_path: str,
    run_name: str,
    sm_ids: list[str],
    overwrite_existing_results: bool,
    n_workers: int,
) -> dict:
    """Run the SM classifier with config parameters from the UI."""
    workers = _validate_workers(n_workers)
    normalized_sm_ids = _normalize_sm_ids(sm_ids)
    final_run_name = run_name
    if overwrite_existing_results:
        final_run_name = f"{run_name}_{datetime.utcnow().strftime('%Y%m%d%H%M%S')}"

    cfg = {
        "Name": "sm_classifier_config_streamlit",
        "use_dask": True,
        "dask": {"local": True, "n_workers": workers},
        "run_name": final_run_name,
        "data_dir_path": data_dir_path,
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


def _render_sm_classifier(data_dir_path: str) -> None:
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
                result = run_sm_classifier(
                    data_dir_path=data_dir_path,
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
