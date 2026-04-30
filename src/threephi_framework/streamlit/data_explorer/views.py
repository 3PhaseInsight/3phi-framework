import re

import pandas as pd
import streamlit as st
from sqlalchemy import func, select, text

import threephi_framework.db.db as threephi_db
from threephi_framework.controllers.topology import TopologyController
from threephi_framework.models.meta.meter import MetaMeterModel
from threephi_framework.models.topology.assets.feeder import FeederModel
from threephi_framework.models.topology.assets.transformer import TransformerModel
from threephi_framework.models.topology.graph.cable import CableModel
from threephi_framework.models.topology.graph.node import NodeModel
from threephi_framework.models.topology.graph.topology_version import TopologyVersionModel
from threephi_framework.streamlit.shared.utils import (
    _days_since,
    _get_feature_name,
    _safe_entity_id,
    get_extractor,
    load_plot_df,
)


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

    date_df = pd.DataFrame([{"date": date_key, "parquet_files": count} for date_key, count in sorted(by_date.items())])
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
        feeders = int(session.execute(select(func.count()).select_from(feeder_ids_subq)).scalar_one() or 0)
        cabinets = int(session.execute(select(func.count()).select_from(cabinet_ids_subq)).scalar_one() or 0)
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
    plot_df["connection_type"] = plot_df["source_type"].astype(str) + " -> " + plot_df["target_type"].astype(str)
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
                    value for value in topology_df[column].dropna().astype(str).tolist() if value.startswith("Cabinet.")
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
        return 'digraph G { label="No topology data available"; }'

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
        "edge [arrowsize=0.7];",
        *node_defs.values(),
        *edge_defs,
        "}",
    ]
    return "\n".join(dot_lines)


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

            st.caption(f"Graph scope: {scope_type}" + ("" if scope_value == "All" else f" | selection: {scope_value}"))
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
                        "Scope behavior": "All topology plots and graph follow the selected scope and scope selection.",
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
                "Topology level applies to aggregated entities only (cabinet/feeder/transformer/substation/zip)."
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
