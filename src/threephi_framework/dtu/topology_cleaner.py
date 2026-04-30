import dask
import dask.dataframe as dd
import numpy as np


def clean_topology(
    topology: dd.DataFrame,
    node1_col: str,
    node2_col: str,
    sec_substation_col: str,
    cable_capacity_col: str,
    cable_length_col: str,
    phase_size_col: str,
    resistance_col: str,
    reactance_col: str,
    lv_feeder_fuse_size_col: str,
    cable_type_col: str,
    phase_material_col: str,
) -> dd.DataFrame:
    """Clean LV topology data by imputing missing values and removing duplicate edges.

    Applies three sequential cleaning steps:
    1. Numerical NaN imputation via per-substation mean, falling back to global mean.
    2. Categorical NaN imputation via per-substation mode, falling back to global mode.
    3. Deduplication of parallel edges between the same node pair, keeping the edge
       with the highest current-carrying capacity.

    Args:
        topology: Raw topology Dask DataFrame in the lv_topology schema.
        node1_col: Column name for the first node identifier.
        node2_col: Column name for the second node identifier.
        sec_substation_col: Column name for the secondary substation identifier.
        cable_capacity_col: Column name for cable current-carrying capacity.
        cable_length_col: Column name for cable length.
        phase_size_col: Column name for conductor cross-section.
        resistance_col: Column name for cable resistance.
        reactance_col: Column name for cable reactance.
        lv_feeder_fuse_size_col: Column name for LV feeder fuse size.
        cable_type_col: Column name for cable type category.
        phase_material_col: Column name for conductor material category.

    Returns:
        Cleaned topology Dask DataFrame with the same schema as the input.
    """
    # CLEANING STEP 0: Add helper columns with the identifier part of NODE1, NODE2, and Substation
    topology["NODE1_value"] = topology[node1_col].str.split(".", n=1).str.get(1).astype("string")
    topology["NODE2_value"] = topology[node2_col].str.split(".", n=1).str.get(1).astype("string")
    topology["Substation"] = topology[sec_substation_col].str.split(".", n=1).str.get(1).astype("string")

    # CLEANING STEP 1: Numerical columns — fill NaN with per-substation mean, fall back to global mean
    # TODO: This is hardcoded to second batch, could be made more generic
    numerical_cols: list[str] = [
        cable_capacity_col,
        cable_length_col,
        phase_size_col,
        resistance_col,
        reactance_col,
        lv_feeder_fuse_size_col,
    ]

    valid: dd.DataFrame = topology.dropna(subset=["Substation"])
    mean_per_sub_dd: dd.DataFrame = valid.groupby("Substation")[numerical_cols].mean()
    mean_global_dd: dd.Series = topology[numerical_cols].mean()
    mean_per_substation, mean_global = dask.compute(mean_per_sub_dd, mean_global_dd)

    # TODO: Rounding to zero decimals matches the original implementation but loses precision for
    # resistance and reactance — consider removing for those columns.
    mean_per_substation = mean_per_substation.round(0)
    mean_global = mean_global.round(0)

    for col in numerical_cols:
        fill_per_sub = topology["Substation"].map(mean_per_substation[col], meta=("Substation", "float32"))
        global_fill = np.float32(mean_global[col])
        topology[col] = topology[col].astype("float32").fillna(fill_per_sub).fillna(global_fill)

    final_defaults = {c: np.float32(mean_global[c]) for c in numerical_cols}
    topology[numerical_cols] = topology[numerical_cols].astype("float32").fillna(final_defaults)

    # CLEANING STEP 2: Categorical columns — fill NaN with per-substation mode, fall back to global mode
    # TODO: This is hardcoded to second batch, could be made more generic
    categories = {
        cable_type_col: ["APB", "MAL", "PEX", "PEX ENLEDER", "PEXMAL", "PSP", "U"],
        phase_material_col: ["AL", "CU", "U"],
    }

    for col, cats in categories.items():
        topology[col] = topology[col].astype("category").cat.set_categories(cats)

    for col in categories:
        counts: dd.DataFrame = (
            topology[["Substation", col]]
            .dropna(subset=[col])
            .groupby(["Substation", col], observed=True)
            .size()
            .compute()
            .reset_index(name="count")
        )
        mode_map = (
            counts.sort_values(["Substation", "count"], ascending=[True, False])
            .drop_duplicates("Substation")
            .set_index("Substation")[col]
            .to_dict()
        )
        topology[col] = topology[col].fillna(topology["Substation"].map(mode_map, meta=("Substation", "object")))
        global_mode = topology[col].value_counts().idxmax().compute()
        topology[col] = topology[col].fillna(global_mode)

    # CLEANING STEP 3: Remove parallel edges between the same node pair, keeping the highest capacity
    keys = ["NODE1_value", "NODE2_value"]
    grp_max = topology.groupby(keys)[cable_capacity_col].max().rename("max_cap").to_frame()
    topology = topology.merge(grp_max, left_on=keys, right_index=True, how="left")
    topology = topology[topology[cable_capacity_col] == topology["max_cap"]]
    topology = topology.drop(columns=["max_cap"]).reset_index(drop=True)
    topology = topology.drop_duplicates(subset=keys, keep="first")

    topology = topology.drop(columns=["NODE1_value", "NODE2_value", "Substation"])
    return topology.reset_index(drop=True)


def clean_sm_cabinet(sm_cab_df: dd.DataFrame, meter_number_col: str) -> dd.DataFrame:
    """Remove cabinet rows that have no meter attached.

    Args:
        sm_cab_df: Smart-meter–cabinet mapping Dask DataFrame.
        meter_number_col: Column name for the meter number identifier.

    Returns:
        Filtered Dask DataFrame containing only rows with a valid meter number.
    """
    return sm_cab_df.dropna(subset=[meter_number_col])
