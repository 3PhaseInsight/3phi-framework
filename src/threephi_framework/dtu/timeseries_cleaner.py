import dask.dataframe as dd
import pandas as pd


def apply_quality_flags(raw_ddf: dd.DataFrame, flags_ddf: dd.DataFrame) -> dd.DataFrame:
    """Apply quality flags to raw timeseries data, nulling flagged measurements.

    Both DataFrames must share the same (meter_number, timestamp) key space.
    The flags dataset must contain one int8 flag column per measurement column
    following the naming convention ``<col>_flag``. Any flag value other than
    0 (QualityFlag.OK) causes the corresponding measurement to be set to NaN.

    Args:
        raw_ddf: Raw timeseries Dask DataFrame (PhaseMeasurementsParquetSchema).
        flags_ddf: Quality-flag Dask DataFrame (QualityFlagParquetSchema),
                   co-partitioned with raw_ddf on dt/shard.

    Returns:
        Cleaned Dask DataFrame with the same schema as raw_ddf.
    """
    join_keys = ["meter_number", "timestamp"]
    merged = raw_ddf.merge(flags_ddf, on=join_keys, how="left")

    measurement_cols = [c for c in raw_ddf.columns if c not in join_keys]
    for col in measurement_cols:
        flag_col = f"{col}_flag"
        if flag_col in merged.columns:
            merged[col] = merged[col].where(merged[flag_col] == 0)

    return merged[raw_ddf.columns]


def apply_phase_map(ddf: dd.DataFrame, phase_map: pd.DataFrame) -> dd.DataFrame:
    """Correct phase misassignment by rearranging measurement columns per meter.

    The phase map must have columns:
        meter_number (str), l1_maps_to (str), l2_maps_to (str), l3_maps_to (str)
    where values of l*_maps_to are one of "l1", "l2", "l3". A row for meter M
    with l1_maps_to="l2" means data currently in the l1 columns actually belongs
    to l2. Meters absent from the phase map are passed through unchanged.

    The rename is applied atomically per meter group to avoid intermediate
    collisions when two phases are swapped.

    Args:
        ddf: Cleaned timeseries Dask DataFrame.
        phase_map: Small pandas DataFrame with phase assignment corrections.
                   Loaded once and broadcast to all partitions.

    Returns:
        Phase-corrected Dask DataFrame with the same schema as the input.
    """
    phase_map_dict: dict[str, dict] = phase_map.set_index("meter_number")[
        ["l1_maps_to", "l2_maps_to", "l3_maps_to"]
    ].to_dict("index")
    all_columns = list(ddf.columns)

    def _correct_partition(pdf: pd.DataFrame) -> pd.DataFrame:
        groups = []
        for meter_id, group in pdf.groupby("meter_number", sort=False):
            correction = phase_map_dict.get(str(meter_id))
            if correction is None or _is_identity(correction):
                groups.append(group)
                continue
            rename = _build_atomic_rename(all_columns, correction)
            groups.append(group.rename(columns=rename))
        return pd.concat(groups) if groups else pdf.iloc[0:0]

    return ddf.map_partitions(_correct_partition, meta=ddf)


def _is_identity(correction: dict) -> bool:
    return correction["l1_maps_to"] == "l1" and correction["l2_maps_to"] == "l2" and correction["l3_maps_to"] == "l3"


def _build_atomic_rename(columns: list[str], correction: dict) -> dict[str, str]:
    """Build old→new column rename map for one meter's phase correction.

    Uses suffixed matching to safely handle swaps (e.g. l1↔l2) without
    collisions. Returns a flat old→new dict for pd.DataFrame.rename().
    """
    phase_map = {
        "l1": correction["l1_maps_to"],
        "l2": correction["l2_maps_to"],
        "l3": correction["l3_maps_to"],
    }
    rename = {}
    for col in columns:
        for src_phase, dst_phase in phase_map.items():
            if src_phase != dst_phase and col.endswith(f"_{src_phase}"):
                rename[col] = col[: -len(src_phase)] + dst_phase
                break
    return rename
