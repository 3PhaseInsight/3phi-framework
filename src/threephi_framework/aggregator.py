import os

import dask.dataframe as dd

from src.threephi_framework import DataExtractor

# variables: Sequence[str] = field(default_factory=lambda: ["V", "P14", "P23", "Q12", "Q34"])


class Aggregator:
    def __init__(self, config):
        self.data_extractor = DataExtractor(os.environ["PHASE_MEASUREMENTS_PATH"])
        schema = self.data_extractor.phase_measurements_parquet_schema
        self.measurement_cols = [
            schema.voltage_l1_col,
            schema.voltage_l2_col,
            schema.voltage_l3_col,
            schema.act_pow_p14_l1_col,
            schema.act_pow_p14_l2_col,
            schema.act_pow_p14_l3_col,
        ]

    def build_cagg_daily_stats(self, ddf: dd.DataFrame) -> dd.DataFrame:
        """
        Per day · (sm_id, phase, variable):
          n, n_nan, n_zero, n_below, sum, sumsq, min, max
        """
        c = [0, 0, 0]  # TODO: temporary placeholder for c
        df = ddf[ddf[c.variable_col].isin(list(c.variables))].copy()
        df["bucket"] = dd.to_datetime(df[c.ts_col]).dt.floor("D")

        is_nan = df[c.value_col].isna()
        is_zero = df[c.value_col] == 0
        is_below = df[c.value_col] < c.v_lim

        # counts as integers, sums for mean/std rollups
        df = df.assign(
            n_one=1,
            n_nan=is_nan.astype("int64"),
            n_zero=is_zero.fillna(False).astype("int64"),
            n_below=is_below.fillna(False).astype("int64"),
            sum=df[c.value_col].fillna(0.0),
            sumsq=(df[c.value_col].fillna(0.0) ** 2),
        )

        gb_cols = ["bucket", c.sm_id_col, c.phase_col, c.variable_col]
        aggs = {
            "n_one": "sum",
            "n_nan": "sum",
            "n_zero": "sum",
            "n_below": "sum",
            "sum": "sum",
            "sumsq": "sum",
            c.value_col: ["min", "max"],
        }
        out = df.groupby(gb_cols).agg(aggs)

        # flatten MultiIndex columns
        out = out.rename(
            columns={
                ("n_one", "sum"): "n",
                ("n_nan", "sum"): "n_nan",
                ("n_zero", "sum"): "n_zero",
                ("n_below", "sum"): "n_below",
                ("sum", "sum"): "sum",
                ("sumsq", "sum"): "sumsq",
                (c.value_col, "min"): "min",
                (c.value_col, "max"): "max",
            }
        )[["n", "n_nan", "n_zero", "n_below", "sum", "sumsq", "min", "max"]]

        return out.reset_index()
