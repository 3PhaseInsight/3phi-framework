from dataclasses import dataclass
from enum import IntEnum

VERSION = "1.0.0"


@dataclass
class PhaseMeasurementsCsvSchema:
    timestamp_col: str = "timestamp_dst"
    meter_col: str = "meter_number"


@dataclass()
class PhaseMeasurementsParquetSchema:
    timestamp_col: str = "timestamp"
    meter_col: str = "meter_number"
    voltage_l1_col: str = "voltage_l1"
    voltage_l2_col: str = "voltage_l2"
    voltage_l3_col: str = "voltage_l3"
    act_pow_p14_l1_col: str = "active_power_p14_l1"
    act_pow_p14_l2_col: str = "active_power_p14_l2"
    act_pow_p14_l3_col: str = "active_power_p14_l3"
    act_pow_p23_l1_col: str = "active_power_p23_l1"
    act_pow_p23_l2_col: str = "active_power_p23_l2"
    act_pow_p23_l3_col: str = "active_power_p23_l3"
    react_pow_q12_l1_col: str = "reactive_power_q12_l1"
    react_pow_q12_l2_col: str = "reactive_power_q12_l2"
    react_pow_q12_l3_col: str = "reactive_power_q12_l3"
    react_pow_q34_l1_col: str = "reactive_power_q34_l1"
    react_pow_q34_l2_col: str = "reactive_power_q34_l2"
    react_pow_q34_l3_col: str = "reactive_power_q34_l3"
    thd_u_l1_col: str = "thdu_l1"
    thd_u_l2_col: str = "thdu_l2"
    thd_u_l3_col: str = "thdu_l3"
    thd_i_l1_col: str = "thdi_l1"
    thd_i_l2_col: str = "thdi_l2"
    thd_i_l3_col: str = "thdi_l3"


class QualityFlag(IntEnum):
    """Int8-encoded data quality flag for a single timeseries measurement."""

    OK = 0
    FROZEN = 1
    ZERO = 2
    BELOW_LIMIT = 3
    CORRUPT = 4


@dataclass
class QualityFlagParquetSchema:
    """Co-partitioned flag dataset.

    Same dt/shard partitioning as PhaseMeasurementsParquetSchema. One int8 flag
    column per measurement column, named <measurement_col>_flag. Values are
    QualityFlag members.
    """

    timestamp_col: str = "timestamp"
    meter_col: str = "meter_number"
    voltage_l1_col: str = "voltage_l1_flag"
    voltage_l2_col: str = "voltage_l2_flag"
    voltage_l3_col: str = "voltage_l3_flag"
    act_pow_p14_l1_col: str = "active_power_p14_l1_flag"
    act_pow_p14_l2_col: str = "active_power_p14_l2_flag"
    act_pow_p14_l3_col: str = "active_power_p14_l3_flag"
    act_pow_p23_l1_col: str = "active_power_p23_l1_flag"
    act_pow_p23_l2_col: str = "active_power_p23_l2_flag"
    act_pow_p23_l3_col: str = "active_power_p23_l3_flag"
    react_pow_q12_l1_col: str = "reactive_power_q12_l1_flag"
    react_pow_q12_l2_col: str = "reactive_power_q12_l2_flag"
    react_pow_q12_l3_col: str = "reactive_power_q12_l3_flag"
    react_pow_q34_l1_col: str = "reactive_power_q34_l1_flag"
    react_pow_q34_l2_col: str = "reactive_power_q34_l2_flag"
    react_pow_q34_l3_col: str = "reactive_power_q34_l3_flag"
    thd_u_l1_col: str = "thdu_l1_flag"
    thd_u_l2_col: str = "thdu_l2_flag"
    thd_u_l3_col: str = "thdu_l3_flag"
    thd_i_l1_col: str = "thdi_l1_flag"
    thd_i_l2_col: str = "thdi_l2_flag"
    thd_i_l3_col: str = "thdi_l3_flag"
