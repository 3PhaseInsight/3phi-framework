from dataclasses import dataclass

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
