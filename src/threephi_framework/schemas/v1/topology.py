import pandas as pd

lv_topology_meta = pd.DataFrame(
    {
        "secondary_substation": pd.Series(dtype="string"),
        "zip_code_secondary_substation": pd.Series(dtype="string"),
        "transformer": pd.Series(dtype="string"),
        "transformer_capacity": pd.Series(dtype="Int64"),
        "lv_feeder": pd.Series(dtype="string"),
        "lv_feeder_fuse_size": pd.Series(dtype="Int64"),
        "node1": pd.Series(dtype="string"),
        "node2": pd.Series(dtype="string"),
        "cable_id": pd.Series(dtype="string"),
        "cable_type": pd.Series(dtype="string"),
        "cable_length": pd.Series(dtype="float64"),
        "phase_size": pd.Series(dtype="Int64"),
        "phase_material": pd.Series(dtype="string"),
        "cable_capacity": pd.Series(dtype="Int64"),
        "resistance": pd.Series(dtype="float64"),
        "reactance": pd.Series(dtype="float64"),
    }
)

# used to read in and cast export
lv_topology_dtype = {
    "secondary_substation": "object",
    "zip_code_secondary_substation": "object",
    "transformer": "object",
    "transformer_capacity": "Int64",
    "lv_feeder": "object",
    "lv_feeder_fuse_size": "Int64",
    "node1": "object",
    "node2": "object",
    "cable_id": "object",
    "cable_type": "object",
    "cable_length": "float64",
    "phase_size": "Int64",
    "phase_material": "object",
    "cable_capacity": "Int64",
    "resistance": "float64",
    "reactance": "float64",
}

# used to cast to "proper" datatypes before ingestion
lv_topology_types = {
    "transformer_capacity": "Int64",
    "lv_feeder_fuse_size": "Int64",
    "phase_size": "Int64",
    "cable_capacity": "Int64",
    "secondary_substation": "string",
    "zip_code_secondary_substation": "string",
    "transformer": "string",
    "lv_feeder": "string",
    "node1": "string",
    "node2": "string",
    "cable_id": "string",
    "cable_type": "string",
    "phase_material": "string",
}

sm_cabinet_meta = pd.DataFrame(
    {
        "meter_number": pd.Series(dtype="Int64"),  # unique numeric id, allow null
        "delivery_point_id": pd.Series(dtype="Int64"),  # numeric id, allow null
        "cabinet": pd.Series(dtype="string"),  # keep as string (prefixes like "Cabinet.")
        "lv_feeder": pd.Series(dtype="string"),  # empty in example, so allow missing
        "has_heat_pump": pd.Series(dtype="boolean"),  # true/false, nullable
        "has_solar_panel": pd.Series(dtype="boolean"),  # true/false, nullable
        "capacity_solar_panel": pd.Series(dtype="float64"),  # numeric, can be NaN if no solar
        "service_fuse_size": pd.Series(dtype="Int64"),  # integer amps, nullable
    }
)

# used to read in and cast export
sm_cabinet_dtype = {
    "meter_number": "Int64",
    "delivery_point_id": "Int64",
    "cabinet": "object",
    "lv_feeder": "object",
    # "has_heat_pump": "string", # don't force booleans to strings
    # "has_solar_panel": "string",
    "capacity_solar_panel": "float64",
    "service_fuse_size": "Int64",
}

# used to cast to "proper" datatypes before ingestion
sm_cabinet_types = {
    "meter_number": "Int64",
    "delivery_point_id": "Int64",
    "cabinet": "string",
    "lv_feeder": "string",
    "has_heat_pump": "boolean",
    "has_solar_panel": "boolean",
    "capacity_solar_panel": "float64",
    "service_fuse_size": "Int64",
}
