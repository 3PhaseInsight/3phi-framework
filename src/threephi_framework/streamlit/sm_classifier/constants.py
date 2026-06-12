SM_CLASSIFIER_CATEGORY_LABELS = {
    "All_SMs": "All meters",
    "SMs_with_dataset_containing_data": "Datasets with usable data",
    "SMs_with_dataset_containing_no_data": "Datasets without usable data",
    "SMs_without_dataset": "Missing dataset",
    "SMs_with_incomplete_topology_info": "Incomplete topology",
    "SMs_with_only_good_data_quality": "Only good data quality",
    "SMs_with_any_medium_or_bad_data_quality": "Any medium or bad data quality",
    "SMs_with_any_bad_data_quality": "Any bad data quality",
    "SMs_with_3-phase_connection": "Three-phase connection",
    "SMs_with_2-phase_connection": "Two-phase connection",
    "SMs_with_1-phase_connection": "One-phase connection",
    "SMs_with_connection_error": "Connection error",
    "SMs_with_on_off_switch": "On/off switching",
}

SM_CLASSIFIER_CATEGORY_ORDER = [
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

VOLTAGE_QUALITY_METRIC_LABELS = {
    "NaN frac": "Missing readings",
    "Zero frac": "Zero readings",
    "Below Vlim frac": "Below voltage limit",
    "Frozen frac": "Frozen readings",
    "Total corruption frac": "Total corrupted readings",
    "summary": "Quality summary",
}

SM_CLASSIFIER_VARIABLE_LABELS = {
    "V": "Voltage",
    "P14": "Active power import",
    "P23": "Active power export",
    "Q12": "Reactive power inductive",
    "Q34": "Reactive power capacitive",
}

SM_CLASSIFIER_STATISTIC_LABELS = {
    "Min": "Minimum",
    "Max": "Maximum",
    "Mean": "Mean",
    "Std": "Standard deviation",
}

SM_CLASSIFIER_PLOT_FILTERS = {
    "All with dataset containing data": "SMs_with_dataset_containing_data",
    "With only good data quality": "SMs_with_only_good_data_quality",
    "With any medium or bad data quality": "SMs_with_any_medium_or_bad_data_quality",
    "With 1-phase connection": "SMs_with_1-phase_connection",
    "With 2-phase connection": "SMs_with_2-phase_connection",
    "With connection error": "SMs_with_connection_error",
    "With on off switch": "SMs_with_on_off_switch",
}
