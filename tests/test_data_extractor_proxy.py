import os

import pandas as pd
import pytest

# S3Connector reads its env vars at construction time but does not connect
os.environ.setdefault("S3_ENDPOINT_URL", "http://localhost:19000")
os.environ.setdefault("S3_ACCESS_KEY", "test")
os.environ.setdefault("S3_SECRET_KEY", "test")

from threephi_framework.data_extractor.data_extractor import DataExtractor  # noqa: E402


@pytest.fixture
def extractor():
    return DataExtractor()


def test_time_series_controller_rooted_at_dataset_base(extractor):
    # phase_measurements/raw → the level-aware controller must sit one level up
    assert extractor.s3_connector.dataset_root_path == "s3://3phi/phase_measurements/raw"
    assert extractor.time_series_controller.connector.dataset_root_path == "s3://3phi/phase_measurements"


def test_shape_single_meter_frame_resamples_and_suffixes(extractor):
    ts = pd.to_datetime(
        ["2025-01-01 00:00:00", "2025-01-01 00:15:00", "2025-01-01 00:45:00"],
        utc=True,
    )
    sub = pd.DataFrame(
        {
            "timestamp": ts,
            "meter_number": ["42"] * 3,
            "shard": [1] * 3,
            "dt": ["2025-01-01"] * 3,
            "voltage_l1": [230.0, 231.0, 229.0],
        }
    )
    extractor.expected_timestamps = pd.date_range(
        start="2025-01-01 00:00:00+00:00", end="2025-01-01 01:00:00+00:00", freq="15min"
    )

    frame = extractor._shape_single_meter_frame(
        sub, "42", ts_col="timestamp", meter_col="meter_number", add_current=False, add_unbalance=False
    )

    # suffixed measurement column, partition columns dropped
    assert list(frame.columns) == ["voltage_l1_42"]
    # full expected range, missing slots are NaN
    assert len(frame) == 5
    assert frame["voltage_l1_42"].iloc[0] == 230.0
    assert pd.isna(frame["voltage_l1_42"].loc["2025-01-01 00:30:00+00:00"])  # gap
    assert pd.isna(frame["voltage_l1_42"].iloc[-1])  # beyond last sample


def test_build_cabinet_sm_mapping_classifies_available_and_missing(extractor, monkeypatch):
    sm_cab = pd.DataFrame(
        {
            "meter_number": [100025.0, 100026.0, None, 100027.0],
            "delivery_point_id": [1.0, 2.0, 3.0, 4.0],
            "cabinet": ["Cabinet.7", "Cabinet.7", "Cabinet.8", None],
            "lv_feeder": [None, None, None, "LvFeeder.9"],
            "has_heat_pump": [True, False, None, None],
            "has_solar_panel": [False, False, None, None],
            "capacity_solar_panel": [None, None, None, None],
            "service_fuse_size": [None, None, None, None],
        }
    )
    monkeypatch.setattr(extractor, "_export_sm_cabinet_pdf", lambda level: sm_cab)
    monkeypatch.setattr(extractor, "_ids_with_data", lambda: {"100025"})

    mapping = extractor._build_cabinet_sm_mapping("raw")

    # Only rows with both cabinet and meter_number survive; IDs are strings
    assert mapping == {
        "7": {
            "METER_NUMBER": ["100025", "100026"],
            "AVAILABLE_METERS": ["100025"],
            "MISSING_METERS": ["100026"],
        }
    }


def test_get_sm_ids_for_cabinet_uses_mapping_and_validates(extractor, monkeypatch):
    mapping = {"7": {"AVAILABLE_METERS": ["1", "2"], "METER_NUMBER": [], "MISSING_METERS": []}}
    monkeypatch.setattr(extractor, "_get_cabinet_sm_mapping", lambda level, overwrite=False: mapping)

    assert extractor.get_sm_ids_for_cabinet(7, None, "raw") == ["1", "2"]

    with pytest.raises(ValueError, match="does not exist"):
        extractor.get_sm_ids_for_cabinet(99, None, "raw")

    # explicit list bypasses the mapping entirely
    assert extractor.get_sm_ids_for_cabinet(7, ["5"], "raw") == ["5"]
