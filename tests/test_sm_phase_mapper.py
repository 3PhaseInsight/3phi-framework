import os
from unittest.mock import MagicMock

import pandas as pd
import pytest

os.environ.setdefault("S3_ENDPOINT_URL", "http://localhost:19000")
os.environ.setdefault("S3_ACCESS_KEY", "test")
os.environ.setdefault("S3_SECRET_KEY", "test")

from threephi_framework.controllers import meta as meta_module  # noqa: E402
from threephi_framework.controllers.meta import MetaController  # noqa: E402
from threephi_framework.dtu.sm_phase_mapper import (  # noqa: E402
    _check_majority_feeder,
    _check_majority_phase,
    _get_feeder,
    _get_phase,
    _initialize_trafo_result_df,
    _unconnected_phase_labels,
)


class _FakeSession:
    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False

    def commit(self):
        pass


def _result_df(rows):
    cols = [
        "SM ID",
        "SM Phase",
        "Feeder Phase",
        "Trafo Phase",
        "True Feeder ID",
        "True Trafo ID",
        "Likely Cabinet ID",
    ]
    return pd.DataFrame(rows, columns=cols)


class TestUpdatePhaseMapping:
    @pytest.fixture
    def captured_rows(self, monkeypatch):
        captured = []
        resource = MagicMock()
        resource.return_value.upsert_many.side_effect = lambda rows: captured.extend(rows)
        monkeypatch.setattr(meta_module, "MetaPhaseMappingResource", resource)
        return captured

    @pytest.fixture
    def controller(self):
        return MetaController(_FakeSession)

    def test_missing_columns_raise(self, controller):
        with pytest.raises(ValueError, match="Missing columns"):
            controller.update_phase_mapping(1, pd.DataFrame({"SM ID": [1]}))

    def test_cleans_sentinels_and_casts(self, controller, captured_rows):
        df = _result_df([["100025", "l1", "L2", "nan", "910228", "other", None]])
        controller.update_phase_mapping(494439, df)

        assert captured_rows == [
            {
                "meter_id": 100025,
                "sm_phase": "L1",
                "feeder_phase": "L2",
                "trafo_phase": None,
                "true_feeder_id": 910228,
                "true_trafo_id": None,  # "other" → None
                "likely_cabinet_id": None,
            }
        ]

    def test_deduplicates_on_meter_and_phase(self, controller, captured_rows):
        df = _result_df(
            [
                ["1", "l1", "L1", "L1", "9", "5", None],
                ["1", "l1", "L2", "L2", "9", "5", None],  # same key — last wins
                ["1", "l2", "L2", "L2", "9", "5", None],
            ]
        )
        controller.update_phase_mapping(5, df)

        assert len(captured_rows) == 2
        l1_row = next(r for r in captured_rows if r["sm_phase"] == "L1")
        assert l1_row["feeder_phase"] == "L2"

    def test_invalid_phase_raises(self, controller, captured_rows):
        df = _result_df([["1", "l4", None, None, None, None, None]])
        with pytest.raises(ValueError, match="Invalid phase value"):
            controller.update_phase_mapping(5, df)

    def test_rows_without_meter_or_phase_are_skipped(self, controller, captured_rows):
        df = _result_df(
            [
                ["nan", "l1", None, None, None, None, None],
                ["1", "nan", None, None, None, None, None],
            ]
        )
        controller.update_phase_mapping(5, df)
        assert captured_rows == []


class TestUnconnectedPhaseLabels:
    def test_labels_match_column_naming(self):
        info = {
            "Connectivity": {"Connected Phases": ["L1", "L3"]},
            "Topology": {"Cabinet ID": 996044, "Feeder ID": 910228},
        }
        assert _unconnected_phase_labels(info, 100025) == ["voltage_l2_100025_996044_910228"]

    def test_missing_cabinet_uses_nan_like_columns(self):
        info = {
            "Connectivity": {"Connected Phases": ["L1", "L2"]},
            "Topology": {"Cabinet ID": None, "Feeder ID": 910228},
        }
        assert _unconnected_phase_labels(info, 7) == ["voltage_l3_7_nan_910228"]

    def test_no_connectivity_yields_nothing(self):
        assert _unconnected_phase_labels({"Connectivity": None, "Topology": {}}, 7) == []
        assert _unconnected_phase_labels({"Connectivity": {"Connected Phases": None}, "Topology": {}}, 7) == []


class TestClusterHelpers:
    def test_column_name_parsing(self):
        col = "voltage_l2_100025_996044_910228"
        assert _get_phase(col) == "l2"
        assert _get_feeder(col) == "910228"

    def test_majority_phase_and_feeder(self):
        cols = [
            "voltage_l1_1_c_910228",
            "voltage_l1_2_c_910228",
            "voltage_l2_3_c_999999",
        ]
        assert _check_majority_phase(cols, majority_phase_threshold=0.5) == "l1"
        assert _check_majority_phase(cols, majority_phase_threshold=0.8) is None
        assert _check_majority_feeder(cols, majority_feeder_threshold=0.5) == "910228"

    def test_initialize_trafo_result_df(self):
        cols = ["voltage_l1_100025_996044_910228", "voltage_l2_100025_996044_910228"]
        df = _initialize_trafo_result_df(cols, trafo_id="494439")

        assert len(df) == 2
        first = df.iloc[0]
        assert first["Trafo ID"] == "494439"
        assert first["SM ID"] == "100025"
        assert first["SM Phase"] == "l1"
        # before evaluation the labels are assumed correct
        assert first["Feeder Phase"] == "l1"
        assert first["True Feeder ID"] == "910228"
        assert first["Trafo Phase"] == "nan"
