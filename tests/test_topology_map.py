from types import SimpleNamespace

from threephi_framework.controllers.topology import _build_topology_map


def _chain_row(*, secondary_substation_id, transformer_id, feeder_id, cabinet_id, delivery_point_id=None):
    """A stand-in for the SQLAlchemy Row returned by get_topology_chain_for_meter."""
    return SimpleNamespace(
        secondary_substation_id=secondary_substation_id,
        transformer_id=transformer_id,
        feeder_id=feeder_id,
        cabinet_id=cabinet_id,
        delivery_point_id=delivery_point_id,
    )


def test_builds_one_labeled_entry_per_meter():
    chains = {
        319672: [
            _chain_row(secondary_substation_id=706039, transformer_id=125763, feeder_id=364204, cabinet_id=86493),
        ],
    }

    result = _build_topology_map(
        meter_ids=[319672],
        chains_by_meter=chains,
        zip_by_substation={706039: 2600},
        meters_with_data={319672},
    )

    assert result == {
        319672: {
            "Zip Code": 2600,
            "Secondary Substation ID": 706039,
            "Transformer ID": 125763,
            "Feeder ID": 364204,
            "Cabinet ID": 86493,
            "Has data": True,
        }
    }


def test_uses_first_chain_row_when_meter_has_multiple_paths():
    chains = {
        606018: [
            _chain_row(secondary_substation_id=706039, transformer_id=125763, feeder_id=364204, cabinet_id=459597),
            _chain_row(secondary_substation_id=706039, transformer_id=125763, feeder_id=999999, cabinet_id=86493),
        ],
    }

    result = _build_topology_map(
        meter_ids=[606018],
        chains_by_meter=chains,
        zip_by_substation={706039: 2600},
        meters_with_data=set(),
    )

    assert result[606018]["Feeder ID"] == 364204
    assert result[606018]["Cabinet ID"] == 459597
    assert result[606018]["Has data"] is False


def test_meter_without_chain_gets_none_topology_but_keeps_has_data():
    result = _build_topology_map(
        meter_ids=[111],
        chains_by_meter={111: []},
        zip_by_substation={},
        meters_with_data={111},
    )

    assert result == {
        111: {
            "Zip Code": None,
            "Secondary Substation ID": None,
            "Transformer ID": None,
            "Feeder ID": None,
            "Cabinet ID": None,
            "Has data": True,
        }
    }


def test_zip_is_none_when_substation_absent_from_lookup():
    chains = {
        42: [_chain_row(secondary_substation_id=706039, transformer_id=1, feeder_id=2, cabinet_id=None)],
    }

    result = _build_topology_map(
        meter_ids=[42],
        chains_by_meter=chains,
        zip_by_substation={},  # substation 706039 missing from the lookup
        meters_with_data=set(),
    )

    assert result[42]["Zip Code"] is None
    assert result[42]["Cabinet ID"] is None


def test_preserves_meter_order_and_label_order():
    row = _chain_row(secondary_substation_id=1, transformer_id=1, feeder_id=1, cabinet_id=1)
    chains = {2: [row], 1: [row]}

    result = _build_topology_map(
        meter_ids=[2, 1],
        chains_by_meter=chains,
        zip_by_substation={1: 9999},
        meters_with_data={1, 2},
    )

    assert list(result.keys()) == [2, 1]
    assert list(result[2].keys()) == [
        "Zip Code",
        "Secondary Substation ID",
        "Transformer ID",
        "Feeder ID",
        "Cabinet ID",
        "Has data",
    ]
