import logging
from dataclasses import dataclass

import threephi_framework.db.db as threephi_db
from threephi_framework import TopologyController
from threephi_framework.data_apps.base import BaseDataApp
from threephi_framework.data_apps.base_config import BaseConfig


@dataclass(frozen=True)
class TopologyTesterConfig(BaseConfig):
    dask: dict
    substation_id: str
    delivery_point_id: int
    cabinet_id: int
    feeder_id: int


class TopologyTester(BaseDataApp):
    def __init__(self, config):
        super().__init__(config)

        self.config = TopologyTesterConfig(**config)

        self.substation_id = config["substation_id"]
        self.delivery_point_id = config["delivery_point_id"]
        self.cabinet_id = config["cabinet_id"]
        self.feeder_id = config["feeder_id"]
        self.topology_controller = TopologyController(threephi_db.new_session)

    def run(self):
        # Test get_meters_for_substation
        meters = self.topology_controller.get_meters_for_substation(self.substation_id)
        logging.info(f"Meters for substation {self.substation_id}: {meters}")

        # Test get_meters
        meters = self.topology_controller.get_meters(True, True)
        logging.info(f"get_meters(has_heat_pump=True, has_solar_panel=True): {meters}")

        # Test get_meters_for_node (delivery_point)
        meters = self.topology_controller.get_meters_for_node(self.delivery_point_id, "delivery_point")
        logging.info(f'get_meters_for_node(node_id={self.delivery_point_id}, "delivery_point"): {meters}')

        # Test get_meters_for_node (cabinet)
        meters = self.topology_controller.get_meters_for_node(self.cabinet_id, "cabinet")
        logging.info(f'get_meters_for_node(node_id={self.cabinet_id}, "cabinet"): {meters}')

        # Test get_meters_for_node (feeder)
        meters = self.topology_controller.get_meters_for_node(self.feeder_id, "lv_feeder")
        logging.info(f'get_meters_for_node(node_id={self.feeder_id}, "lv_feeder"): {meters}')

if __name__ == "__main__":
    config = {
        "dask": {
            "local": True,
            "n_workers": 6,
        },
        "substation_id": 147237,
        "delivery_point_id": 999994,
        "cabinet_id": 996044,
        "feeder_id": 910228,
    }
    with TopologyTester(config) as app:
        app.run()
