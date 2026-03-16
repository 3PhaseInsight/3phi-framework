from sqlalchemy.dialects.postgresql import ENUM

NodeTypeEnum = ENUM(
    "LvFeeder",
    "Cabinet",
    "DeliveryPoint",
    name="node_type",
    schema="lv",
    create_type=False,
)
