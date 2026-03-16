import os


class LvSchemaMixin:
    __table_args__ = {"schema": os.getenv("LV_SCHEMA", "lv")}
