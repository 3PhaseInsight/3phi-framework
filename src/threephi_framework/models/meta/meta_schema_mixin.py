import os


class MetaSchemaMixin:
    __table_args__ = {"schema": os.getenv("META_SCHEMA", "meta")}
