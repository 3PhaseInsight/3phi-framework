# __init__.py
from __future__ import annotations

# Optionally expose the package version dynamically
try:
    from importlib.metadata import PackageNotFoundError, version  # py3.8+
except ImportError:  # pragma: no cover
    from importlib_metadata import PackageNotFoundError, version  # type: ignore

try:
    __version__ = version("threephi-framework")
except PackageNotFoundError:
    __version__ = "0.0.0"

# Re-export light, library-facing APIs only
from .controllers.topology import TopologyController
from .data_apps.base import BaseDataApp
from .data_apps.sm_classifier import SMClassifier
from .data_apps.timeseries_ingestor import TimeseriesIngestor
from .data_apps.topology_cleaner import TopologyCleaner
from .data_apps.topology_ingestor import TopologyIngestor
from .data_apps.topology_tester import TopologyTester
from .data_extractor.data_extractor import DataExtractor
from .object_storage.azure_blob_connector import AzureBlobConnector
from .object_storage.s3_connector import S3Connector

__all__ = [
    "S3Connector",
    "AzureBlobConnector",
    "BaseDataApp",
    "TimeseriesIngestor",
    "TopologyCleaner",
    "TopologyIngestor",
    "TopologyTester",
    "SMClassifier",
    "DataExtractor",
    "TopologyController",
    "__version__",
]
