"""Backend-agnostic connector creation.

Data apps receive an optional connector instance; when none is injected, this
factory decides which backend to construct. Precedence:

1. explicit ``backend`` argument (e.g. from a data app config)
2. the ``OBJECT_STORAGE_BACKEND`` environment variable
3. default: ``"s3"``

Connector credentials always come from the environment (see the connector
classes / README for the variables each backend requires).
"""

from __future__ import annotations

import os

from threephi_framework.object_storage.base_connector import BaseConnector

BACKEND_ENV_VAR = "OBJECT_STORAGE_BACKEND"
_S3_ALIASES = {"s3", "minio"}
_AZURE_ALIASES = {"azure", "az", "azure_blob"}


def create_connector(data_dir_path: str, backend: str | None = None) -> BaseConnector:
    """Create an object-storage connector rooted at ``data_dir_path``.

    Args:
        data_dir_path: Dataset root within the bucket/container,
            e.g. ``phase_measurements/raw``.
        backend: Backend name ("s3"/"minio" or "azure"/"az"/"azure_blob").
            Falls back to the ``OBJECT_STORAGE_BACKEND`` environment variable,
            then to "s3".

    Returns:
        BaseConnector: A connector of the selected backend.

    Raises:
        ValueError: If the backend name is not recognized.
    """
    name = (backend or os.getenv(BACKEND_ENV_VAR) or "s3").strip().lower()

    if name in _S3_ALIASES:
        from threephi_framework.object_storage.s3_connector import S3Connector

        return S3Connector(data_dir_path=data_dir_path)

    if name in _AZURE_ALIASES:
        from threephi_framework.object_storage.azure_blob_connector import AzureBlobConnector

        return AzureBlobConnector(data_dir_path=data_dir_path)

    raise ValueError(f"Unknown object storage backend '{name}'. Supported: {sorted(_S3_ALIASES | _AZURE_ALIASES)}")
