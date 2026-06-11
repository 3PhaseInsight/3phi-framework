import os

import pytest

# Connector constructors read env vars but do not connect
os.environ.setdefault("S3_ENDPOINT_URL", "http://localhost:19000")
os.environ.setdefault("S3_ACCESS_KEY", "test")
os.environ.setdefault("S3_SECRET_KEY", "test")
os.environ.setdefault("AZURE_STORAGE_ACCOUNT_NAME", "testaccount")
os.environ.setdefault("AZURE_STORAGE_CONTAINER_NAME", "3phi")
os.environ.setdefault("AZURE_STORAGE_ACCOUNT_KEY", "dGVzdA==")

from threephi_framework.object_storage.azure_blob_connector import AzureBlobConnector  # noqa: E402
from threephi_framework.object_storage.factory import BACKEND_ENV_VAR, create_connector  # noqa: E402
from threephi_framework.object_storage.s3_connector import S3Connector  # noqa: E402


class TestFactory:
    def test_defaults_to_s3(self, monkeypatch):
        monkeypatch.delenv(BACKEND_ENV_VAR, raising=False)
        connector = create_connector("phase_measurements/raw")
        assert isinstance(connector, S3Connector)
        assert connector.dataset_root_path == "s3://3phi/phase_measurements/raw"

    def test_explicit_backend_argument(self):
        connector = create_connector("phase_measurements/raw", backend="azure")
        assert isinstance(connector, AzureBlobConnector)
        assert connector.dataset_root_path == "az://3phi/phase_measurements/raw"

    def test_env_var_fallback(self, monkeypatch):
        monkeypatch.setenv(BACKEND_ENV_VAR, "azure")
        connector = create_connector("phase_measurements/raw")
        assert isinstance(connector, AzureBlobConnector)

    def test_argument_beats_env_var(self, monkeypatch):
        monkeypatch.setenv(BACKEND_ENV_VAR, "azure")
        connector = create_connector("phase_measurements/raw", backend="s3")
        assert isinstance(connector, S3Connector)

    def test_unknown_backend_raises(self):
        with pytest.raises(ValueError, match="Unknown object storage backend"):
            create_connector("phase_measurements/raw", backend="gcs")


class TestConnectorContract:
    def test_storage_base_is_uniform(self):
        assert S3Connector("x").storage_base == "s3://3phi"
        assert AzureBlobConnector("x").storage_base == "az://3phi"

    def test_with_data_dir_returns_same_type_at_new_root(self):
        s3 = S3Connector("phase_measurements/raw")
        sibling = s3.with_data_dir("phase_measurements")
        assert type(sibling) is S3Connector
        assert sibling.dataset_root_path == "s3://3phi/phase_measurements"

        az = AzureBlobConnector("phase_measurements/raw")
        assert az.with_data_dir("phase_measurements").dataset_root_path == "az://3phi/phase_measurements"

    def test_save_plot_is_shared_base_implementation(self):
        assert "save_plot" not in S3Connector.__dict__
        assert "save_plot" not in AzureBlobConnector.__dict__


class TestInjection:
    def test_data_extractor_uses_injected_connector(self):
        from threephi_framework.data_extractor.data_extractor import DataExtractor

        injected = AzureBlobConnector("phase_measurements/raw")
        de = DataExtractor(connector=injected)

        assert de.connector is injected
        assert de.s3_connector is injected  # legacy alias
        # derived paths follow the injected backend
        assert de.s3_base == "az://3phi"
        assert de.time_series_controller.connector.dataset_root_path == "az://3phi/phase_measurements"

    def test_data_extractor_backend_argument(self):
        from threephi_framework.data_extractor.data_extractor import DataExtractor

        de = DataExtractor(backend="azure")
        assert isinstance(de.connector, AzureBlobConnector)

    def test_base_data_app_uses_injected_connector(self):
        from threephi_framework.data_apps.base import BaseDataApp

        injected = AzureBlobConnector("phase_measurements/raw")
        app = BaseDataApp({"data_dir_path": "phase_measurements/raw"}, connector=injected)

        assert app.connector is injected
        assert app.data_extractor.connector is injected
        assert app.time_series_controller.connector.dataset_root_path == "az://3phi/phase_measurements"

    def test_base_data_app_backend_from_config(self):
        from threephi_framework.data_apps.base import BaseDataApp

        app = BaseDataApp({"object_storage_backend": "azure"})
        assert isinstance(app.connector, AzureBlobConnector)

    def test_base_data_app_defaults_to_s3(self, monkeypatch):
        monkeypatch.delenv(BACKEND_ENV_VAR, raising=False)
        from threephi_framework.data_apps.base import BaseDataApp

        app = BaseDataApp({})
        assert isinstance(app.connector, S3Connector)
        assert app.connector.dataset_root_path == "s3://3phi/phase_measurements/raw"
