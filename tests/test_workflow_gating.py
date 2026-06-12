import json
import os
from unittest.mock import MagicMock

import pytest

os.environ.setdefault("S3_ENDPOINT_URL", "http://localhost:19000")
os.environ.setdefault("S3_ACCESS_KEY", "test")
os.environ.setdefault("S3_SECRET_KEY", "test")

from threephi_framework.data_apps.base import BaseDataApp  # noqa: E402
from threephi_framework.data_apps.timeseries_ingestor import TimeseriesIngestor  # noqa: E402
from threephi_framework.data_apps.topology_cleaner import TopologyCleaner  # noqa: E402
from threephi_framework.data_apps.topology_ingestor import TopologyIngestor  # noqa: E402


class _ScopedApp(BaseDataApp):
    WORKFLOW = "scoped_step"
    IDENTITY_KEYS = ("source_path", "threshold")

    def run(self):
        pass


class _UnscopedApp(BaseDataApp):
    WORKFLOW = "unscoped_step"

    def run(self):
        pass


def _app(cls, config):
    app = cls(config)
    app.meta_controller = MagicMock()  # overrides the cached_property
    return app


class TestWorkflowName:
    def test_identity_uses_only_declared_keys(self):
        app = _app(_ScopedApp, {"source_path": "/data", "threshold": 3, "dask": {"n_workers": 8}})
        assert app.workflow_identity() == {"source_path": "/data", "threshold": 3}

    def test_name_is_stable_and_ignores_irrelevant_config(self):
        a = _app(_ScopedApp, {"source_path": "/data", "threshold": 3, "save_plots": True})
        b = _app(_ScopedApp, {"threshold": 3, "source_path": "/data", "save_plots": False})
        assert a.workflow_name() == b.workflow_name()
        assert a.workflow_name().startswith("scoped_step:")

    def test_name_changes_with_relevant_config(self):
        a = _app(_ScopedApp, {"source_path": "/data", "threshold": 3})
        b = _app(_ScopedApp, {"source_path": "/data", "threshold": 4})
        assert a.workflow_name() != b.workflow_name()

    def test_unscoped_app_keeps_bare_name(self):
        assert _app(_UnscopedApp, {}).workflow_name() == "unscoped_step"

    def test_undeclared_workflow_raises(self):
        class NoWorkflow(BaseDataApp):
            def run(self):
                pass

        with pytest.raises(NotImplementedError):
            _app(NoWorkflow, {}).workflow_name()


class TestWorkflowCompletion:
    def test_completed_under_scoped_name(self):
        app = _app(_ScopedApp, {"source_path": "/data", "threshold": 3})
        app.meta_controller.is_workflow_completed.side_effect = lambda name: name == app.workflow_name()
        assert app.workflow_completed() is True

    def test_legacy_bare_name_counts_as_completed(self):
        app = _app(_ScopedApp, {"source_path": "/data", "threshold": 3})
        app.meta_controller.is_workflow_completed.side_effect = lambda name: name == "scoped_step"
        assert app.workflow_completed() is True

    def test_not_completed(self):
        app = _app(_ScopedApp, {"source_path": "/data", "threshold": 3})
        app.meta_controller.is_workflow_completed.return_value = False
        assert app.workflow_completed() is False

    def test_mark_stores_identity_as_description(self):
        app = _app(_ScopedApp, {"source_path": "/data", "threshold": 3})
        app.mark_workflow_completed()

        name = app.workflow_name()
        app.meta_controller.start_workflow.assert_called_once_with(
            name, description=json.dumps({"source_path": "/data", "threshold": 3}, sort_keys=True, default=str)
        )
        app.meta_controller.complete_workflow.assert_called_once_with(name)


class TestAppDeclarations:
    def test_apps_declare_workflows(self):
        assert TimeseriesIngestor.WORKFLOW == "timeseries_csv_to_parquet_partitions"
        assert TimeseriesIngestor.IDENTITY_KEYS == (
            "csv_source_path",
            "csv_file_pattern",
            "parquet_destination_path",
        )
        assert TopologyIngestor.WORKFLOW == "topology_ingestion"
        assert TopologyIngestor.IDENTITY_KEYS == ("topology_source_path", "sm_cab_source_path")
        # the cleaner always works on the current topology version → unscoped
        assert TopologyCleaner.WORKFLOW == "topology_cleaning"
        assert TopologyCleaner.IDENTITY_KEYS == ()

    def test_skip_path_does_not_mark_completion(self):
        config = {
            "override": False,
            "topology_source_path": "/data/topo.csv",
            "sm_cab_source_path": "/data/sm.csv",
        }
        app = _app(TopologyIngestor, config)
        app.meta_controller.is_workflow_completed.return_value = True

        app.run()

        # skipping must not stamp completion (under any name)
        app.meta_controller.complete_workflow.assert_not_called()
        app.meta_controller.start_workflow.assert_not_called()
