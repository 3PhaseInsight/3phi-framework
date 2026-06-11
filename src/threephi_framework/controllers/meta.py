import logging
from collections.abc import Callable
from typing import Any
from uuid import UUID

import pandas as pd
from sqlalchemy.orm import Session

from threephi_framework.models.meta.run_result import RunResultModel
from threephi_framework.resources.meta.meter import MetaMeterResource
from threephi_framework.resources.meta.run_result import RunResultResource
from threephi_framework.resources.meta.workflow_state import WorkflowStateResource
from threephi_framework.resources.topology.assets.meter import MeterResource


class MetaController:
    """
    Controller responsible for accessing and aggregating metadata associated
    with smart meters.

    The controller provides:
    - Methods to update meter metadata (data quality, statistics, connectivity)
    - Methods to compute and return a smart meter characterization summary
    - Methods to insert and query run results
    - Airflow-aware convenience methods for automatically resolving dag_id/run_id

    Each public method creates its own short-lived session via ``session_factory``
    and owns the transaction boundary (commits on write methods, no commit on
    read methods). This matches :class:`IngestionController` and avoids holding
    long-lived sessions across DAG runs.

    Args:
        session_factory (Callable[[], Session]):
            A factory returning new SQLAlchemy sessions.
    """

    def __init__(self, session_factory: Callable[[], Session]):
        """
        Initialize the MetaController.

        Args:
            session_factory (Callable[[], Session]):
                Factory that returns new SQLAlchemy sessions.
        """
        self._sf = session_factory

    def update_sm_characterization(self, meter_id: int, data: dict) -> None:
        """
        Update metadata fields of an existing smart meter using classifier output.

        This method updates:
        - Data quality information
        - Statistical descriptions
        - Connectivity metadata

        Args:
            meter_id (int):
                The meter ID to update.
            data (dict):
                Smart Meter Classifier output containing "Data Quality",
                "Data Statistics", and "Connectivity" keys.

        Returns:
            None
        """
        logging.info(f"Update meter: {meter_id}")
        meter_data = {
            "data_quality": data["Data Quality"],
            "data_statistics": data["Data Statistics"],
            "connectivity": data["Connectivity"],
        }
        logging.info(f"With Data: {meter_data}")
        with self._sf() as s:
            MetaMeterResource(s).update(meter_id, meter_data)
            s.commit()

    def get_sm_characterization(self, meter_id: int) -> dict[str, Any]:
        """
        Return a characterization summary for the specified smart meter.

        The summary includes:
        - Topology information (substation, transformer, feeder, cabinet)
        - Dataset availability metrics
        - Data quality and statistical information
        - Connectivity metadata

        If the meter does not exist, returned metadata fields are ``None`` and
        availability is marked as ``False``.

        Args:
            meter_id (int):
                The meter ID.

        Returns:
            dict[str, Any]:
                A dictionary containing topology, availability, quality,
                statistics, and connectivity details.
        """
        with self._sf() as s:
            meta_meter = MetaMeterResource(s)
            topology_meter = MeterResource(s)

            topology_chain_result = topology_meter.get_topology_chain_for_meter(meter_id=meter_id)
            if not topology_chain_result:
                topology_chain = {
                    "Secondary Substation ID": None,
                    "Transformer ID": None,
                    "Feeder ID": None,
                    "Cabinet ID": None,
                }
            else:
                first = topology_chain_result[0]
                rest = topology_chain_result[1:]

                topology_chain = {
                    "Secondary Substation ID": first.secondary_substation_id,
                    "Transformer ID": first.transformer_id,
                    "Delivery Point ID": first.delivery_point_id,
                    "Feeder ID": first.feeder_id,
                    "Cabinet ID": first.cabinet_id,
                    "Additional Feeder IDs": [r.feeder_id for r in rest],
                    "Additional Cabinet IDs": [r.cabinet_id for r in rest],
                }

            meta = meta_meter.get(meter_id)

            if meta is None:
                logging.warning("Cannot fetch SM Characterization for non-existent meter!")

                dataset_availability = {
                    "Available": False,
                    "Contains Data": None,
                    "Relative Length": None,
                    "Absolute Length": None,
                }

                return {
                    "Topology": topology_chain,
                    "Dataset Availability": dataset_availability,
                    "Data Quality": None,
                    "Data Statistics": None,
                    "Connectivity": None,
                }

            max_total_rows = meta_meter.get_max_total_rows()
            relative_length = (
                None if max_total_rows is None or max_total_rows == 0 else meta.total_rows / max_total_rows
            )

            dataset_availability = {
                "Available": True,
                "Contains Data": meta.total_rows > 0,
                "Relative Length": relative_length,
                "Absolute Length": meta.total_rows,
            }

            return {
                "Topology": topology_chain,
                "Dataset Availability": dataset_availability,
                "Data Quality": meta.data_quality,
                "Data Statistics": meta.data_statistics,
                "Connectivity": meta.connectivity,
            }

    def insert_run_result(
        self,
        *,
        dag_id: str | None,
        run_id: str | None,
        meter_id: int | None,
        phase: str,
        label_type: str,
        label_value: str,
        confidence: float,
        source: str,
        result: dict[str, Any] | None = None,
        id: UUID | None = None,
        topology_version: int | None,
        node_id: int | None,
        edge_id: int | None,
        cable_id: int | None,
    ):
        """
        Insert a single run result row with explicitly provided dag_id and run_id.

        This method commits immediately.

        Args:
            dag_id (str):
                DAG identifier (part 1 of the unique DAG run identifier).
            run_id (str):
                Run identifier (part 2 of the unique DAG run identifier).
            meter_id (int | None):
                Meter ID this result pertains to.
            phase (str):
                Phase label (must be a valid ``public.result_phase`` enum value).
            label_type (str):
                Label type identifier (e.g., classification type).
            label_value (str):
                Label value (e.g., predicted label).
            confidence (float):
                Confidence score for the result.
            source (str):
                Origin of the result (e.g., algorithm name, data app, etc.).
            result (dict[str, Any] | None):
                Optional JSON blob with additional data.
            id (UUID | None):
                Optional UUID primary key. If omitted, the resource layer will
                generate one.
            topology_version (int | None):
                Topology version this result pertains to.
            node_id (int | None):
                Node ID this result pertains to.
            edge_id (int | None):
                Edge ID this result pertains to.
            cable_id (int | None):
                Cable ID this result pertains to.

        Returns:
            RunResultModel:
                The newly inserted run result ORM object.
        """
        payload: dict[str, Any] = {
            "dag_id": dag_id,
            "run_id": run_id,
            "meter_id": meter_id,
            "phase": phase,
            "label_type": label_type,
            "label_value": label_value,
            "confidence": confidence,
            "source": source,
            "result": result,
            "topology_version": topology_version,
            "node_id": node_id,
            "edge_id": edge_id,
            "cable_id": cable_id,
        }

        if id is not None:
            payload["id"] = id

        with self._sf() as s:
            obj = RunResultResource(s).insert(payload)
            s.commit()
            return obj

    def get_run_result(self, result_id: UUID):
        """
        Fetch a run result by primary key.

        Args:
            result_id (UUID):
                Primary key of the run result.

        Returns:
            RunResultModel | None:
                The run result if found, otherwise ``None``.
        """
        with self._sf() as s:
            return RunResultResource(s).get(result_id)

    def query_run_results(
        self,
        *,
        dag_id: str | None = None,
        run_id: str | None = None,
        meter_id: int | None = None,
        phase: str | None = None,
        label_type: str | None = None,
        label_value: str | None = None,
        source: str | None = None,
        topology_version: int | None = None,
        node_id: int | None = None,
        edge_id: int | None = None,
        cable_id: int | None = None,
        min_confidence: float | None = None,
        max_confidence: float | None = None,
        limit: int | None = None,
        offset: int | None = None,
        order_by: str | list[str] | None = None,
        descending: bool = False,
    ):
        """
        Query run results with flexible filtering and ordering.

        This controller method provides a simple interface for users by allowing
        string-based ordering fields.

        Args:
            dag_id (str | None):
                Filter by DAG ID.
            run_id (str | None):
                Filter by run ID.
            meter_id (int | None):
                Filter by meter ID.
            phase (str | None):
                Filter by phase enum value.
            label_type (str | None):
                Filter by label type.
            label_value (str | None):
                Filter by label value.
            source (str | None):
                Filter by result source.
            topology_version (int | None):
                Filter by topology version.
            node_id (int | None):
                Filter by node ID.
            edge_id (int | None):
                Filter by edge ID.
            cable_id (int | None):
                Filter by cable ID.
            min_confidence (float | None):
                Filter for confidence >= this value.
            max_confidence (float | None):
                Filter for confidence <= this value.
            limit (int | None):
                Limit number of returned rows.
            offset (int | None):
                Offset for pagination.
            order_by (str | list[str] | None):
                Order results by one or more fields. Supported values are:
                ``id``, ``dag_id``, ``run_id``, ``meter_id``, ``phase``,
                ``label_type``, ``label_value``, ``confidence``, ``source``.
            descending (bool):
                If ``True``, order descending.

        Returns:
            list[RunResultModel]:
                List of matching run results.
        """

        order_cols = None
        if order_by is not None:
            fields = [order_by] if isinstance(order_by, str) else list(order_by)

            allowed = {
                "id": RunResultModel.id,
                "dag_id": RunResultModel.dag_id,
                "run_id": RunResultModel.run_id,
                "meter_id": RunResultModel.meter_id,
                "phase": RunResultModel.phase,
                "label_type": RunResultModel.label_type,
                "label_value": RunResultModel.label_value,
                "confidence": RunResultModel.confidence,
                "source": RunResultModel.source,
            }

            invalid = [f for f in fields if f not in allowed]
            if invalid:
                raise ValueError(f"Unsupported order_by field(s): {invalid}")

            order_cols = [allowed[f] for f in fields]

        with self._sf() as s:
            return RunResultResource(s).query(
                dag_id=dag_id,
                run_id=run_id,
                meter_id=meter_id,
                phase=phase,
                label_type=label_type,
                label_value=label_value,
                source=source,
                topology_version=topology_version,
                node_id=node_id,
                edge_id=edge_id,
                cable_id=cable_id,
                min_confidence=min_confidence,
                max_confidence=max_confidence,
                limit=limit,
                offset=offset,
                order_by=order_cols,
                descending=descending,
            )

    @staticmethod
    def _get_airflow_dag_and_run_id() -> tuple[str, str]:
        """
        Resolve ``dag_id`` and ``run_id`` from the current Airflow task context.

        This method must be called during Airflow task execution (e.g., within a
        PythonOperator or TaskFlow task). It will fail when called outside of an
        Airflow runtime context.

        Returns:
            tuple[str, str]:
                ``(dag_id, run_id)`` resolved from the Airflow context.

        Raises:
            RuntimeError:
                If Airflow is not available or if dag_id/run_id cannot be
                resolved from the current context.
        """
        try:
            from airflow.providers.standard.operators.python import get_current_context
        except Exception as e:  # pragma: no cover
            raise RuntimeError("Airflow is not available; cannot resolve DAG context.") from e

        ctx = get_current_context()

        dag = ctx.get("dag")
        dag_run = ctx.get("dag_run")

        dag_id = dag.dag_id if dag is not None else None
        run_id = getattr(dag_run, "run_id", None) if dag_run is not None else None

        if not dag_id or not run_id:
            raise RuntimeError(
                f"Failed to resolve dag_id/run_id from Airflow context (dag_id={dag_id}, run_id={run_id})"
            )

        return dag_id, run_id

    def insert_run_result_current_run(
        self,
        *,
        meter_id: int,
        phase: str,
        label_type: str,
        label_value: str,
        confidence: float,
        source: str,
        result: dict[str, Any] | None = None,
        id: UUID | None = None,
        topology_version: int | None,
        node_id: int | None,
        edge_id: int | None,
        cable_id: int | None,
    ):
        """
        Insert a run result using dag_id and run_id from the current Airflow run.

        This is a convenience wrapper around :meth:`insert_run_result` that
        automatically resolves ``dag_id`` and ``run_id`` from Airflow's runtime
        task context.

        Args:
            meter_id (int):
                Meter ID this result pertains to.
            phase (str):
                Phase label (must be a valid ``public.result_phase`` enum value).
            label_type (str):
                Label type identifier (e.g., classification type).
            label_value (str):
                Label value (e.g., predicted label).
            confidence (float):
                Confidence score for the result.
            source (str):
                Origin of the result (e.g., algorithm name, data app, etc.).
            result (dict[str, Any] | None):
                Optional JSON blob with additional data.
            id (UUID | None):
                Optional UUID primary key. If omitted, the resource layer will
                generate one.

        Returns:
            RunResultModel:
                The newly inserted run result ORM object.

        Raises:
            RuntimeError:
                If called outside of an Airflow runtime context.
        """
        dag_id, run_id = self._get_airflow_dag_and_run_id()

        return self.insert_run_result(
            dag_id=dag_id,
            run_id=run_id,
            meter_id=meter_id,
            phase=phase,
            label_type=label_type,
            label_value=label_value,
            confidence=confidence,
            source=source,
            result=result,
            id=id,
            topology_version=topology_version,
            node_id=node_id,
            edge_id=edge_id,
            cable_id=cable_id,
        )

    def upsert_meter_stats(self, df: pd.DataFrame) -> None:
        """
        Upsert meter inventory stats into the meta meter table.

        Args:
            df (pd.DataFrame):
                DataFrame with columns ``id``, ``first_seen``, ``last_seen``,
                ``total_rows``. Typically the output of a per-meter aggregation
                over an ingested time series file.
        """
        with self._sf() as s:
            MetaMeterResource(s).upsert_meter_stats(df)
            s.commit()

    def is_workflow_completed(self, workflow: str) -> bool:
        with self._sf() as s:
            return WorkflowStateResource(s).is_completed(workflow)

    def start_workflow(self, workflow: str, description: str | None = None) -> None:
        logging.info("Starting workflow: %s", workflow)
        with self._sf() as s:
            WorkflowStateResource(s).get_or_create(workflow, description)
            s.commit()

    def complete_workflow(self, workflow: str) -> None:
        logging.info("Completing workflow: %s", workflow)
        with self._sf() as s:
            WorkflowStateResource(s).mark_completed(workflow)
            s.commit()

    def get_time_series_meta_info(self) -> dict[str, Any]:
        with self._sf() as s:
            min_ts, max_ts, meter_ids = MetaMeterResource(s).get_timeseries_info()
        return {
            "min_timestamp": min_ts,
            "max_timestamp": max_ts,
            "id_list_of_sms_with_data": meter_ids,
        }
