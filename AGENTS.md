# AGENTS.md — 3phi-framework

Quick-reference for AI agents working in this repository. Read this before exploring files.

## What this repo is

`threephi_framework` (PyPI: `3phi-framework`) is a Python library for **LV electrical grid data management**: ingesting smart meter time series, managing network topology, classifying meter data quality, and running distributed analytics via Dask. Python 3.12+, MIT license.

## Directory map

```
src/threephi_framework/
├── __init__.py                   # Public API (connectors, BaseDataApp, data apps, DataExtractor, TopologyController, ProcessingLevel)
├── processing_level.py           # ProcessingLevel StrEnum (raw / cleaned / cleaned_and_corrected)
├── db/db.py                      # SQLAlchemy engine + new_session() factory
├── object_storage/
│   ├── base_connector.py         # Abstract storage interface (+ raw/flags/corrections convenience methods)
│   ├── s3_connector.py           # MinIO/S3 impl (s3fs + Dask storage options)
│   └── azure_blob_connector.py   # Azure Blob Storage impl (adlfs)
├── models/
│   ├── base.py                   # BaseModel (DeclarativeBase)
│   ├── topology/lv_schema_mixin.py   # Sets schema="lv" for topology tables
│   ├── topology/assets/          # SecondarySubstation, Transformer, Feeder, Cabinet, DeliveryPoint, Meter
│   ├── topology/graph/           # Node, Edge, Cable, EdgeCable, TopologyVersion
│   ├── topology/utilities.py     # NodeCurrentModel / EdgeCurrentModel (map the *_current views)
│   ├── meta/meta_schema_mixin.py     # Sets schema for metadata tables
│   └── meta/                     # MetaMeter, FileIndex, IngestBatch, RunResult, WorkflowState
├── resources/
│   ├── base.py                   # BaseResource: session=self.s, bulk_insert(), _log_*()
│   ├── staging.py                # Temp tables for bulk ingestion
│   ├── sanity.py                 # Pre-commit data validation checks
│   ├── topology/                 # Mirrors models/topology/ (assets + graph) + topology_export.py
│   └── meta/                     # MetaMeter, FileIndex, IngestBatch, RunResult, WorkflowState resources
├── controllers/
│   ├── topology.py               # TopologyController: version management, ingestion, queries, exports
│   ├── meta.py                   # MetaController: meter metadata, run results, workflow states
│   ├── time_series.py            # TimeSeriesController: processing-level-aware timeseries reads
│   └── ingestion.py              # IngestionController: ingest batch + file index lifecycle
├── data_apps/
│   ├── base.py                   # BaseDataApp: context manager, Dask lifecycle, cached controllers
│   ├── base_config.py            # BaseConfig frozen dataclass → .to_dict()
│   ├── timeseries_ingestor.py
│   ├── topology_ingestor.py
│   ├── topology_cleaner.py       # Re-ingests the current topology at level "cleaned"
│   ├── topology_tester.py
│   ├── sm_classifier.py
│   └── stat_labeler.py
├── data_extractor/
│   ├── data_extractor.py         # CSV→Parquet ingestion + legacy-compatible proxy over the controllers
│   └── schemas/phase_measurements/v1.py  # Parquet/CSV column schemas + QualityFlag
├── schemas/v1/                   # Pandas dtype definitions (topology, phase_measurements)
├── dtu/                          # Domain logic (sm_classifier, stat_labeler, timeseries_cleaner, topology_cleaner)
└── util/util.py                  # v1_get_shard_for_meter_id() — xxhash % 3 sharding
```

Unit tests live in `tests/` (pure-Python, no DB required): run with `pytest`. Data apps
additionally have `if __name__ == "__main__"` blocks for integration runs against a live stack.

## Architecture: three-tier pattern

Every domain follows the same layering. Never skip tiers.

```
Data App  →  Controller  →  Resource  →  Model (ORM)
```

| Layer | Responsibility | Base class |
|---|---|---|
| Data App | Orchestration, config, Dask lifecycle | `BaseDataApp` |
| Controller | Domain operations, coordinates resources | — |
| Resource | SQL/S3 CRUD, domain queries | `BaseResource` |
| Model | SQLAlchemy ORM table definition | `BaseModel` |

**Adding a new domain entity** means adding a Model, a Resource, wiring into a Controller, and optionally a Data App — in that order.

## Key abstractions

### BaseDataApp (`data_apps/base.py`)
Context manager. Always use as `with MyApp(config) as app: app.run()`.
- `__enter__`: starts Dask client (local or remote TCP)
- `__exit__`: closes Dask client, logs any exception
- Controllers are `@cached_property` — instantiated lazily on first access
- Subclasses must implement `run()`

### BaseResource (`resources/base.py`)
- Constructor takes a `Session`; stored as `self.s`
- `bulk_insert(table, rows: Iterable[dict])` — wraps SQLAlchemy `insert().values()`
- `_log_debug/info/warning/error()` — prefer these over bare `logging` calls

### TopologyController (`controllers/topology.py`)
- Central controller for topology lifecycle
- `ingest(topology_ddf, sm_cab_ddf)`: full ingestion pipeline with staging, sanity checks, and atomic version flip via `flip_current_to(version)`
- `TopologyVersion` with `is_current` flag enables reproducible historical queries

### Storage connectors (`object_storage/`)
`BaseConnector` defines the full interface. Two implementations ship out of the box:
- `S3Connector` — wraps s3fs; targets MinIO/S3; bucket hardcoded to `3phi`
- `AzureBlobConnector` — wraps adlfs; targets Azure Blob Storage; falls back to `DefaultAzureCredential` when no account key is set

Both use the same Parquet sharding scheme: meter_id → `util.v1_get_shard_for_meter_id()` → 3 shards.

`TimeSeriesController` accepts any `BaseConnector` — swap the implementation without changing application code. `get_meter_data(meter_ids, dataset_root_path=None)` accepts an optional path override; if omitted, falls back to the path set at construction time.

### Config dict shape

```python
config = {
    "result_name": "optional_run_label",   # defaults to unix timestamp
    "dask": {
        # Remote cluster:
        "host": "dask-scheduler",
        "port": "8786",
        # OR local cluster:
        "local": True,
        "n_workers": 2,
    }
}
```

## Domain model

LV topology hierarchy (parent → child):
```
SecondarySubstation → Transformer → Feeder → Cabinet → DeliveryPoint → Meter
```
All live in the `lv` DB schema. Meter metadata (JSONB columns `data_quality`, `data_statistics`, `connectivity`) lives in the `meta`/public schema.

## Naming conventions

| Thing | Convention | Example |
|---|---|---|
| ORM model | `*Model` | `MeterModel`, `MetaMeterModel` |
| Resource | `*Resource` | `MeterResource`, `MetaMeterResource` |
| Controller | `*Controller` | `TopologyController` |
| Data app | descriptive noun | `TimeseriesIngestor`, `SMClassifier` |
| DB schemas | `lv` (topology), `public` (meta) | — |

## Code style

- Ruff enforced: `E, F, W, UP, B, SIM, I` rules
- Line length: 120 characters
- Quote style: double quotes
- Python 3.12+ — use modern syntax (match, `X | Y` unions, `type` aliases)
- Run `ruff check` and `ruff format` before committing; pre-commit hook does this automatically

## Environment

All connection settings come from `.env`. The committed `.env` contains local-dev defaults
matching the `docker/` stack — real deployments override these values:

```bash
# S3 / MinIO
S3_ENDPOINT_URL=http://localhost:19000
S3_ACCESS_KEY=minioadmin
S3_SECRET_KEY=minioadmin

# Azure Blob Storage (alternative to S3)
AZURE_STORAGE_ACCOUNT_NAME=myaccount
AZURE_STORAGE_CONTAINER_NAME=3phi
AZURE_STORAGE_ACCOUNT_KEY=           # optional — omit to use DefaultAzureCredential

# Database
DB_TYPE=POSTGRES
DB_USER=postgres
DB_PASSWORD=password
DB_NAME=3phi-db
DB_HOST=localhost
DB_PORT=5432
```

Local dev: `make up` inside `docker/` spins up PostgreSQL (schema deployed from the canonical
data-platform sqitch migrations) + MinIO. See `docker/README.md` for seeding and variants.

## What NOT to do

- Do not bypass the Controller/Resource tiers by writing SQL directly in Data Apps
- Do not create a new session manually — always use `threephi_db.new_session` factory
- Do not run Dask operations outside a `BaseDataApp` context manager
- Do not add a new model schema without a corresponding mixin (see `LvSchemaMixin`, `MetaSchemaMixin`)
