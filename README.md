# 3phi Framework

Utility classes for **DB** access, **S3** interactions, and **data processing** via Controller Classes.  
Distributed on PyPi.

> **Install name:** `3phi-framework`  
> **Import package:** `threephi_framework`

---

## Installation

### Install from PyPi

```
pip install 3phi-framework
```

### Installing a Development Build (from CI)

Development builds are generated for each pull request and attached as workflow artifacts.
#### Download the artifact

1. Open the pull request on GitHub
2. Go to the Checks tab
3. Open the CI and Release workflow run
4. Download the artifact named dist-pr
5. Extract the archive locally

It will contain files like:

dist/
  3phi_framework-<version>.whl
  3phi_framework-<version>.tar.gz

#### Install the wheel (recommended)

From the extracted directory:
```
pip install dist/3phi_framework-*.whl
```

Alternatively, install the source distribution:
```
pip install dist/3phi_framework-*.tar.gz
```

Notes:
Wheels are preferred and install faster.
Make sure you are using Python ≥ 3.12 (project requirement).
Dev builds are temporary and may be deleted after 7 days.

## Quickstart

The framework is set up for local development as well as for being used in a deployment.
To set up your environment for local development, follow these steps:

### Set up virtual environment

[execute_data_app.sh](execute_data_app.sh) expects a virtual environment to be set up under [.venv]. See the [python docs](https://docs.python.org/3/library/venv.html) on how to set it up.

### Seed data

Obtain seed data for the database and the object storage and copy it to:
- [3_db_seed.sql](docker/db/init/3_db_seed.sql): This should be a sql dump/snapshot of a working 3phi Database, PSQL will automatically seed the DB when it is created using docker compose.
- [3phi](docker/object_storage/3phi): This should be a copy of a bucket from a working object storage. It will be mounted in the minio object storage as a bucket.

### Spin up DB and Object Storage

Navigate to [docker](./docker) and run 
```
docker compose up -d
```

This will bring up a local DB and a MinIO Object Storage seeded with the data you provided.

### Run a data app locally

Use the utility script [execute_data_app.sh](execute_data_app.sh) and pass the data app name as an argument, e.g.:
```
./exectue_data_app.sh sm_classifier
```
In case the script is not executable, make it executable:
```
chmod +x execute_data_app.sh
```

The script will install the dependencies in [requirements.txt](requirements.txt) in your virtual environment, set up environment variables as they are listed in [.env](.env) and execute the data app as a python module.

## Data Model

The currently assumed datamodel is illustrated in the diagram below:

![Data Model](docs/Data_Model_Jan_2026.png)
