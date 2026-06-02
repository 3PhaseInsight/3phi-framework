# Local development stack

Standalone Postgres + MinIO for developing the framework, with the **canonical** DB schema
deployed from the `data-platform` sqitch migrations (no `data-platform` checkout required).

## Prerequisites
- Docker
- `git` able to clone `3PhaseInsight/data-platform` over HTTPS (your normal git credentials apply)

## Usage
- `make up` — schema only (empty tables); then ingest via the data apps.
- `make up-seeded` — schema **and** load a local `db/seed/seed.sql` (see `db/seed/README.md`).
- `make migrations-refresh` — force re-fetch of the migration cache.
- `make down` — stop the stack. `make clean` — wipe the local DB volume.

## Pinning the schema
Migrations are pulled from `3PhaseInsight/data-platform` at `MIGRATIONS_REF` (default `main`).
Override via the environment or a gitignored `dev.env` (copy from `dev.env.example`), e.g. to track
an unmerged branch:

    MIGRATIONS_REF=feature/api-layer make up

## How the fetch works
`make` populates the gitignored `.migrations/sqitch/` cache with a shallow, blobless sparse
`git clone` of `3PhaseInsight/data-platform` at `MIGRATIONS_REF` — no tooling beyond `git`.
`MIGRATIONS_REF` accepts a **branch or tag** (an arbitrary commit SHA would need a non-shallow fetch).
