# Local seed data (optional)

Place a **data-only** dump at `db/seed/seed.sql` (this file is gitignored). It is loaded by
`make up-seeded` *after* the sqitch schema is deployed. `make up` (no seed) leaves the tables empty.

Generate it from a populated source DB:

    pg_dump "$SOURCE_URI" \
      --data-only --no-owner --no-acl --disable-triggers \
      --schema=meta --schema=lv \
      -f db/seed/seed.sql

Notes:
- `--data-only` keeps schema ownership with sqitch (no CREATE statements to collide).
- `--schema=meta --schema=lv` excludes the `airflow` and `sqitch` registry schemas.
- `--no-owner`/`--no-acl` drop `OWNER TO` and `GRANT`/`REVOKE` lines that reference roles
  absent locally (they would abort the load under `ON_ERROR_STOP`).
- `--disable-triggers` bypasses FK ordering; fine because the local `postgres` role is superuser.
- This is real customer data — keep it local; it must never be committed.

## Re-seeding

`make up-seeded` runs the loader once; a cleanly-exited container is not re-run on a second
`make up-seeded`. After updating `seed.sql`, force it:

    docker compose --profile seed up -d --force-recreate seed-runner

Loader output goes to the container log — check it with `docker logs local-3phi-seed`.
