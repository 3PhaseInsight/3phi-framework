# Local-dev DB schema provisioning for 3phi-framework.
#   make up         -> bring up the stack with schema only (sqitch migrations)
#   make up-seeded  -> same, plus load a local data-only db/seed/seed.sql
#   make up-db      -> like up, but DB + schema only (no MinIO object storage)
#   make up-db-seeded -> like up-seeded, but without MinIO object storage
#   make migrations-refresh -> force re-fetch of the migration cache
-include dev.env

MIGRATIONS_REF ?= main
DP_REPO        ?= 3PhaseInsight/data-platform
SQITCH_SUBPATH ?= data-platform-infrastructure/sqitch
MIGRATIONS_DIR := ./.migrations
SQITCH_DIR     := $(MIGRATIONS_DIR)/sqitch
STAMP          := $(MIGRATIONS_DIR)/.ref-$(subst /,-,$(MIGRATIONS_REF)).stamp

.PHONY: up up-seeded up-db up-db-seeded migrations migrations-refresh down clean

up: migrations
	docker compose up -d

up-seeded: migrations
	docker compose --profile seed up -d

up-db: migrations
	docker compose up -d local-db sqitch-runner

up-db-seeded: migrations
	docker compose --profile seed up -d local-db sqitch-runner seed-runner

# The cache is keyed on MIGRATIONS_REF; a mutable ref like `main` is not re-checked
# once cached. Run `make migrations-refresh` to pull the latest for the same ref.
migrations: $(STAMP)

$(STAMP):
	@command -v git >/dev/null 2>&1 || { echo "ERROR: git not found."; exit 1; }
	rm -rf $(MIGRATIONS_DIR)
	mkdir -p $(MIGRATIONS_DIR)
	git clone --depth 1 --filter=blob:none --sparse --branch $(MIGRATIONS_REF) https://github.com/$(DP_REPO).git $(MIGRATIONS_DIR)/_dp
	git -C $(MIGRATIONS_DIR)/_dp sparse-checkout set $(SQITCH_SUBPATH)
	mv $(MIGRATIONS_DIR)/_dp/$(SQITCH_SUBPATH) $(SQITCH_DIR)
	rm -rf $(MIGRATIONS_DIR)/_dp
	touch $@
	@echo "✅ migrations fetched: $(DP_REPO)@$(MIGRATIONS_REF) -> $(SQITCH_DIR)"

migrations-refresh:
	rm -rf $(MIGRATIONS_DIR) && $(MAKE) migrations

down:
	docker compose --profile seed down

clean:
	./clean_volume.sh
