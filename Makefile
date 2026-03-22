.PHONY: install lint format typecheck test check prebuild dev-api dev-ui dev docker-prepare docker-build docker-up docker-down docker

# --- Setup and Dev Tool Commands ---
# install all dev and analytics dependencies with pip
install:
	pip install -e ".[dev,analytics]"


# run pyright on src/ (not tests and api yet, TODO: it)
typecheck:
	pyright src/

# run pytest unit tests with coverage 
test:
	pytest tests/ -v --cov=src/opl --cov-report=term-missing

# run ruff linting on /src, /tests, /api
lint:
	ruff check src/ tests/ api/

# run ruff formatting on /src, /tests, /api
format:
	ruff format src/ tests/ api/

# --- Run (dev with HMR) ---

# Run the FastAPI server with uvicorn hot-reload
dev-api:
	python -m uvicorn api.main:app --reload --host 0.0.0.0 --port 8000

# Run the Vite dev server with HMR
dev-ui:
	cd ui && npm run dev

# Run both api and ui dev servers in parallel
dev:
	$(MAKE) dev-api & $(MAKE) dev-ui & wait


# --- Run (docker production builds) ---

# Copy DuckDB from platform default path into build context
docker-prepare:
	@echo "Copying DuckDB to build context..."
	mkdir -p data
	cp "$$(python -c 'import platformdirs; print(platformdirs.user_data_dir("opl-py"))')/opl.duckdb" data/opl.duckdb
	@echo "DB copied to data/opl.duckdb"

# Build both containers
docker-build: docker-prepare
	docker compose build

# Start the stack
docker-up:
	docker compose up -d

# Stop the stack
docker-down:
	docker compose down

# Build and start
docker: docker-build docker-up
	@echo "OPL running at http://localhost:3000"
