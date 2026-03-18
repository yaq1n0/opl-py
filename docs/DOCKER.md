# Docker

Multi-container deployment of opl-py with a FastAPI backend and React frontend.

## Architecture

```
┌─────────────────────────────────────────┐
│  docker compose                         │
│                                         │
│  ┌──────────┐       ┌───────────────┐   │
│  │ ui       │ /api/ │ api           │   │
│  │ nginx    │──────>│ FastAPI +     │   │
│  │ :80      │       │ uvicorn :8000 │   │
│  └──────────┘       └───────────────┘   │
│   port 3000          port 8000          │
└─────────────────────────────────────────┘
```

- **api** — Python 3.12, FastAPI serving opl-py endpoints. Contains the DuckDB database and pretrained models.
- **ui** — Static React build served by nginx. Reverse-proxies `/api/` requests to the api service.

## Prerequisites

- Docker and Docker Compose
- Python 3.12+ with opl-py installed (`make install`)
- Data initialized (`opl init`) and models trained (`opl train`)

## Quick Start

```bash
# 1. Install opl-py on the host
make install

# 2. Initialize the database (downloads OpenPowerlifting data)
opl init

# 3. Train pretrained models
opl train

# 4. Build and start the containers
make docker
```

The UI is available at **http://localhost:3000** and the API at **http://localhost:8000**.

## What `make docker` Does

1. **`docker-prepare`** — Copies `opl.duckdb` from the platform default path into `data/` in the build context
2. **`docker-build`** — Runs `docker compose build` (multi-stage builds for both services)
3. **`docker-up`** — Starts both containers in detached mode

## Makefile Targets

| Target                | Description                                    |
| --------------------- | ---------------------------------------------- |
| `make docker`         | Prepare data, build images, and start stack    |
| `make docker-build`   | Prepare data and build images only             |
| `make docker-up`      | Start the stack (images must be built already) |
| `make docker-down`    | Stop and remove containers                     |
| `make docker-prepare` | Copy DuckDB into `data/` build context         |

## Environment Variables

| Variable             | Default                | Description                          |
| -------------------- | ---------------------- | ------------------------------------ |
| `OPL_DB_PATH`        | `/app/data/opl.duckdb` | Path to DuckDB database in container |
| `OPL_PRETRAINED_DIR` | `/app/pretrained`      | Root directory for pretrained models |

## Slim Build (without PyTorch)

By default the API image includes PyTorch for neural network models. To build without it (smaller image, ~1GB vs ~3-4GB):

```bash
docker compose build --build-arg INSTALL_TORCH=false
```

This disables the `neural_network` approach but all scikit-learn-based approaches remain available.

## Port Mappings

| Service | Container Port | Host Port | URL                   |
| ------- | -------------- | --------- | --------------------- |
| ui      | 80             | 3000      | http://localhost:3000 |
| api     | 8000           | 8000      | http://localhost:8000 |

## API Endpoints

| Method | Path                 | Description                |
| ------ | -------------------- | -------------------------- |
| GET    | `/api/health`        | Health check               |
| GET    | `/api/stats`         | Database statistics        |
| GET    | `/api/search?q=`     | Search lifters by name     |
| GET    | `/api/lifter/{name}` | Lifter details and history |
| GET    | `/api/approaches`    | List prediction approaches |
| POST   | `/api/predict`       | Predict next performance   |

## Troubleshooting

**"No such file: data/opl.duckdb"** — Run `opl init` on the host first, then `make docker-prepare`.

**"No pretrained models found"** — Run `opl train` on the host. The `pretrained/` directory must exist with trained models.

**Port already in use** — Change the host port in `docker-compose.yml` (e.g., `"3001:80"`).

**UI shows connection errors** — The ui container waits for the api health check. Run `docker compose logs api` to check for startup errors.
