# Development

## Setup

```bash
pip install -e ".[dev,analytics]"
```

Or via Make:

```bash
make install
```

## Quality Checks

Run all checks (lint + typecheck + tests)

```bash
make check
```

### Individual checks

| Command          | Description                                                |
| ---------------- | ---------------------------------------------------------- |
| `make lint`      | `ruff check src/ tests/`                                   |
| `make format`    | `ruff format src/ tests/`                                  |
| `make typecheck` | `pyright src/`                                             |
| `make test`      | `pytest tests/ -v --cov=src/opl --cov-report=term-missing` |

## Database Lifecycle

```bash
make init                           # opl init (download + create DB)
make update                         # opl update (re-download + refresh DB)
make init DB_PATH=/path/to/opl.db   # against a specific path
```

## Training

Train a trajectory model on the full dataset. The model is saved to `pretrained/{csv_date}/model.joblib`, where `csv_date` is read from the database metadata.

```bash
make train                          # train on full dataset
make train LIMIT=5000               # cap lifters (for testing)
make train DB_PATH=/path/to/opl.db  # against a specific database
```

Training on the full dataset (50k–200k qualifying lifters) takes several minutes.

Or directly via Python:

```bash
python -m opl.analytics.scripts.train
python -m opl.analytics.scripts.train --db-path /path/to/opl.duckdb
python -m opl.analytics.scripts.train --output-dir /path/to/pretrained
python -m opl.analytics.scripts.train --limit 5000
```

## E2E Demos

Run demos against a live database:

```bash
make demo                          # core + analytics demos
make demo DB_PATH=/path/to/opl.db  # against a specific database
make demo-init                     # includes downloading the full dataset
```

## Cleanup

Remove `__pycache__`, build artifacts, and caches:

```bash
make clean
```

## Install Extras

| Extra                               | What it adds                                                            |
| ----------------------------------- | ----------------------------------------------------------------------- |
| `pip install opl-py[analytics]`     | `scikit-learn`, `polars` — percentiles, features, trajectory prediction |
| `pip install opl-py[dev]`           | `pytest`, `ruff`, `pyright`, etc.                                       |
| `pip install opl-py[analytics,dev]` | Everything                                                              |
