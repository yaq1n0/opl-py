# opl-py

A typed Python SDK for working with [OpenPowerlifting](https://www.openpowerlifting.org/) data. 

Query lifters, meets, and rankings from a local DuckDB database built from the official OpenPowerlifting bulk CSV. 

Includes optional ML-based trajectory prediction analytics module.

## 0a. PyPI Installation (for usage)

To install opl-py from PyPI for usage:

```bash
pip install opl-py
```

For optional analytics features (trajectory prediction, percentile rankings):

```bash
pip install opl-py[analytics]
```

## 0b. Local/Editable installation (for development):

To install opl-py as an editable package, not from PyPI, use:

```bash
make install
```

or directly with pip:

```bash
pip install -e ".[dev,analytics]"
```

## 1. Quick Start

Download and ingest the full OPL dataset (~250 MB) into a local DuckDB file:
DuckDB path is `{platform_dirs.user_data_dir}/opl-py/opl.duckdb`

```bash
opl init
```

Update the database

```bash
opl update
```

Then query it with the opl Python sdk.

```python
import opl

client = opl.OPL()

# Look up a lifter
lifter = client.lifter("Ray Williams #1")
print(lifter.best_total_kg)
print(lifter.competition_count)

for entry in lifter.history():
    print(f"{entry.date}: {entry.total_kg}kg @ {entry.bodyweight_kg}kg")

# Query rankings
top_raw = client.rankings(
    sex=opl.Sex.MALE,
    equipment=opl.Equipment.RAW,
    event=opl.Event.SBD,
    weight_class="93",
    order_by="dots",
    limit=10,
)

# Raw SQL for custom queries
results = client.query(
    "SELECT Federation, COUNT(*) as n FROM entries GROUP BY Federation ORDER BY n DESC LIMIT 10"
)
```

## Web App Development (with HMR)

To run the API and UI locally with hot-reload:

```bash
make dev
```

This starts both servers in parallel:

- **API** — uvicorn on `http://localhost:8000` with `--reload` (restarts on Python file changes)
- **UI** — Vite dev server on `http://localhost:5173` with HMR (React fast refresh)

## Additional Documentation

| Guide                          | Description                                                    |
| ------------------------------ | -------------------------------------------------------------- |
| [SDK](docs/SDK.md)             | Python client, models, and enums                               |
| [Analytics](docs/ANALYTICS.md) | Percentile rankings, feature extraction, trajectory prediction |
| [Docker](docs/DOCKER.md)       | Docker setup for the web app ui                                |

## Attribution

This project uses data from the [OpenPowerlifting](https://www.openpowerlifting.org/) project. OpenPowerlifting data is contribueted to the public domain.
