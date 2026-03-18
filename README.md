# opl-py

A typed Python SDK for working with [OpenPowerlifting](https://www.openpowerlifting.org/) data. Query lifters, meets, and rankings from a local DuckDB database built from the official OpenPowerlifting bulk CSV. Includes optional ML-based trajectory prediction analytics module.

## 0. Installation

```bash
pip install opl-py
```

For analytics features (trajectory prediction, percentile rankings):

```bash
pip install opl-py[analytics]
```

For dev features (type-checking, testing, linting, formatting, publishing):

```bash
pip install opl-py[dev]
```

## 1. Quick Start

Download and ingest the full OPL dataset (~250 MB) into a local DuckDB file:

```bash
opl init
```

Then query it from Python:

```python
import opl

client = opl.OPL()

# Look up a lifter
lifter = client.lifter("Ray Williams #1")
print(lifter.best_total_kg)       # 1104.5
print(lifter.competition_count)   # 30

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

## Documentation

| Guide                          | Description                                                    |
| ------------------------------ | -------------------------------------------------------------- |
| [CLI](docs/CLI.md)             | Command-line interface — `opl init`, `opl update`, `opl info`  |
| [SDK](docs/SDK.md)             | Python client, models, and enums                               |
| [Analytics](docs/ANALYTICS.md) | Percentile rankings, feature extraction, trajectory prediction |
| [Development](docs/DEV.md)     | Local setup, linting, testing, Make targets                    |

## Attribution

This project uses data from the [OpenPowerlifting](https://www.openpowerlifting.org/) project. OpenPowerlifting data is contributed to the public domain.
