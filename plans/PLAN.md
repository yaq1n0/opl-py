# opl-py: OpenPowerlifting Python SDK & Analytics

## Overview

`opl-py` is a Python library published to PyPI that provides a typed, ergonomic SDK for working with [OpenPowerlifting](https://www.openpowerlifting.org/) data. It includes optional extras for ML-based lifter trajectory prediction (`[analytics]`).

OpenPowerlifting is an open-source project on GitLab that maintains a public-domain dataset of the world's powerlifting competition results (~3M+ rows, ~250MB+ CSV). The project provides bulk CSV downloads at `https://data.openpowerlifting.org` but has no official Python SDK. More information at `https://openpowerlifting.gitlab.io/opl-csv/introduction.html`

## Build & Packaging

### pyproject.toml

- Build backend: `hatchling`
- Package name on PyPI: `opl-py`
- Import name: `opl`
- Python: `>=3.12`
- Single source of version truth in `src/opl/__init__.py` via `__version__`

### Dependencies

**Core (bare `pip install opl-py`):**

- `duckdb` — local columnar database, CSV ingestion, query engine
- `pydantic` — typed models, validation, serialization
- `httpx` — downloading the bulk CSV
- `platformdirs` — OS-appropriate data directory resolution
- `rich` — CLI progress bars for download/init
- `click` — CLI interface

**Optional extras:**

```toml
[project.optional-dependencies]
analytics = ["scikit-learn", "polars>=1.0"]
dev = ["pytest", "pytest-cov", "ruff", "mypy", "pre-commit"]
```

### CLI Entry Point

```toml
[project.scripts]
opl = "opl.core.cli:main"
```

Commands:

- `opl init` — download latest CSV from data.openpowerlifting.org, ingest into local DuckDB file
- `opl update` — re-download CSV, diff and apply changes to existing DB
- `opl info` — print DB location, row count, last updated timestamp

---

## Phase 1: Core SDK (`opl.core`)

This is the foundation. Everything else depends on it.

### Data Source

- Bulk CSV download from `https://data.openpowerlifting.org`
- This returns a ZIP file containing a dated CSV (e.g., `openpowerlifting-2025-01-01/openpowerlifting-2025-01-01.csv`)
- The CSV schema is documented at: https://openpowerlifting.gitlab.io/opl-csv/bulk-csv-docs.html
- Data is public domain. Include the OPL attribution statement in README and in the library's `__init__.py` docstring.

### DuckDB Backend (`opl.core.db`)

- Store the DuckDB file at `platformdirs.user_data_dir("opl-py") / "opl.duckdb"`
- On `opl init`:
  1. Download the ZIP via httpx with a progress bar (rich)
  2. Extract the CSV
  3. Create a DuckDB database with a single `entries` table using DuckDB's native `read_csv_auto()`
  4. Create a `metadata` table storing: `last_updated` timestamp, `source_url`, `row_count`, `csv_date`
  5. Create appropriate indexes (Name, Federation, Date, Equipment)
  6. Clean up the downloaded ZIP/CSV after ingestion
- On `opl update`:
  1. Download the latest CSV
  2. Drop and recreate the `entries` table (the CSV is a full snapshot, not incremental)
  3. Update metadata
- Provide a `get_connection()` function that returns a DuckDB connection, raising a clear error if the DB doesn't exist yet ("Run `opl init` first")

### Pydantic Models (`opl.core.models`)

Map the OPL CSV columns to typed Python objects. Key models:

```python
class Entry(BaseModel):
    """A single competition entry (one row in the OPL dataset)."""
    name: str
    sex: Sex
    event: Event
    equipment: Equipment
    age: float | None
    age_class: str | None
    birth_year_class: str | None
    division: str | None
    bodyweight_kg: float | None
    weight_class_kg: str | None
    squat1_kg: float | None
    squat2_kg: float | None
    squat3_kg: float | None
    best3_squat_kg: float | None
    bench1_kg: float | None
    bench2_kg: float | None
    bench3_kg: float | None
    best3_bench_kg: float | None
    deadlift1_kg: float | None
    deadlift2_kg: float | None
    deadlift3_kg: float | None
    best3_deadlift_kg: float | None
    total_kg: float | None
    place: str  # can be "DQ", "DD", "NS", or a number
    dots: float | None
    wilks: float | None
    glossbrenner: float | None
    goodlift: float | None
    tested: bool | None
    country: str | None
    state: str | None
    federation: str
    parent_federation: str | None
    date: date
    meet_country: str | None
    meet_state: str | None
    meet_town: str | None
    meet_name: str
    sanctioned: bool | None

class Lifter(BaseModel):
    """Aggregated view of a lifter across all competitions."""
    name: str
    entries: list[Entry]

    @computed_field
    def competition_count(self) -> int: ...

    @computed_field
    def best_total_kg(self) -> float | None: ...

    @computed_field
    def best_squat_kg(self) -> float | None: ...

    @computed_field
    def best_bench_kg(self) -> float | None: ...

    @computed_field
    def best_deadlift_kg(self) -> float | None: ...

    def history(self) -> list[Entry]:
        """Entries sorted chronologically."""
        ...

class Meet(BaseModel):
    """A competition meet."""
    name: str
    date: date
    federation: str
    country: str | None
    state: str | None
    town: str | None
    entries: list[Entry]
```

### Enums (`opl.core.enums`)

```python
class Sex(str, Enum):
    MALE = "M"
    FEMALE = "F"
    MX = "Mx"

class Equipment(str, Enum):
    RAW = "Raw"
    WRAPS = "Wraps"
    SINGLE_PLY = "Single-ply"
    MULTI_PLY = "Multi-ply"
    UNLIMITED = "Unlimited"
    STRAPS = "Straps"

class Event(str, Enum):
    SBD = "SBD"  # Full power
    BD = "BD"    # Push-pull
    SD = "SD"    # Squat-deadlift
    SB = "SB"    # Squat-bench
    S = "S"      # Squat only
    B = "B"      # Bench only
    D = "D"      # Deadlift only
```

### Client Interface (`opl.core.client`)

This is the primary user-facing API. All methods return typed Pydantic models.

```python
class OPL:
    """Main client for querying OpenPowerlifting data."""

    def __init__(self, db_path: Path | None = None):
        """Connect to local DuckDB. Uses default path if not specified."""
        ...

    def lifter(self, name: str) -> Lifter | None:
        """Look up a lifter by name. Returns None if not found."""
        ...

    def search_lifters(self, query: str, limit: int = 20) -> list[Lifter]:
        """Fuzzy search lifters by name."""
        ...

    def meet(self, name: str, federation: str | None = None, date: date | None = None) -> Meet | None:
        """Look up a specific meet."""
        ...

    def meets(
        self,
        federation: str | None = None,
        country: str | None = None,
        year: int | None = None,
        limit: int = 100,
        offset: int = 0,
    ) -> list[Meet]:
        """Query meets with filters."""
        ...

    def rankings(
        self,
        sex: Sex | None = None,
        equipment: Equipment | None = None,
        event: Event | None = None,
        weight_class: str | None = None,
        federation: str | None = None,
        year: int | None = None,
        tested: bool | None = None,
        order_by: str = "total_kg",
        limit: int = 100,
        offset: int = 0,
    ) -> list[Entry]:
        """Query rankings with filters. Mirrors openpowerlifting.org rankings page."""
        ...

    def federations(self) -> list[str]:
        """List all federations in the dataset."""
        ...

    def query(self, sql: str, params: dict | None = None) -> list[dict]:
        """Raw SQL escape hatch. Returns list of dicts."""
        ...

    def to_polars(self, sql: str, params: dict | None = None):
        """Execute SQL and return a Polars DataFrame. Requires polars."""
        ...

    def stats(self) -> dict:
        """Return DB metadata: row count, last updated, etc."""
        ...
```

### Usage Examples (for README)

```python
import opl

# Initialise data (first time only)
# Or run: opl init

client = opl.OPL()

# Look up a lifter
lifter = client.lifter("Ray Williams")
print(lifter.best_total_kg)         # 1083.5
print(lifter.competition_count)     # 42
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

# Raw SQL for power users
results = client.query(
    "SELECT Federation, COUNT(*) as n FROM entries GROUP BY Federation ORDER BY n DESC LIMIT 10"
)

# Polars integration
df = client.to_polars("SELECT * FROM entries WHERE Country = 'UK'")
```

---

## Phase 2: Analytics (`opl.analytics`)

Requires `pip install opl-py[analytics]`.

### Trajectory Prediction (`opl.analytics.trajectory`)

Given a lifter's competition history, predict their future performance. This is the novel ML component.

**Approach:**

- Use each lifter's time series of competition results as training data
- Features: age at competition, bodyweight, time since first competition, time since last competition, competition frequency, equipment type, lift ratios (squat:bench:deadlift), historical rate of progression
- Target: next competition total (or per-lift predictions)
- Model: start with gradient boosted trees (scikit-learn HistGradientBoostingRegressor) — interpretable, fast to train, handles missing values natively
- Training data: all lifters with 3+ competition entries (cross-sectional model, not per-lifter)

```python
from opl.analytics import predict_trajectory, TrajectoryPrediction

client = opl.OPL()
lifter = client.lifter("Some Lifter")

prediction: TrajectoryPrediction = predict_trajectory(lifter)
# prediction.next_total_kg -> 525.0
# prediction.next_squat_kg -> 210.0
# prediction.next_bench_kg -> 135.0
# prediction.next_deadlift_kg -> 180.0
# prediction.confidence_interval -> (510.0, 540.0)
# prediction.percentile -> 85.2  (among similar lifters)
# prediction.trajectory_curve -> list of (months_from_now, predicted_total) points
```

### Feature Engineering (`opl.analytics.features`)

Extract ML-ready features from a lifter's Entry history:

- `career_length_days` — days between first and most recent competition
- `competition_count` — total meets
- `competition_frequency` — meets per year
- `best_total_kg`, `best_squat_kg`, `best_bench_kg`, `best_deadlift_kg`
- `latest_total_kg`, `latest_bodyweight_kg`
- `total_progression_rate` — kg gained per year of competing
- `squat_to_total_ratio`, `bench_to_total_ratio`, `deadlift_to_total_ratio`
- `age_at_latest`, `age_at_first`
- `weight_class_numeric` — parsed from weight class string
- `is_tested` — most recent tested status
- `equipment_mode` — most common equipment used
- `days_since_last_comp`

### Normative Data (`opl.analytics.normative`)

Percentile rankings — "where does this lifter stand relative to their peers?"

```python
from opl.analytics import percentile

# What percentile is a 200kg squat for a tested raw 93kg male?
percentile(
    lift="squat",
    weight=200.0,
    sex=opl.Sex.MALE,
    equipment=opl.Equipment.RAW,
    weight_class="93",
    tested=True,
)
# -> 82.5 (82.5th percentile)
```

### Pre-trained Model Distribution

- Train the model as part of a CI/release step
- Serialize with joblib or pickle
- Ship the trained model as a package data file (small, <10MB)
- Provide `opl.analytics.retrain()` for users who want to train on their local (possibly more recent) data

---

## Implementation Order

1. **Scaffold the project** — pyproject.toml, src layout, empty packages, dev tooling (ruff, mypy, pytest)
2. **`opl.core.download`** — download and extract the OPL CSV
3. **`opl.core.db`** — DuckDB ingestion, schema creation, indexes
4. **`opl.core.enums` + `opl.core.models`** — all Pydantic models and enums
5. **`opl.core.client`** — the OPL class with all query methods
6. **CLI** — `opl init`, `opl update`, `opl info`
7. **Tests for all of the above** — use a small fixture CSV (100-200 rows) committed to the repo
8. **`opl.analytics.features`** — feature engineering
9. **`opl.analytics.normative`** — percentile calculations
10. **`opl.analytics.trajectory`** — ML model training and prediction

## Development Tooling

- **Formatter/Linter**: ruff (replaces black + isort + flake8)
- **Type checking**: mypy (strict mode)
- **Testing**: pytest + pytest-cov
- **Pre-commit hooks**: ruff, mypy
- **CI**: GitHub Actions — lint, typecheck, test on Python 3.11/3.12/3.13

## Key Design Principles

- **Typed everywhere.** No `dict[str, Any]` in the public API. Every return value is a Pydantic model or a known type.
- **Lazy by default.** Don't load the entire dataset into memory. Use DuckDB queries and return bounded results.
- **Fail clearly.** If the DB doesn't exist, say "run `opl init`". If a lifter isn't found, return `None`, don't raise.
- **No scraping.** Use the official bulk CSV download. OPL explicitly says not to scrape.
- **Attribution.** Include the OPL attribution statement as specified in their docs.

## Attribution

This project uses data from the [OpenPowerlifting](https://www.openpowerlifting.org/) project. You may download a copy of the data at https://data.openpowerlifting.org.
