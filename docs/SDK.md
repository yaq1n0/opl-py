# SDK

The core Python interface for querying OpenPowerlifting data.

## OPL Client

```python
import opl

client = opl.OPL()                        # uses default DB path
client = opl.OPL(db_path=Path("my.db"))   # custom path
```

### Methods

| Method                                                                                             | Returns            | Description                                                     |
| -------------------------------------------------------------------------------------------------- | ------------------ | --------------------------------------------------------------- |
| `lifter(name)`                                                                                     | `Lifter \| None`   | Look up a lifter by exact name                                  |
| `lifters_bulk(min_meets, limit)`                                                                   | `list[Lifter]`     | Load many lifters in one SQL query (use for training)           |
| `search_lifters(query, limit=20)`                                                                  | `list[Lifter]`     | Case-insensitive substring search                               |
| `meet(name, federation, date)`                                                                     | `Meet \| None`     | Look up a specific meet                                         |
| `meets(federation, country, year, limit, offset)`                                                  | `list[Meet]`       | Filter meets                                                    |
| `rankings(sex, equipment, event, weight_class, federation, year, tested, order_by, limit, offset)` | `list[Entry]`      | Query rankings                                                  |
| `federations()`                                                                                    | `list[str]`        | All federations in the dataset                                  |
| `query(sql, params)`                                                                               | `list[dict]`       | Raw SQL escape hatch                                            |
| `to_polars(sql, params)`                                                                           | `polars.DataFrame` | SQL result as a Polars DataFrame (requires `opl-py[analytics]`) |
| `stats()`                                                                                          | `dict`             | Row count, last updated, source URL                             |

## Models

### Entry

A single competition result (one row in the OPL dataset).

```python
entry.name              # str
entry.sex               # Sex
entry.equipment         # Equipment
entry.event             # Event
entry.total_kg          # float | None
entry.best3_squat_kg    # float | None
entry.best3_bench_kg    # float | None
entry.best3_deadlift_kg # float | None
entry.dots              # float | None
entry.wilks             # float | None
entry.date              # datetime.date
entry.federation        # str
entry.meet_name         # str
```

### Lifter

Aggregated view across all competitions.

```python
lifter.name               # str
lifter.entries            # list[Entry]
lifter.competition_count  # int  (computed)
lifter.best_total_kg      # float | None  (computed)
lifter.best_squat_kg      # float | None  (computed)
lifter.best_bench_kg      # float | None  (computed)
lifter.best_deadlift_kg   # float | None  (computed)
lifter.history()          # list[Entry] sorted chronologically
```

### Meet

A competition.

```python
meet.name        # str
meet.date        # datetime.date
meet.federation  # str
meet.country     # str | None
meet.entries     # list[Entry]
```

## Enums

```python
opl.Sex.MALE / opl.Sex.FEMALE / opl.Sex.MX

opl.Equipment.RAW / opl.Equipment.WRAPS / opl.Equipment.SINGLE_PLY
opl.Equipment.MULTI_PLY / opl.Equipment.UNLIMITED / opl.Equipment.STRAPS

opl.Event.SBD   # Full power
opl.Event.B     # Bench only
opl.Event.D     # Deadlift only
# ... SD, SB, BD, S
```
