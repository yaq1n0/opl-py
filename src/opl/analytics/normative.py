"""Normative percentile calculations for powerlifting performance."""

from pathlib import Path

from opl.core.db import get_connection
from opl.core.enums import Equipment, Sex


def percentile(
    lift: str,
    weight: float,
    sex: Sex,
    equipment: Equipment,
    weight_class: str | None = None,
    tested: bool | None = None,
    db_path: Path | None = None,
) -> float:
    """Calculate what percentile a given lift weight falls at.

    Args:
        lift: One of "squat", "bench", "deadlift", "total".
        weight: The lift weight in kg.
        sex: Sex category.
        equipment: Equipment category.
        weight_class: Optional weight class filter.
        tested: Optional tested status filter.
        db_path: Optional custom database path.

    Returns:
        Percentile as a float (0-100).
    """
    column_map = {
        "squat": "Best3SquatKg",
        "bench": "Best3BenchKg",
        "deadlift": "Best3DeadliftKg",
        "total": "TotalKg",
    }

    if lift not in column_map:
        raise ValueError(f"lift must be one of {list(column_map.keys())}, got '{lift}'")

    col = column_map[lift]

    conditions = [
        f'"{col}" IS NOT NULL',
        f'"{col}" > 0',
        '"Sex" = ?',
        '"Equipment" = ?',
    ]
    params: list[object] = [sex.value, equipment.value]

    if weight_class is not None:
        conditions.append('"WeightClassKg" = ?')
        params.append(weight_class)

    if tested is not None:
        conditions.append('"Tested" = ?')
        params.append("Yes" if tested else "")

    where = " AND ".join(conditions)

    con = get_connection(db_path)
    try:
        # Count how many lifts are below the given weight
        count_below = con.execute(
            f'SELECT COUNT(*) FROM entries WHERE {where} AND "{col}" <= ?',
            params + [weight],
        ).fetchone()
        total_count = con.execute(
            f"SELECT COUNT(*) FROM entries WHERE {where}",
            params,
        ).fetchone()
    finally:
        con.close()

    if not total_count or total_count[0] == 0:
        return 0.0
    if not count_below:
        return 0.0

    return round((count_below[0] / total_count[0]) * 100, 1)
