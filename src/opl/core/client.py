from __future__ import annotations

from datetime import date
from pathlib import Path
from typing import Any

import duckdb

from opl.core.db import get_connection, get_db_info
from opl.core.enums import Equipment, Event, Sex
from opl.core.models import Entry, Lifter, Meet

# Mapping from Python model field names to OPL CSV column names
_COLUMN_MAP: dict[str, str] = {
    "name": "Name",
    "sex": "Sex",
    "event": "Event",
    "equipment": "Equipment",
    "age": "Age",
    "age_class": "AgeClass",
    "birth_year_class": "BirthYearClass",
    "division": "Division",
    "bodyweight_kg": "BodyweightKg",
    "weight_class_kg": "WeightClassKg",
    "squat1_kg": "Squat1Kg",
    "squat2_kg": "Squat2Kg",
    "squat3_kg": "Squat3Kg",
    "best3_squat_kg": "Best3SquatKg",
    "bench1_kg": "Bench1Kg",
    "bench2_kg": "Bench2Kg",
    "bench3_kg": "Bench3Kg",
    "best3_bench_kg": "Best3BenchKg",
    "deadlift1_kg": "Deadlift1Kg",
    "deadlift2_kg": "Deadlift2Kg",
    "deadlift3_kg": "Deadlift3Kg",
    "best3_deadlift_kg": "Best3DeadliftKg",
    "total_kg": "TotalKg",
    "place": "Place",
    "dots": "Dots",
    "wilks": "Wilks",
    "glossbrenner": "Glossbrenner",
    "goodlift": "Goodlift",
    "tested": "Tested",
    "country": "Country",
    "state": "State",
    "federation": "Federation",
    "parent_federation": "ParentFederation",
    "date": "Date",
    "meet_country": "MeetCountry",
    "meet_state": "MeetState",
    "meet_town": "MeetTown",
    "meet_name": "MeetName",
    "sanctioned": "Sanctioned",
}

_SELECT_COLS = ", ".join(f'"{col}" AS {field}' for field, col in _COLUMN_MAP.items())


def _row_to_entry(row: dict[str, object]) -> Entry:
    """Convert a raw DB row dict to an Entry model."""
    # Handle tested/sanctioned as bools
    if row.get("tested") is not None:
        val = row["tested"]
        row["tested"] = val == "Yes" if isinstance(val, str) else bool(val)
    if row.get("sanctioned") is not None:
        val = row["sanctioned"]
        row["sanctioned"] = val == "Yes" if isinstance(val, str) else bool(val)
    # Coerce fields that DuckDB may infer as int but are strings in the model
    for str_field in ("weight_class_kg", "place"):
        if row.get(str_field) is not None:
            row[str_field] = str(row[str_field])
    # Replace empty strings with None for optional fields
    for key, val in row.items():
        if val == "":
            row[key] = None
    return Entry.model_validate(row)


class OPL:
    """Main client for querying OpenPowerlifting data."""

    def __init__(self, db_path: Path | None = None) -> None:
        """Connect to local DuckDB. Uses default path if not specified."""
        self._db_path = db_path
        # Validate connection on init
        con = get_connection(db_path)
        con.close()

    def _connect(self) -> duckdb.DuckDBPyConnection:
        return get_connection(self._db_path)

    def _fetch_entries(
        self,
        where: str = "",
        params: list[object] | None = None,
        order_by: str = "date DESC",
        limit: int = 100,
        offset: int = 0,
    ) -> list[Entry]:
        """Internal helper to fetch entries with filters."""
        sql = f"SELECT {_SELECT_COLS} FROM entries"
        if where:
            sql += f" WHERE {where}"
        sql += f" ORDER BY {order_by} LIMIT {limit} OFFSET {offset}"

        con = self._connect()
        try:
            result = con.execute(sql, params or [])
            cols = [desc[0] for desc in result.description]
            rows = result.fetchall()
            return [_row_to_entry(dict(zip(cols, row, strict=True))) for row in rows]
        finally:
            con.close()

    def lifter(self, name: str) -> Lifter | None:
        """Look up a lifter by name. Returns None if not found."""
        entries = self._fetch_entries(where='"Name" = ?', params=[name], limit=10000)
        if not entries:
            return None
        return Lifter(name=name, entries=entries)

    def lifters_bulk(
        self,
        min_meets: int = 1,
        limit: int | None = None,
    ) -> list[Lifter]:
        """Load many lifters efficiently in a single query.

        Fetches all qualifying entries in one SQL round-trip and groups them
        in Python. Orders of magnitude faster than calling lifter() in a loop.

        Args:
            min_meets: Only include lifters with at least this many meets
                       that have a recorded TotalKg.
            limit: Cap the number of lifters returned (useful for testing).

        Returns:
            List of Lifter objects, each with their full competition history.
        """
        from collections import defaultdict

        limit_sql = f"LIMIT {limit}" if limit else ""

        sql = f"""
            WITH qualifying AS (
                SELECT "Name" FROM entries
                WHERE "TotalKg" IS NOT NULL AND "TotalKg" > 0
                GROUP BY "Name"
                HAVING COUNT(*) >= {min_meets}
                ORDER BY "Name"
                {limit_sql}
            )
            SELECT {_SELECT_COLS} FROM entries
            WHERE "Name" IN (SELECT "Name" FROM qualifying)
            ORDER BY "Name", "Date"
        """

        con = self._connect()
        try:
            result = con.execute(sql)
            cols = [desc[0] for desc in result.description]
            rows = result.fetchall()
        finally:
            con.close()

        entries_by_name: dict[str, list[Entry]] = defaultdict(list)
        for row in rows:
            entry = _row_to_entry(dict(zip(cols, row, strict=True)))
            entries_by_name[entry.name].append(entry)

        return [Lifter(name=name, entries=ents) for name, ents in entries_by_name.items()]

    def search_lifters(self, query: str, limit: int = 20) -> list[Lifter]:
        """Search lifters by name (case-insensitive prefix/substring match)."""
        con = self._connect()
        try:
            result = con.execute(
                """
                SELECT DISTINCT "Name" FROM entries
                WHERE "Name" ILIKE ?
                LIMIT ?
                """,
                [f"%{query}%", limit],
            )
            names = [row[0] for row in result.fetchall()]
        finally:
            con.close()

        lifters: list[Lifter] = []
        for name in names:
            lifter = self.lifter(name)
            if lifter:
                lifters.append(lifter)
        return lifters

    def meet(
        self, name: str, federation: str | None = None, date: date | None = None
    ) -> Meet | None:
        """Look up a specific meet."""
        conditions = ['"MeetName" = ?']
        params: list[object] = [name]
        if federation:
            conditions.append('"Federation" = ?')
            params.append(federation)
        if date:
            conditions.append('"Date" = ?')
            params.append(str(date))

        where = " AND ".join(conditions)
        entries = self._fetch_entries(where=where, params=params, limit=10000)
        if not entries:
            return None

        first = entries[0]
        return Meet(
            name=first.meet_name,
            date=first.date,
            federation=first.federation,
            country=first.meet_country,
            state=first.meet_state,
            town=first.meet_town,
            entries=entries,
        )

    def meets(
        self,
        federation: str | None = None,
        country: str | None = None,
        year: int | None = None,
        limit: int = 100,
        offset: int = 0,
    ) -> list[Meet]:
        """Query meets with filters."""
        conditions: list[str] = []
        params: list[object] = []

        if federation:
            conditions.append('"Federation" = ?')
            params.append(federation)
        if country:
            conditions.append('"MeetCountry" = ?')
            params.append(country)
        if year:
            conditions.append('YEAR("Date") = ?')
            params.append(year)

        where = " AND ".join(conditions) if conditions else ""

        # Get distinct meets first
        sql = 'SELECT DISTINCT "MeetName", "Date", "Federation" FROM entries'
        if where:
            sql += f" WHERE {where}"
        sql += f' ORDER BY "Date" DESC LIMIT {limit} OFFSET {offset}'

        con = self._connect()
        try:
            result = con.execute(sql, params)
            meet_keys = result.fetchall()
        finally:
            con.close()

        meets: list[Meet] = []
        for meet_name, meet_date, meet_fed in meet_keys:
            m = self.meet(meet_name, federation=meet_fed, date=meet_date)
            if m:
                meets.append(m)
        return meets

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
        """Query rankings with filters."""
        conditions: list[str] = []
        params: list[object] = []

        if sex:
            conditions.append('"Sex" = ?')
            params.append(sex.value)
        if equipment:
            conditions.append('"Equipment" = ?')
            params.append(equipment.value)
        if event:
            conditions.append('"Event" = ?')
            params.append(event.value)
        if weight_class:
            conditions.append('"WeightClassKg" = ?')
            params.append(weight_class)
        if federation:
            conditions.append('"Federation" = ?')
            params.append(federation)
        if year:
            conditions.append('YEAR("Date") = ?')
            params.append(year)
        if tested is not None:
            conditions.append('"Tested" = ?')
            params.append("Yes" if tested else "")

        # Map Python field name to SQL column for ordering
        order_col = _COLUMN_MAP.get(order_by, order_by)
        where = " AND ".join(conditions) if conditions else ""

        return self._fetch_entries(
            where=where,
            params=params,
            order_by=f'"{order_col}" DESC NULLS LAST',
            limit=limit,
            offset=offset,
        )

    def federations(self) -> list[str]:
        """List all federations in the dataset."""
        con = self._connect()
        try:
            result = con.execute('SELECT DISTINCT "Federation" FROM entries ORDER BY "Federation"')
            return [row[0] for row in result.fetchall()]
        finally:
            con.close()

    def query(self, sql: str, params: dict[str, object] | None = None) -> list[dict[str, object]]:
        """Raw SQL escape hatch. Returns list of dicts."""
        con = self._connect()
        try:
            result = con.execute(sql, list(params.values())) if params else con.execute(sql)
            cols = [desc[0] for desc in result.description]
            return [dict(zip(cols, row, strict=True)) for row in result.fetchall()]
        finally:
            con.close()

    def to_polars(self, sql: str, params: dict[str, object] | None = None) -> Any:
        """Execute SQL and return a Polars DataFrame. Requires polars."""
        try:
            import polars as pl
        except ImportError as err:
            raise ImportError(
                "polars is required for to_polars(). Install with: pip install opl-py[analytics]"
            ) from err
        con = self._connect()
        try:
            result = con.execute(sql, list(params.values())) if params else con.execute(sql)
            return pl.from_arrow(result.fetch_arrow_table())
        finally:
            con.close()

    def stats(self) -> dict[str, str | int]:
        """Return DB metadata: row count, last updated, etc."""
        info = get_db_info(self._db_path)
        return dict(info)
