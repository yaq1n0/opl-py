#!/usr/bin/env python
"""E2E demo: Core SDK functionality against a live DuckDB.

Usage:
    python -m demo.demo_core [--db-path /path/to/opl.duckdb]

Demonstrates:
    - OPL client stats
    - Lifter lookup and history
    - Lifter search
    - Rankings with filters
    - Meet lookup and listing
    - Federation listing
    - Raw SQL queries
"""

import sys
from pathlib import Path

import opl


def show_stats(client: opl.OPL) -> None:
    print("=== DB Stats ===")
    stats = client.stats()
    for k, v in stats.items():
        print(f"  {k}: {v}")


def show_lifter(client: opl.OPL) -> None:
    print("=== Lifter Lookup: Ray Williams #1 ===")
    lifter = client.lifter("Ray Williams #1")
    if lifter:
        print(f"  Name: {lifter.name}")
        print(f"  Competitions: {lifter.competition_count}")
        print(f"  Best total: {lifter.best_total_kg} kg")
        print(f"  Best squat: {lifter.best_squat_kg} kg")
        print(f"  Best bench: {lifter.best_bench_kg} kg")
        print(f"  Best deadlift: {lifter.best_deadlift_kg} kg")
        print("  Recent history:")
        for entry in lifter.history()[-5:]:
            print(
                f"    {entry.date}: {entry.total_kg}kg "
                f"@ {entry.bodyweight_kg}kg ({entry.federation})"
            )
    else:
        print("  Not found (try running with the full OPL database)")


def show_search(client: opl.OPL) -> None:
    print('=== Search: "Haack" ===')
    results = client.search_lifters("Haack", limit=5)
    for r in results:
        print(f"  {r.name} — {r.competition_count} meets, best total: {r.best_total_kg}kg")
    if not results:
        print("  No results found")


def show_rankings(client: opl.OPL) -> None:
    print("=== Top 10 Raw Male SBD by Total ===")
    top = client.rankings(
        sex=opl.Sex.MALE,
        equipment=opl.Equipment.RAW,
        event=opl.Event.SBD,
        order_by="total_kg",
        limit=10,
    )
    for i, e in enumerate(top, 1):
        print(f"  {i}. {e.name} — {e.total_kg}kg @ {e.bodyweight_kg}kg ({e.date}, {e.federation})")


def show_meets(client: opl.OPL) -> None:
    print("=== Meet Lookup ===")
    meets = client.meets(federation="USAPL", limit=3)
    for m in meets:
        print(f"  {m.name} ({m.date}, {m.federation}) — {len(m.entries)} entries")


def show_federations(client: opl.OPL) -> None:
    feds = client.federations()
    print(f"=== Federations: {len(feds)} total ===")
    print(f"  First 10: {feds[:10]}")


def show_raw_sql(client: opl.OPL) -> None:
    print("=== Raw SQL: Top federations by entry count ===")
    results = client.query(
        'SELECT "Federation", COUNT(*) as n FROM entries '
        'GROUP BY "Federation" ORDER BY n DESC LIMIT 10'
    )
    for r in results:
        print(f"  {r['Federation']}: {r['n']:,}")


def main(db_path: Path | None = None) -> None:
    client = opl.OPL(db_path=db_path)

    show_stats(client)
    print()
    show_lifter(client)
    print()
    show_search(client)
    print()
    show_rankings(client)
    print()
    show_meets(client)
    print()
    show_federations(client)
    print()
    show_raw_sql(client)
    print()
    print("SUCCESS: All core SDK operations completed.")


if __name__ == "__main__":
    custom_path = None
    if "--db-path" in sys.argv:
        idx = sys.argv.index("--db-path")
        custom_path = Path(sys.argv[idx + 1])
    main(custom_path)
