import shutil
import tempfile
from pathlib import Path

import click
from rich.console import Console

from opl.core.db import default_db_path, get_db_info, ingest_csv, update_db
from opl.core.download import download_and_extract

console = Console()


@click.group()
def main() -> None:
    """opl — OpenPowerlifting data CLI."""


@main.command()
@click.option("--db-path", type=click.Path(), default=None, help="Custom database path.")
def init(db_path: str | None) -> None:
    """Download latest OPL data and create local database."""
    dest = Path(db_path) if db_path else default_db_path()

    if dest.exists():
        console.print(
            f"[yellow]Database already exists at {dest}. Use `opl update` to refresh.[/yellow]"
        )
        return

    console.print("[bold]Downloading OpenPowerlifting data...[/bold]")
    tmp_dir = Path(tempfile.mkdtemp())
    try:
        csv_path = download_and_extract(tmp_dir)
        console.print("[bold]Ingesting into DuckDB...[/bold]")
        db = ingest_csv(csv_path, dest)
        console.print(f"[green]Database created at {db}[/green]")

        info = get_db_info(dest)
        console.print(f"  Rows: {info.get('row_count', 'unknown'):,}")
        console.print(f"  Source date: {info.get('csv_date', 'unknown')}")
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)


@main.command()
@click.option("--db-path", type=click.Path(), default=None, help="Custom database path.")
def update(db_path: str | None) -> None:
    """Re-download OPL data and update the local database."""
    dest = Path(db_path) if db_path else default_db_path()

    if not dest.exists():
        console.print("[red]Database not found. Run `opl init` first.[/red]")
        return

    console.print("[bold]Downloading latest OpenPowerlifting data...[/bold]")
    tmp_dir = Path(tempfile.mkdtemp())
    try:
        csv_path = download_and_extract(tmp_dir)
        console.print("[bold]Updating DuckDB...[/bold]")
        update_db(csv_path, dest)
        console.print("[green]Database updated successfully.[/green]")

        info = get_db_info(dest)
        console.print(f"  Rows: {info.get('row_count', 'unknown'):,}")
        console.print(f"  Source date: {info.get('csv_date', 'unknown')}")
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)


@main.command()
@click.option("--db-path", type=click.Path(), default=None, help="Custom database path.")
def info(db_path: str | None) -> None:
    """Print database info."""
    dest = Path(db_path) if db_path else default_db_path()

    if not dest.exists():
        console.print(f"[red]Database not found at {dest}. Run `opl init` first.[/red]")
        return

    info_data = get_db_info(dest)
    console.print("[bold]OPL Database Info[/bold]")
    console.print(f"  Location:    {dest}")
    console.print(f"  Rows:        {info_data.get('row_count', 'unknown'):,}")
    console.print(f"  Last updated: {info_data.get('last_updated', 'unknown')}")
    console.print(f"  Source date:  {info_data.get('csv_date', 'unknown')}")
    console.print(f"  Source URL:   {info_data.get('source_url', 'unknown')}")


@main.command()
@click.option("--db-path", type=click.Path(), default=None, help="Custom database path.")
@click.option(
    "--output-dir",
    type=click.Path(),
    default="pretrained",
    show_default=True,
    help="Root directory for saved models.",
)
@click.option(
    "--min-meets",
    type=int,
    default=4,
    show_default=True,
    help="Minimum meets a lifter must have to be included.",
)
@click.option(
    "--approach",
    type=str,
    default=None,
    help="Train only a specific approach (default: all).",
)
@click.option(
    "--limit",
    type=int,
    default=None,
    help="Limit number of training lifters (useful for testing).",
)
def train(
    db_path: str | None,
    output_dir: str,
    min_meets: int,
    approach: str | None,
    limit: int | None,
) -> None:
    """Train trajectory models on the full OPL dataset."""
    from opl.analytics.scripts.train import train

    train(
        db_path=Path(db_path) if db_path else None,
        output_dir=Path(output_dir),
        min_meets=min_meets,
        approach=approach,
        limit=limit,
    )


@main.group()
def demo() -> None:
    """Run E2E demos (require an initialised database)."""


@demo.command("core")
@click.option("--db-path", type=click.Path(), default=None, help="Custom database path.")
def demo_core(db_path: str | None) -> None:
    """Run core SDK demo."""
    from demo.demo_core import main as run

    run(Path(db_path) if db_path else None)


@demo.command("analytics")
@click.option("--db-path", type=click.Path(), default=None, help="Custom database path.")
def demo_analytics(db_path: str | None) -> None:
    """Run analytics demo."""
    from demo.demo_analytics import main as run

    run(Path(db_path) if db_path else None)


@demo.command("all")
@click.option("--db-path", type=click.Path(), default=None, help="Custom database path.")
@click.option(
    "--include-init", is_flag=True, default=False, help="Include init demo (~160 MB download)."
)
def demo_all(db_path: str | None, include_init: bool) -> None:
    """Run all demos in sequence."""
    from demo.demo_analytics import main as run_analytics
    from demo.demo_core import main as run_core

    db = Path(db_path) if db_path else None
    separator = "=" * 60

    if include_init:
        from demo.demo_init import main as run_init

        print(separator)
        print("  RUNNING: demo_init")
        print(separator)
        print()
        run_init(db)
        print()

    for name, run in [("demo_core", run_core), ("demo_analytics", run_analytics)]:
        print(separator)
        print(f"  RUNNING: {name}")
        print(separator)
        print()
        run(db)
        print()

    print(separator)
    print("  ALL DEMOS PASSED")
    print(separator)
