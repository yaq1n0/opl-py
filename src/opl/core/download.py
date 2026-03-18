import tempfile
import zipfile
from pathlib import Path

import httpx
from rich.progress import (
    BarColumn,
    DownloadColumn,
    Progress,
    TextColumn,
    TransferSpeedColumn,
)

OPL_DATA_URL = "https://openpowerlifting.gitlab.io/opl-csv/files/openpowerlifting-latest.zip"


def download_and_extract(target_dir: Path) -> Path:
    """Download the OPL bulk CSV ZIP and extract it. Returns path to the CSV file."""
    target_dir.mkdir(parents=True, exist_ok=True)

    with tempfile.NamedTemporaryFile(suffix=".zip", delete=False) as tmp:
        zip_path = Path(tmp.name)

    try:
        _download_zip(zip_path)
        csv_path = _extract_csv(zip_path, target_dir)
    finally:
        zip_path.unlink(missing_ok=True)

    return csv_path


def _download_zip(dest: Path) -> None:
    """Download the OPL ZIP file with a progress bar."""
    with (
        httpx.stream("GET", OPL_DATA_URL, follow_redirects=True, timeout=300) as resp,
        Progress(
            TextColumn("[bold blue]Downloading OPL data..."),
            BarColumn(),
            DownloadColumn(),
            TransferSpeedColumn(),
        ) as progress,
    ):
        resp.raise_for_status()
        total = int(resp.headers.get("content-length", 0))
        task = progress.add_task("download", total=total or None)

        with open(dest, "wb") as f:
            for chunk in resp.iter_bytes(chunk_size=1024 * 64):
                f.write(chunk)
                progress.update(task, advance=len(chunk))


def _extract_csv(zip_path: Path, target_dir: Path) -> Path:
    """Extract the CSV from the OPL ZIP. Returns path to the extracted CSV."""
    with zipfile.ZipFile(zip_path) as zf:
        csv_names = [n for n in zf.namelist() if n.endswith(".csv")]
        if not csv_names:
            raise RuntimeError("No CSV file found in the downloaded ZIP")
        csv_name = csv_names[0]
        zf.extract(csv_name, target_dir)
        return target_dir / csv_name
