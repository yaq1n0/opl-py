# CLI

The `opl` command-line tool manages the local DuckDB database that powers the SDK.

## Commands

### `opl init`

Download the latest OPL CSV (~250 MB) and create the local DuckDB database.

```bash
opl init
opl init --db-path /path/to/custom.duckdb
```

### `opl update`

Re-download the latest OPL CSV and refresh the existing database.

```bash
opl update
opl update --db-path /path/to/custom.duckdb
```

### `opl info`

Print database location, row count, dataset date, and source URL.

```bash
opl info
opl info --db-path /path/to/custom.duckdb
```

## Database Location

By default the database is stored in your OS user data directory, determined by `platformdirs.user_data_dir("opl-py")`:

| OS      | Default path                                      |
| ------- | ------------------------------------------------- |
| macOS   | `~/Library/Application Support/opl-py/opl.duckdb` |
| Linux   | `~/.local/share/opl-py/opl.duckdb`                |
| Windows | `C:\Users\<you>\AppData\Local\opl-py\opl.duckdb`  |

All commands accept `--db-path` to override this location.
