#!/usr/bin/env python
"""Run all E2E demos in sequence.

Usage:
    python -m demo.demo_all [--db-path /path/to/opl.duckdb]
    python -m demo.demo_all --include-init [--db-path /tmp/demo.duckdb]

By default skips demo_init (which downloads ~160MB). Pass --include-init to run it.
"""

import sys
from pathlib import Path


def main() -> None:
    include_init = "--include-init" in sys.argv
    db_path = None
    if "--db-path" in sys.argv:
        idx = sys.argv.index("--db-path")
        db_path = Path(sys.argv[idx + 1])

    demos = []

    if include_init:
        from demo.demo_init import main as init_main

        demos.append(("demo_init", lambda: init_main(db_path)))

    from demo.demo_analytics import main as analytics_main
    from demo.demo_core import main as core_main

    demos.append(("demo_core", lambda: core_main(db_path)))
    demos.append(("demo_analytics", lambda: analytics_main(db_path)))

    separator = "=" * 60

    for name, run in demos:
        print(separator)
        print(f"  RUNNING: {name}")
        print(separator)
        print()
        run()
        print()

    print(separator)
    print("  ALL DEMOS PASSED")
    print(separator)


if __name__ == "__main__":
    main()
