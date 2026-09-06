"""Fail a release that has no benchmark history entry for the version being tagged.

`benchmarks/perf_history.json` is appended by a local `TQDB_TRACK=1` benchmark run,
so nothing enforces that it happened. v0.8.4 shipped with no entry at all, leaving a
four-month hole in the trend data that nobody noticed until the next release. This
gate makes the release fail instead.

Usage:
    python scripts/ci/check_perf_history.py 0.9.0
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

HISTORY = Path(__file__).resolve().parents[2] / "benchmarks" / "perf_history.json"
PRERELEASE_MARKERS = ("alpha", "beta", "rc")


def main(argv: list[str]) -> int:
    if len(argv) != 2:
        print(f"usage: {argv[0]} <version>", file=sys.stderr)
        return 2
    version = argv[1]

    if any(marker in version for marker in PRERELEASE_MARKERS):
        print(f"pre-release {version} — perf history entry not required")
        return 0

    if not HISTORY.exists():
        print(f"::error::{HISTORY} is missing", file=sys.stderr)
        return 1

    history = json.loads(HISTORY.read_text(encoding="utf-8"))
    entries = [e for e in history if e.get("version") == version]
    if entries:
        stamps = ", ".join(sorted(e.get("timestamp", "?")[:10] for e in entries))
        print(f"{version}: {len(entries)} benchmark history entry(ies) on record ({stamps})")
        return 0

    recent = sorted({e.get("version", "?") for e in history})[-5:]
    print(
        f"::error::no benchmark history entry for {version}. Run\n"
        f"  TQDB_TRACK=1 python benchmarks/paper_recall_bench.py --update-readme --track\n"
        f"and land the updated benchmarks/perf_history.json before tagging.\n"
        f"Versions on record: {', '.join(recent)}",
        file=sys.stderr,
    )
    return 1


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
