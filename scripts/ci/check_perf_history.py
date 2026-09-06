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


def fail(message: str) -> int:
    """Emit a GitHub Actions error annotation, plus plain text for local runs.

    `%0A` is how an annotation encodes a newline; the stderr copy converts it back
    so the message stays readable outside CI.
    """
    print(f"::error::{message}")
    print(message.replace("%0A", "\n"), file=sys.stderr)
    return 1


def main(argv: list[str]) -> int:
    if len(argv) != 2:
        print(f"usage: {argv[0]} <version>", file=sys.stderr)
        return 2
    version = argv[1]

    if any(marker in version for marker in PRERELEASE_MARKERS):
        print(f"pre-release {version} — perf history entry not required")
        return 0

    # This runs as a release gate, so every failure mode has to arrive as a readable
    # annotation rather than a traceback buried in the job log.
    if not HISTORY.exists():
        return fail(f"{HISTORY} is missing")
    try:
        history = json.loads(HISTORY.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError, UnicodeDecodeError) as exc:
        return fail(f"{HISTORY} could not be read: {exc}")
    if not isinstance(history, list):
        return fail(
            f"{HISTORY} should hold a list of benchmark runs, "
            f"found {type(history).__name__}"
        )

    entries = [e for e in history if isinstance(e, dict) and e.get("version") == version]
    if entries:
        stamps = ", ".join(sorted(str(e.get("timestamp", "?"))[:10] for e in entries))
        print(f"{version}: {len(entries)} benchmark history entry(ies) on record ({stamps})")
        return 0

    recent = sorted({str(e.get("version", "?")) for e in history if isinstance(e, dict)})[-5:]
    return fail(
        f"no benchmark history entry for {version}. Run%0A"
        f"  TQDB_TRACK=1 python benchmarks/paper_recall_bench.py --update-readme --track%0A"
        f"and land the updated benchmarks/perf_history.json before tagging.%0A"
        f"Versions on record: {', '.join(recent) or '(none)'}"
    )


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
