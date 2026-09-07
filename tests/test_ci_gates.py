"""The release gates in scripts/ci/, exercised against this repository's own history.

These run before a tag is turned into a GitHub release and a PyPI upload, so their
failure modes matter as much as their success ones: a gate that crashes, or that
passes something it should stop, is worse than no gate.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
PERF_GATE = REPO / "scripts" / "ci" / "check_perf_history.py"
MIGRATION_GATE = REPO / "scripts" / "ci" / "check_migration_bump.py"


def run(script: Path, *args: str, cwd: Path | None = None) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, str(script), *args],
        capture_output=True,
        text=True,
        cwd=str(cwd or REPO),
    )


# ---------------------------------------------------------------------------
# check_perf_history.py — a release must carry a tracked benchmark run
# ---------------------------------------------------------------------------


def test_perf_history_accepts_a_version_on_record():
    assert run(PERF_GATE, "0.9.0").returncode == 0


def test_perf_history_rejects_a_version_with_no_entry():
    """0.8.4 really shipped without one — that is why this gate exists."""
    result = run(PERF_GATE, "0.8.4")
    assert result.returncode == 1
    assert "TQDB_TRACK=1" in result.stdout, "must print the command that fixes it"


def test_perf_history_skips_prereleases():
    assert run(PERF_GATE, "0.9.1rc1").returncode == 0


def test_perf_history_needs_a_version_argument():
    assert run(PERF_GATE).returncode == 2


def test_perf_history_reports_a_broken_file_cleanly(tmp_path):
    """A corrupt history must annotate, not traceback."""
    (tmp_path / "benchmarks").mkdir()
    (tmp_path / "benchmarks" / "perf_history.json").write_text("{not json", encoding="utf-8")
    (tmp_path / "scripts" / "ci").mkdir(parents=True)
    script = tmp_path / "scripts" / "ci" / "check_perf_history.py"
    script.write_text(PERF_GATE.read_text(encoding="utf-8"), encoding="utf-8")

    result = run(script, "1.0.0", cwd=tmp_path)
    assert result.returncode == 1
    assert "::error::" in result.stdout
    assert "Traceback" not in result.stderr


# ---------------------------------------------------------------------------
# check_migration_bump.py — a migration cannot ride in on a patch bump
# ---------------------------------------------------------------------------


def test_migration_gate_catches_the_0_8_4_incident():
    """0.8.4 documented a Migration and shipped as a patch; every pre-0.8.4 store
    then failed to open (#110). The gate must catch it in the real changelog."""
    result = run(MIGRATION_GATE, "0.8.4")
    assert result.returncode == 1
    assert "PATCH bump over 0.8.3" in result.stdout
    assert "#110" in result.stdout


def test_migration_gate_passes_releases_without_a_migration():
    for version in ("0.8.5", "0.9.0"):
        assert run(MIGRATION_GATE, version).returncode == 0, version


def test_migration_gate_skips_prereleases():
    assert run(MIGRATION_GATE, "0.9.1rc1").returncode == 0


def test_migration_gate_rejects_an_unknown_version():
    result = run(MIGRATION_GATE, "42.0.0")
    assert result.returncode == 1
    assert "no section for 42.0.0" in result.stdout
