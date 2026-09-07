"""Refuse to release a PATCH version whose changelog section has a Migration heading.

A change that needs a Migration note is, by definition, one that existing databases
cannot survive untouched. Semantic versioning says that is not a patch. v0.8.4 did
exactly this: it moved the dense rotation matrix from f32 to bf16, documented the
migration correctly, and shipped as z+1 — so users upgraded without reading release
notes and hit `io error: unexpected end of file` on every store they had (issue #110).

Usage:
    python scripts/ci/check_migration_bump.py 0.9.1
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

CHANGELOG = Path(__file__).resolve().parents[2] / "CHANGELOG.md"
SECTION_RE = re.compile(r"^## \[(?P<version>\d+\.\d+\.\d+)\]", re.MULTILINE)
MIGRATION_RE = re.compile(r"^###+\s+Migration\b", re.MULTILINE | re.IGNORECASE)
PRERELEASE_MARKERS = ("alpha", "beta", "rc")


def fail(message: str) -> int:
    """Emit a GitHub Actions error annotation plus plain text for local runs."""
    print(f"::error::{message}")
    print(message.replace("%0A", "\n"), file=sys.stderr)
    return 1


def sections(text: str) -> list[tuple[str, str]]:
    """Released version sections, newest first, as (version, body)."""
    matches = list(SECTION_RE.finditer(text))
    out = []
    for i, m in enumerate(matches):
        end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        out.append((m.group("version"), text[m.end() : end]))
    return out


def main(argv: list[str]) -> int:
    if len(argv) != 2:
        print(f"usage: {argv[0]} <version>", file=sys.stderr)
        return 2
    version = argv[1]

    if any(marker in version for marker in PRERELEASE_MARKERS):
        print(f"pre-release {version} — migration/bump check skipped")
        return 0

    if not CHANGELOG.exists():
        return fail(f"{CHANGELOG} is missing")
    try:
        text = CHANGELOG.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError) as exc:
        return fail(f"{CHANGELOG} could not be read: {exc}")

    parsed = sections(text)
    index = next((i for i, (v, _) in enumerate(parsed) if v == version), None)
    if index is None:
        return fail(
            f"CHANGELOG.md has no section for {version}. Add one before tagging."
        )

    _, body = parsed[index]
    if not MIGRATION_RE.search(body):
        print(f"{version}: no Migration section — nothing to check")
        return 0

    if index + 1 >= len(parsed):
        print(f"{version}: Migration section present, no earlier release to compare")
        return 0

    previous = parsed[index + 1][0]
    cur = tuple(int(p) for p in version.split("."))
    prev = tuple(int(p) for p in previous.split("."))
    if cur[:2] == prev[:2]:
        return fail(
            f"{version} is a PATCH bump over {previous} but its changelog section has a "
            f"Migration heading.%0AA release that existing databases cannot survive "
            f"untouched is not a patch — bump the MINOR version instead, so users see it "
            f"in the version they upgrade to.%0A(See issue #110: v0.8.4 shipped a format "
            f"change as a patch and every pre-0.8.4 store failed to open.)"
        )

    print(f"{version}: Migration section present, and it is not a patch bump over {previous}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
