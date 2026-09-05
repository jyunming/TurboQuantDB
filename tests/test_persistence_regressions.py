"""
Persistence regression tests for delete -> reinsert flows.
"""

from __future__ import annotations

import numpy as np

from tqdb import Database


def _vec(d: int = 16) -> np.ndarray:
    return np.arange(d, dtype=np.float32)


def test_insert_persists_across_reopen_smoke(tmp_path):
    """
    Control case: a straightforward insert must survive close/reopen.
    """
    path = str(tmp_path / "db")
    db = Database.open(path, 16, bits=4, metric="ip")
    db.insert("x", _vec())
    assert db.count() == 1
    db.close()

    db2 = Database.open(path, 16)
    assert db2.count() == 1
    assert db2.get("x") is not None


def test_delete_then_reinsert_should_persist_across_reopen(tmp_path):
    """
    Minimal repro:
    1) insert/upsert ID
    2) delete ID
    3) reinsert same ID
    4) close + reopen

    Expected: ID is present.
    Current behavior: ID is missing after reopen.
    """
    path = str(tmp_path / "db")
    db = Database.open(path, 16, bits=4, metric="ip")
    v = _vec()

    db.upsert("x", v, metadata={"phase": 1}, document="first")
    assert db.count() == 1
    assert db.get("x") is not None

    deleted = db.delete("x")
    assert deleted is True
    assert db.count() == 0

    db.upsert("x", v, metadata={"phase": 2}, document="second")
    assert db.count() == 1
    assert db.get("x") is not None

    db.close()

    reopened = Database.open(path, 16)
    assert reopened.count() == 1
    got = reopened.get("x")
    assert got is not None
    assert got["metadata"]["phase"] == 2
    assert got["document"] == "second"


# ---------------------------------------------------------------------------
# Issue #102 — close() must release the memory mapping deterministically, and a
# truncated live_codes.bin must be reported as a catchable error, not a Rust
# panic (PanicException inherits BaseException, so `except Exception` misses it).
# ---------------------------------------------------------------------------


def _make_store(tmp_path, name="store"):
    path = tmp_path / name
    db = Database.open(str(path), 16, bits=4, metric="ip")
    db.insert("a", _vec(), {"k": "v"})
    db.flush()
    return path, db


def _can_truncate(path):
    """True when live_codes.bin can be resized — i.e. no mapping is left open."""
    try:
        with open(path / "live_codes.bin", "wb") as f:
            f.write(b"")
        return True
    except OSError:
        return False


def test_close_releases_memory_mapping(tmp_path):
    path, db = _make_store(tmp_path)
    db.close()
    assert _can_truncate(path), (
        "close() must release the mapping; on Windows a live mapping fails "
        "resize with [Errno 22] / os error 1224"
    )


def test_close_is_idempotent(tmp_path):
    _, db = _make_store(tmp_path)
    db.close()
    db.close()  # no-op, must not raise


def test_operations_after_close_raise_catchable_error(tmp_path):
    _, db = _make_store(tmp_path)
    db.close()
    for op, args in [
        (db.insert, ("b", _vec())),
        (db.get, ("a",)),
        (db.count, ()),
        (db.search, (_vec(), 1)),
    ]:
        try:
            op(*args)
        except Exception as e:
            assert "closed" in str(e), f"{op.__name__}: unexpected error {e}"
        else:
            raise AssertionError(f"{op.__name__} should raise after close()")


def test_context_manager_closes_on_exit(tmp_path):
    path = tmp_path / "ctx"
    with Database.open(str(path), 16, bits=4, metric="ip") as db:
        db.insert("a", _vec())
        assert db.count() == 1
    assert _can_truncate(path), "`with` block must close the database on exit"


def test_context_manager_does_not_swallow_exceptions(tmp_path):
    path = tmp_path / "ctx_raise"
    try:
        with Database.open(str(path), 16, bits=4, metric="ip") as db:
            db.insert("a", _vec())
            raise ValueError("boom")
    except ValueError:
        pass
    else:
        raise AssertionError("__exit__ must not suppress exceptions")


def test_truncated_live_codes_reports_catchable_error(tmp_path):
    """An empty live_codes.bin used to panic on first search (uncatchable)."""
    path, db = _make_store(tmp_path, "damaged")
    db.close()
    (path / "live_codes.bin").write_bytes(b"")

    try:
        db2 = Database.open(str(path), 16, bits=4, metric="ip")
    except Exception as e:  # noqa: BLE001 — the point is that it *is* catchable
        assert "corrupt store" in str(e)
        assert "live_codes.bin" in str(e)
        return
    # If a future version chooses to open a damaged store, the first query must
    # still raise a normal exception rather than a PanicException.
    try:
        db2.search(_vec(), 1)
    except Exception as e:  # noqa: BLE001
        assert "panic" not in type(e).__name__.lower()
    else:
        raise AssertionError("damaged store must be reported at open or on search")


def test_close_releases_rerank_sidecar_mapping(tmp_path):
    """The reporter's config: rerank=True also maps live_vectors.bin."""
    path = tmp_path / "rerank_store"
    db = Database.open(str(path), 16, bits=8, metric="ip", normalize=True, rerank=True)
    db.insert("a", _vec(), {"k": "v"})
    db.flush()
    db.close()
    for name in ("live_codes.bin", "live_vectors.bin"):
        f = path / name
        if not f.exists():
            continue
        with open(f, "wb") as fh:
            fh.write(b"")  # raises OSError if a mapping is still open
