"""Tests for the asyncio facade at :mod:`tqdb.aio`.

The wrapper has no Rust-side counterpart — the value-add is purely the
``run_in_executor`` dispatch + the right method shapes. Tests focus on:

1. **API shape**: every wrapped method round-trips a value the sync API would
   return.
2. **Concurrency**: many ``await`` calls genuinely run in parallel without
   blocking the event loop. The Rust code already releases the GIL inside
   each PyO3 method (``py.allow_threads``), so the only risk is the wrapper
   accidentally serializing things.
3. **Lifecycle**: ``async with`` works; ``close()`` shuts the auto-created
   executor.
"""

from __future__ import annotations

import asyncio
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pytest

from tqdb.aio import AsyncDatabase


@pytest.fixture
def tmp_db_path():
    with tempfile.TemporaryDirectory() as tmp:
        yield tmp


def _vec(d: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.standard_normal(d).astype(np.float32)


# ── basic API shape ─────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_open_insert_search_close(tmp_db_path):
    d = 16
    db = await AsyncDatabase.open(tmp_db_path, dimension=d, bits=2)
    try:
        await db.insert("a", _vec(d, 1), document="alpha")
        await db.insert("b", _vec(d, 2), document="beta")
        results = await db.search(_vec(d, 1), top_k=2)
        ids = {r["id"] for r in results}
        assert "a" in ids
    finally:
        await db.close()


@pytest.mark.asyncio
async def test_async_context_manager(tmp_db_path):
    async with await AsyncDatabase.open(tmp_db_path, dimension=8, bits=2) as db:
        await db.insert("only", _vec(8, 7))
        assert "only" in db
        assert len(db) == 1
    # Executor should be shut after exit; calling close again must be safe.
    # (Don't error on double-close because cleanup ordering in tests is
    # sometimes unpredictable.)


@pytest.mark.asyncio
async def test_batch_and_filter(tmp_db_path):
    d = 8
    async with await AsyncDatabase.open(tmp_db_path, dimension=d, bits=2) as db:
        ids = [f"d{i}" for i in range(10)]
        vecs = np.stack([_vec(d, i) for i in range(10)])
        metas = [{"bucket": i % 3} for i in range(10)]
        await db.insert_batch(ids, vecs, metas, None, "insert")

        bucket0 = await db.list_ids(where_filter={"bucket": 0})
        assert len(bucket0) == 4  # 0, 3, 6, 9

        deleted = await db.delete_batch(where_filter={"bucket": 0})
        assert deleted == 4
        assert await db.count() == 6


@pytest.mark.asyncio
async def test_query_batch(tmp_db_path):
    d = 8
    async with await AsyncDatabase.open(tmp_db_path, dimension=d, bits=2) as db:
        for i in range(5):
            await db.insert(f"d{i}", _vec(d, i))
        emb = np.stack([_vec(d, 0), _vec(d, 1)])
        batch = await db.query(emb, n_results=3)
        assert len(batch) == 2
        assert all(len(r) <= 3 for r in batch)


@pytest.mark.asyncio
async def test_get_and_metadata_update(tmp_db_path):
    d = 8
    async with await AsyncDatabase.open(tmp_db_path, dimension=d, bits=2) as db:
        await db.insert("x", _vec(d, 1), {"v": 1}, "doc-x")
        rec = await db.get("x")
        assert rec["metadata"]["v"] == 1
        await db.update_metadata("x", {"v": 2})
        rec = await db.get("x")
        assert rec["metadata"]["v"] == 2


# ── concurrency ─────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_concurrent_searches_dont_serialize(tmp_db_path):
    """Concurrent awaits should fan out through the executor.

    Use a tiny blocking fake DB instead of timing real Rust searches. The real
    search path can be sub-millisecond on small corpora, making ratio checks
    mostly scheduler noise rather than a test of the Python wrapper.
    """

    class _SleepyDb:
        def search(self, query, top_k):
            time.sleep(0.02)
            return [{"id": str(query), "score": 1.0}]

        def close(self):
            return None

    pool = ThreadPoolExecutor(max_workers=8)
    db = AsyncDatabase(_SleepyDb(), executor=pool, owns_executor=True)
    try:
        n_tasks = 16
        t0 = time.perf_counter()
        results = await asyncio.gather(
            *(db.search(i, top_k=1) for i in range(n_tasks))
        )
        elapsed = time.perf_counter() - t0

        assert len(results) == n_tasks
        assert elapsed < 0.20, (
            f"{n_tasks} executor-backed searches took {elapsed:.3f}s; "
            "the wrapper appears to serialize."
        )
    finally:
        await db.close()


@pytest.mark.asyncio
async def test_event_loop_remains_responsive_during_search(tmp_db_path):
    """While a long-ish search runs in the executor, an unrelated coroutine
    must still get scheduling slices."""
    d = 32
    async with await AsyncDatabase.open(tmp_db_path, dimension=d, bits=4) as db:
        ids = [f"d{i}" for i in range(1_000)]
        vecs = np.stack([_vec(d, i) for i in range(1_000)])
        await db.insert_batch(ids, vecs, None, None, "insert")

        ticks: list[float] = []

        async def heartbeat():
            for _ in range(20):
                ticks.append(time.perf_counter())
                await asyncio.sleep(0.005)

        async def workload():
            for i in range(20):
                await db.search(_vec(d, i), top_k=5)

        await asyncio.gather(heartbeat(), workload())

        # If the loop blocked on workload(), heartbeat would skip ticks.
        # Verify monotonic spacing within reasonable bounds.
        spacings = [b - a for a, b in zip(ticks, ticks[1:])]
        assert max(spacings) < 0.5, (
            f"event loop was blocked: max heartbeat gap {max(spacings):.3f}s"
        )


# ── lifecycle ───────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_external_executor_is_not_shutdown(tmp_db_path):
    """When the caller supplies an executor, AsyncDatabase must not close it
    on `close()`. The caller manages its own lifecycle."""
    from concurrent.futures import ThreadPoolExecutor

    pool = ThreadPoolExecutor(max_workers=2)
    db = await AsyncDatabase.open(tmp_db_path, dimension=8, bits=2, executor=pool)
    await db.insert("x", _vec(8, 1))
    await db.close()
    # External pool must still be usable after db.close().
    fut = pool.submit(lambda: 42)
    assert fut.result(timeout=2) == 42
    pool.shutdown()


@pytest.mark.asyncio
async def test_sync_escape_hatch(tmp_db_path):
    async with await AsyncDatabase.open(tmp_db_path, dimension=8, bits=2) as db:
        await db.insert("x", _vec(8, 1))
        # Cheap sync ops can use the underlying Database directly.
        assert "x" in db.sync
        assert len(db.sync) == 1
