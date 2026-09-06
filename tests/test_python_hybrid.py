"""Python-boundary validation tests for the hybrid search kwarg.

The Rust side (`parse_hybrid` / `parse_hybrid_batch` in src/python/mod.rs) must
turn malformed input into clean `ValueError`s, never a panic. This file covers
the realistic mistakes a user can make from Python.
"""

from __future__ import annotations

import tempfile

import numpy as np
import pytest

from tqdb import Database


@pytest.fixture
def db():
    """A tiny database with one vector + doc, just enough for the boundary tests
    to make `search`/`query` calls that actually touch the hybrid parser."""
    with tempfile.TemporaryDirectory() as tmp:
        d = 8
        instance = Database.open(tmp, dimension=d, bits=2)
        instance.insert("d0", np.zeros(d, dtype=np.float32), document="hello world")
        yield instance
        instance.close()


def _q(d: int = 8) -> np.ndarray:
    return np.zeros(d, dtype=np.float32)


# ── parse_hybrid (single search) ─────────────────────────────────────────────


def test_unknown_key_rejected(db):
    with pytest.raises(ValueError, match="unknown key"):
        db.search(_q(), top_k=1, hybrid={"text": "x", "bogus": 1})


def test_text_missing_rejected(db):
    with pytest.raises(ValueError, match="text"):
        db.search(_q(), top_k=1, hybrid={"weight": 0.5})


def test_text_wrong_type_rejected(db):
    # `text` must be str; passing int is rejected by PyO3 type extraction.
    with pytest.raises((ValueError, TypeError)):
        db.search(_q(), top_k=1, hybrid={"text": 123})


def test_weight_out_of_range(db):
    with pytest.raises(ValueError, match="weight"):
        db.search(_q(), top_k=1, hybrid={"text": "x", "weight": 1.5})
    with pytest.raises(ValueError, match="weight"):
        db.search(_q(), top_k=1, hybrid={"text": "x", "weight": -0.1})


def test_rrf_k_below_one_rejected(db):
    with pytest.raises(ValueError, match="rrf_k"):
        db.search(_q(), top_k=1, hybrid={"text": "x", "rrf_k": 0.5})


def test_oversample_zero_rejected(db):
    with pytest.raises(ValueError, match="oversample"):
        db.search(_q(), top_k=1, hybrid={"text": "x", "oversample": 0})


def test_oversample_negative_rejected(db):
    # PyO3's usize extraction will reject the negative; the exact error type
    # is OverflowError but ValueError is also acceptable.
    with pytest.raises((ValueError, OverflowError)):
        db.search(_q(), top_k=1, hybrid={"text": "x", "oversample": -3})


def test_empty_text_falls_back_silently(db):
    # Empty text is not an error — it collapses to dense-only.
    out = db.search(_q(), top_k=1, hybrid={"text": "", "weight": 0.5})
    assert isinstance(out, list)
    assert len(out) == 1


def test_unicode_text_accepted(db):
    # Non-ASCII / surrogate-prone characters must round-trip cleanly.
    out = db.search(_q(), top_k=1, hybrid={"text": "café résumé naïve"})
    assert isinstance(out, list)


# ── parse_hybrid_batch (query) ───────────────────────────────────────────────


def test_query_text_broadcasts(db):
    emb = np.stack([_q(), _q()], axis=0)
    out = db.query(emb, n_results=1, hybrid={"text": "hello"})
    assert isinstance(out, list)
    assert len(out) == 2


def test_query_texts_list_must_match_rows(db):
    emb = np.stack([_q(), _q()], axis=0)
    with pytest.raises(ValueError, match="texts"):
        db.query(emb, n_results=1, hybrid={"texts": ["one"]})


def test_query_text_and_texts_both_set_rejected(db):
    emb = np.stack([_q(), _q()], axis=0)
    with pytest.raises(ValueError, match="either"):
        db.query(emb, n_results=1, hybrid={"text": "a", "texts": ["a", "b"]})


def test_query_neither_text_nor_texts_rejected(db):
    emb = np.stack([_q(), _q()], axis=0)
    with pytest.raises(ValueError, match="missing"):
        db.query(emb, n_results=1, hybrid={"weight": 0.5})


def test_query_all_empty_texts_falls_back(db):
    emb = np.stack([_q(), _q()], axis=0)
    out = db.query(emb, n_results=1, hybrid={"texts": ["", ""]})
    assert isinstance(out, list)
    assert len(out) == 2


# ---------------------------------------------------------------------------
# db.explain() — per-retriever score breakdown for hybrid tuning
# ---------------------------------------------------------------------------


def _corpus(tmp, d=16):
    """Three docs whose dense and sparse rankings deliberately disagree."""
    instance = Database.open(tmp, dimension=d, bits=4, metric="ip")
    instance.insert("a", np.ones(d, dtype=np.float32), document="rust vector database quantization")
    instance.insert("b", (np.arange(d) / d).astype(np.float32), document="python sqlite embedded storage")
    instance.insert("c", np.full(d, 0.5, dtype=np.float32), document="rust storage engine")
    return instance


def test_explain_returns_per_leg_breakdown():
    with tempfile.TemporaryDirectory() as tmp:
        db_ = _corpus(tmp)
        rows = db_.explain(np.ones(16, dtype=np.float32), text="rust storage", top_k=3)
        assert len(rows) == 3
        for r in rows:
            for key in (
                "id", "score", "fused_score", "dense_score", "dense_rank",
                "sparse_score", "sparse_rank", "metadata", "document",
            ):
                assert key in r, f"missing {key}"
            assert r["score"] == r["fused_score"]
        # Fused scores are returned best-first.
        assert [r["fused_score"] for r in rows] == sorted(
            (r["fused_score"] for r in rows), reverse=True
        )
        db_.close()


def test_explain_matches_search_ordering_and_scores():
    """The card's contract: same params -> same ordering as search(hybrid=...)."""
    with tempfile.TemporaryDirectory() as tmp:
        db_ = _corpus(tmp)
        q = np.ones(16, dtype=np.float32)
        explained = db_.explain(q, text="rust storage", top_k=3)
        searched = db_.search(q, 3, hybrid={"text": "rust storage"})
        assert [r["id"] for r in explained] == [r["id"] for r in searched]
        for e, s in zip(explained, searched):
            assert e["fused_score"] == s["score"]
        db_.close()


def test_hybrid_ranking_is_deterministic_across_calls():
    """Regression: BM25 and RRF collected from HashMaps and sorted on score alone,
    so tied documents came back in whatever order the map iterated — identical
    queries produced different rankings and different fused scores."""
    with tempfile.TemporaryDirectory() as tmp:
        db_ = _corpus(tmp)
        q = np.ones(16, dtype=np.float32)
        seen = {
            tuple(
                (r["id"], round(r["fused_score"], 12), r["dense_rank"], r["sparse_rank"])
                for r in db_.explain(q, text="rust storage", top_k=3)
            )
            for _ in range(20)
        }
        assert len(seen) == 1, f"hybrid ranking is not reproducible: {seen}"
        db_.close()


def test_explain_marks_missing_leg_as_none():
    """A doc only the dense leg can reach has no sparse score, and vice versa."""
    with tempfile.TemporaryDirectory() as tmp:
        d = 16
        db_ = Database.open(tmp, dimension=d, bits=4, metric="ip")
        db_.insert("dense_only", np.ones(d, dtype=np.float32), document="alpha")
        db_.insert("sparse_only", -np.ones(d, dtype=np.float32), document="zebra quokka")
        rows = {r["id"]: r for r in db_.explain(np.ones(d, dtype=np.float32), text="zebra", top_k=2)}
        assert rows["dense_only"]["sparse_score"] is None
        assert rows["dense_only"]["sparse_rank"] is None
        assert rows["dense_only"]["dense_rank"] == 1
        assert rows["sparse_only"]["sparse_rank"] == 1
        db_.close()


def test_explain_rejects_weight_outside_unit_interval():
    with tempfile.TemporaryDirectory() as tmp:
        db_ = _corpus(tmp)
        with pytest.raises(ValueError, match="weight"):
            db_.explain(np.ones(16, dtype=np.float32), text="rust", weight=1.5)
        db_.close()


def test_explain_dimension_mismatch_raises():
    with tempfile.TemporaryDirectory() as tmp:
        db_ = _corpus(tmp)
        with pytest.raises(ValueError, match="dimension mismatch"):
            db_.explain(np.ones(4, dtype=np.float32), text="rust")
        db_.close()


def test_explain_honours_metadata_filter():
    with tempfile.TemporaryDirectory() as tmp:
        d = 16
        db_ = Database.open(tmp, dimension=d, bits=4, metric="ip")
        db_.insert("keep", np.ones(d, dtype=np.float32), {"lang": "rust"}, document="storage engine")
        db_.insert("drop", np.ones(d, dtype=np.float32), {"lang": "python"}, document="storage engine")
        rows = db_.explain(
            np.ones(d, dtype=np.float32), text="storage", top_k=5, filter={"lang": "rust"}
        )
        assert [r["id"] for r in rows] == ["keep"]
        db_.close()
