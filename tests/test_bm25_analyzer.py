"""Production-grade BM25 text analysis: stemming, stopwords, tokenizer split.

Roadmap card [v0.9.0] feat(bm25). Its acceptance criteria are the first two tests:
"running shoes" and "run shoe" must retrieve each other, and the stopword list must
be passable at open time.
"""

from __future__ import annotations

import json
import pathlib

import numpy as np
import pytest

from tqdb import Database

DIM = 8


def _vec() -> np.ndarray:
    return np.ones(DIM, dtype=np.float32)


def _corpus(path, **kwargs) -> Database:
    db = Database.open(str(path), dimension=DIM, bits=4, metric="ip", **kwargs)
    db.insert("shoe", _vec(), document="We sell a comfortable run shoe for athletes")
    db.insert("boot", _vec(), document="Waterproof hiking boots and jackets")
    db.flush()
    return db


def _sparse_hits(db) -> list[str]:
    """IDs the BM25 leg actually surfaced, best-first."""
    rows = db.explain(_vec(), text="running shoes", top_k=5)
    return [r["id"] for r in rows if r["sparse_rank"] is not None]


# ---------------------------------------------------------------------------
# Acceptance criteria from the roadmap card
# ---------------------------------------------------------------------------


def test_stemming_matches_inflected_query(tmp_path):
    """'running shoes' must find the document that says 'run shoe'."""
    db = _corpus(tmp_path / "stem")
    assert _sparse_hits(db) == ["shoe"]
    db.close()


def test_stopwords_are_passable_at_open_time(tmp_path):
    db = Database.open(str(tmp_path / "sw"), dimension=DIM, stopwords=["shoe"])
    db.insert("a", _vec(), document="run shoe")
    rows = db.explain(_vec(), text="shoe", top_k=1)
    assert rows[0]["sparse_rank"] is None, "a term listed as a stopword must not match"
    db.close()


# ---------------------------------------------------------------------------
# Configuration surface
# ---------------------------------------------------------------------------


def test_language_none_restores_pre_0_9_behaviour(tmp_path):
    db = _corpus(tmp_path / "plain", text_language="none")
    assert _sparse_hits(db) == [], "without stemming, 'running' does not match 'run'"
    db.close()


def test_bundled_stopwords_are_dropped_by_default(tmp_path):
    db = Database.open(str(tmp_path / "bundled"), dimension=DIM)
    db.insert("a", _vec(), document="the quick brown fox")
    rows = db.explain(_vec(), text="the", top_k=1)
    assert rows[0]["sparse_rank"] is None
    db.close()


def test_empty_stopword_list_keeps_every_token(tmp_path):
    db = Database.open(str(tmp_path / "nosw"), dimension=DIM, stopwords=[])
    db.insert("a", _vec(), document="the quick brown fox")
    rows = db.explain(_vec(), text="the", top_k=1)
    assert rows[0]["sparse_rank"] == 1
    db.close()


def test_whitespace_only_split_keeps_hyphenated_terms(tmp_path):
    db = Database.open(
        str(tmp_path / "split"),
        dimension=DIM,
        text_language="none",
        stopwords=[],
        split_on_punctuation=False,
    )
    db.insert("a", _vec(), document="error-code 42")
    assert db.explain(_vec(), text="error-code", top_k=1)[0]["sparse_rank"] == 1
    # With whitespace-only splitting, half of a hyphenate is not a token.
    assert db.explain(_vec(), text="error", top_k=1)[0]["sparse_rank"] is None
    db.close()


def test_unknown_language_is_rejected(tmp_path):
    with pytest.raises(ValueError, match="unknown text language"):
        Database.open(str(tmp_path / "bad"), dimension=DIM, text_language="klingon")


def test_non_english_language_stems(tmp_path):
    db = Database.open(str(tmp_path / "de"), dimension=DIM, text_language="german")
    db.insert("a", _vec(), document="Kinder spielen")
    assert db.explain(_vec(), text="kind", top_k=1)[0]["sparse_rank"] == 1
    db.close()


# ---------------------------------------------------------------------------
# Persistence: the index and the analyzer must never disagree
# ---------------------------------------------------------------------------


def test_analyzer_is_persisted_and_reused_on_reopen(tmp_path):
    path = tmp_path / "persist"
    db = _corpus(path, text_language="french", stopwords=["le"])
    db.close()

    manifest = json.loads((path / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["analyzer"]["language"] == "french"
    assert manifest["analyzer"]["stopwords"] == ["le"]

    # Reopening without text options keeps what the database was created with.
    reopened = Database.open(str(path), dimension=DIM)
    after = json.loads((path / "manifest.json").read_text(encoding="utf-8"))
    assert after["analyzer"]["language"] == "french"
    reopened.close()


def test_changing_the_analyzer_reindexes_stored_documents(tmp_path):
    path = tmp_path / "reindex"
    db = _corpus(path, text_language="none")
    assert _sparse_hits(db) == []
    db.close()

    # Switching to English must re-analyse the documents already in the store.
    upgraded = Database.open(str(path), dimension=DIM, text_language="english")
    assert _sparse_hits(upgraded) == ["shoe"]
    upgraded.close()


def test_pre_0_9_store_without_analyzer_migrates_on_open(tmp_path):
    """A manifest written before 0.9 has no analyzer key at all."""
    path = tmp_path / "legacy"
    db = _corpus(path, text_language="none")
    db.close()

    manifest_path = pathlib.Path(path) / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest.pop("analyzer", None)
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    migrated = Database.open(str(path), dimension=DIM)
    assert _sparse_hits(migrated) == ["shoe"], "documents must be re-analysed on upgrade"
    assert json.loads(manifest_path.read_text(encoding="utf-8"))["analyzer"]["language"] == "english"
    migrated.close()


def test_hybrid_search_still_returns_dense_results_for_stopword_only_query(tmp_path):
    """A query of nothing but stopwords must not break the dense leg."""
    db = _corpus(tmp_path / "dense_only")
    hits = db.search(_vec(), 2, hybrid={"text": "the of and"})
    assert len(hits) == 2
    db.close()
