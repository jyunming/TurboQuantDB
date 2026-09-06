"""
tqdb — high-performance embedded vector database.

Implements the TurboQuant algorithm (arXiv:2504.19874) for zero-training-time
vector quantization: 2–4 bits per coordinate, 8–16× less RAM than float32,
with provably unbiased inner-product estimation via QJL transforms.

Quick start::

    import numpy as np, tqdb

    vector = np.random.rand(1536).astype("f4")   # your embedding model's output
    db = tqdb.open("mydb", 1536)
    db.insert("doc1", vector)
    results = db.search(vector, top_k=5)

``tqdb.open`` is the zero-configuration entry point; ``Database.open`` exposes every
knob (bits, metric, rerank, fast_mode, ...) and ``Database.search`` documents the
filter syntax.
"""
from typing import Any, Optional

from .tqdb import Database, TurboQuantDB
from .chroma_compat import CompatClient as ChromaCompatClient, PersistentClient
from .lancedb_compat import connect as lancedb_connect
from .aio import AsyncDatabase
from .multivector import MultiVectorStore
from importlib.metadata import version, PackageNotFoundError

try:
    __version__ = version("tqdb")
except PackageNotFoundError:
    __version__ = "0.0.0"

def open(path: str, dimension: Optional[int] = None, **kwargs: Any) -> Database:
    """Open (or create) a database with sensible defaults — the one-liner entry point.

    ``tqdb.open("mydb", 1536)`` is all a new user needs: ``bits=4``, ``metric="ip"``
    and ``fast_mode=True`` are the right answer for the common case, and reopening an
    existing store senses every parameter from its ``manifest.json``, so ``dimension``
    can be omitted there.

    Args:
        path: Directory for the database files.
        dimension: Vector dimensionality. Required only when creating a new database;
            omit it to reopen an existing one with its persisted settings.
        **kwargs: Any :meth:`Database.open` parameter (``bits``, ``seed``, ``metric``,
            ``rerank``, ``fast_mode``, ``rerank_precision``, ``collection``,
            ``normalize``, ``quantizer_type``, ``wal_flush_threshold``).

    Returns:
        An open :class:`Database`.

    Raises:
        ValueError: If the database is new and ``dimension`` was not supplied.

    Example::

        import tqdb

        db = tqdb.open("mydb", 1536)      # create
        db = tqdb.open("mydb")            # reopen, settings sensed from the manifest

    Note:
        ``rerank`` keeps :meth:`Database.open`'s compression-first default of
        ``False``; pass ``rerank=True`` to trade ~2x disk for ~+15 pp R@1.
    """
    return Database.open(path, dimension, **kwargs)


__all__ = [
    "open",
    "Database",
    "TurboQuantDB",
    "AsyncDatabase",
    "MultiVectorStore",
    "__version__",
    "ChromaCompatClient",
    "PersistentClient",
    "lancedb_connect",
]
