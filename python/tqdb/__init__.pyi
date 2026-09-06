"""Type stubs for the tqdb package."""

from tqdb.tqdb import Database as Database, TurboQuantDB as TurboQuantDB
from tqdb.chroma_compat import (
    CompatClient as ChromaCompatClient,
    PersistentClient as PersistentClient,
)
from tqdb.lancedb_compat import connect as lancedb_connect
from tqdb.aio import AsyncDatabase as AsyncDatabase
from tqdb.multivector import MultiVectorStore as MultiVectorStore

from typing import Any, Optional

__version__: str

def open(path: str, dimension: Optional[int] = ..., **kwargs: Any) -> Database:
    """Open (or create) a database with sensible defaults."""
    ...

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
