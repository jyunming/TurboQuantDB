# TurboQuantDB — Copilot Instructions

TurboQuantDB (`tqdb`) is an embedded vector database written in Rust with Python bindings via PyO3/Maturin. It implements the TurboQuant algorithm (arXiv:2504.19874): two-stage quantization (MSE via QR rotation + residual QJL via dense Gaussian projection) achieving near-optimal vector compression with zero training time.

Two deployment modes:
- **Embedded** — `tqdb` Python package, runs in-process (like DuckDB)
- **Server** — Axum HTTP service in `server/` (separate Cargo workspace)

---

## Build & Test Commands

```bash
# Primary workflow — compile Rust and install into active venv
maturin develop --release

# Fast type/compile check only (no .pyd output)
cargo check -q

# Format Rust code
cargo fmt --all

# Run a single Rust unit test (tests live in src/storage/engine/tests.rs)
cargo test -q --lib <test_name>

# Run all Rust unit tests (319 tests)
cargo test -q --lib

# Run a specific Rust integration test file
cargo test --test integration_tests
cargo test --test bench_search
cargo test --test bench_batch_crud

# Run all Python tests (requires maturin develop first)
.venv\Scripts\python -m pytest tests/ --basetemp="tmp_pytest" -q

# Run a single Python test
.venv\Scripts\python -m pytest tests/test_python_api.py::TestSearch::test_basic_search -v --basetemp="tmp_pytest"

# Python coverage
.venv\Scripts\python -m pytest tests/ --cov=tqdb --cov-report=term-missing --basetemp="tmp_pytest" -q

# CI quality gates (min recall 0.60, max latency 100ms)
python benchmarks/ci_quality_gate.py

# Pre-merge benchmark — updates README tables and perf_history.json
python benchmarks/paper_recall_bench.py --update-readme --track
```

> **Windows note:** Always use `.venv\Scripts\python` — `python` resolves to a system install with an older wheel.

---

## Architecture

### Module Map

| Path | Responsibility |
|------|---------------|
| `src/python/mod.rs` | `Database` PyO3 class — entire Python-facing API surface |
| `src/storage/engine/mod.rs` | `TurboQuantEngine` — insert/search/delete/flush orchestration |
| `src/storage/engine/filter.rs` | Metadata filter evaluation and scoring helpers (`pub(crate)`) |
| `src/storage/engine/tests.rs` | All 319 Rust unit tests (`use super::*`) |
| `src/storage/wal.rs` | Write-ahead log for crash recovery |
| `src/storage/segment.rs` | Immutable append-only segment files |
| `src/storage/live_codes.rs` | Memory-mapped hot vector cache (`live_codes.bin`) |
| `src/storage/graph.rs` | HNSW graph index (`graph.bin`, memory-mapped) |
| `src/storage/id_pool.rs` | ID ↔ slot hash table (FNV-1a) |
| `src/storage/metadata.rs` | Per-vector metadata and documents |
| `src/storage/compaction.rs` | Segment merging |
| `src/quantizer/prod.rs` | `ProdQuantizer` — orchestrates MSE + QJL stages |
| `src/quantizer/mse.rs` | `MseQuantizer` — QR rotation + Lloyd-Max codebook |
| `src/quantizer/qjl.rs` | `QjlQuantizer` — 1-bit dense Gaussian projection, bit-packed |
| `python/tqdb/chroma_compat.py` | ChromaDB-compatible client (`CompatClient`, `PersistentClient`) |
| `python/tqdb/lancedb_compat.py` | LanceDB-compatible connection (`connect()`, `CompatTable`) |
| `python/tqdb/rag.py` | `TurboQuantRetriever` — LangChain-style wrapper |

### Data Flow

**Write:** `insert_batch()` → quantize (QR → MSE centroids + Gaussian QJL bits) → WAL entry → `live_codes.bin` → periodic flush to immutable segment

**Search (brute-force, default):** query → precompute MSE lookup table + QJL scale → score all live vectors → top-k

**Search (ANN, `_use_ann=True`):** query → HNSW beam search → optional rerank → top-k. Requires `create_index()` first; `_use_ann` defaults to `False`.

**Index build:** `create_index()` → reads all live vectors → builds HNSW graph → writes `graph.bin` (memory-mapped)

### Storage Files

```
<db_path>/
├── manifest.json       — dimension, bits, seed, metric, rerank_precision, normalize
├── quantizer.bin       — serialised ProdQuantizer (bincode)
├── live_codes.bin      — mmap'd quantised vectors (MSE codes + QJL bits + gamma + norm + deleted flag)
├── live_vectors.bin    — raw f16/f32 vectors (only when rerank_precision="f16"/"f32")
├── wal.log             — write-ahead log
├── metadata.bin        — metadata + documents per slot
├── live_ids.bin        — serialised IdPool
├── graph.bin           — HNSW adjacency (memory-mapped)
├── graph_ids.json      — slot list for indexed nodes
└── seg-XXXXXXXX.bin    — immutable segment files
```

---

## Key Conventions

### Python API additions go in `src/python/mod.rs` only
Rust engine methods are `pub(crate)`; the Python surface is entirely defined in `src/python/mod.rs`. New Python methods must add a `#[pyo3(signature = ...)]` annotation and update `python/tqdb/tqdb.pyi`.

### Rerank modes — three distinct behaviours
- `rerank=True, rerank_precision=None` (default): **dequantization** reranking — zero extra disk/RAM, recall improvement via reconstructed vectors
- `rerank=True, rerank_precision="f16"`: raw f16 vectors stored, exact reranking, +n×d×2 bytes
- `rerank=True, rerank_precision="f32"`: raw f32 vectors, maximum precision, +n×d×4 bytes
- `rerank=False`: no reranking at all

### Quantizer is data-oblivious
`ProdQuantizer` uses seed-deterministic QR rotation (MSE stage) and dense Gaussian projection (QJL stage) — no training data needed. The seed and dimension must match on every `Database.open()`. Legacy databases where `len(rotation_signs) == d` use the SRHT path; all new databases use the full d×d matrix path.

### Windows mmap / file handle semantics
`live_codes.bin` and `graph.bin` are memory-mapped. On Windows the OS holds the file handle open until the Python object is garbage-collected. Tests use a `conftest.py` autouse fixture that calls `gc.collect()` twice after each test. Any code that renames or overwrites these files **must** drop the mmap handle first.

### Engine decomposition
`engine.rs` was split into a sub-module directory:
- `engine/mod.rs` — all `impl TurboQuantEngine` methods + data types
- `engine/filter.rs` — filter/scoring helpers (`pub(crate)`)
- `engine/tests.rs` — all unit tests (`use super::*` to access private fields)

### Thread safety
`Database` wraps `Arc<RwLock<TurboQuantEngine>>`. Concurrent reads are allowed; writes are serialised. Never call `write_engine()` inside a `read_engine()` closure.

### Metadata filter syntax
Both Rust (`metadata_matches_filter` in `filter.rs`) and Python compat layers share the same filter dict format:
```python
{"field": "value"}                         # bare equality
{"field": {"$gte": 2023}}                  # $eq $ne $gt $gte $lt $lte $in $nin $exists $contains
{"$and": [...]} / {"$or": [...]}           # logical combinators
```

### Versioning — single source of truth
Version is defined only in `pyproject.toml`. `Cargo.toml` must be kept in sync manually. Python reads it via `importlib.metadata`. Never bump on a feature branch — bump only in the merge commit to `main`.

| Commits in merge | Bump |
|-----------------|------|
| Any `feat:` | MINOR, reset PATCH |
| Only `fix:` / `perf:` / `refactor:` | PATCH |
| Breaking API change | MAJOR |

### Commit style
`type(scope): summary` — e.g. `fix(storage): release mmap before rename on Windows`, `perf(quantizer): faster bit-unpack`

### Git push
```bash
GITHUB_TOKEN="" git push origin <branch>
```
Plain `git push` returns 403 on the development machine.

### Pre-commit hook
Runs `cargo fmt --check`, `cargo check`, and `cargo test --lib` on every commit. All three must pass. Perf checks only run when `src/` files are staged.

---

## Testing Patterns

- Rust tests in `src/storage/engine/tests.rs` use `use super::*` — they can access all private engine fields
- Python tests use `tmp_path` (pytest fixture); `conftest.py` forces GC after each test to release mmap handles on Windows
- Run Python tests with `--basetemp="tmp_pytest"` locally to avoid the locked `AppData/Local/Temp/pytest-of-<user>` issue on Windows
- Compat layer tests (`test_chroma_compat*.py`, `test_lancedb_compat*.py`) use the real Rust extension — no mocking of `Database`

---

## Benchmark Workflow (pre-merge to main)

```bash
# Full benchmark run — updates README.md tables + perf_history.json + plots
python benchmarks/paper_recall_bench.py --update-readme --track

# Plots only (no run)
python benchmarks/paper_recall_bench.py --plots-only
```

Both `benchmarks/benchmark_plots.png` and `benchmarks/perf_history.json` are tracked in git and must be committed with any perf-relevant change.

---

## Roadmap & Development Cycle

TurboQuantDB follows a weekly sprint cycle tracked on **GitHub Project board #3** (`https://github.com/users/jyunming/projects/3`).

### Sprint schedule

| Sprint | Milestone | Due | Status |
|--------|-----------|-----|--------|
| 1 | v0.3 · Quality Foundation | Apr 12, 2026 | Released v0.3.0 |
| 2 | v0.4 · Index + Data Safety | Apr 19, 2026 | In progress |
| 3 | v0.5 · Ecosystem + Search Quality | Apr 26, 2026 | |
| 4 | v0.6 · Performance + Structure | May 3, 2026 | |
| 5 | v0.7–0.9 · Feature Expansion | May 10, 2026 | |
| 6 | v1.0 · Stable Release | May 17, 2026 | |

### Issue → PR cycle (follow this order exactly)

1. **Open an issue** on `jyunming/TurboQuantDB` using the feature request or bug template.
2. **Link to the project board**: `GITHUB_TOKEN="" gh project item-add 3 --owner jyunming --url <issue-url>`
3. **Create a branch** (never commit to `main` directly): `git checkout -b feat/<name>`
4. **Implement** — add tests, update `docs/PYTHON_API.md`, update `python/tqdb/tqdb.pyi` stubs.
5. **Commit** — pre-commit hook runs automatically:
   - `cargo fmt --check` → `cargo check` → `cargo test --lib` (always)
   - Perf gate via `benchmarks/precommit_perf_check.py` (only when `src/` files are staged)
   - Doc-only commits (all staged files are `.md`, `docs/`, `website/`) skip all Rust checks
6. **Before pushing** (when `src/` changed), run the paper benchmark:
   ```bash
   TQDB_TRACK=1 python benchmarks/paper_recall_bench.py --update-readme --track
   git add benchmarks/perf_history.json benchmarks/perf_baseline.json benchmarks/benchmark_plots.png
   ```
7. **Push and open PR**: `GITHUB_TOKEN="" git push origin <branch>` then `gh pr create --base main`
8. **Monitor CI** — Rust (ubuntu + windows), Python smoke, benchmark-artifacts. Fix all failures.
9. **Address Copilot review comments** — fix every comment, then resolve threads via GraphQL:
   ```bash
   # Get thread IDs
   gh api graphql -f query='{ repository(owner:"jyunming",name:"TurboQuantDB") { pullRequest(number:N) { reviewThreads(first:20) { nodes { id isResolved } } } } }'
   # Resolve each
   GITHUB_TOKEN="" gh api graphql -f query='mutation { resolveReviewThread(input:{threadId:"PRRT_..."}) { thread { isResolved } } }'
   ```
10. **Do not push or merge without explicit user approval.**
11. **After merge**: tag release, update project board items to Done:
    ```bash
    git tag vX.Y.Z && GITHUB_TOKEN="" git push origin vX.Y.Z
    ```

### v0.3 completed items (reference for what's available in v0.3.0)

- `Database.open(path)` — parameterless reopen reads all params from `manifest.json`
- `delete_batch(where_filter=...)` — filter-based atomic bulk delete
- `list_metadata_values(field)` — enumerate distinct metadata values with counts
- `normalize=True` on `Database.open()` — auto L2-normalise vectors and queries
- Hybrid ANN + brute-force for post-index inserts ("dark vectors")
- ChromaDB shim (`tqdb.chroma_compat`), LanceDB shim (`tqdb.lancedb_compat`)
- S3 segment backend (`--features cloud`), Prometheus `/metrics`, restore endpoint
- `.pyi` type stubs, WAL v5 CRC32, segment CRC32, engine sub-module decomposition

### v0.4 open issues (Sprint 2, due Apr 19)

- **#16** — Delta index: brute-force overlay for post-index inserts, merged on `create_index()`
- **#17** — Parallel batch ingest via Rayon
- **#18** — Compaction crash recovery (atomic rename, `.tmp` rollback on reopen)
- **#19** — `.unwrap()` audit pass 1 (write-path + Python-reachable sites)
