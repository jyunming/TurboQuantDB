# TurboQuantDB — Paper Conformance Assessment

**Assessed against:** TurboQuant paper `2504.19874v1` (bundled at `2504.19874v1.md`)  
**Assessment date:** 2026-04-01  
**Scope:** current repository state only; stale findings from earlier revisions were removed when contradicted by current code

---

## Executive Summary

| Area | Current status | Severity |
|------|----------------|----------|
| Core quantization pipeline (MSE + residual QJL) | ✓ Implemented and credible | — |
| Lloyd-Max / Beta-distribution codebook construction | ✓ Implemented | — |
| Asymmetric query scoring + SIMD LUT path | ✓ Implemented | — |
| Rotation step vs paper | ✓ Fixed: active path now uses QR rotation (MSE) and dense Gaussian projection (QJL) | — |
| Packed live-code storage / `bits > 8` support | ✓ Earlier issues appear fixed in current code | — |
| ANN graph implementation | ✓ Present; current code is multi-level HNSW-like | — |
| Paper-aligned recall benchmark harness | ✓ Present | — |
| Repository-wide coherence (tests/server/wrappers/docs) | ✗ Major drift from current engine API | HIGH |

TurboQuantDB is currently best described as a **credible TurboQuant-conformant embedded vector database prototype**. The core Rust library now closely follows the paper's algorithms (QR rotation for MSE, dense Gaussian projection for QJL, Lloyd-Max codebook, residual QJL, asymmetric LUT scoring). The repository as a whole is **not yet fully robust or cleanly synchronized**: several tests, server endpoints, wrappers, benchmark scripts, and README claims remain stale relative to the current engine.

---

## 1. What currently matches the paper well

### 1.1 Lloyd-Max scalar quantization on Beta-like coordinate marginals

The paper's MSE path depends on quantizing rotated coordinates with an optimal scalar codebook derived from the coordinate distribution on the unit sphere.

**Current implementation**
- `src/quantizer/codebook.rs`
  - `beta_pdf(...)` implements the Beta-density-style marginal used in the paper
  - `lloyd_max(...)` solves the 1-D continuous k-means approximation for the codebook
- `src/quantizer/mse.rs`
  - stores per-dimension code indices against that codebook

**Verdict:** **Aligned in substance.** This is one of the strongest paper-to-code correspondences in the repository.

### 1.2 Two-stage PROD quantizer (MSE quantizer + QJL on residual)

The paper's inner-product quantizer first applies an MSE quantizer with one less bit, then quantizes the residual using 1-bit QJL.

**Current implementation**
- `src/quantizer/prod.rs`
  - `quantize(...)` first quantizes with the MSE quantizer
  - dequantizes the MSE result
  - computes the residual
  - applies QJL to the residual
  - stores `gamma = ||residual||_2`
- `src/quantizer/qjl.rs`
  - performs sign-based bit-packing for the residual sketch
  - dequantization rescales by the QJL factor involving `sqrt(pi / (2d))`

**Verdict:** **Aligned in substance.** This is a faithful implementation of the paper's high-level PROD design.

### 1.3 Asymmetric query scoring with lookup tables

The paper relies on scoring compressed database vectors against a full-precision query using precomputed per-query structures.

**Current implementation**
- `src/quantizer/prod.rs`
  - `prepare_ip_query(...)` builds an MSE lookup table plus the QJL-side query sketch
  - `score_ip_encoded(...)` combines LUT-based centroid scoring with the QJL correction term
  - includes AVX2/FMA acceleration on x86_64 when available

**Verdict:** **Aligned in substance.** The current scoring path is well thought out and is one of the more convincing parts of the implementation.

---

## 2. Important current divergences from the paper

### 2.1 ~~Active rotation path uses SRHT~~ — **RESOLVED**

This gap has been closed. The active paths in `mse.rs` and `qjl.rs` now use the paper-exact transforms:

- **MSE quantizer** (`src/quantizer/mse.rs`): `new()` generates a full QR-orthogonal rotation matrix via `generate_random_rotation()`; stored as a flattened d×d matrix in `rotation_signs`. The `apply_rotation()` / `apply_rotation_transpose()` helpers replace all `srht()` calls. Legacy databases (d-element sign vector) are detected at load time and routed to the old SRHT path for backward compatibility.

- **QJL quantizer** (`src/quantizer/qjl.rs`): `new()` generates a dense Gaussian iid N(0,1) projection matrix via `generate_projection_matrix()`; stored as a flattened d×d matrix in `projection_signs`. Same format-detection backward-compat pattern. The `scale_base` bug (`sqrt(π/(2d))` → `sqrt(π/2)/d`) was also fixed here.

- **Query scoring** (`src/quantizer/prod.rs`): `srht()` calls in `prepare_ip_query()` and `prepare_ip_query_lite()` replaced with `apply_rotation()` / `apply_projection()` sub-quantizer calls. The `srht` import removed.

**Performance note:** QR/Gaussian transforms are O(d²) per vector vs SRHT's O(d log d). For d=1536 this is ~140× slower on insert/dequantize. This is correct per the paper and can be optimized with a BLAS sgemm call if needed.

**Verdict:** **Resolved.** Both rotation and projection now conform exactly to the paper's Algorithm 1 and Algorithm 2.

### 2.2 Database-specific extensions are substantial

The paper is about quantization; this repository extends it into a storage/search system with:
- WAL persistence (`src/storage/wal.rs`)
- memory-mapped live-code storage (`src/storage/live_codes.rs`)
- metadata/documents (`src/storage/metadata.rs`)
- ID indirection (`src/storage/id_pool.rs`)
- ANN graph indexing (`src/storage/graph.rs`)
- optional raw-vector reranking (`src/storage/engine.rs`)

**Verdict:** **Expected adaptation.** This is not a problem by itself, but it means the repository should be framed as an application of the paper's ideas, not simply a line-by-line implementation of the paper.

---

## 3. Earlier assessment findings that are no longer supported by current code

The previous version of this document contained several findings that are stale relative to the current tree.

### 3.1 Packed live-code stride issue appears fixed

The previous assessment claimed `open()` used an unpacked `d`-byte MSE stride while the rest of the engine used packed bit-width-based sizing.

**Current code**
- `src/storage/engine.rs` now computes:
  - `let mse_len = (manifest.d * (manifest.b - 1)).div_ceil(8);`
  - `let qjl_len = manifest.d.div_ceil(8);`
  - `let stride = mse_len + qjl_len + LIVE_GAMMA_BYTES + LIVE_NORM_BYTES + LIVE_DELETED_BYTES;`
- This matches the current `live_mse_len()` / `live_qjl_len()` / `live_stride()` path.

**Verdict:** The previously reported stride inconsistency is **not supported by the current code**.

### 3.2 `bits > 8` overflow finding is no longer supported by current code

The previous assessment claimed `CodeIndex = u8` caused failures above 8 bits.

**Current code**
- `src/quantizer/mod.rs` now defines:
  - `pub type CodeIndex = u16;`

**Verdict:** The previous overflow finding is **stale**.  
**Caution:** current passing end-to-end validation still does not exercise higher bit-widths thoroughly, so this should be treated as **apparently fixed, but under-tested** rather than fully retired.

### 3.3 Single-layer ANN finding is no longer supported by current code

The previous assessment described the graph as single-layer.

**Current code**
- `src/storage/graph.rs`
  - assigns random per-node levels
  - stores per-level adjacency
  - tracks `entry_point` and `max_level`
  - performs top-down multi-level navigation before level-0 search

**Verdict:** The current graph is **multi-level HNSW-like**, not single-layer.

### 3.4 Benchmark-metric gap is smaller than previously stated

The previous assessment claimed the repo lacked a paper-style Recall@1@k benchmark.

**Current code**
- `benchmarks/recall_bench.py` now explicitly measures **Recall@1@k**
- `benchmarks/run_recall_bench.py` also measures exact-reference recall against NumPy top-k
- `benchmarks/ci_quality_gate.py` defines quality gates around recall and latency

**Verdict:** The repository now has a **meaningful benchmark harness**, though not all benchmark scripts are current or runnable without additional setup.

---

## 4. Current repository health problems

The core library is in better shape than the repository around it.

### 4.1 What currently validates cleanly

Observed on the current tree:

- `cargo check -q` ✓
- `cargo test -q --lib` ✓

This supports the claim that the **core Rust library** is in materially better shape than the surrounding repo surfaces.

### 4.2 What is currently stale or broken around the engine

Observed on the current tree:

- `cargo test -q --test integration_tests` failed to compile because parts of `tests/integration_tests.rs` still reference engine APIs and stats fields that do not exist in the current engine.
- `cargo test -q --test cloud_tests` failed to compile because the test calls `engine.create_index(...)`, while the current engine exposes `create_index_with_params(...)`.
- `cargo test -q --test bench_search` failed to compile because it expects `compact()`, `create_index(...)`, and old `DbStats` fields.
- `cargo test -q --test bench_batch_crud` failed to compile because it expects `delete_many(...)`.
- `cd server && cargo test -q` failed to compile because `server/src/main.rs` expects collection/snapshot/reporting APIs that are not present in the current engine.

### 4.3 Python/docs/benchmark drift

Several Python-side or doc-side surfaces are also stale:

- `README.md` still says the project does **not** have an ANN graph like HNSW, which is false for the current code.
- `python/turboquantdb/rag.py` appears out of sync with the current `Database.open(...)` binding signature in `src/python/mod.rs`.
- Some benchmark scripts still call `db.flush()`, but the current Python binding does not expose `flush`.

**Verdict:** The repository currently has a **major coherence problem**. The engine, tests, server, wrappers, benchmarks, and docs are not evolving together.

---

## 5. Credibility and feasibility assessment

### 5.1 Credibility

**Core algorithmic credibility:** **Good**
- The quantizer logic is real, nontrivial, and thoughtfully implemented.
- The code clearly reflects the paper's main ideas rather than merely borrowing terminology.

**Repository-level credibility:** **Mixed**
- Public claims are weakened by stale docs and broken adjacent surfaces.
- A technically literate reviewer will notice quickly that the engine appears healthier than the rest of the repo.

### 5.2 Feasibility

**Core embedded-engine feasibility:** **Good**
- The architecture is plausible for an embedded vector DB:
  - quantize on ingest
  - persist through WAL + local files
  - search via exhaustive or ANN path
  - rerank with stored raw vectors when enabled

**Full product feasibility:** **Not yet demonstrated**
- The optional server and many end-to-end validation surfaces are not currently synchronized with the engine.

### 5.3 Robustness

**Strengths**
- WAL replay path exists
- packed on-disk structures are explicit
- Windows mmap handling in `src/storage/live_codes.rs` is careful

**Weaknesses**
- broken/stale integration surfaces dominate the current repo health picture
- many important behaviors are not covered by passing end-to-end tests
- benchmark and wrapper drift means advertised workflows are not uniformly trustworthy

---

## 6. Current action items

### Action 1 (HIGH) — Reconcile engine API with the rest of the repository

Choose a single source of truth:

1. **Restore the engine APIs** that tests/server/wrappers expect, or
2. **Update/remove the stale surfaces** so they match the current engine.

This includes at least:
- `tests/integration_tests.rs`
- `tests/cloud_tests.rs`
- `tests/bench_search.rs`
- `tests/bench_batch_crud.rs`
- `server/src/main.rs`
- `python/turboquantdb/rag.py`
- stale benchmark scripts and README examples

### Action 2 ~~(MEDIUM)~~ — ✅ RESOLVED: Switch to paper-conformant QR rotation and Gaussian QJL projection

Completed in commit `feat(quantizer): switch to paper-conformant QR rotation and Gaussian QJL projection`.
- `mse.rs`: QR rotation matrix active, SRHT legacy fallback preserved
- `qjl.rs`: dense Gaussian projection matrix active, scale bug fixed, SRHT legacy fallback preserved
- `prod.rs`: `srht` import removed, query path delegates to sub-quantizer helpers

### Action 3 (MEDIUM) — Refresh public docs

At minimum:
- remove the README claim that there is no ANN graph
- align Python usage examples with the current binding signature
- ensure benchmark instructions only mention current, supported scripts

### Action 4 (MEDIUM) — Re-establish end-to-end validation

Priority validation targets:
- current Python binding smoke tests
- current ANN build/search flow
- current higher-bit-width regression coverage
- current benchmark path for recall and compression claims

---

## 7. Overall verdict

**As a paper-inspired core implementation:** **credible**
- The current core library captures the paper's main quantization ideas well.

**As an exact implementation of the paper:** **not quite**
- The active rotation path diverges from the paper's QR-based construction.

**As a clean, robust repository today:** **not yet**
- The largest issue is no longer the previously reported packed-storage bug set; it is **repository coherence drift**.

The correct current description is:

> **TurboQuantDB is a promising TurboQuant-inspired embedded vector database prototype with a credible core quantization implementation, but significant documentation, validation, and repository-surface drift still needs to be resolved.**
