# TurboQuantDB

[![License](https://img.shields.io/badge/license-Apache%202.0-blue.svg)](https://github.com/jyunming/TurboQuantDB/blob/main/LICENSE)
[![PyPI](https://img.shields.io/pypi/v/tqdb)](https://pypi.org/project/tqdb/)

An embedded vector database written in Rust with Python bindings, built around the **TurboQuant** paper ([arXiv:2504.19874](https://arxiv.org/abs/2504.19874)). TurboQuantDB uses a structured SRHT quantizer by default for practical vector-search workloads, and also offers `quantizer_type="exact"` for the paper-faithful QR + dense Gaussian path.

**Goal:** make massive embedding datasets practical on lightweight hardware. A 100k-vector, 1536-dim collection that would occupy 586 MB as raw float32 fits in **108 MB on disk** with TQDB b=4, or just **59 MB** with b=2 — enabling laptop-scale RAG over millions of documents without a dedicated server.

Two deployment modes:
- **Embedded** — `tqdb` Python package (`pip install tqdb`), runs in-process (no daemon)
- **Server** — Axum HTTP service in `server/`, with multi-tenancy, RBAC, quotas, and async jobs

---

## Key Properties

- **Zero training** — No `train()` step. Vectors are quantized and stored immediately on insert.
- **5–10× compression** — b=4 reduces 1536-dim float32 embeddings from 586 MB to 108 MB (5.4×); b=2 reaches 59 MB (9.9×) at 100k vectors.
- **Two quantizer modes** — default `srht` for practical production use; optional `exact` for paper-faithful comparisons.
- **Optional ANN index** — Build an HNSW graph after loading data for fast approximate search.
- **Metadata filtering** — MongoDB-style filter operators on any metadata field.
- **Crash recovery** — Write-ahead log (WAL) ensures durability without explicit flushing.
- **Python native** — Built with PyO3 and Maturin; no server or sidecar required.

---

## Quantizer Modes

TurboQuantDB exposes the same two-stage MSE + residual-QJL layout through two quantizer families:

- **`None` / `"srht"` (default)** — structured Walsh-Hadamard + random-sign transforms, `n = next_power_of_two(d)`, and `O(d log d)` setup/apply cost. This is the practical default and the mode used by the main README presets and benchmark tables unless noted otherwise.
- **`"exact"`** — QR-random orthogonal rotation + dense i.i.d. Gaussian QJL with `n = d`. This is the paper-faithful path for research, audits, and exact-vs-default comparisons, but it uses `O(d²)` time/state.
- **`fast_mode=True` (default)** — All `b` bits go to the MSE codebook. No QJL residual is stored or scored. This is the recommended mode for RAG and ANN search: it matches the bit allocation in the paper's Figure 5 and achieves the paper's recall numbers.
- **`fast_mode=False`** — Splits the budget: `b-1` bits to MSE and 1 bit to a QJL Johnson-Lindenstrauss residual sketch. The QJL gives an *unbiased inner-product estimator*, useful for LLM KV-cache attention scoring where absolute inner-product accuracy matters more than ranking order. For RAG/ANN (where only rank order matters), it reduces recall by taking budget away from MSE.

If you do not set `quantizer_type`, you are using the default SRHT family.

---

## Installation

### Prerequisites

- [Rust](https://rustup.rs/) stable toolchain
- Python 3.10+
- C++ compiler: Visual Studio Build Tools (Windows) · `xcode-select --install` (macOS) · `build-essential` (Linux)

### Build from source

```bash
python -m venv venv
source venv/bin/activate        # Windows: .\venv\Scripts\activate
pip install maturin
maturin develop --release
```

### Install pre-built wheel

```bash
pip install tqdb
```

---

## Recommended Setup

These starter presets use the default `fast_mode=True` (all bits → MSE, paper-aligned recall).
Pass `fast_mode=False` only for LLM KV-cache inner-product estimation.

```python
from tqdb import Database

# Recommended — brute-force with dequantization reranking
db = Database.open(path, dimension=DIM, bits=4, rerank=True)
results = db.search(query, top_k=10)
# 96.2% Recall@1, 100% Recall@4 at 100k×1536  |  108 MB disk  |  ~48ms p50

# Minimum disk — 9.9× compression, still excellent recall
db = Database.open(path, dimension=DIM, bits=2, rerank=True)
results = db.search(query, top_k=10)
# 86.2% Recall@1, 99.7% Recall@4 at 100k×1536  |  60 MB disk  |  ~44ms p50

# Optional: build an HNSW index after bulk load for sub-10ms queries
db.create_index()
results = db.search(query, top_k=10, _use_ann=True)

# Paper-faithful exact mode — use for research or paper comparisons
db = Database.open(path, dimension=DIM, bits=4, rerank=True, quantizer_type="exact")
results = db.search(query, top_k=10)
# Exact mode uses dense QR + Gaussian transforms (O(d²)); expect slower ingest at high d.
```

Full parameter reference: [`docs/PYTHON_API.md`](https://github.com/jyunming/TurboQuantDB/blob/main/docs/PYTHON_API.md)

---

## Quick Start

```python
import numpy as np
from tqdb import Database

db = Database.open("./my_db", dimension=1536, bits=4, metric="ip", rerank=True)  # default SRHT mode

db.insert("doc-1", np.random.randn(1536).astype("f4"), metadata={"topic": "ml"}, document="Machine learning intro")
db.insert("doc-2", np.random.randn(1536).astype("f4"), metadata={"topic": "systems"}, document="Rust memory model")

results = db.search(np.random.randn(1536).astype("f4"), top_k=5)
for r in results:
    print(r["id"], r["score"], r["document"])
```

---

## Python API

> Full reference: **[`docs/PYTHON_API.md`](https://github.com/jyunming/TurboQuantDB/blob/main/docs/PYTHON_API.md)**

```python
# Open / create
db = Database.open(path, dimension, bits=4, seed=42, metric="ip",
                   rerank=True, fast_mode=True, rerank_precision=None,
                   collection=None, wal_flush_threshold=None,
                   quantizer_type=None)  # None/"srht" = default structured fast path; "exact" = paper-faithful QR + Gaussian (O(d²))

# Write
db.insert(id, vector, metadata=None, document=None)
db.insert_batch(ids, vectors, metadatas=None, documents=None, mode="insert")  # "insert"|"upsert"|"update"
db.upsert(id, vector, metadata=None, document=None)
db.update(id, vector, metadata=None, document=None)        # RuntimeError if not found
db.update_metadata(id, metadata=None, document=None)       # RuntimeError if not found

# Delete & retrieve
db.delete(id)                        # → bool
db.delete_batch(ids)                 # → int (count deleted)
db.get(id)                           # → {id, metadata, document} | None
db.get_many(ids)                     # → list[dict | None]
db.list_all()                        # → list[str]
db.list_ids(where_filter=None, limit=None, offset=0)       # paginated
db.count(filter=None)                # → int
db.stats()                           # → dict
len(db) / "id" in db                 # container protocol

# Search — brute-force by default; pass _use_ann=True to use HNSW index
results = db.search(query, top_k=10, filter=None, _use_ann=False,
                    ann_search_list_size=None, include=None)
# include: list of "id"|"score"|"metadata"|"document" (default all)
# ann_search_list_size: HNSW ef_search override (only used when _use_ann=True)

all_results = db.query(query_embeddings, n_results=10, where_filter=None)
# query_embeddings: np.ndarray (N, D) — returns list[list[dict]]

# Index
db.create_index(max_degree=32, ef_construction=200, n_refinements=5,
                search_list_size=128, alpha=1.2)

# Metadata filter operators
# $eq $ne $gt $gte $lt $lte $in $nin $exists $and $or
db.search(query, top_k=5, filter={"year": {"$gte": 2023}})
db.search(query, top_k=5, filter={"$and": [{"topic": "ml"}, {"year": {"$gte": 2023}}]})
```

---

## Recommended Presets

### Recommended — brute-force + reranking

```python
db = Database.open(path, dimension=DIM, bits=4, rerank=True)
results = db.search(query, top_k=10, _use_ann=False)
# 96.2% Recall@1, 100% Recall@4 at 100k×1536  |  108 MB disk  |  ~48ms p50 (brute-force)
```

### Minimum Disk — compress aggressively

```python
db = Database.open(path, dimension=DIM, bits=2, rerank=True)
results = db.search(query, top_k=10, _use_ann=False)
# 86.8% Recall@1, 99.3% Recall@4 at 100k×1536  |  60 MB disk (9.9× smaller)  |  ~43ms p50
```

### Optional — ANN index for lower latency

```python
# Build once after inserting data; recall scales with ann_search_list_size
db.create_index()
results = db.search(query, top_k=10, _use_ann=True, ann_search_list_size=200)
```

---

## Benchmarks

Three datasets from [arXiv:2504.19874](https://arxiv.org/abs/2504.19874) — n=100k vectors each. Full script: [`benchmarks/paper_recall_bench.py`](https://github.com/jyunming/TurboQuantDB/blob/main/benchmarks/paper_recall_bench.py).

Unless explicitly labeled `exact`, the tables below refer to the default `quantizer_type=None/"srht"` engine. Paper curves are included for context, and the brute-force rows use exhaustive scoring over the selected quantizer family.

<!-- PAPER_BENCH_START -->
### Algorithm Validation — Recall vs Paper

![Benchmark recall curves — TQDB vs paper](https://raw.githubusercontent.com/jyunming/TurboQuantDB/main/benchmarks/benchmark_plots.png)

Brute-force recall across all three datasets from [arXiv:2504.19874](https://arxiv.org/abs/2504.19874) Figure 5 — n=100k vectors, paper values read visually from plots (approximate).

**GloVe-200** (d=200, 100,000 corpus, 10,000 queries)

| Config | @k=1 | @k=2 | @k=4 | @k=8 | @k=16 | @k=32 | @k=64 |
|---|---:|---:|---:|---:|---:|---:|---:|
| TurboQuant 2-bit (paper Fig. 5a) | ≈55.0% | ≈70.0% | ≈83.0% | ≈91.0% | ≈96.0% | ≈99.0% | ≈100.0% |
| **TQDB b=2 rerank=F** | 55.3% | 71.5% | 83.9% | 91.9% | 96.8% | 99.1% | 99.7% |
| **TQDB b=2 rerank=T** | 55.3% | 71.5% | 83.9% | 91.9% | 96.8% | 99.1% | 99.7% |
| TurboQuant 4-bit (paper Fig. 5a) | ≈86.0% | ≈96.0% | ≈99.0% | ≈100.0% | ≈100.0% | ≈100.0% | ≈100.0% |
| **TQDB b=4 rerank=F** | 84.1% | 95.2% | 99.2% | 100.0% | 100.0% | 100.0% | 100.0% |
| **TQDB b=4 rerank=T** | 84.1% | 95.2% | 99.2% | 100.0% | 100.0% | 100.0% | 100.0% |

**DBpedia OpenAI3 d=1536** (d=1536, 100,000 corpus, 1,000 queries)

| Config | @k=1 | @k=2 | @k=4 | @k=8 | @k=16 | @k=32 | @k=64 |
|---|---:|---:|---:|---:|---:|---:|---:|
| TurboQuant 2-bit (paper Fig. 5b) | ≈89.5% | ≈98.0% | ≈99.5% | ≈100.0% | ≈100.0% | ≈100.0% | ≈100.0% |
| **TQDB b=2 rerank=F** | 86.2% | 96.6% | 99.7% | 99.9% | 100.0% | 100.0% | 100.0% |
| **TQDB b=2 rerank=T** | 86.2% | 96.6% | 99.7% | 99.9% | 100.0% | 100.0% | 100.0% |
| TurboQuant 4-bit (paper Fig. 5b) | ≈97.0% | ≈100.0% | ≈100.0% | ≈100.0% | ≈100.0% | ≈100.0% | ≈100.0% |
| **TQDB b=4 rerank=F** | 96.2% | 99.9% | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% |
| **TQDB b=4 rerank=T** | 96.2% | 99.9% | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% |

**DBpedia OpenAI3 d=3072** (d=3072, 100,000 corpus, 1,000 queries)

| Config | @k=1 | @k=2 | @k=4 | @k=8 | @k=16 | @k=32 | @k=64 |
|---|---:|---:|---:|---:|---:|---:|---:|
| TurboQuant 2-bit (paper Fig. 5c) | ≈90.5% | ≈98.5% | ≈99.5% | ≈100.0% | ≈100.0% | ≈100.0% | ≈100.0% |
| **TQDB b=2 rerank=F** | 90.2% | 97.6% | 99.6% | 100.0% | 100.0% | 100.0% | 100.0% |
| **TQDB b=2 rerank=T** | 90.2% | 97.6% | 99.6% | 100.0% | 100.0% | 100.0% | 100.0% |
| TurboQuant 4-bit (paper Fig. 5c) | ≈97.5% | ≈100.0% | ≈100.0% | ≈100.0% | ≈100.0% | ≈100.0% | ≈100.0% |
| **TQDB b=4 rerank=F** | 98.0% | 99.7% | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% |
| **TQDB b=4 rerank=T** | 98.0% | 99.7% | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% |

All TQDB rows use `fast_mode=True` (MSE-only: all `b` bits go to the MSE codebook, no QJL residual). This is the same allocation as the paper's Figure 5 — b MSE bits/dim. Any residual gap at GloVe k=1 (~0–3%) is attributable to dataset sampling (we use the first 100k vectors from the 1.18M-token corpus; the paper used a random sample). DBpedia results match within 1–2% across all k values.

### Performance & Config Trade-offs

![Config trade-off overview — latency, disk, RAM, CPU](https://raw.githubusercontent.com/jyunming/TurboQuantDB/main/benchmarks/benchmark_plots_perf.png)

All 8 configs — brute-force and ANN (HNSW md=32, ef=128), all using `fast_mode=True` (MSE-only). Disk MB for ANN includes `graph.bin`. RAM = peak RSS during query phase. Index = HNSW build time (ANN only).

**GloVe-200** (d=200, 100,000 corpus, 10,000 queries)

| Config | Mode | Ingest | Index | Disk MB | RAM MB | p50 ms | p99 ms | R@1 | MRR |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| b=2 rerank=F | Brute | 1.2s | — | 19.8 | 210 | 10.89 | 13.09 | 55.3% | 0.690 |
| b=2 rerank=T | Brute | 1.2s | — | 19.8 | 212 | 13.34 | 15.35 | 55.3% | 0.690 |
| b=4 rerank=F | Brute | 2.3s | — | 25.9 | 218 | 11.33 | 13.11 | 84.1% | 0.910 |
| b=4 rerank=T | Brute | 2.0s | — | 25.9 | 217 | 13.46 | 15.63 | 84.1% | 0.910 |
| b=2 rerank=F | ANN | 1.2s | 11.6s | 28.4 | 241 | 0.68 | 1.56 | 34.5% | 0.419 |
| b=2 rerank=T | ANN | 1.1s | 11.6s | 28.4 | 243 | 4.15 | 6.61 | 40.8% | 0.501 |
| b=4 rerank=F | ANN | 1.6s | 11.4s | 34.5 | 249 | 0.69 | 1.58 | 51.2% | 0.545 |
| b=4 rerank=T | ANN | 1.6s | 11.4s | 34.5 | 250 | 4.34 | 6.99 | 62.6% | 0.670 |

**DBpedia OpenAI3 d=1536** (d=1536, 100,000 corpus, 1,000 queries)

| Config | Mode | Ingest | Index | Disk MB | RAM MB | p50 ms | p99 ms | R@1 | MRR |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| b=2 rerank=F | Brute | 4.9s | — | 83.9 | 838 | 38.92 | 45.89 | 86.2% | 0.924 |
| b=2 rerank=T | Brute | 4.9s | — | 83.9 | 834 | 43.82 | 48.99 | 86.2% | 0.924 |
| b=4 rerank=F | Brute | 10.4s | — | 132.8 | 882 | 42.35 | 48.40 | 96.2% | 0.981 |
| b=4 rerank=T | Brute | 8.3s | — | 132.8 | 881 | 48.06 | 54.00 | 96.2% | 0.981 |
| b=2 rerank=F | ANN | 5.0s | 43.8s | 92.5 | 793 | 5.25 | 8.77 | 81.7% | 0.875 |
| b=2 rerank=T | ANN | 5.1s | 44.0s | 92.5 | 794 | 27.05 | 41.71 | 84.0% | 0.901 |
| b=4 rerank=F | ANN | 8.8s | 43.6s | 141.3 | 843 | 5.13 | 8.75 | 91.7% | 0.935 |
| b=4 rerank=T | ANN | 8.6s | 43.6s | 141.3 | 843 | 29.54 | 44.84 | 94.9% | 0.967 |

**DBpedia OpenAI3 d=3072** (d=3072, 100,000 corpus, 1,000 queries)

| Config | Mode | Ingest | Index | Disk MB | RAM MB | p50 ms | p99 ms | R@1 | MRR |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| b=2 rerank=F | Brute | 8.9s | — | 157.2 | 1539 | 70.84 | 80.70 | 90.2% | 0.946 |
| b=2 rerank=T | Brute | 9.0s | — | 157.2 | 1469 | 81.93 | 92.75 | 90.2% | 0.946 |
| b=4 rerank=F | Brute | 16.5s | — | 254.8 | 1652 | 80.65 | 95.95 | 98.0% | 0.989 |
| b=4 rerank=T | Brute | 16.1s | — | 254.8 | 1567 | 92.12 | 105.25 | 98.0% | 0.989 |
| b=2 rerank=F | ANN | 9.0s | 76.4s | 165.8 | 1459 | 9.40 | 15.28 | 86.0% | 0.900 |
| b=2 rerank=T | ANN | 8.9s | 78.6s | 165.8 | 1465 | 53.60 | 82.38 | 89.3% | 0.935 |
| b=4 rerank=F | ANN | 17.1s | 77.0s | 263.4 | 1558 | 11.35 | 17.98 | 93.4% | 0.943 |
| b=4 rerank=T | ANN | 16.7s | 76.8s | 263.4 | 1559 | 63.35 | 93.64 | 96.3% | 0.972 |

**Reproduction:** `maturin develop --release && python benchmarks/paper_recall_bench.py --update-readme --track`  (requires `pip install datasets psutil matplotlib`)

<!-- PAPER_BENCH_END -->

### When to use brute-force vs. ANN

The benchmark tables above show a clear pattern: the best search mode depends on vector dimensionality.

**Use brute-force (`_use_ann=False`, the default) when d <= 256**

At low dimensionality, TurboQuant's quantization must pack each dimension into very few bits relative to the total information content. On GloVe-200 (d=200), the gap between brute-force and ANN is stark:

| Config | Brute-force R@1 | ANN R@1 | ANN latency gain |
|--------|:-----------:|:------:|:------:|
| b=4, rerank=T | **84.1%** | 62.6% | ~2x faster p50 |
| b=2, rerank=T | **55.3%** | 40.8% | ~1x (no gain) |

ANN at d=200 loses 22 percentage points of recall versus brute-force because the HNSW graph is built on quantized distances, which are less accurate at low dimension. The latency advantage does not compensate for this recall collapse. Use brute-force for d <= 256.

**Use ANN (`_use_ann=True`) when d >= 512**

At high dimensionality, quantization is more accurate and the ANN approximation is much tighter. On DBpedia d=1536:

| Config | Brute-force R@1 | ANN R@1 | ANN latency gain |
|--------|:-----------:|:------:|:------:|
| b=4, rerank=T | 96.2% | **94.9%** | ~5x faster p50 |
| b=4, rerank=F | 92.6% | **87.9%** | ~3x faster p50 |

ANN costs only ~2 points of recall while cutting latency from ~48ms to ~14ms p50. For production RAG at d=1536 or d=3072, ANN is the right default — build the index once after initial load, then queries are sub-linear.

**Summary:**

| Dimension range | Recommended mode | Reason |
|-----------------|-----------------|--------|
| d <= 256 | Brute-force | Quantization noise at low-d collapses ANN recall |
| d = 512–1024 | Either (test both) | Moderate quantization quality; ANN gain is partial |
| d >= 1536 | ANN | High-d quantization is accurate; ANN gives 3–5x latency speedup with <3% recall cost |

---

## Release Comparison

Performance delta vs. previous version across key metrics (100k vectors, best config per mode).

### v0.4.0 vs v0.3.0 (2026-04-08)

**DBpedia d=1536, brute-force b=4 rerank=T**

| Metric | v0.3.0 | v0.4.0 | Delta |
|--------|--------|--------|-------|
| Ingest | — | ~7s | — |
| p50 query | — | 47.7ms | — |
| p99 query | — | 50.9ms | — |
| R@1 | — | 96.2% | — |
| Disk | — | 108.3 MB | — |
| RAM | — | 860 MB | — |

> v0.3.0 benchmark data was not tracked. Tracking started at v0.4.0 via `benchmarks/perf_history.json`. From v0.5.0 onward, each release row will include a delta column against the prior release.

---

## RAG Integration

```python
from tqdb.rag import TurboQuantRetriever

retriever = TurboQuantRetriever(db_path="./rag_db", dimension=1536, bits=4)
retriever.add_texts(texts=texts, embeddings=embeddings, metadatas=metadatas)

results = retriever.similarity_search(query_embedding=query_vec, k=5)
for r in results:
    print(r["score"], r["text"])
```

---

## Architecture

TurboQuantDB is an embedded database — it runs in-process with no daemon.

```
./my_db/
├── manifest.json        — DB config (dimension, bits, seed, metric)
├── quantizer.bin        — Serialized quantizer state
├── live_codes.bin       — Memory-mapped quantized vectors (hot path)
├── live_vectors.bin     — Raw vectors for exact reranking (only if rerank_precision="f16" or "f32")
├── wal.log              — Write-ahead log
├── metadata.bin         — Per-vector metadata and documents
├── live_ids.bin         — ID → slot index
├── graph.bin            — HNSW adjacency list (if index built)
└── seg-XXXXXXXX.bin     — Immutable flushed segment files
```

**Write path (default, fast_mode=True):** `insert()` → SRHT rotation → MSE quantize → WAL → `live_codes.bin` → flush to segment

**Write path (fast_mode=False):** `insert()` → SRHT rotation → MSE quantize → QJL residual sketch → WAL → `live_codes.bin` → flush to segment

**Write path (quantizer_type="exact"):** `insert()` → QR rotation → MSE quantize → (QJL residual sketch if fast_mode=False) → WAL → `live_codes.bin` → flush to segment

**Search (brute-force):** query → precompute lookup tables → score all live vectors → top-k

**Search (ANN):** query → HNSW beam search → rerank → top-k

**Quantization structure:** Both quantizer families use the same two-stage layout:
1. **MSE stage** — rotate the vector and apply a Lloyd-Max scalar codebook
2. **Residual stage** — encode the MSE residual with a 1-bit QJL-style sketch plus `gamma`

**Default `srht` mode:** uses Walsh-Hadamard × random-sign transforms for both the rotation and the residual sketch. It runs in `O(d log d)`, pads `d` to the next power of two, and is the recommended production mode.

**Optional `exact` mode:** uses QR-random orthogonal rotation and dense i.i.d. `N(0,1)` Gaussian QJL with `n = d`, matching the paper's algorithmic contract more closely. It is slower and larger at high dimension, but useful for paper-faithful comparisons and algorithm audits.

With `fast_mode=True` (default), all `b` bits go to the MSE codebook — no QJL residual is stored or scored, matching the paper's Figure 5 bit allocation. With `fast_mode=False`, the budget is split `b-1` bits MSE + 1 bit QJL, which provides an unbiased inner-product estimator useful for LLM KV-cache scoring.

**Note on "zero indexing time":** The paper's claim (Table 2: TurboQuant 0.0013 s vs PQ 239 s vs RabitQ 2268 s at d=1536) measures **codebook/matrix construction time only** — i.e., how long to set up the quantizer before inserting any data. TurboQuant is near-zero because the quantizer is constructed analytically from a random seed with no data dependency; PQ and RabitQ require expensive training passes (k-means, SVD) over the corpus. The per-vector encoding cost (applying the rotation and projection to each inserted vector) still exists and scales as O(d log d) for SRHT or O(d²) for the exact mode.

### What comes from the paper vs. what is added here

The TurboQuant paper contributes the **quantization algorithm** — how to compress vectors and estimate inner products accurately. Its experiments use flat (exhaustive) search: all database vectors are scored against every query using the LUT-based asymmetric scorer. The paper's "indexing time virtually zero" claim refers to the quantizer requiring no training data, not to graph construction.

**From the paper:** two-stage MSE + QJL quantization, Lloyd-Max codebook, asymmetric LUT scoring, exhaustive flat-search evaluation, and the QR + dense Gaussian formulation that TurboQuantDB exposes as `quantizer_type="exact"`.

**TurboQuantDB default:** `None/"srht"` keeps the same two-stage quantization skeleton but swaps the dense random maps for structured transforms to reduce time/state in practical vector DB workloads.

**Added by TurboQuantDB (not in the paper):** WAL persistence, memory-mapped storage, metadata/documents, HNSW graph index, reranking, Python bindings, and the HTTP server.

The brute-force search path (`_use_ann=False`, the default) matches the paper's exhaustive-search evaluation style: every vector is scored with the quantizer's LUT scorer. Pair it with `quantizer_type="exact"` when you want the closest paper-faithful path in this repo. The HNSW index is a practical engineering addition that reduces the candidate set before scoring, enabling sub-linear search at the cost of approximate recall. Pass `_use_ann=True` to engage the HNSW index (requires `create_index()` to have been called first).

### Module Map

| Path | Responsibility |
|------|---------------|
| `src/python/mod.rs` | `Database` class — Python-facing API |
| `src/storage/engine.rs` | `TurboQuantEngine` — insert/search/delete orchestration |
| `src/storage/wal.rs` | Write-ahead log |
| `src/storage/segment.rs` | Immutable append-only segments |
| `src/storage/live_codes.rs` | Memory-mapped hot vector cache |
| `src/storage/graph.rs` | HNSW graph index |
| `src/quantizer/prod.rs` | `ProdQuantizer` — MSE + QJL orchestrator |
| `src/quantizer/mse.rs` | `MseQuantizer` — Lloyd-Max MSE stage (structured SRHT by default; QR in exact mode) |
| `src/quantizer/qjl.rs` | `QjlQuantizer` — 1-bit residual sketch (structured SRHT by default; dense Gaussian in exact mode) |
| `python/tqdb/rag.py` | `TurboQuantRetriever` — LangChain-style wrapper |
| `server/` | Optional Axum HTTP service (separate Cargo workspace) |

---

## Server Mode

> **Status: experimental.** The server crate compiles and the core endpoints work, but it has not been hardened for production use. The embedded library (`tqdb` Python package, `from tqdb import Database`) is the primary supported interface.

An optional Axum-based HTTP server is available in `server/` for multi-tenant deployments. It adds API key authentication, quota enforcement, and async job management (compaction, index building, snapshots).

```bash
cd server && cargo build --release
TQ_SERVER_ADDR=0.0.0.0:8080 TQ_LOCAL_ROOT=./data ./target/release/tqdb-server
```

See [`server/README.md`](https://github.com/jyunming/TurboQuantDB/blob/main/server/README.md) for the full endpoint reference. Key env vars:

| Variable | Default | Description |
|----------|---------|-------------|
| `TQ_SERVER_ADDR` | `127.0.0.1:8080` | Bind address |
| `TQ_LOCAL_ROOT` | `./data` | Storage root |
| `TQ_JOB_WORKERS` | `2` | Async job thread count |

---

## Performance Roadmap

The current implementation uses SIMD-accelerated scoring (AVX2) for the brute-force search inner loop, the MSE centroid scan,
and the QJL bit-unpack inner product. The FWHT transform (legacy SRHT path) also has an AVX2 fast path.

**GPU acceleration** — batch ingest would benefit from cuBLAS GEMM (~3–5× for
large batches on high-end cards). The ANN search path is memory-bound, not
compute-bound, so GPU benefit there is minimal; the bottleneck is random cache
misses during HNSW graph traversal rather than floating-point throughput.

**AVX-512 codebook scan** — on modern Intel CPUs the MSE centroid lookup can be
vectorised 2× wider with AVX-512, potentially halving scoring latency per batch.

**Persistent HNSW** — incremental graph updates (no full rebuild after each ingest
batch) would allow streaming use cases without periodic `create_index()` calls.

---

## Research Basis

This is an independent implementation of ideas from the TurboQuant paper. The algorithm itself was authored by the original researchers.

> Zandieh, A., Daliri, M., Hadian, M., & Mirrokni, V. (2025). *TurboQuant: Online Vector Quantization with Near-optimal Distortion Rate*. [arXiv:2504.19874](https://arxiv.org/abs/2504.19874)

```bibtex
@article{zandieh2025turboquant,
  title={TurboQuant: Online Vector Quantization with Near-optimal Distortion Rate},
  author={Zandieh, Amir and Daliri, Majid and Hadian, Majid and Mirrokni, Vahab},
  journal={arXiv preprint arXiv:2504.19874},
  year={2025}
}
```

---

## License

Apache License 2.0 — see [LICENSE](https://github.com/jyunming/TurboQuantDB/blob/main/LICENSE).
