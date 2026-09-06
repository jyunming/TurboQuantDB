# TurboQuantDB Benchmarks

Three datasets from [arXiv:2504.19874](https://arxiv.org/abs/2504.19874) — n=100k vectors each.
Full script: [`benchmarks/paper_recall_bench.py`](https://github.com/jyunming/TurboQuantDB/blob/main/benchmarks/paper_recall_bench.py).

The benchmark script opens databases with `quantizer_type=None`, so current runs
`"dense"` below d=1024 and `"srht"` at d>=1024. All rows use `fast_mode=True`
(MSE-only, matching paper Figure 5 bit allocation); rerank varies by config.
ANN rows use HNSW (md=32, ef=128).

To regenerate:
```bash
python benchmarks/paper_recall_bench.py --update-readme --track
```
(requires `pip install datasets psutil matplotlib`)

CI perf gate:
- PR CI runs a fast smoke perf gate (`benchmarks/sprint_smoke.py`) in [`.github/workflows/ci.yml`](../.github/workflows/ci.yml).
- This is a regression tripwire (latency/recall/ingest), not a full publish benchmark replacement.
- `sprint_smoke.py` prints a compact PASS/FAIL summary for ingest, Recall@10,
  p50, and p95, then exits non-zero only when a gate fails. It does not write
  JSON or update history files.
- `paper_recall_bench.py` always writes `benchmarks/_bench_recall_results.json`
  after a real run. `--update-readme` patches only the marked section in this
  file, `--track` appends `benchmarks/perf_history.json` and regenerates the
  HTML report, and `--plots-only` regenerates plots from saved JSON without
  rerunning the benchmark.

---

<!-- PAPER_BENCH_START -->
### Algorithm Validation — Recall vs Paper

![Benchmark recall curves — TQDB vs paper](https://raw.githubusercontent.com/jyunming/TurboQuantDB/main/benchmarks/benchmark_plots.png)

Brute-force recall across all three datasets from [arXiv:2504.19874](https://arxiv.org/abs/2504.19874) Figure 5 — n=100k vectors, paper values read visually from plots (approximate).

**GloVe-200** (d=200, 100,000 corpus, 10,000 queries)

| Config | @k=1 | @k=2 | @k=4 | @k=8 | @k=16 | @k=32 | @k=64 |
|---|---:|---:|---:|---:|---:|---:|---:|
| TurboQuant 2-bit (paper Fig. 5a) | ≈55.0% | ≈70.0% | ≈83.0% | ≈91.0% | ≈96.0% | ≈99.0% | ≈100.0% |
| **TQDB b=2 rerank=F** | 51.1% | 67.2% | 79.9% | 88.7% | 94.6% | 97.8% | 99.2% |
| **TQDB b=2 rerank=T** | 97.1% | 98.2% | 98.2% | 98.2% | 98.2% | 98.2% | 98.2% |
| TurboQuant 4-bit (paper Fig. 5a) | ≈86.0% | ≈96.0% | ≈99.0% | ≈100.0% | ≈100.0% | ≈100.0% | ≈100.0% |
| **TQDB b=4 rerank=F** | 81.9% | 94.5% | 99.1% | 100.0% | 100.0% | 100.0% | 100.0% |
| **TQDB b=4 rerank=T** | 98.7% | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% |

**DBpedia OpenAI3 d=1536** (d=1536, 100,000 corpus, 1,000 queries)

| Config | @k=1 | @k=2 | @k=4 | @k=8 | @k=16 | @k=32 | @k=64 |
|---|---:|---:|---:|---:|---:|---:|---:|
| TurboQuant 2-bit (paper Fig. 5b) | ≈89.5% | ≈98.0% | ≈99.5% | ≈100.0% | ≈100.0% | ≈100.0% | ≈100.0% |
| **TQDB b=2 rerank=F** | 86.2% | 96.6% | 99.7% | 99.9% | 100.0% | 100.0% | 100.0% |
| **TQDB b=2 rerank=T** | 99.7% | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% |
| TurboQuant 4-bit (paper Fig. 5b) | ≈97.0% | ≈100.0% | ≈100.0% | ≈100.0% | ≈100.0% | ≈100.0% | ≈100.0% |
| **TQDB b=4 rerank=F** | 96.2% | 99.9% | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% |
| **TQDB b=4 rerank=T** | 99.7% | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% |

**DBpedia OpenAI3 d=3072** (d=3072, 100,000 corpus, 1,000 queries)

| Config | @k=1 | @k=2 | @k=4 | @k=8 | @k=16 | @k=32 | @k=64 |
|---|---:|---:|---:|---:|---:|---:|---:|
| TurboQuant 2-bit (paper Fig. 5c) | ≈90.5% | ≈98.5% | ≈99.5% | ≈100.0% | ≈100.0% | ≈100.0% | ≈100.0% |
| **TQDB b=2 rerank=F** | 90.2% | 97.6% | 99.6% | 100.0% | 100.0% | 100.0% | 100.0% |
| **TQDB b=2 rerank=T** | 99.7% | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% |
| TurboQuant 4-bit (paper Fig. 5c) | ≈97.5% | ≈100.0% | ≈100.0% | ≈100.0% | ≈100.0% | ≈100.0% | ≈100.0% |
| **TQDB b=4 rerank=F** | 98.0% | 99.7% | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% |
| **TQDB b=4 rerank=T** | 99.7% | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% |

All TQDB rows use `fast_mode=True` (MSE-only: all `b` bits go to the MSE codebook, no QJL residual). This is the same allocation as the paper's Figure 5 — b MSE bits/dim. Any residual gap at GloVe k=1 (~0–3%) is attributable to dataset sampling (we use the first 100k vectors from the 1.18M-token corpus; the paper used a random sample). DBpedia results match within 1–2% across all k values.

### Performance & Config Trade-offs

![Config trade-off overview — latency, disk, RAM, CPU](https://raw.githubusercontent.com/jyunming/TurboQuantDB/main/benchmarks/benchmark_plots_perf.png)

All 8 configs — brute-force and ANN (HNSW md=32, ef=128), all using `fast_mode=True` (MSE-only). Disk MB for ANN includes `graph.bin`. RAM = peak RSS during query phase. Index = HNSW build time (ANN only).

**GloVe-200** (d=200, 100,000 corpus, 10,000 queries)

| Config | Mode | Ingest | Index | Disk MB | RAM MB | p50 ms | p99 ms | R@1 | MRR |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| b=2 rerank=F | Brute | 2.1s | — | 6.2 | 197 | 3.47 | 6.67 | 51.1% | 0.651 |
| b=2 rerank=T | Brute | 7.7s | — | 25.8 | 218 | 4.45 | 7.30 | 97.1% | 0.977 |
| b=4 rerank=F | Brute | 2.6s | — | 10.9 | 204 | 3.51 | 7.19 | 81.9% | 0.898 |
| b=4 rerank=T | Brute | 7.5s | — | 30.5 | 225 | 5.90 | 10.87 | 98.7% | 0.994 |
| b=2 rerank=F | ANN | 2.2s | 30.0s | 14.8 | 232 | 0.90 | 2.49 | 33.6% | 0.412 |
| b=2 rerank=T | ANN | 7.1s | 46.0s | 34.4 | 258 | 4.41 | 12.36 | 59.0% | 0.593 |
| b=4 rerank=F | ANN | 2.9s | 27.6s | 19.5 | 238 | 0.72 | 2.07 | 52.6% | 0.565 |
| b=4 rerank=T | ANN | 8.7s | 41.8s | 39.1 | 261 | 4.37 | 13.14 | 72.9% | 0.733 |

**DBpedia OpenAI3 d=1536** (d=1536, 100,000 corpus, 1,000 queries)

| Config | Mode | Ingest | Index | Disk MB | RAM MB | p50 ms | p99 ms | R@1 | MRR |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| b=2 rerank=F | Brute | 7.5s | — | 50.2 | 806 | 9.90 | 22.67 | 86.2% | 0.924 |
| b=2 rerank=T | Brute | 12.6s | — | 197.0 | 956 | 11.41 | 25.55 | 99.7% | 0.999 |
| b=4 rerank=F | Brute | 13.4s | — | 99.0 | 855 | 21.24 | 40.70 | 96.2% | 0.981 |
| b=4 rerank=T | Brute | 16.6s | — | 245.9 | 1005 | 21.81 | 38.08 | 99.7% | 0.999 |
| b=2 rerank=F | ANN | 7.2s | 192.5s | 58.8 | 771 | 6.06 | 14.45 | 81.6% | 0.873 |
| b=2 rerank=T | ANN | 11.5s | 173.7s | 205.6 | 916 | 8.47 | 16.51 | 95.6% | 0.958 |
| b=4 rerank=F | ANN | 11.7s | 188.6s | 107.6 | 818 | 4.47 | 10.82 | 91.5% | 0.933 |
| b=4 rerank=T | ANN | 18.9s | 172.9s | 254.4 | 966 | 7.64 | 16.02 | 96.3% | 0.965 |

**DBpedia OpenAI3 d=3072** (d=3072, 100,000 corpus, 1,000 queries)

| Config | Mode | Ingest | Index | Disk MB | RAM MB | p50 ms | p99 ms | R@1 | MRR |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| b=2 rerank=F | Brute | 15.0s | — | 99.0 | 1405 | 15.78 | 31.21 | 90.2% | 0.946 |
| b=2 rerank=T | Brute | 23.9s | — | 392.4 | 1778 | 17.70 | 33.42 | 99.7% | 0.999 |
| b=4 rerank=F | Brute | 28.6s | — | 196.7 | 1589 | 36.01 | 56.54 | 98.0% | 0.989 |
| b=4 rerank=T | Brute | 38.3s | — | 490.0 | 1882 | 37.61 | 61.56 | 99.7% | 0.999 |
| b=2 rerank=F | ANN | 17.0s | 317.7s | 107.6 | 1412 | 13.10 | 29.77 | 85.7% | 0.898 |
| b=2 rerank=T | ANN | 27.3s | 339.2s | 401.0 | 1705 | 22.61 | 54.59 | 96.3% | 0.965 |
| b=4 rerank=F | ANN | 30.7s | 345.5s | 205.3 | 1511 | 10.99 | 26.72 | 93.4% | 0.943 |
| b=4 rerank=T | ANN | 54.9s | 360.1s | 498.6 | 1803 | 25.52 | 51.66 | 96.9% | 0.971 |

**Reproduction:** `maturin develop --release && python benchmarks/paper_recall_bench.py --update-readme --track`  (requires `pip install datasets psutil matplotlib`)

<!-- PAPER_BENCH_END -->

---

## When to use brute-force vs. ANN

The best search mode depends on vector dimensionality.

**Use brute-force (`_use_ann=False`, the default) when d ≤ 256**

At low dimensionality, quantization noise dominates. On GloVe-200 (d=200), ANN loses significant recall versus brute-force:

| Config | Brute-force R@1 | ANN R@1 | ANN latency gain |
|--------|:-----------:|:------:|:------:|
| b=4, rerank=T | **72.4%** | 53.8% | ~1.2× faster p50 |
| b=2, rerank=T | **33.7%** | 24.3% | ~1× (no gain) |

The HNSW graph built on quantized distances is inaccurate at low dimension. Use brute-force for d ≤ 256.

**Use ANN (`_use_ann=True`) when d ≥ 512**

At high dimensionality, quantization is accurate and the ANN approximation is tight. On DBpedia d=1536:

| Config | Brute-force R@1 | ANN R@1 | ANN latency gain |
|--------|:-----------:|:------:|:------:|
| b=4, rerank=T | 92.2% | **90.4%** | ~1.7× faster p50 |
| b=4, rerank=F | 92.2% | **87.8%** | ~4× faster p50 |

ANN costs ~2 points of recall while cutting latency from ~51ms to ~37ms p50. For production RAG at d=1536+, build the index once after initial load.

**Summary:**

| Dimension range | Recommended mode | Reason |
|-----------------|-----------------|--------|
| d ≤ 256 | Brute-force | Quantization noise collapses ANN recall |
| d = 512–1024 | Either (test both) | Moderate quantization quality; ANN gain is partial |
| d ≥ 1536 | ANN | High-d quantization is accurate; ANN gives 1.5–4× latency gain with <3% recall cost |

---

## Hybrid retrieval evaluation

Dense retrieval misses keyword-heavy queries; sparse retrieval misses
paraphrases. The harness at [`benchmarks/retrieval_eval.py`](https://github.com/jyunming/TurboQuantDB/blob/main/benchmarks/retrieval_eval.py)
grades all three paths (dense, BM25, hybrid RRF) on the same query sets so
new retrieval features can be judged against measured wins instead of
anecdotes. Append-only history lives at `benchmarks/retrieval_eval_history.json`.

```bash
python benchmarks/retrieval_eval.py             # 1k synthetic corpus, ~3s
python benchmarks/retrieval_eval.py --n 5000    # bigger corpus, ~15s
python benchmarks/retrieval_eval.py --no-history # don't append to history
```

Three query sets are generated from the synthetic corpus:

| Query set | What's in the query | Ideal retriever |
|-----------|---------------------|-----------------|
| `semantic` | A perturbed corpus vector, no text | Pure dense |
| `lexical`  | A rare token + a random orthogonal vector | Pure BM25 |
| `mixed`    | Half of each, interleaved | Hybrid (RRF) |

Representative output (N=1k, D=128, Q=100 per set, weight=0.5):

| query set | path | R@1 | R@10 | MRR@10 | NDCG@10 |
|-----------|------|----:|-----:|-------:|--------:|
| semantic  | dense  | 1.000 | 1.000 | 1.000 | 1.000 |
| semantic  | bm25   | 0.000 | 0.000 | 0.000 | 0.000 |
| semantic  | hybrid | 1.000 | 1.000 | 1.000 | 1.000 |
| lexical   | dense  | 0.010 | 0.010 | 0.010 | 0.010 |
| lexical   | bm25   | 1.000 | 1.000 | 1.000 | 1.000 |
| lexical   | hybrid | 0.200 | 1.000 | 0.466 | 0.597 |
| mixed     | dense  | 0.500 | 0.500 | 0.500 | 0.500 |
| mixed     | bm25   | 0.500 | 0.500 | 0.500 | 0.500 |
| mixed     | hybrid | 0.550 | 1.000 | 0.689 | 0.765 |

Read the table as: **on the mixed workload, hybrid raises R@10 from 0.500
to 1.000** (+50 pp) over either path alone, while losing nothing on
semantic-only queries. On lexical queries hybrid is dominated by pure BM25
on R@1, but still recovers the gold doc in the top-10 every time, which is
what most RAG pipelines actually consume.

---

## Embedded Python features not yet benchmarked

Several Python-side features have correctness coverage in the test suite but
aren't yet tracked in the longitudinal perf history:

- **Multi-vector / MaxSim retrieval at scale** (`MultiVectorStore`) — the wrapper is a Python layer over the existing single-vector engine. Per-config recall and latency at 10k+ docs / 80k+ tokens are pending and should be measured before native engine integration work.
- **`AsyncDatabase` concurrent-search throughput** — `tests/test_async_api.py` verifies executor fan-out with a blocking fake DB and event-loop responsiveness with a real DB, but there are no longitudinal throughput numbers. The actual ceiling depends on user thread-pool sizing, CPU cores, storage, and embedder latency.
- **Migration toolkit throughput** (`tqdb.migrate`) — the CLI prints rows/sec per run but those numbers aren't tracked in perf_history. For now, expect rates close to a plain `Database.insert_batch` loop on the same hardware.

LangChain / LlamaIndex / hybrid-via-integration overhead is also not benchmarked separately. The wrappers add a small Python-layer cost per call (one filter-dict translation + one numpy conversion) which has been negligible in practice on the 1k-5k corpora the test suite uses.
