"""
SRHT vs. Exact-Paper Quantizer Comparison
==========================================
Compares recall@k, query latency, and ingest throughput for:
  - "srht"  : SRHT rotation (H·D) + SRHT projection — O(d log d), what the code actually uses
  - "exact" : QR-random orthogonal + dense Gaussian — O(d²), what the paper specifies

Usage:
    python benchmarks/compare_quantizers.py
"""

import time
import tempfile

import numpy as np
import tqdb

DATASETS = [
    # (name,       D,    N,      Q,    bits)
    ("random-128", 128,  20_000, 500,  4),
    ("random-768", 768,  10_000, 200,  4),
    ("random-1536",1536, 5_000,  100,  4),
]

TOP_KS = [1, 4, 8]


def run_variant(qt: str, D: int, N: int, corpus: np.ndarray, queries: np.ndarray,
                ids: list[str], bits: int) -> dict:
    with tempfile.TemporaryDirectory() as tmp:
        db = tqdb.Database.open(
            tmp, dimension=D, bits=bits, metric="ip",
            rerank=False, quantizer_type=qt
        )

        # Ingest
        t0 = time.perf_counter()
        batch = 2000
        for s in range(0, N, batch):
            db.insert_batch(ids[s:s+batch], corpus[s:s+batch])
        db.flush()
        ingest_s = time.perf_counter() - t0

        # Query
        latencies = []
        hit_ids_per_q = []
        for q in queries:
            t0 = time.perf_counter()
            hits = db.search(q, top_k=max(TOP_KS))
            latencies.append(time.perf_counter() - t0)
            hit_ids_per_q.append([r["id"] for r in hits])

    return {
        "ingest_vps": N / ingest_s,
        "p50_ms":     np.median(latencies) * 1000,
        "p95_ms":     np.percentile(latencies, 95) * 1000,
        "hit_ids":    hit_ids_per_q,
    }


def compute_recall(hit_ids_per_q, ground_truth, k):
    recalls = []
    for i, hit_ids in enumerate(hit_ids_per_q):
        hit_set = set(hit_ids[:k])
        gt_set  = set(ground_truth[i][:k])
        recalls.append(len(hit_set & gt_set) / k)
    return np.mean(recalls)


def ground_truth_topk(corpus: np.ndarray, queries: np.ndarray, k: int) -> list[list[str]]:
    """Exact inner-product top-k on unit-sphere-normalised corpus."""
    normed = corpus / (np.linalg.norm(corpus, axis=1, keepdims=True) + 1e-12)
    scores = queries @ normed.T          # (Q, N)
    top_indices = np.argsort(-scores, axis=1)[:, :k]
    return [[f"v{j}" for j in row] for row in top_indices]


print("=" * 72)
print(f"{'Dataset':<16} {'Metric':<22} {'SRHT':>10} {'Exact':>10} {'Delta':>10}")
print("=" * 72)

for name, D, N, Q, bits in DATASETS:
    rng = np.random.RandomState(42)
    corpus  = rng.randn(N, D).astype("f4")
    queries = rng.randn(Q, D).astype("f4")
    ids     = [f"v{i}" for i in range(N)]

    gt = ground_truth_topk(corpus, queries, max(TOP_KS))

    results = {}
    for qt in ["srht", "exact"]:
        print(f"  running {qt:5s} on {name} (D={D}, N={N})...", flush=True)
        results[qt] = run_variant(qt, D, N, corpus, queries, ids, bits)

    print(f"\n--- {name}  (N={N}, D={D}, bits={bits}) ---")

    def row(label, key):
        s = results["srht"][key]
        e = results["exact"][key]
        pct = (e - s) / (s + 1e-12) * 100
        arrow = "+" if pct > 1 else ("-" if pct < -1 else "~")
        print(f"  {label:<22} {s:>10.4f} {e:>10.4f} {pct:>+8.1f}% {arrow}")

    for k in TOP_KS:
        for qt in ["srht", "exact"]:
            results[qt][f"recall@{k}"] = compute_recall(results[qt]["hit_ids"], gt, k)
        row(f"recall@{k}", f"recall@{k}")

    row("p50 latency (ms)", "p50_ms")
    row("p95 latency (ms)", "p95_ms")
    row("ingest (vps)",     "ingest_vps")
    print()

print("=" * 72)
print("Done.")
