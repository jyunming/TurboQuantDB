"""
SRHT vs. Exact-Paper Quantizer — Full Paper-Metrics Comparison
===============================================================
Uses the same GloVe-200 and DBpedia-1536/3072 corpora as the paper benchmark
(arXiv:2504.19874).  Compares the full metric set that appears in the README:

  Recall@1@k for k in [1, 2, 4, 8, 16, 32, 64]
  MRR  (Mean Reciprocal Rank of true top-1)
  Ingest throughput (vps)
  Query latency  p50 / p99 (ms)
  Disk footprint (MB)
  RAM ingest peak / RAM query peak (MB)   — requires psutil
  CPU ingest % / CPU query %              — requires psutil

Quantizer modes compared:
  srht  : SRHT rotation (H·D) + SRHT projection — O(d log d), n = next_power_of_two(d)
  exact : QR-random orthogonal + dense Gaussian  — O(d²),     n = d

Usage:
    python benchmarks/compare_quantizers_paper.py
    python benchmarks/compare_quantizers_paper.py --datasets glove
    python benchmarks/compare_quantizers_paper.py --datasets glove dbpedia-1536 --bits 4
"""

from __future__ import annotations

import argparse
import sys
import tempfile
import time
import os
from pathlib import Path

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from bench_core import (
    K_VALUES,
    compute_recalls,
    compute_mrr,
    disk_size_mb,
    CpuRamSampler,
    load_glove,
    load_dbpedia,
)

import tqdb

DATASETS_ALL = ["glove", "dbpedia-1536", "dbpedia-3072"]
TOP_K = 64   # retrieve this many neighbours; recall@k uses prefix of returned list


def run_variant(
    qt: str,
    vecs: np.ndarray,
    qvecs: np.ndarray,
    true_top1: np.ndarray,
    bits: int,
) -> dict:
    N, D = vecs.shape
    ids = [str(i) for i in range(N)]

    with tempfile.TemporaryDirectory() as tmp:
        db = tqdb.Database.open(
            tmp, dimension=D, bits=bits, metric="ip",
            rerank=False, quantizer_type=qt,
        )

        # --- Ingest ---
        with CpuRamSampler() as ingest_sampler:
            t0 = time.perf_counter()
            batch = 5_000
            for s in range(0, N, batch):
                db.insert_batch(ids[s:s+batch], vecs[s:s+batch])
            db.flush()
            ingest_s = time.perf_counter() - t0

        disk_mb = disk_size_mb(tmp)

        # --- Query ---
        latencies: list[float] = []
        all_results: list[list[str]] = []
        with CpuRamSampler() as query_sampler:
            for q in qvecs:
                t0 = time.perf_counter()
                hits = db.search(q, top_k=TOP_K)
                latencies.append(time.perf_counter() - t0)
                all_results.append([r["id"] for r in hits])

    recalls = compute_recalls(all_results, true_top1)
    mrr     = compute_mrr(all_results, true_top1)

    return {
        "ingest_vps":        N / ingest_s,
        "p50_ms":            float(np.median(latencies)) * 1000,
        "p99_ms":            float(np.percentile(latencies, 99)) * 1000,
        "disk_mb":           disk_mb,
        "ram_ingest_mb":     ingest_sampler.peak_ram_mb,
        "ram_query_mb":      query_sampler.peak_ram_mb,
        "cpu_ingest_pct":    ingest_sampler.avg_cpu_pct,
        "cpu_query_pct":     query_sampler.avg_cpu_pct,
        "mrr":               mrr,
        "recalls":           recalls,
    }


def pct_delta(s: float, e: float) -> tuple[float, str]:
    pct = (e - s) / (abs(s) + 1e-12) * 100
    arrow = "+" if pct > 1 else ("-" if pct < -1 else "~")
    return pct, arrow


def print_comparison(name: str, bits: int, srht: dict, exact: dict) -> None:
    W = 72
    print(f"\n{'=' * W}")
    print(f"  {name}  bits={bits}")
    print(f"{'=' * W}")
    print(f"  {'Metric':<26} {'SRHT':>10} {'Exact':>10}  {'Delta':>10}")
    print(f"  {'-' * 60}")

    def row(label: str, sk: float, ek: float, fmt: str = ".4f") -> None:
        pct, arrow = pct_delta(sk, ek)
        print(f"  {label:<26} {sk:>10{fmt}} {ek:>10{fmt}}  {pct:>+8.1f}% {arrow}")

    # Recall@k
    for k in K_VALUES:
        if k not in srht["recalls"]:
            continue
        row(f"recall@{k}", srht["recalls"][k], exact["recalls"][k])

    print(f"  {'-' * 60}")

    # MRR
    row("mrr",                  srht["mrr"],          exact["mrr"])
    print(f"  {'-' * 60}")

    # Performance
    row("ingest (vps)",         srht["ingest_vps"],   exact["ingest_vps"], ".0f")
    row("p50 latency (ms)",     srht["p50_ms"],        exact["p50_ms"])
    row("p99 latency (ms)",     srht["p99_ms"],        exact["p99_ms"])
    row("disk (MB)",            srht["disk_mb"],       exact["disk_mb"])
    row("RAM ingest peak (MB)", srht["ram_ingest_mb"], exact["ram_ingest_mb"], ".1f")
    row("RAM query peak (MB)",  srht["ram_query_mb"],  exact["ram_query_mb"], ".1f")
    row("CPU ingest (%)",       srht["cpu_ingest_pct"],exact["cpu_ingest_pct"], ".1f")
    row("CPU query (%)",        srht["cpu_query_pct"], exact["cpu_query_pct"], ".1f")


def print_srht_only(name: str, bits: int, srht: dict) -> None:
    """Print SRHT results when exact was skipped (too expensive for D>=3072)."""
    W = 72
    print(f"\n{'=' * W}")
    print(f"  {name}  bits={bits}  [exact skipped — O(d^2) too slow at this dimension]")
    print(f"{'=' * W}")
    print(f"  {'Metric':<26} {'SRHT':>10}")
    print(f"  {'-' * 38}")
    for k in K_VALUES:
        if k in srht["recalls"]:
            print(f"  {'recall@'+str(k):<26} {srht['recalls'][k]:>10.4f}")
    print(f"  {'mrr':<26} {srht['mrr']:>10.4f}")
    print(f"  {'-' * 38}")
    print(f"  {'ingest (vps)':<26} {srht['ingest_vps']:>10.0f}")
    print(f"  {'p50 latency (ms)':<26} {srht['p50_ms']:>10.4f}")
    print(f"  {'p99 latency (ms)':<26} {srht['p99_ms']:>10.4f}")
    print(f"  {'disk (MB)':<26} {srht['disk_mb']:>10.4f}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", nargs="+", choices=DATASETS_ALL, default=DATASETS_ALL,
                    help="Which datasets to run (default: all three)")
    ap.add_argument("--bits", nargs="+", type=int, default=[2, 4],
                    help="Bit widths to test (default: 2 4)")
    ap.add_argument("--save-json", default=None,
                    help="Save results as JSON to this path")
    args = ap.parse_args()

    print("=" * 72)
    print("SRHT vs. Exact-Paper Quantizer  —  Full Paper Metric Comparison")
    print("=" * 72)

    dataset_loaders: list[tuple[str, callable]] = []
    if "glove" in args.datasets:
        dataset_loaders.append(("glove-200", load_glove))
    if "dbpedia-1536" in args.datasets:
        dataset_loaders.append(("dbpedia-1536", lambda: load_dbpedia(1536)))
    if "dbpedia-3072" in args.datasets:
        dataset_loaders.append(("dbpedia-3072", lambda: load_dbpedia(3072)))

    # Accumulate results for optional JSON save
    all_results: dict[str, dict] = {}

    for ds_name, loader in dataset_loaders:
        print(f"\nLoading {ds_name} ...", flush=True)
        vecs, qvecs, truth = loader()
        N, D = vecs.shape
        Q    = len(qvecs)
        print(f"  corpus {N:,} x {D},  queries {Q:,}")
        all_results[ds_name] = {}

        for bits in args.bits:
            print(f"\n  [{ds_name}  bits={bits}]", flush=True)
            skip_exact = (D >= 3072)

            srht_res = None
            exact_res = None

            for qt in ["srht", "exact"]:
                if qt == "exact" and skip_exact:
                    print(f"    exact: skipped (D={D} — O(d^2)={D*D:,} ops/vec is prohibitive)")
                    continue
                print(f"    {qt}: running ...", flush=True)
                res = run_variant(qt, vecs, qvecs, truth, bits)
                if qt == "srht":
                    srht_res = res
                else:
                    exact_res = res
                print(f"    {qt}: done  (ingest {res['ingest_vps']:,.0f} vps,"
                      f"  p50 {res['p50_ms']:.2f} ms,"
                      f"  recall@1 {res['recalls'].get(1, 0):.3f})")

            key = f"b{bits}"
            all_results[ds_name][key] = {"srht": srht_res, "exact": exact_res}

            if srht_res and exact_res:
                print_comparison(ds_name, bits, srht_res, exact_res)
            elif srht_res:
                print_srht_only(ds_name, bits, srht_res)

    print("\n" + "=" * 72)
    print("Done.")

    if args.save_json:
        import json as _json
        out = Path(args.save_json)
        # Convert recalls dict keys to strings for JSON serialization
        def _jsonify(r):
            if r is None:
                return None
            rc = dict(r)
            rc["recalls"] = {str(k): v for k, v in r["recalls"].items()}
            return rc
        serializable = {
            ds: {b: {"srht": _jsonify(v["srht"]), "exact": _jsonify(v["exact"])}
                 for b, v in bits_data.items()}
            for ds, bits_data in all_results.items()
        }
        out.write_text(_json.dumps(serializable, indent=2), encoding="utf-8")
        print(f"  Results saved to {out}", flush=True)


if __name__ == "__main__":
    main()
