"""BM25 retrieval quality on a BEIR subset, before and after text analysis.

Roadmap card [v0.9.0] feat(bm25) asks for "BM25 recall on BEIR subset before/after".
"Before" is the pre-0.9 analyzer (split + lowercase only, `text_language="none"`);
"after" is the 0.9 default (Snowball stemming + bundled stopwords).

Only the sparse leg is measured: every document gets the same dense vector and the
hybrid weight is pinned to 1.0, so the fused ranking is the BM25 ranking. That keeps
the number attributable to text analysis alone.

Usage:
    python benchmarks/bench_bm25_beir.py                 # scifact, k=10
    python benchmarks/bench_bm25_beir.py --dataset nfcorpus --k 10 20
"""

from __future__ import annotations

import argparse
import json
import tempfile
import time
from pathlib import Path

import numpy as np

from tqdb import Database

DIM = 8  # the dense leg is inert here; keep it tiny
RESULTS_PATH = Path(__file__).parent / "_bm25_beir_results.json"


def load_beir(dataset: str) -> tuple[dict[str, str], dict[str, str], dict[str, set[str]]]:
    """Return (corpus, queries, qrels) for a BEIR dataset, keyed by string ids."""
    from datasets import load_dataset  # noqa: PLC0415 — optional benchmark dep

    corpus_rows = load_dataset(f"BeIR/{dataset}", "corpus", split="corpus")
    query_rows = load_dataset(f"BeIR/{dataset}", "queries", split="queries")
    qrel_rows = load_dataset(f"BeIR/{dataset}-qrels", split="test")

    corpus = {
        str(r["_id"]): (f"{r['title']} {r['text']}".strip() if r["title"] else r["text"])
        for r in corpus_rows
    }
    qrels: dict[str, set[str]] = {}
    for r in qrel_rows:
        if int(r["score"]) > 0:
            qrels.setdefault(str(r["query-id"]), set()).add(str(r["corpus-id"]))
    # Only queries with judgements are scorable.
    queries = {str(r["_id"]): r["text"] for r in query_rows if str(r["_id"]) in qrels}
    return corpus, queries, qrels


def evaluate(corpus, queries, qrels, k_values, **open_kwargs) -> dict:
    """Index the corpus with one analyzer config and score BM25-only retrieval."""
    ids = list(corpus)
    docs = [corpus[i] for i in ids]
    vec = np.ones(DIM, dtype=np.float32)
    vectors = np.tile(vec, (len(ids), 1))

    with tempfile.TemporaryDirectory(prefix="bm25_beir_") as tmp:
        db = Database.open(tmp, dimension=DIM, bits=4, metric="ip", **open_kwargs)
        t0 = time.perf_counter()
        for start in range(0, len(ids), 2000):
            end = start + 2000
            db.insert_batch(ids[start:end], vectors[start:end], documents=docs[start:end])
        db.flush()
        index_s = time.perf_counter() - t0

        max_k = max(k_values)
        hits = {k: 0 for k in k_values}
        total = {k: 0 for k in k_values}
        latencies = []
        for qid, text in queries.items():
            relevant = qrels[qid]
            t1 = time.perf_counter()
            # weight=1.0 gives the dense leg zero weight: this is BM25 alone.
            results = db.search(vec, max_k, hybrid={"text": text, "weight": 1.0})
            latencies.append((time.perf_counter() - t1) * 1000.0)
            returned = [r["id"] for r in results]
            for k in k_values:
                found = len(relevant & set(returned[:k]))
                hits[k] += found
                total[k] += len(relevant)
        db.close()

    return {
        "index_s": round(index_s, 2),
        "p50_ms": round(float(np.percentile(latencies, 50)), 3),
        "recall": {str(k): round(hits[k] / max(total[k], 1), 4) for k in k_values},
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dataset", default="scifact", help="BEIR dataset name (default: scifact)")
    ap.add_argument("--k", nargs="+", type=int, default=[10], help="recall cutoffs")
    args = ap.parse_args()

    print(f"Loading BeIR/{args.dataset} ...", flush=True)
    corpus, queries, qrels = load_beir(args.dataset)
    print(f"  {len(corpus):,} documents, {len(queries):,} judged queries")

    configs = {
        "before (pre-0.9: split + lowercase)": {"text_language": "none", "stopwords": []},
        "after (0.9 default: stemming + stopwords)": {},
        "stemming only (no stopwords)": {"stopwords": []},
    }

    results = {}
    for label, kwargs in configs.items():
        print(f"\n{label} ...", flush=True)
        r = evaluate(corpus, queries, qrels, args.k, **kwargs)
        results[label] = r
        recalls = "  ".join(f"R@{k}={r['recall'][str(k)]:.4f}" for k in args.k)
        print(f"  {recalls}   index={r['index_s']}s  p50={r['p50_ms']}ms")

    before = results["before (pre-0.9: split + lowercase)"]
    after = results["after (0.9 default: stemming + stopwords)"]
    print("\n" + "=" * 64)
    print(f"BEIR/{args.dataset} — BM25-only retrieval")
    print("=" * 64)
    print(f"{'config':<44}" + "".join(f"R@{k:<8}" for k in args.k))
    for label, r in results.items():
        print(f"{label:<44}" + "".join(f"{r['recall'][str(k)]:<10.4f}" for k in args.k))
    for k in args.k:
        b = before["recall"][str(k)]
        a = after["recall"][str(k)]
        delta = a - b
        pct = (delta / b * 100.0) if b else float("nan")
        print(f"\nR@{k}: {b:.4f} -> {a:.4f}   ({delta:+.4f}, {pct:+.1f}%)")

    payload = {"dataset": args.dataset, "k": args.k, "results": results}
    RESULTS_PATH.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"\nSaved to {RESULTS_PATH}")


if __name__ == "__main__":
    main()
