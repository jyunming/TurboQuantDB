"""
Scale-matrix benchmark — multiple vector counts × multiple engines.

Sizes tested: 10k, 25k, 50k, 100k, 500k
Engines: TQDB (HQ/Balanced/FastBuild), FAISS HNSW, FAISS IVF-PQ, ChromaDB, Qdrant, LanceDB

Metrics per cell: ingest time, vec/s, disk MB, RAM (ingest peak + query),
                  CPU% (ingest + query), p50/p97/p99 latency, Recall@10, MRR@10

Usage:
    python benchmarks/scale_matrix_bench.py
    python benchmarks/scale_matrix_bench.py --sizes 10000,50000,100000
    python benchmarks/scale_matrix_bench.py --engines tqdb_hq,tqdb_bal,faiss_hnsw
"""

import argparse
import gc
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time

import numpy as np
import psutil

sys.stdout.reconfigure(encoding="utf-8")

PYTHON   = sys.executable
WORKER   = __file__
DATASET  = "KShivendu/dbpedia-entities-openai-1M"
VEC_FIELD = "openai"

DEFAULT_SIZES   = [10_000, 25_000, 50_000, 100_000, 500_000]
DEFAULT_ENGINES = [
    "tqdb_hq", "tqdb_bal", "tqdb_fast",
    "faiss_hnsw", "faiss_ivfpq",
    "chromadb", "qdrant", "lancedb",
]

ENGINE_LABELS = {
    "tqdb_hq":    "TQDB b=8 HQ",
    "tqdb_bal":   "TQDB b=4 Balanced",
    "tqdb_fast":  "TQDB b=4 FastBuild",
    "faiss_hnsw": "FAISS HNSW",
    "faiss_ivfpq":"FAISS IVF-PQ",
    "chromadb":   "ChromaDB (HNSW)",
    "qdrant":     "Qdrant (HNSW)",
    "lancedb":    "LanceDB (IVF-PQ)",
}

# TQDB engine params: (bits, fast_mode, rerank, max_degree, ef_construction, n_refinements, ann_sl)
TQDB_PARAMS = {
    "tqdb_hq":   (8, False, True,  32, 200, 8, 200),
    "tqdb_bal":  (4, False, True,  32, 200, 5, 128),
    "tqdb_fast": (4, True,  False, 32, 200, 5, 128),
}


# ── Worker ─────────────────────────────────────────────────────────────────────

def run_worker_mode(args):
    engine  = args.worker_engine
    top_k   = args.top_k
    data_dir = args.worker_data_dir

    try:
        if sys.platform == "win32":
            import ctypes
            ctypes.windll.kernel32.SetPriorityClass(
                ctypes.windll.kernel32.GetCurrentProcess(), 0x00000080)
    except Exception:
        pass

    vecs  = np.load(os.path.join(data_dir, "embeddings.npy"))
    qvecs = np.load(os.path.join(data_dir, "query_vecs.npy"))
    truth = np.load(os.path.join(data_dir, "ground_truth.npy"))
    with open(os.path.join(data_dir, "meta.json"), encoding="utf-8") as f:
        meta = json.load(f)

    ids       = meta["ids"]
    documents = meta["documents"]
    N, DIM    = vecs.shape
    NQ        = len(qvecs)

    def rss_mb():
        gc.collect()
        return psutil.Process().memory_info().rss / 1024 / 1024

    tmp = tempfile.mkdtemp(prefix=f"sm_{engine}_")
    ingest_peak_rss = rss_mb()
    found_indices   = []
    id_to_idx       = {pid: i for i, pid in enumerate(ids)}

    try:
        proc = psutil.Process()

        # ── INGEST ────────────────────────────────────────────────────────────
        t0   = time.perf_counter()
        cpu0 = proc.cpu_times()

        if engine in TQDB_PARAMS:
            bits, fast_mode, rerank, max_degree, ef_construction, n_refinements, ann_sl = \
                TQDB_PARAMS[engine]
            import turboquantdb
            db_path = os.path.join(tmp, "tqdb")
            db = turboquantdb.Database.open(
                db_path, dimension=DIM, bits=bits, metric="ip",
                rerank=rerank, fast_mode=fast_mode)
            db.insert_batch(ids, vecs, documents=documents)
            ingest_peak_rss = max(ingest_peak_rss, rss_mb())
            db.create_index(max_degree=max_degree, ef_construction=ef_construction,
                            search_list_size=ann_sl, n_refinements=n_refinements)
            ingest_peak_rss = max(ingest_peak_rss, rss_mb())

        elif engine == "faiss_hnsw":
            import faiss
            index = faiss.IndexHNSWFlat(DIM, 32, faiss.METRIC_INNER_PRODUCT)
            index.hnsw.efConstruction = 200
            # normalize for inner product (already normalized at load time)
            faiss.normalize_L2(vecs)
            index.add(vecs)
            ingest_peak_rss = max(ingest_peak_rss, rss_mb())
            faiss.write_index(index, os.path.join(tmp, "faiss.index"))
            ingest_peak_rss = max(ingest_peak_rss, rss_mb())

        elif engine == "faiss_ivfpq":
            import faiss
            nlist   = min(4096, max(64, N // 40))
            m       = max(1, DIM // 8)   # subvectors (must divide DIM)
            quantizer = faiss.IndexFlatIP(DIM)
            index   = faiss.IndexIVFPQ(quantizer, DIM, nlist, m, 8,
                                        faiss.METRIC_INNER_PRODUCT)
            faiss.normalize_L2(vecs)
            index.train(vecs)
            index.add(vecs)
            index.nprobe = max(1, nlist // 16)
            ingest_peak_rss = max(ingest_peak_rss, rss_mb())
            faiss.write_index(index, os.path.join(tmp, "faiss.index"))
            ingest_peak_rss = max(ingest_peak_rss, rss_mb())

        elif engine == "chromadb":
            import chromadb
            _client = chromadb.PersistentClient(path=tmp)
            col = _client.create_collection("vecs", metadata={"hnsw:space": "ip"})
            BS = 2000
            for i in range(0, N, BS):
                end = min(i + BS, N)
                col.add(ids=ids[i:end],
                        embeddings=vecs[i:end].tolist(),
                        documents=documents[i:end])
                ingest_peak_rss = max(ingest_peak_rss, rss_mb())

        elif engine == "qdrant":
            from qdrant_client import QdrantClient
            from qdrant_client.models import (Distance, VectorParams, PointStruct,
                                               SearchParams, OptimizersConfigDiff,
                                               HnswConfigDiff)
            _qclient = QdrantClient(path=os.path.join(tmp, "qdrant"))
            _qclient.create_collection(
                "bench",
                vectors_config=VectorParams(size=DIM, distance=Distance.DOT),
                hnsw_config=HnswConfigDiff(m=32, ef_construct=200),
                optimizers_config=OptimizersConfigDiff(indexing_threshold=N + 1))
            BS = 2000
            for i in range(0, N, BS):
                end = min(i + BS, N)
                _qclient.upsert("bench", points=[
                    PointStruct(id=j, vector=vecs[j].tolist())
                    for j in range(i, end)])
                ingest_peak_rss = max(ingest_peak_rss, rss_mb())
            _qclient.update_collection(
                "bench", optimizers_config=OptimizersConfigDiff(indexing_threshold=0))
            for _ in range(300):
                if _qclient.get_collection("bench").status.value == "green":
                    break
                time.sleep(1.0)
            ingest_peak_rss = max(ingest_peak_rss, rss_mb())

        elif engine == "lancedb":
            import lancedb
            _ldb = lancedb.connect(tmp)
            BS = 5000
            rows, _ltable = [], None
            for i in range(0, N, BS):
                end = min(i + BS, N)
                rows = [{"vector": vecs[j].tolist(), "id": ids[j], "doc": documents[j]}
                        for j in range(i, end)]
                if _ltable is None:
                    _ltable = _ldb.create_table("v", data=rows)
                else:
                    _ltable.add(rows)
                ingest_peak_rss = max(ingest_peak_rss, rss_mb())
            nlist = min(256, max(4, N // 40))
            _ltable.create_index(num_partitions=nlist,
                                 num_sub_vectors=max(1, DIM // 8))
            ingest_peak_rss = max(ingest_peak_rss, rss_mb())

        ingest_wall = time.perf_counter() - t0
        cpu1 = proc.cpu_times()
        ingest_cpu = ((cpu1.user - cpu0.user) + (cpu1.system - cpu0.system)) / ingest_wall * 100

        disk_mb = sum(
            os.path.getsize(os.path.join(dp, fn))
            for dp, _, fnames in os.walk(tmp)
            for fn in fnames
            if not os.path.islink(os.path.join(dp, fn))
        ) / 1024 / 1024

        gc.collect()
        retrieve_rss = rss_mb()

        # ── QUERY ─────────────────────────────────────────────────────────────
        latencies = []
        cpu0q = proc.cpu_times()
        tq0   = time.perf_counter()

        for qi, qvec in enumerate(qvecs):
            t0 = time.perf_counter()

            if engine in TQDB_PARAMS:
                _, _, _, _, _, _, ann_sl = TQDB_PARAMS[engine]
                res  = db.search(qvec, top_k=top_k, ann_search_list_size=ann_sl)
                hits = [id_to_idx.get(r.get("id", ""), -1) for r in res]

            elif engine == "faiss_hnsw":
                qn = qvec.reshape(1, -1).copy()
                faiss.normalize_L2(qn)
                index.hnsw.efSearch = 128
                _, I = index.search(qn, top_k)
                hits = I[0].tolist()

            elif engine == "faiss_ivfpq":
                qn = qvec.reshape(1, -1).copy()
                faiss.normalize_L2(qn)
                _, I = index.search(qn, top_k)
                hits = I[0].tolist()

            elif engine == "chromadb":
                res  = col.query(query_embeddings=[qvec.tolist()], n_results=top_k)
                hits = [id_to_idx.get(x, -1) for x in res["ids"][0]]

            elif engine == "qdrant":
                res  = _qclient.query_points("bench", query=qvec.tolist(), limit=top_k,
                                              search_params=SearchParams(hnsw_ef=128))
                hits = [p.id for p in res.points]

            elif engine == "lancedb":
                rows = _ltable.search(qvec).limit(top_k).nprobes(20).to_list()
                hits = [id_to_idx.get(str(r.get("id", "")), -1) for r in rows]

            latencies.append((time.perf_counter() - t0) * 1000)
            found_indices.append(hits)

        query_wall = time.perf_counter() - tq0
        cpu1q = proc.cpu_times()
        query_cpu = ((cpu1q.user - cpu0q.user) + (cpu1q.system - cpu0q.system)) / query_wall * 100

        # ── RECALL ────────────────────────────────────────────────────────────
        recall_scores, mrr_scores = [], []
        for qi in range(NQ):
            gt  = set(int(x) for x in truth[qi])
            got = set(int(x) for x in found_indices[qi] if x >= 0)
            recall_scores.append(len(gt & got) / len(gt) if gt else 0.0)
            mrr = 0.0
            for rank, idx in enumerate(found_indices[qi], 1):
                if int(idx) in gt:
                    mrr = 1.0 / rank
                    break
            mrr_scores.append(mrr)

        print(json.dumps({
            "ingest_time":     ingest_wall,
            "ingest_speed":    N / ingest_wall,
            "disk_mb":         disk_mb,
            "ingest_rss_mb":   ingest_peak_rss,
            "retrieve_rss_mb": retrieve_rss,
            "ingest_cpu_util": ingest_cpu,
            "query_cpu_util":  query_cpu,
            "p50_ms":  float(np.percentile(latencies, 50)),
            "p97_ms":  float(np.percentile(latencies, 97)),
            "p99_ms":  float(np.percentile(latencies, 99)),
            "recall_at_k": float(np.mean(recall_scores)),
            "mrr":         float(np.mean(mrr_scores)),
        }))

    finally:
        gc.collect()
        if engine == "chromadb":
            try:
                del col, _client; gc.collect(); time.sleep(0.5)
            except Exception:
                pass
        if engine == "qdrant":
            try:
                del _qclient; gc.collect(); time.sleep(0.5)
            except Exception:
                pass
        if engine == "lancedb":
            try:
                del _ltable, _ldb; gc.collect(); time.sleep(0.5)
            except Exception:
                pass
        shutil.rmtree(tmp, ignore_errors=True)


# ── Harness ────────────────────────────────────────────────────────────────────

def load_or_download(n_vecs, n_queries, top_k, ckpt_dir):
    os.makedirs(ckpt_dir, exist_ok=True)
    tag       = f"dbpedia_{n_vecs}"
    ckpt_meta = os.path.join(ckpt_dir, f"{tag}_meta.json")
    ckpt_vecs = os.path.join(ckpt_dir, f"{tag}_vecs.npy")

    if os.path.exists(ckpt_meta) and os.path.exists(ckpt_vecs):
        print(f"    Loading from checkpoint ({n_vecs:,} vecs)...", flush=True)
        with open(ckpt_meta, encoding="utf-8") as f:
            saved = json.load(f)
        vecs = np.load(ckpt_vecs)
    else:
        print(f"    Downloading {n_vecs:,} vectors from HuggingFace...", flush=True)
        from datasets import load_dataset
        t0 = time.perf_counter()
        ds = load_dataset(DATASET, split="train", streaming=True)
        ids_list, docs_list, raw_vecs = [], [], []
        for i, row in enumerate(ds):
            if i >= n_vecs:
                break
            ids_list.append(str(row["_id"]))
            docs_list.append(f"{row['title']} — {row['text'][:200]}")
            raw_vecs.append(row[VEC_FIELD])
            if (i + 1) % 10_000 == 0:
                print(f"      {i+1:>7,}/{n_vecs:,}", flush=True)
        vecs = np.array(raw_vecs, dtype=np.float32)
        norms = np.linalg.norm(vecs, axis=1, keepdims=True)
        vecs /= np.where(norms > 0, norms, 1.0)
        saved = {"ids": ids_list, "documents": docs_list}
        np.save(ckpt_vecs, vecs)
        with open(ckpt_meta, "w", encoding="utf-8") as f:
            json.dump(saved, f)
        print(f"    Downloaded & saved ({time.perf_counter()-t0:.1f}s)")

    ids       = saved["ids"]
    documents = saved["documents"]
    N, DIM    = vecs.shape

    # query/truth
    ckpt_qvecs = os.path.join(ckpt_dir, f"{tag}_qvecs.npy")
    ckpt_truth = os.path.join(ckpt_dir, f"{tag}_truth_{top_k}.npy")

    if os.path.exists(ckpt_qvecs) and os.path.exists(ckpt_truth):
        qvecs        = np.load(ckpt_qvecs)
        ground_truth = np.load(ckpt_truth)
    else:
        print(f"    Building {n_queries} queries + brute-force truth...", flush=True)
        rng = np.random.default_rng(42)
        q_idx = rng.choice(N, size=n_queries, replace=False)
        qvecs = vecs[q_idx].copy()
        t0 = time.perf_counter()
        BF_BS = 50
        truth_rows = []
        for i in range(0, n_queries, BF_BS):
            scores = qvecs[i:i+BF_BS] @ vecs.T
            truth_rows.append(np.argsort(-scores, axis=1)[:, :top_k])
        ground_truth = np.vstack(truth_rows)
        print(f"    Ground truth in {time.perf_counter()-t0:.1f}s")
        np.save(ckpt_qvecs, qvecs)
        np.save(ckpt_truth, ground_truth)

    return vecs, qvecs, ground_truth, ids, documents, DIM


def spawn_worker(engine, data_dir, top_k):
    cmd = [PYTHON, WORKER,
           "--worker_engine",   engine,
           "--worker_data_dir", data_dir,
           "--top_k",           str(top_k)]
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=7200)
        for line in proc.stdout.strip().splitlines():
            try:
                return json.loads(line)
            except json.JSONDecodeError:
                continue
        print(f"    [ERROR] {engine}:\n{proc.stderr[-1500:]}", file=sys.stderr)
        return None
    except subprocess.TimeoutExpired:
        print(f"    [TIMEOUT] {engine}", file=sys.stderr)
        return None


def fmt(v, unit="", d=1):
    return f"{v:.{d}f}{unit}" if v is not None else "n/a"


def print_size_table(n_vecs, dim, top_k, engines, results):
    bar = "=" * 100
    print(f"\n{bar}")
    print(f"  n_vecs={n_vecs:,}  dim={dim}  top_k={top_k}  (float32 raw = {n_vecs*dim*4/1024/1024:.0f} MB)")
    print(bar)
    hdr = f"  {'Engine':<22}  {'Ingest':>8}  {'vec/s':>6}  {'Disk':>8}  {'RAM-i':>7}  {'RAM-q':>7}  {'CPU-i':>6}  {'CPU-q':>6}  {'p50':>7}  {'p97':>7}  {'p99':>7}  {'R@10':>6}  {'MRR':>6}"
    sep = f"  {'-'*22}  {'-'*8}  {'-'*6}  {'-'*8}  {'-'*7}  {'-'*7}  {'-'*6}  {'-'*6}  {'-'*7}  {'-'*7}  {'-'*7}  {'-'*6}  {'-'*6}"
    print(hdr)
    print(sep)
    for eng in engines:
        label = ENGINE_LABELS.get(eng, eng)
        r = results.get(eng)
        if not r:
            print(f"  {label:<22}  FAILED / SKIPPED")
            continue
        print(f"  {label:<22}  "
              f"{fmt(r['ingest_time'],'s'):>8}  "
              f"{int(r['ingest_speed']):>6,}  "
              f"{fmt(r['disk_mb'],'MB'):>8}  "
              f"{fmt(r['ingest_rss_mb'],'MB',0):>7}  "
              f"{fmt(r['retrieve_rss_mb'],'MB',0):>7}  "
              f"{fmt(r['ingest_cpu_util'],'%',0):>6}  "
              f"{fmt(r['query_cpu_util'],'%',0):>6}  "
              f"{fmt(r['p50_ms'],'ms',2):>7}  "
              f"{fmt(r['p97_ms'],'ms',2):>7}  "
              f"{fmt(r['p99_ms'],'ms',2):>7}  "
              f"{r['recall_at_k']*100:>5.1f}%  "
              f"{r['mrr']:>6.4f}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sizes", default=",".join(str(s) for s in DEFAULT_SIZES),
                        help="Comma-separated list of vector counts")
    parser.add_argument("--engines", default=",".join(DEFAULT_ENGINES),
                        help="Comma-separated engine keys")
    parser.add_argument("--n_queries", type=int, default=200)
    parser.add_argument("--top_k",     type=int, default=10)
    parser.add_argument("--checkpoint_dir", default=os.path.join(
        os.path.dirname(__file__), "..", "tmp_ls_checkpoint"))
    parser.add_argument("--out", default="bench_results_scale_matrix.json")
    # Worker-mode args (internal use)
    parser.add_argument("--worker_engine",   default=None)
    parser.add_argument("--worker_data_dir", default=None)
    args = parser.parse_args()

    if args.worker_engine:
        run_worker_mode(args)
        return

    sizes   = [int(x) for x in args.sizes.split(",")]
    engines = [x.strip() for x in args.engines.split(",")]
    ckpt    = os.path.abspath(args.checkpoint_dir)

    all_results = {}  # {n_vecs: {engine: result_dict}}

    print(f"\nScale-Matrix Benchmark")
    print(f"Engines : {', '.join(ENGINE_LABELS.get(e,e) for e in engines)}")
    print(f"Sizes   : {', '.join(f'{s:,}' for s in sizes)}")
    print(f"Queries : {args.n_queries}  top_k={args.top_k}\n")

    for n_vecs in sizes:
        print(f"\n{'─'*60}")
        print(f"  SIZE: {n_vecs:,} vectors")
        print(f"{'─'*60}")

        vecs, qvecs, truth, ids, documents, DIM = load_or_download(
            n_vecs, args.n_queries, args.top_k, ckpt)

        # Write shared worker data
        data_dir = os.path.join(ckpt, f"worker_data_{n_vecs}")
        os.makedirs(data_dir, exist_ok=True)
        np.save(os.path.join(data_dir, "embeddings.npy"),   vecs)
        np.save(os.path.join(data_dir, "query_vecs.npy"),   qvecs)
        np.save(os.path.join(data_dir, "ground_truth.npy"), truth)
        with open(os.path.join(data_dir, "meta.json"), "w", encoding="utf-8") as f:
            json.dump({"ids": ids, "documents": documents}, f)

        size_results = {}
        for eng in engines:
            label = ENGINE_LABELS.get(eng, eng)
            print(f"  [{n_vecs:,}] Running {label} ...", flush=True)
            time.sleep(2)
            r = spawn_worker(eng, data_dir, args.top_k)
            if r:
                print(f"    ingest={r['ingest_time']:.1f}s  "
                      f"disk={r['disk_mb']:.1f}MB  "
                      f"RAM={r['retrieve_rss_mb']:.0f}MB  "
                      f"p50={r['p50_ms']:.2f}ms  "
                      f"recall@{args.top_k}={r['recall_at_k']*100:.1f}%  "
                      f"MRR={r['mrr']:.4f}")
            else:
                print(f"    FAILED")
            size_results[eng] = r

        all_results[str(n_vecs)] = size_results
        print_size_table(n_vecs, DIM, args.top_k, engines, size_results)

        # Save incremental results
        with open(args.out, "w", encoding="utf-8") as f:
            json.dump(all_results, f, indent=2)

        # Clean up per-size worker data dir to save disk
        shutil.rmtree(data_dir, ignore_errors=True)

    print(f"\nResults saved to {args.out}")


if __name__ == "__main__":
    main()
