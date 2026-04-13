# TurboQuantDB Server

Optional Axum HTTP service providing TurboQuantDB in multi-tenant server mode. Use this when you need REST API access, multi-tenancy, authentication, quotas, or async job management. For single-process Python use, the embedded `tqdb` package is simpler.

## Quick Start

The server binary ships pre-built inside the `tqdb` wheel on all supported platforms
(Linux x86-64, macOS, Windows). **Linux arm64/aarch64** is the exception — see
[Building from Source](#building-from-source-development-only) for that platform.

```bash
pip install tqdb
tqdb-server          # listens on 127.0.0.1:8080 by default
```

Configure via environment variables before launching (see [Environment Variables](#environment-variables) below).

## Building from Source (development only)

```bash
cd server
cargo run --release
```

## Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `TQ_SERVER_ADDR` | `127.0.0.1:8080` | Listen address |
| `TQ_LOCAL_ROOT` | `./data` | Root directory for all data files |
| `TQ_STORAGE_URI` | `TQ_LOCAL_ROOT` | Storage URI (file:// path) |
| `TQ_AUTH_STORE_PATH` | `<TQ_LOCAL_ROOT>/auth_store.json` | API key + RBAC store |
| `TQ_QUOTA_STORE_PATH` | `<TQ_LOCAL_ROOT>/quota_store.json` | Quota limits store |
| `TQ_JOB_STORE_PATH` | `<TQ_LOCAL_ROOT>/job_store.json` | Async job state store |
| `TQ_JOB_WORKERS` | `2` | Concurrent async job workers |

## API Endpoints

### Health
- `GET /healthz`
- `GET /metrics` — Prometheus text metrics (requires `Authorization: ApiKey <key>`)
- `GET/POST /v1/tenants/:tenant/databases/:database/collections`
- `DELETE /v1/tenants/:tenant/databases/:database/collections/:collection`
- `POST /v1/tenants/:tenant/databases/:database/collections/:collection/add`
- `POST /v1/tenants/:tenant/databases/:database/collections/:collection/upsert`
- `POST /v1/tenants/:tenant/databases/:database/collections/:collection/delete`
- `POST /v1/tenants/:tenant/databases/:database/collections/:collection/get`
- `POST /v1/tenants/:tenant/databases/:database/collections/:collection/query`
- `POST /v1/tenants/:tenant/databases/:database/collections/:collection/compact`
- `POST /v1/tenants/:tenant/databases/:database/collections/:collection/index`
- `POST /v1/tenants/:tenant/databases/:database/collections/:collection/snapshot`
- `POST /v1/tenants/:tenant/databases/:database/collections/:collection/restore`
- `GET /v1/tenants/:tenant/databases/:database/collections/:collection/jobs`
- `GET /v1/tenants/:tenant/databases/:database/quota_usage`
- `GET /v1/jobs/:job_id`
- `POST /v1/jobs/:job_id/cancel`
- `POST /v1/jobs/:job_id/retry`

### Collection Management
- `GET /v1/tenants/:tenant/databases/:database/collections` — List collections
- `POST /v1/tenants/:tenant/databases/:database/collections` — Create collection
- `DELETE /v1/tenants/:tenant/databases/:database/collections/:collection` — Delete collection

### Data Plane
- `POST .../add` — Batch insert; supports `report=true` for partial-failure reporting
- `POST .../upsert` — Batch insert-or-update; supports `report=true`
- `POST .../delete` — Delete vectors by IDs
- `POST .../get` — Fetch vectors by IDs; supports `include` (`ids`, `metadatas`, `documents`), `offset`, `limit`
- `POST .../query` — Vector similarity search; supports `include` (`ids`, `scores`, `metadatas`, `documents`), `offset`

### Async Jobs
- `POST .../compact` — Start background compaction
- `POST .../index` — Start background HNSW index build
- `POST .../snapshot` — Start background snapshot
- `POST .../restore` — Start background restore from a snapshot
- `GET .../jobs` — List jobs for a collection
- `GET /v1/jobs/:job_id` — Get job status
- `POST /v1/jobs/:job_id/cancel` — Cancel a job
- `POST /v1/jobs/:job_id/retry` — Retry a failed job

## Features

- **Authentication** — API keys with RBAC scoped to tenant/database/collection level, persisted in `auth_store.json`
- **Quotas** — Per-collection limits on vector count, disk bytes, and concurrent jobs, persisted in `quota_store.json`
- **Async jobs** — Compaction, index build, snapshots, and restores run in background workers; restart-safe with up to 3 retry attempts, state persisted in `job_store.json`
- **Partial-failure reporting** — `add` and `upsert` with `report=true` return `{applied: N, failed: [...]}` instead of fail-fast

### Data-Plane Request Notes

- `POST .../add` and `POST .../upsert` support optional `report` (when `true`, returns partial-failure report with `applied` and `failed[]` instead of fail-fast).
- `POST .../get` supports optional selectors (`ids`, `filter`, `where_filter`) plus `include`, `offset`, `limit`.
- `POST .../query` supports `top_k` (or alias `n_results`), `filter` (or alias `where_filter`), optional `include`, and `offset`.
- `include` defaults to all allowed fields for each endpoint.

## Request Notes

- `include` defaults to all allowed fields if omitted
- `offset` and `limit` in `get` enable pagination
- Job state survives server restarts (job store is persisted to disk)
