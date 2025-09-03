## 01 — Data Architecture

### Scope
This document defines where EcoMetricx stores raw extractor outputs, normalized documents, retrieval chunks, indices, and manifests, and records storage/retention decisions for Railway deployment.

### Storage decisions
- Object storage: Cloudflare R2 (S3-compatible), region: WNAM.
- No AWS dependency. Access via standard S3 SDK credentials.
- Keep unredacted raw data only under `data/raw/`. Redacted text exists in chunks.
- Retention policy: keep last 10 runs or cap total DB size at ~2 GB for embeddings/chunks (whichever first). Oldest superseded runs pruned.

Environment variables (Railway):
```
R2_ACCESS_KEY_ID=...
R2_SECRET_ACCESS_KEY=...
R2_ENDPOINT=https://<accountid>.r2.cloudflarestorage.com
R2_REGION=wnam
R2_BUCKET=ecometricx-artifacts
R2_PUBLIC_BASE_URL=https://pub-<domain>/ecometricx-artifacts
```

### Directory layout
- `data/raw/{source}/{run_id}/...`: Immutable copy of current extractor outputs.
- `data/normalized/{source}/{run_id}/documents.jsonl`: Canonical documents.
- `data/chunks/{source}/{run_id}/chunks.jsonl`: Retrieval units (redacted when enabled).
- `data/index/pgvector/{run_id}/index_manifest.json`: Embedding/index stats.
- `data/manifests/{run_id}.json`: Inventory with checksums, sizes, file list.

Notes:
- `source` examples: `visual_extraction`, `enhanced_pdf`.
- `run_id` format: `YYYYMMDD_HHMMSS`.

### Manifests
Each run produces `data/manifests/{run_id}.json` with:
- run_id, source, raw_dir, created_at
- totals: files, bytes
- by_extension: count and bytes per extension
- files: relative path, size, sha256, mime_type, modified

### Governance and PII
- No external governance constraints provided.
- Redaction occurs at chunking stage for names, addresses, account numbers.
- Raw unredacted data is kept only in `data/raw/` and not exposed to retrieval.

### Retrieval targets and acceptance
- Retrieval baseline: Recall@10 ≥ 0.70 (MVP); stretch ≥ 0.80 with reranker.
- Latency target: P95 < 500 ms for top-20 retrieval without reranker on CPU.

### Future integration (summary)
- DB: Postgres 16 on Railway with pgvector + FTS.
- Embeddings: `bge-small-en-v1.5` (CPU), cached.
- Hybrid scoring: 0.6 * cosine + 0.4 * BM25; optional reranker flag.


