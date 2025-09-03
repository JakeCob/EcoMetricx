## 04 — Retrieval and Indexing (Postgres FTS + Qdrant)

### Postgres schema
Postgres is FTS-only (documents + chunks + tsvector), vectors live in Qdrant.

**Tables:**
- `documents(document_id PK, ..., metadata jsonb)`
- `chunks(chunk_id PK, parent_document_id FK, text, page_num, section_path, ..., text_tsv tsvector)`

**Indexes:**
- FTS: GIN on `chunks.text_tsv` (auto-maintained trigger)

### Qdrant
Vector storage and similarity search.

**Configuration:**
- QDRANT_URL, QDRANT_API_KEY, QDRANT_COLLECTION=ecometricx, QDRANT_DIM=384
- Ingest embeddings: `python scripts/ingest_to_qdrant.py`

### Ingestion
- `scripts/ingest_to_postgres.py` reads normalized/chunks for current `run_id` and upserts (FTS-only).
- `scripts/ingest_to_qdrant.py` reads embeddings and upserts to Qdrant vector store.
- Requires `DATABASE_URL` env var for Postgres and Qdrant env vars.

### Hybrid search (example)
```sql
-- FTS (Postgres)
SELECT chunk_id, ts_rank_cd(text_tsv, plainto_tsquery('english', $1)) AS bm25
FROM chunks
WHERE text_tsv @@ plainto_tsquery('english', $1)
ORDER BY bm25 DESC LIMIT 50;
```

**Vector search** is handled via Qdrant HTTP API. The API blends Qdrant cosine with Postgres BM25 (alpha=FUSION_ALPHA) and returns citations.