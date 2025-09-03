## 04 — Retrieval and Indexing (Postgres + pgvector + FTS)

### Tables
- `documents(document_id PK, ..., metadata jsonb)`
- `chunks(chunk_id PK, parent_document_id FK, text, page_num, section_path, ..., text_tsv tsvector)`
- `chunk_embeddings(chunk_id PK, embedding vector(384), embedding_model)`

### Indexes
- FTS: GIN on `chunks.text_tsv` (auto-maintained trigger)
- Vector: HNSW if available, else IVFFLAT on `chunk_embeddings.embedding`

### Ingestion
- `scripts/ingest_to_postgres.py` reads normalized/chunks/embeddings for current `run_id` and upserts.
- Requires `DATABASE_URL` env var.

### Hybrid search (example)
```
-- FTS
SELECT chunk_id, ts_rank_cd(text_tsv, plainto_tsquery('english', $1)) AS bm25
FROM chunks
WHERE text_tsv @@ plainto_tsquery('english', $1)
ORDER BY bm25 DESC LIMIT 50;

-- Vector
SELECT ce.chunk_id, 1 - (ce.embedding <=> $1::vector) AS cosine
FROM chunk_embeddings ce
ORDER BY ce.embedding <=> $1::vector ASC LIMIT 50;
```
Blend scores in app (e.g., 0.6*cosine + 0.4*bm25) and return citations.


