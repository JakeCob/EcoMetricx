-- Enable extensions
CREATE EXTENSION IF NOT EXISTS vector;

-- Documents
CREATE TABLE IF NOT EXISTS documents (
  document_id TEXT PRIMARY KEY,
  source_id TEXT NOT NULL,
  source_type TEXT NOT NULL,
  uri TEXT NOT NULL,
  title TEXT NOT NULL,
  language TEXT NOT NULL,
  mime_type TEXT NOT NULL,
  created_at TIMESTAMPTZ NOT NULL,
  extracted_at TIMESTAMPTZ NOT NULL,
  version TEXT NOT NULL,
  checksum TEXT NOT NULL,
  schema_version TEXT NOT NULL,
  pipeline_version TEXT NOT NULL,
  run_id TEXT NOT NULL,
  metadata JSONB
);

-- Chunks
CREATE TABLE IF NOT EXISTS chunks (
  chunk_id TEXT PRIMARY KEY,
  parent_document_id TEXT NOT NULL REFERENCES documents(document_id) ON DELETE CASCADE,
  chunk_index INT NOT NULL,
  start_char INT NOT NULL,
  end_char INT NOT NULL,
  text TEXT NOT NULL,
  page_num INT,
  section_path TEXT,
  content_type TEXT,
  tokens INT,
  quality_score REAL,
  schema_version TEXT NOT NULL,
  pipeline_version TEXT NOT NULL,
  run_id TEXT NOT NULL,
  metadata JSONB
);

-- Embeddings (vector)
-- bge-small-en v1.5 -> 384 dims
CREATE TABLE IF NOT EXISTS chunk_embeddings (
  chunk_id TEXT PRIMARY KEY REFERENCES chunks(chunk_id) ON DELETE CASCADE,
  embedding_model TEXT NOT NULL,
  embedding VECTOR(384) NOT NULL
);

-- FTS index on chunks.text
ALTER TABLE IF EXISTS chunks
  ADD COLUMN IF NOT EXISTS text_tsv tsvector;

CREATE INDEX IF NOT EXISTS idx_chunks_text_tsv ON chunks USING GIN (text_tsv);

-- Triggers to keep tsvector updated
CREATE OR REPLACE FUNCTION chunks_tsvector_update() RETURNS trigger AS $$
BEGIN
  NEW.text_tsv := to_tsvector('english', coalesce(NEW.text,''));
  RETURN NEW;
END
$$ LANGUAGE plpgsql;

DROP TRIGGER IF EXISTS trg_chunks_tsvector_update ON chunks;
CREATE TRIGGER trg_chunks_tsvector_update BEFORE INSERT OR UPDATE OF text ON chunks
FOR EACH ROW EXECUTE PROCEDURE chunks_tsvector_update();

-- Vector index (HNSW if available; else IVFFLAT)
DO $$
BEGIN
  BEGIN
    EXECUTE 'CREATE INDEX IF NOT EXISTS idx_chunk_embeddings_hnsw ON chunk_embeddings USING hnsw (embedding vector_cosine_ops)';
  EXCEPTION WHEN others THEN
    EXECUTE 'CREATE INDEX IF NOT EXISTS idx_chunk_embeddings_ivf ON chunk_embeddings USING ivfflat (embedding vector_cosine_ops) WITH (lists = 100)';
  END;
END
$$;


