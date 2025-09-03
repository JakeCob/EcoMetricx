#!/usr/bin/env python3
import os, json
from pathlib import Path
from datetime import datetime
from typing import Iterable
from dotenv import load_dotenv
import psycopg
from psycopg.types.json import Json


def iter_jsonl(path: Path) -> Iterable[dict]:
    with path.open('r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                yield json.loads(line)


def main():
    load_dotenv()
    dsn = os.environ.get('DATABASE_URL') or os.environ.get('POSTGRES_DSN')
    assert dsn, 'Set DATABASE_URL or POSTGRES_DSN'

    root = Path('.')
    run_id = (root / '.current_run_id').read_text().strip()
    source = 'visual_extraction'

    docs_path = root / 'data' / 'normalized' / source / run_id / 'documents.jsonl'
    chunks_path = root / 'data' / 'chunks' / source / run_id / 'chunks.jsonl'
    assert docs_path.exists() and chunks_path.exists(), 'Missing inputs for ingestion'

    with psycopg.connect(dsn) as conn:
        conn.execute('BEGIN')
        
        # Upsert documents
        for d in iter_jsonl(docs_path):
            payload = d.copy()
            payload['metadata'] = Json(d.get('metadata'))
            conn.execute(
                '''INSERT INTO documents (
                    document_id, source_id, source_type, uri, title, language, mime_type,
                    created_at, extracted_at, version, checksum, schema_version, pipeline_version,
                    run_id, metadata
                ) VALUES (
                    %(document_id)s, %(source_id)s, %(source_type)s, %(uri)s, %(title)s, %(language)s, %(mime_type)s,
                    %(created_at)s, %(extracted_at)s, %(version)s, %(checksum)s, %(schema_version)s, %(pipeline_version)s,
                    %(run_id)s, %(metadata)s
                ) ON CONFLICT (document_id) DO UPDATE SET
                    source_id=EXCLUDED.source_id,
                    source_type=EXCLUDED.source_type,
                    uri=EXCLUDED.uri,
                    title=EXCLUDED.title,
                    language=EXCLUDED.language,
                    mime_type=EXCLUDED.mime_type,
                    created_at=EXCLUDED.created_at,
                    extracted_at=EXCLUDED.extracted_at,
                    version=EXCLUDED.version,
                    checksum=EXCLUDED.checksum,
                    schema_version=EXCLUDED.schema_version,
                    pipeline_version=EXCLUDED.pipeline_version,
                    run_id=EXCLUDED.run_id,
                    metadata=EXCLUDED.metadata''', payload)

        # Upsert chunks
        for c in iter_jsonl(chunks_path):
            payload = c.copy()
            payload['metadata'] = Json(c.get('metadata'))
            conn.execute(
                '''INSERT INTO chunks (
                    chunk_id, parent_document_id, chunk_index, start_char, end_char, text,
                    page_num, section_path, content_type, tokens, quality_score,
                    schema_version, pipeline_version, run_id, metadata
                ) VALUES (
                    %(chunk_id)s, %(parent_document_id)s, %(chunk_index)s, %(start_char)s, %(end_char)s, %(text)s,
                    %(page_num)s, %(section_path)s, %(content_type)s, %(tokens)s, %(quality_score)s,
                    %(schema_version)s, %(pipeline_version)s, %(run_id)s, %(metadata)s
                ) ON CONFLICT (chunk_id) DO UPDATE SET
                    text=EXCLUDED.text,
                    tokens=EXCLUDED.tokens,
                    quality_score=EXCLUDED.quality_score,
                    page_num=EXCLUDED.page_num,
                    section_path=EXCLUDED.section_path,
                    content_type=EXCLUDED.content_type,
                    run_id=EXCLUDED.run_id,
                    metadata=EXCLUDED.metadata''', payload)

        conn.execute('COMMIT')
        print('✅ Ingested documents and chunks (FTS-only)')


if __name__ == '__main__':
    main()