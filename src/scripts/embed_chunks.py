#!/usr/bin/env python3
import json
from pathlib import Path
from datetime import datetime, timezone

MODEL_NAME = "BAAI/bge-small-en-v1.5"


def main():
    root = Path('.')
    run_id = (root / '.current_run_id').read_text().strip()
    source = 'visual_extraction'
    chunks_file = root / 'data' / 'chunks' / source / run_id / 'chunks.jsonl'
    out_dir = root / 'data' / 'index' / 'pgvector' / run_id
    out_dir.mkdir(parents=True, exist_ok=True)
    emb_file = out_dir / 'embeddings.jsonl'
    manifest_file = out_dir / 'index_manifest.json'

    assert chunks_file.exists(), f"Missing chunks: {chunks_file}"

    # Lazy import to avoid env errors
    from fastembed import TextEmbedding
    embedder = TextEmbedding(MODEL_NAME)

    # Read chunks
    chunks = [json.loads(l) for l in chunks_file.read_text(encoding='utf-8').splitlines() if l.strip()]
    texts = [c['text'] for c in chunks]

    # Compute embeddings (generator yields batches)
    vectors = []
    for out in embedder.embed(texts, batch_size=64):
        try:
            # If a batch array is returned (N, D), flatten rows
            if hasattr(out, 'shape') and len(getattr(out, 'shape', ())) == 2:
                for row in out:
                    vectors.append(row)
            else:
                vectors.append(out)
        except Exception:
            vectors.append(out)

    # Write embeddings aligned by index
    with emb_file.open('w', encoding='utf-8') as f:
        for c, v in zip(chunks, vectors):
            rec = {
                'chunk_id': c['chunk_id'],
                'parent_document_id': c['parent_document_id'],
                'embedding_model': MODEL_NAME,
                'embedding_vector': list(map(float, v)),
            }
            f.write(json.dumps(rec) + '\n')

    manifest = {
        'run_id': run_id,
        'source': source,
        'model': MODEL_NAME,
        'embedding_dim': len(vectors[0]) if vectors else 0,
        'chunks_embedded': min(len(vectors), len(chunks)),
        'created_at': datetime.now(timezone.utc).isoformat()
    }
    manifest_file.write_text(json.dumps(manifest, indent=2), encoding='utf-8')
    print(f"Wrote {emb_file} and {manifest_file}")


if __name__ == '__main__':
    main()


