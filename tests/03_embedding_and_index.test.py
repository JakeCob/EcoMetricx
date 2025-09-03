import json
from pathlib import Path


def test_embeddings_manifest_and_counts():
	root = Path(__file__).resolve().parents[1]
	run_id = (root / '.current_run_id').read_text().strip()
	chunks = (root / 'data' / 'chunks' / 'visual_extraction' / run_id / 'chunks.jsonl').read_text().splitlines()
	emb_dir = root / 'data' / 'index' / 'pgvector' / run_id
	manifest = json.loads((emb_dir / 'index_manifest.json').read_text())
	embs = (emb_dir / 'embeddings.jsonl').read_text().splitlines()
	assert len(embs) == len(chunks)
	assert manifest['embedding_dim'] > 0
	assert manifest['chunks_embedded'] == len(chunks)


