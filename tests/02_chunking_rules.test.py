import json
from pathlib import Path
from jsonschema import validate


def test_chunks_schema_and_redaction():
	root = Path(__file__).resolve().parents[1]
	run_id = (root / '.current_run_id').read_text().strip()
	chunk_path = root / 'data' / 'chunks' / 'visual_extraction' / run_id / 'chunks.jsonl'
	assert chunk_path.exists(), 'chunks.jsonl missing'
	schema = json.loads((root / 'schemas' / 'chunk.schema.json').read_text(encoding='utf-8'))
	lines = chunk_path.read_text(encoding='utf-8').splitlines()
	assert len(lines) > 0
	# Validate first few lines
	for line in lines[:5]:
		obj = json.loads(line)
		validate(instance=obj, schema=schema)
		text = obj['text']
		assert 'Account number' not in text or 'ACCOUNT_NUMBER_REDACTED' in text
		assert 'Service address' not in text or 'SERVICE_ADDRESS_REDACTED' in text
		assert 'Dear ' not in text or 'NAME_REDACTED' in text


