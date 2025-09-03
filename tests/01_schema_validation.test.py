import json
from pathlib import Path
from jsonschema import validate


def test_documents_schema_validates():
	root = Path(__file__).resolve().parents[1]
	source = 'visual_extraction'
	run_id = Path(root / '.current_run_id').read_text().strip()
	doc_path = root / 'data' / 'normalized' / source / run_id / 'documents.jsonl'
	assert doc_path.exists(), 'documents.jsonl missing'
	schema = json.loads((root / 'schemas' / 'document.schema.json').read_text(encoding='utf-8'))
	with doc_path.open('r', encoding='utf-8') as f:
		for line in f:
			if not line.strip():
				continue
			obj = json.loads(line)
			validate(instance=obj, schema=schema)
