## 03 — Processing Pipeline (Chunking + Redaction)

### Chunking
- Page-aware sliding window, ~1000 tokens target, 120-token overlap.
- Generates `data/chunks/{source}/{run_id}/chunks.jsonl`.
- Fields include: ids, char spans, page number, section path, tokens, quality score, metadata.

### Redaction
- Applied at chunk time; raw text preserved in `data/raw` only.
- Default patterns: account numbers, service address lines, greeting name.
- Configurable patterns can be extended later.

### Quality Filters
- Min tokens: 30.
- Non-alphanumeric ratio threshold: ≤ 0.5.

### Scripts
- `scripts/chunk_and_redact.py`: reads normalized `documents.jsonl`, writes `chunks.jsonl`.

### Outputs
- `data/chunks/{source}/{run_id}/chunks.jsonl`


