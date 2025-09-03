## 02 — Canonical Schemas (v1.0.0)

### Document
- Required: `schema_version(1.0.0)`, `pipeline_version`, `run_id`, `document_id`, `source_id`, `source_type`, `uri`, `title`, `language(en)`, `mime_type`, `created_at`, `extracted_at`, `version`, `checksum`, `text`
- Optional: `markdown`, `html`, `pages[{page_number,text,confidence}]`, `metadata{...}`

### Chunk
- Required: `schema_version(1.0.0)`, `pipeline_version`, `run_id`, `chunk_id`, `parent_document_id`, `chunk_index`, `start_char`, `end_char`, `text`
- Optional: `page_num`, `section_path`, `content_type`, `tokens`, `quality_score`, `metadata{...}`

### IDs and versions
- `document_id = emx:{source}:{checksum[:12]}` for stability.
- `chunk_id = {document_id}:c{chunk_index}`.
- Include `schema_version` and `pipeline_version` in every record.

### Compatibility policy
- Minor schema changes must be backward compatible.
- Breaking changes bump `schema_version` major and require migration notes.


