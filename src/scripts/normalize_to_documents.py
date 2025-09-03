#!/usr/bin/env python3
import json, hashlib, os
from pathlib import Path
from datetime import datetime, timezone

PIPELINE_VERSION = "1.0.0"
SCHEMA_VERSION = "1.0.0"

def sha256_str(s: str) -> str:
    h = hashlib.sha256()
    h.update(s.encode('utf-8'))
    return h.hexdigest()

def load_current_run() -> tuple[str, str]:
    run_id = Path('.current_run_id').read_text().strip()
    source = 'visual_extraction'
    return source, run_id

def assemble_from_ocr_pages(raw_dir: Path):
    pages = []
    # find OCR results under any nested visual_extraction/ocr_results
    candidates = sorted(raw_dir.glob('**/ocr_results/*_ocr.json'))
    if not candidates:
        return []
    for p in candidates:
        try:
            data = json.loads(p.read_text(encoding='utf-8'))
            pages.append({
                'page_number': int(data.get('page_number', len(pages))),
                'text': data.get('text', ''),
                'confidence': data.get('confidence', 0)
            })
        except Exception:
            continue
    # sort by page_number
    pages.sort(key=lambda x: x.get('page_number', 0))
    return pages

def md_summary_path(raw_dir: Path):
    cands = list(raw_dir.glob('**/final_text/*.md'))
    return cands[0] if cands else None

def infer_source_id(raw_dir: Path, md_path: Path | None, ocr_pages: list) -> str:
    # Prefer MD filename stem without suffixes
    if md_path is not None:
        stem = md_path.stem
        # e.g., test_info_extract_visual -> test_info_extract
        if stem.endswith('_visual'):
            stem = stem[:-7]
        return stem or 'document'
    # Else try from OCR file naming convention
    ocr_dir = raw_dir.glob('**/ocr_results/*_ocr.json')
    try:
        first = next(ocr_dir)
        name = first.name
        # test_info_extract_page0_ocr.json -> test_info_extract
        base = name.split('_page')[0]
        return base or 'document'
    except StopIteration:
        return 'document'

def main():
    source, run_id = load_current_run()
    raw_dir = Path('data/raw') / source / run_id
    out_dir = Path('data/normalized') / source / run_id
    out_dir.mkdir(parents=True, exist_ok=True)

    # We build one logical document from the OCR pages + optional MD summary
    pages = assemble_from_ocr_pages(raw_dir)
    if not pages:
        print("No OCR pages found; nothing to normalize.")
        return
    full_text = "\n\n".join(p.get('text', '') for p in sorted(pages, key=lambda x: x['page_number']))
    md_path = md_summary_path(raw_dir)
    markdown = md_path.read_text(encoding='utf-8') if md_path else None

    # derive ids
    checksum = sha256_str(full_text)
    source_id = infer_source_id(raw_dir, md_path, pages)
    document_id = f"emx:{source}:{checksum[:12]}"
    now_iso = datetime.now(timezone.utc).isoformat()

    doc = {
        'schema_version': SCHEMA_VERSION,
        'pipeline_version': PIPELINE_VERSION,
        'run_id': run_id,
        'document_id': document_id,
        'source_id': source_id,
        'source_type': source,
        'uri': str(raw_dir / 'output'),
        'title': source_id,
        'language': 'en',
        'mime_type': 'text/plain',
        'created_at': now_iso,
        'extracted_at': now_iso,
        'version': '1',
        'checksum': checksum,
        'text': full_text,
        'pages': pages,
        'metadata': {
            'page_count': len(pages)
        }
    }
    if markdown:
        doc['markdown'] = markdown

    out_file = out_dir / 'documents.jsonl'
    with out_file.open('w', encoding='utf-8') as f:
        f.write(json.dumps(doc, ensure_ascii=False) + '\n')
    print(f"Wrote {out_file}")

if __name__ == '__main__':
    main()


