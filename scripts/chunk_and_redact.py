#!/usr/bin/env python3
import json
import re
from pathlib import Path
from datetime import datetime, timezone

PIPELINE_VERSION = "1.0.0"
SCHEMA_VERSION = "1.0.0"


def redact_text(text: str) -> str:
    patterns = [
        # Account number token
        (re.compile(r"(?i)(account\s*number\s*[:#]?\s*)(\S+)"), r"\1ACCOUNT_NUMBER_REDACTED"),
        # Service address label and value (not line-anchored)
        (re.compile(r"(?i)(service\s*address\s*[:#]?\s*)([^\n\r]+)"), r"\1SERVICE_ADDRESS_REDACTED"),
        # Greeting with name: Dear JILL DOE,
        (re.compile(r"(?i)(dear\s+)([A-Z][A-Z\s]+)(,)"), r"\1NAME_REDACTED\3"),
    ]
    redacted = text
    for pat, repl in patterns:
        redacted = pat.sub(repl, redacted)
    return redacted


def simple_tokenize(text: str):
    return re.findall(r"\w+", text)


def chunk_text(full_text: str, pages: list, target_tokens: int = 1000, overlap_tokens: int = 120):
    """Create page-aware chunks using sliding windows over words per page."""
    chunks = []
    global_offset = 0
    for page in sorted(pages, key=lambda p: p.get('page_number', 0)):
        page_text = page.get('text', '')
        tokens = re.findall(r"\w+|\W", page_text)
        # Build mapping from token indices to char offsets
        char_offsets = []
        c = 0
        for t in tokens:
            char_offsets.append(c)
            c += len(t)
        words = [w for w in re.findall(r"\w+", page_text)]
        if not words:
            global_offset += len(page_text)
            continue
        # Windowing in word space
        word_positions = [m.start() for m in re.finditer(r"\w+", page_text)]
        start_idx = 0
        while start_idx < len(word_positions):
            end_idx = min(len(word_positions), start_idx + target_tokens)
            start_char = word_positions[start_idx]
            end_char = len(page_text) if end_idx == len(word_positions) else word_positions[end_idx]
            chunk_txt = page_text[start_char:end_char].strip()
            if chunk_txt:
                chunks.append({
                    'page_number': page.get('page_number', 0),
                    'start_char': global_offset + start_char,
                    'end_char': global_offset + end_char,
                    'text': chunk_txt
                })
            if end_idx == len(word_positions):
                break
            start_idx += max(1, target_tokens - overlap_tokens)
        global_offset += len(page_text)
    return chunks


def quality_filter(text: str) -> bool:
    tokens = simple_tokenize(text)
    if len(tokens) < 30:
        return False
    non_alnum = sum(1 for ch in text if not ch.isalnum() and not ch.isspace())
    ratio = non_alnum / max(1, len(text))
    if ratio > 0.5:
        return False
    return True


def main():
    root = Path('.')
    run_id = (root / '.current_run_id').read_text().strip()
    source = 'visual_extraction'
    norm_path = root / 'data' / 'normalized' / source / run_id / 'documents.jsonl'
    out_dir = root / 'data' / 'chunks' / source / run_id
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / 'chunks.jsonl'

    assert norm_path.exists(), f"Missing normalized documents: {norm_path}"
    now_iso = datetime.now(timezone.utc).isoformat()

    with norm_path.open('r', encoding='utf-8') as fin, out_file.open('w', encoding='utf-8') as fout:
        for line in fin:
            if not line.strip():
                continue
            doc = json.loads(line)
            doc_id = doc['document_id']
            pages = doc.get('pages', [])
            full_text = doc.get('text', '')
            raw_chunks = chunk_text(full_text, pages)
            chunk_index = 0
            for rc in raw_chunks:
                txt = rc['text']
                red = redact_text(txt)
                if not quality_filter(red):
                    continue
                chunk = {
                    'schema_version': SCHEMA_VERSION,
                    'pipeline_version': PIPELINE_VERSION,
                    'run_id': run_id,
                    'chunk_id': f"{doc_id}:c{chunk_index}",
                    'parent_document_id': doc_id,
                    'chunk_index': chunk_index,
                    'start_char': rc['start_char'],
                    'end_char': rc['end_char'],
                    'text': red,
                    'page_num': rc['page_number'],
                    'section_path': f"page/{rc['page_number']}",
                    'content_type': 'prose',
                    'tokens': len(simple_tokenize(red)),
                    'quality_score': 1.0,
                    'metadata': {
                        'created_at': now_iso
                    }
                }
                fout.write(json.dumps(chunk, ensure_ascii=False) + '\n')
                chunk_index += 1

    print(f"Wrote {out_file}")


if __name__ == '__main__':
    main()


