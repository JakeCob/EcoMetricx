#!/usr/bin/env python3
"""
HTTP-based Qdrant ingestion script with robust retry logic.

Expected successful output:
  Warming up Qdrant connection...
  ✓ Warmup successful - Qdrant is reachable
  ✓ Collection 'ecometricx' ensured (384 dimensions, Cosine distance)
  Processing embeddings from: data/index/pgvector/20250101_120000/embeddings.jsonl
  ✓ Upserted batch 1/4 (256 points)
  ✓ Upserted batch 2/4 (256 points)
  ✓ Upserted batch 3/4 (256 points)
  ✓ Upserted batch 4/4 (128 points)
  ✓ Upserted 896 points into Qdrant collection ecometricx
"""

import os
import json
import time
from pathlib import Path
from typing import Dict, List, Any
from dotenv import load_dotenv
import httpx


def retry_with_backoff(func, max_attempts=5, base_delay=0.5, max_delay=8.0):
    """Retry function with exponential backoff."""
    for attempt in range(max_attempts):
        try:
            return func()
        except (httpx.ConnectTimeout, httpx.ReadTimeout, httpx.HTTPStatusError) as e:
            if attempt == max_attempts - 1:
                raise
            
            # Don't retry on 4xx errors except 409 (conflict)
            if hasattr(e, 'response') and e.response is not None:
                if 400 <= e.response.status_code < 500 and e.response.status_code != 409:
                    raise
            
            delay = min(base_delay * (2 ** attempt), max_delay)
            print(f"  Attempt {attempt + 1} failed: {e}. Retrying in {delay:.1f}s...")
            time.sleep(delay)


def warmup_qdrant(client: httpx.Client, url: str) -> None:
    """Test Qdrant connectivity with helpful error messages."""
    print("Warming up Qdrant connection...")
    
    def _warmup():
        response = client.get(f"{url}/collections", timeout=15.0)
        if response.status_code == 401:
            raise Exception("Authentication failed - check QDRANT_API_KEY")
        response.raise_for_status()
        return response
    
    try:
        retry_with_backoff(_warmup)
        print("✓ Warmup successful - Qdrant is reachable")
    except httpx.ConnectTimeout:
        raise Exception(f"Connection timeout to {url} - check QDRANT_URL")
    except Exception as e:
        raise Exception(f"Warmup failed: {e}")


def ensure_collection(client: httpx.Client, url: str, collection: str, dim: int) -> None:
    """Create collection if it doesn't exist (idempotent)."""
    
    def _create_collection():
        # PUT /collections/{collection}
        payload = {
            "vectors": {
                "size": dim,
                "distance": "Cosine"
            }
        }
        response = client.put(
            f"{url}/collections/{collection}",
            json=payload,
            timeout=30.0
        )
        
        # 200/201 = success, 409 = already exists (treat as success)
        if response.status_code == 409:
            print(f"✓ Collection '{collection}' already exists")
            return response
        
        response.raise_for_status()
        return response
    
    try:
        result = retry_with_backoff(_create_collection)
        if result.status_code in (200, 201):
            print(f"✓ Collection '{collection}' created ({dim} dimensions, Cosine distance)")
        else:
            print(f"✓ Collection '{collection}' ensured ({dim} dimensions, Cosine distance)")
    except Exception as e:
        raise Exception(f"Failed to ensure collection: {e}")


def upsert_points_batch(client: httpx.Client, url: str, collection: str, points: List[Dict[str, Any]]) -> None:
    """Upsert a batch of points to Qdrant."""
    
    def _upsert_batch():
        # PUT /collections/{collection}/points
        payload = {"points": points}
        response = client.put(
            f"{url}/collections/{collection}/points",
            json=payload,
            timeout=30.0
        )
        response.raise_for_status()
        return response
    
    retry_with_backoff(_upsert_batch)


def main():
    load_dotenv()
    
    # Environment variables
    url = os.environ.get('QDRANT_URL')
    api_key = os.environ.get('QDRANT_API_KEY')
    collection = os.environ.get('QDRANT_COLLECTION', 'ecometricx')
    dim = int(os.environ.get('QDRANT_DIM', '384'))
    
    if not url:
        raise Exception('QDRANT_URL not set')
    if not api_key:
        raise Exception('QDRANT_API_KEY not set')
    
    # Setup HTTP client with headers
    headers = {
        "Content-Type": "application/json",
        "api-key": api_key
    }
    
    with httpx.Client(headers=headers, http2=False) as client:
        # 1. Warmup: Test connectivity
        warmup_qdrant(client, url)
        
        # 2. Ensure collection exists
        ensure_collection(client, url, collection, dim)
        
        # 3. Load embeddings
        root = Path('.')
        run_id = (root / '.current_run_id').read_text().strip()
        emb_dir = root / 'data' / 'index' / 'pgvector' / run_id
        embs_path = emb_dir / 'embeddings.jsonl'
        
        if not embs_path.exists():
            raise Exception(f'Missing embeddings: {embs_path}')
        
        print(f"Processing embeddings from: {embs_path}")
        
        # 4. Read and prepare points
        points = []
        point_id = 1  # Sequential integer IDs per run
        
        with embs_path.open('r', encoding='utf-8') as f:
            for line in f:
                if not line.strip():
                    continue
                    
                rec = json.loads(line)
                point = {
                    "id": point_id,
                    "vector": rec['embedding_vector'],
                    "payload": {
                        "chunk_id": rec['chunk_id'],
                        "parent_document_id": rec['parent_document_id']
                    }
                }
                points.append(point)
                point_id += 1
        
        if not points:
            print("No points to upsert")
            return
        
        # 5. Upsert in batches of 256
        batch_size = 256
        total_batches = (len(points) + batch_size - 1) // batch_size
        
        for i in range(0, len(points), batch_size):
            batch = points[i:i + batch_size]
            batch_num = i // batch_size + 1
            
            try:
                upsert_points_batch(client, url, collection, batch)
                print(f"✓ Upserted batch {batch_num}/{total_batches} ({len(batch)} points)")
            except Exception as e:
                raise Exception(f"Failed to upsert batch {batch_num}: {e}")
        
        print(f"✓ Upserted {len(points)} points into Qdrant collection {collection}")


if __name__ == '__main__':
    main()