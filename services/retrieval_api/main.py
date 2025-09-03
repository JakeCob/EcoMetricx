import os, json, math
from typing import List, Optional, Dict
from fastapi import FastAPI, Depends, Header, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import psycopg2
import httpx
from dotenv import load_dotenv

# Load environment variables at startup
load_dotenv()

API_KEY = os.environ.get('API_KEY')
FUSION_ALPHA = float(os.environ.get('FUSION_ALPHA', '0.6'))
ENABLE_RERANKER = os.environ.get('ENABLE_RERANKER', 'false').lower() == 'true'
DATABASE_URL = os.environ.get('DATABASE_URL') or os.environ.get('POSTGRES_DSN')
QDRANT_URL = os.environ.get('QDRANT_URL')
QDRANT_API_KEY = os.environ.get('QDRANT_API_KEY')
QDRANT_COLLECTION = os.environ.get('QDRANT_COLLECTION', 'ecometricx')

app = FastAPI(
    title="EcoMetricx Retrieval API",
    version="1.0.0",
    description="AI-powered document search API with hybrid FTS and vector search"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)

@app.on_event("startup")
async def startup_event():
    """Validate configuration on startup."""
    print("🚀 Starting EcoMetricx Retrieval API...")
    
    # Validate required environment variables
    required_vars = ["DATABASE_URL", "API_KEY"]
    missing_vars = [var for var in required_vars if not os.environ.get(var)]
    
    if missing_vars:
        print(f"❌ Missing required environment variables: {missing_vars}")
        print("⚠️  Application will not function properly without these variables")
    else:
        print("✅ All required environment variables are set")
    
    # Optional variables
    optional_vars = ["QDRANT_URL", "QDRANT_API_KEY", "QDRANT_COLLECTION"]
    missing_optional = [var for var in optional_vars if not os.environ.get(var)]
    
    if missing_optional:
        print(f"⚠️  Optional variables not set (vector search disabled): {missing_optional}")
    else:
        print("✅ Vector search configuration complete")
    
    print("🔍 EcoMetricx API ready for hybrid document search!")

# Global HTTP client for Qdrant (lazy initialization)
_qdrant_client: Optional[httpx.Client] = None


async def require_api_key(x_api_key: Optional[str] = Header(default=None)):
    if API_KEY is None:
        raise HTTPException(status_code=500, detail="API key not configured")
    # constant-time compare
    if not x_api_key or not (len(x_api_key) == len(API_KEY) and sum(ord(a) ^ ord(b) for a, b in zip(x_api_key, API_KEY)) == 0):
        raise HTTPException(status_code=403, detail="Forbidden")
    return True


class SearchRequest(BaseModel):
    query: str
    k: int = 10
    filter_document_id: Optional[str] = None


class SimilarRequest(BaseModel):
    chunk_id: str
    k: int = 10


def _get_conn():
    if not DATABASE_URL:
        raise HTTPException(status_code=500, detail="DATABASE_URL not set")
    return psycopg2.connect(DATABASE_URL)


def _get_qdrant_client() -> Optional[httpx.Client]:
    """Lazily initialize and warm up Qdrant HTTP client with fallback handling."""
    global _qdrant_client
    
    if _qdrant_client is not None:
        return _qdrant_client
    
    if not QDRANT_URL or not QDRANT_API_KEY:
        return None
    
    try:
        headers = {
            "Content-Type": "application/json", 
            "api-key": QDRANT_API_KEY
        }
        client = httpx.Client(headers=headers, timeout=15.0, http2=False)
        
        # Warm up: test connectivity with fast GET /collections
        response = client.get(f"{QDRANT_URL}/collections")
        response.raise_for_status()
        
        _qdrant_client = client
        return _qdrant_client
    except Exception:
        # Silently fail - we'll fall back to FTS-only
        return None


def _search_qdrant(query_vector: List[float], k: int, filter_document_id: Optional[str] = None) -> dict:
    """Search Qdrant via HTTP API with proper error handling."""
    client = _get_qdrant_client()
    if not client:
        return {}
    
    try:
        # Prepare the search payload
        search_payload = {
            "vector": query_vector,
            "limit": k,
            "with_payload": True
        }
        
        # Add filter if document_id is specified
        if filter_document_id:
            search_payload["filter"] = {
                "must": [
                    {
                        "key": "document_id",
                        "match": {"value": filter_document_id}
                    }
                ]
            }
        
        # Perform the search
        response = client.post(f"/collections/chunks/points/search", json=search_payload)
        
        if response.status_code == 200:
            result = response.json()
            # Convert results to {chunk_id: score} mapping
            vec_scores = {}
            for point in result.get("result", []):
                chunk_id = point.get("payload", {}).get("chunk_id")
                score = point.get("score", 0.0)
                if chunk_id:
                    vec_scores[chunk_id] = score
            return vec_scores
        else:
            print(f"Qdrant search failed with status {response.status_code}: {response.text}")
            return {}
    except Exception as e:
        print(f"Error in Qdrant search: {e}")
        return {}


def _fetch_snippets(cur, chunk_ids: List[str], max_chars: int = 300) -> Dict[str, str]:
    """Fetch text snippets for chunk_ids from Postgres."""
    if not chunk_ids:
        return {}
    cur.execute(
        "SELECT chunk_id, text FROM chunks WHERE chunk_id = ANY(%s)", (chunk_ids,)
    )
    rows = cur.fetchall()
    mapping = {}
    for cid, text in rows:
        if text:
            snippet = text.strip()
            if len(snippet) > max_chars:
                snippet = snippet[:max_chars] + "..."
            mapping[cid] = snippet
    return mapping


@app.get("/health")
def health():
    """Health check endpoint for Railway monitoring."""
    try:
        # Basic check - ensure required environment variables are set
        if not DATABASE_URL:
            return {"status": "error", "message": "DATABASE_URL not configured"}
        if not API_KEY:
            return {"status": "error", "message": "API_KEY not configured"}
        
        return {
            "status": "ok", 
            "message": "EcoMetricx API is running",
            "version": "1.0.0"
        }
    except Exception as e:
        return {"status": "error", "message": f"Health check failed: {str(e)}"}


@app.get("/debug/config")
def debug_config():
    """Debug endpoint to check API configuration and connectivity."""
    config = {}
    
    # Test FastEmbed import
    try:
        from fastembed import TextEmbedding
        config["fastembed_ok"] = True
    except ImportError:
        config["fastembed_ok"] = False
    
    # Check Qdrant configuration
    config["qdrant_configured"] = bool(QDRANT_URL and QDRANT_API_KEY)
    
    # Test Qdrant connection
    config["qdrant_warm"] = False
    config["qdrant_points"] = 0
    
    if config["qdrant_configured"]:
        try:
            client = _get_qdrant_client()
            if client:
                config["qdrant_warm"] = True
                
                # Get collection info
                response = client.get(f"{QDRANT_URL}/collections/{QDRANT_COLLECTION}")
                if response.status_code == 200:
                    collection_info = response.json()
                    config["qdrant_points"] = collection_info.get("result", {}).get("points_count", 0)
        except Exception:
            pass
    
    # Check Postgres data
    config["documents"] = 0
    config["chunks"] = 0
    config["tsv_ready"] = 0
    
    try:
        conn = _get_conn()
        try:
            cur = conn.cursor()
            # Count documents
            cur.execute("SELECT COUNT(*) FROM documents")
            row = cur.fetchone()
            if row:
                config["documents"] = row[0]
            
            # Count chunks
            cur.execute("SELECT COUNT(*) FROM chunks")
            row = cur.fetchone()
            if row:
                config["chunks"] = row[0]
                
            # Count chunks with tsvector ready
            cur.execute("SELECT COUNT(*) FROM chunks WHERE text_tsv IS NOT NULL")
            row = cur.fetchone()
            if row:
                config["tsv_ready"] = row[0]
        finally:
            conn.close()
    except Exception:
        pass
    
    return config


@app.post("/search", dependencies=[Depends(require_api_key)])
def search(req: SearchRequest):
    """
    Hybrid search: FTS from Postgres + vector search from Qdrant via HTTP API.
    Gracefully falls back to FTS-only if Qdrant is unavailable.
    """
    conn = _get_conn()
    try:
        cur = conn.cursor()
        # Multi-strategy FTS search from Postgres
        # Strategy 1: Try exact phrase match
        fts_sql_exact = """
        SELECT c.chunk_id, c.parent_document_id, c.page_num,
               ts_rank_cd(c.text_tsv, plainto_tsquery('english', %(q)s)) AS bm25
        FROM chunks c
        WHERE c.text_tsv @@ plainto_tsquery('english', %(q)s)
        {doc_filter}
        ORDER BY bm25 DESC
        LIMIT %(k)s
        """.format(doc_filter="AND c.parent_document_id = %(doc)s" if req.filter_document_id else "")
        
        params = {"q": req.query, "k": req.k}
        if req.filter_document_id:
            params["doc"] = req.filter_document_id
        cur.execute(fts_sql_exact, params)
        fts_rows = cur.fetchall()
        
        # Strategy 2: If no exact matches, try OR search with individual words
        if not fts_rows and len(req.query.split()) > 1:
            words = req.query.split()
            or_query = " | ".join(words)  # Create OR query: word1 | word2 | word3
            
            fts_sql_or = """
            SELECT c.chunk_id, c.parent_document_id, c.page_num,
                   ts_rank_cd(c.text_tsv, to_tsquery('english', %(or_q)s)) AS bm25
            FROM chunks c
            WHERE c.text_tsv @@ to_tsquery('english', %(or_q)s)
            {doc_filter}
            ORDER BY bm25 DESC
            LIMIT %(k)s
            """.format(doc_filter="AND c.parent_document_id = %(doc)s" if req.filter_document_id else "")
            
            or_params = {"or_q": or_query, "k": req.k}
            if req.filter_document_id:
                or_params["doc"] = req.filter_document_id
            
            try:
                cur.execute(fts_sql_or, or_params)
                fts_rows = cur.fetchall()
                if fts_rows:
                    print(f"INFO: FTS OR-search activated for query '{req.query}' - {len(fts_rows)} results")
            except Exception:
                # Fall back to original if OR search fails
                pass

        # Vector search from Qdrant (with graceful fallback)
        vec_scores = {}
        try:
            from fastembed import TextEmbedding
            emb = TextEmbedding('BAAI/bge-small-en-v1.5')
            qv = list(emb.embed([req.query]))[0]
            vec_scores = _search_qdrant(qv, req.k, req.filter_document_id)
            if vec_scores:
                print(f"INFO: Vector search for '{req.query}' found {len(vec_scores)} matches")
        except Exception as e:
            # Fall back to FTS-only mode with cosine=0.0
            print(f"INFO: Vector search failed for '{req.query}': {e}")
            pass

        # Fusion: combine FTS and vector scores
        results = []
        fts_ids = set()
        for r in fts_rows:
            chunk_id, doc_id, page_num, bm25 = r
            fts_ids.add(chunk_id)
            cosine = vec_scores.get(chunk_id, 0.0) if vec_scores else 0.0  # 0.0 if Qdrant unavailable
            score = FUSION_ALPHA * cosine + (1 - FUSION_ALPHA) * float(bm25)
            results.append({
                "chunk_id": chunk_id,
                "document_id": doc_id,
                "page_num": page_num,
                "score": score,
            })

        # If FTS returned fewer than k (or 0), include top vector-only hits
        if vec_scores and len(results) < req.k:
            remaining = req.k - len(results)
            # pick highest cosine not already in results
            extra_ids = [cid for cid, _ in sorted(vec_scores.items(), key=lambda x: x[1], reverse=True)
                         if cid not in fts_ids][:max(0, remaining)]
            if extra_ids:
                cur.execute(
                    "SELECT chunk_id, parent_document_id, page_num FROM chunks WHERE chunk_id = ANY(%s)",
                    (extra_ids,)
                )
                rows = cur.fetchall()
                meta = {row[0]: (row[1], row[2]) for row in rows}
                for cid in extra_ids:
                    doc_id, page_num = meta.get(cid, (None, None))
                    results.append({
                        "chunk_id": cid,
                        "document_id": doc_id,
                        "page_num": page_num,
                        "score": FUSION_ALPHA * float(vec_scores.get(cid, 0.0))
                    })
                
                # Log vector-only fallback for debugging
                if len(fts_rows) == 0 and extra_ids:
                    print(f"INFO: Vector-only fallback activated for query '{req.query}' - {len(extra_ids)} results")

        # Sort by score and trim
        results.sort(key=lambda r: r.get("score", 0.0), reverse=True)
        results = results[:req.k]

        # Attach snippets
        chunk_ids = [r["chunk_id"] for r in results]
        snippets = _fetch_snippets(cur, chunk_ids)
        for r in results:
            r["snippet"] = snippets.get(r["chunk_id"])  # may be None
        return {"results": results}
    finally:
        conn.close()


@app.post("/similar", dependencies=[Depends(require_api_key)])
def similar(req: SimilarRequest):
    """Find similar chunks using Qdrant vector similarity."""
    conn = _get_conn()
    try:
        cur = conn.cursor()
        # Verify chunk exists in Postgres
        cur.execute("SELECT chunk_id FROM chunks WHERE chunk_id=%s", (req.chunk_id,))
        row = cur.fetchone()
        if not row:
            raise HTTPException(status_code=404, detail="chunk_id not found")
        
        # Get similar chunks from Qdrant using chunk_id as reference
        try:
            client = _get_qdrant_client()
            if not client:
                raise HTTPException(status_code=503, detail="Vector search unavailable")
            
            # Find the reference point in Qdrant by chunk_id
            search_payload = {
                "filter": {
                    "must": [
                        {
                            "key": "chunk_id", 
                            "match": {"value": req.chunk_id}
                        }
                    ]
                },
                "limit": 1,
                "with_payload": True,
                "with_vector": True
            }
            
            response = client.post(
                f"{QDRANT_URL}/collections/{QDRANT_COLLECTION}/points/scroll",
                json=search_payload,
                timeout=15.0
            )
            response.raise_for_status()
            
            result = response.json()
            points = result.get("result", {}).get("points", [])
            
            if not points:
                raise HTTPException(status_code=404, detail="chunk_id not found in vector store")
            
            reference_vector = points[0].get("vector", [])
            
            # Now search for similar vectors
            similar_payload = {
                "vector": reference_vector,
                "limit": req.k,
                "with_payload": True
            }
            
            response = client.post(
                f"{QDRANT_URL}/collections/{QDRANT_COLLECTION}/points/search", 
                json=similar_payload,
                timeout=15.0
            )
            response.raise_for_status()
            
            result = response.json()
            similar_results = []
            
            for hit in result.get("result", []):
                payload_data = hit.get("payload", {})
                chunk_id = payload_data.get("chunk_id")
                score = hit.get("score", 0.0)
                if chunk_id:
                    similar_results.append({
                        "chunk_id": chunk_id, 
                        "cosine": float(score)
                    })
            
            return {"results": similar_results}
            
        except HTTPException:
            raise
        except Exception:
            raise HTTPException(status_code=503, detail="Vector search temporarily unavailable")
    finally:
        conn.close()


class AnswerRequest(BaseModel):
    query: str
    k: int = 3
    filter_document_id: Optional[str] = None


@app.post("/answer", dependencies=[Depends(require_api_key)])
def answer(req: AnswerRequest):
    """Simple extractive answer: returns top snippet(s) as an answer with citations."""
    # Reuse search logic to get ranked results
    search_req = SearchRequest(query=req.query, k=req.k, filter_document_id=req.filter_document_id)
    search_res = search(search_req)
    hits = search_res.get("results", [])
    if not hits:
        return {"answer": "No relevant content found.", "citations": []}
    # Concatenate top snippets for a brief answer
    snippets = [h.get("snippet") for h in hits if h.get("snippet")]
    answer_text = "\n\n".join(snippets) if snippets else "No snippet available."
    citations = [
        {
            "chunk_id": h["chunk_id"],
            "document_id": h.get("document_id"),
            "page_num": h.get("page_num"),
            "score": h.get("score"),
        }
        for h in hits
    ]
    return {"answer": answer_text, "citations": citations}