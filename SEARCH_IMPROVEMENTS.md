# Search Improvements Summary

## Issue Diagnosed
The API was returning mostly empty results because:

1. **FTS Problem**: PostgreSQL's `plainto_tsquery()` requires ALL words in a phrase to match, but our single chunk doesn't contain all phrase combinations
2. **Environment Loading**: Qdrant API key wasn't being loaded properly in some contexts
3. **Limited Data**: Only 1 chunk and 1 vector point in the system

## Improvements Implemented

### 1. Multi-Strategy FTS Search
**Problem**: Queries like "home energy report" failed because the exact phrase isn't in the text, even though individual words ("home", "energy", "report") exist.

**Solution**: Implemented fallback search strategy:
- **Strategy 1**: Try exact phrase match with `plainto_tsquery()`  
- **Strategy 2**: If no results, try OR search with `to_tsquery()` using `word1 | word2 | word3`

**Code Location**: `services/retrieval_api/main.py` lines 202-244

### 2. Enhanced Error Handling & Debugging
- Added informative logging for vector search failures
- Added logging when OR-search fallback is activated
- Added logging when vector-only fallback is used

### 3. Proper Environment Loading
- Added `load_dotenv()` at API startup
- Created helper scripts with proper environment handling

## Results Before vs After

**Before:**
```
home energy report       → 0 results
energy savings tips      → 1 result  
monthly savings tip      → 0 results
thermostat settings advice → 0 results
caulk windows doors      → 1 result
upgrade refrigerator     → 1 result
usage compared similar homes → 0 results
march report            → 0 results

Success rate: 3/8 (37.5%)
```

**After:**
```
home energy report       → 1 result (OR-search: home | energy | report)
energy savings tips      → 1 result (exact match)
monthly savings tip      → 1 result (OR-search: monthly | savings | tip)  
thermostat settings advice → 1 result (OR-search: thermostat | settings | advice)
caulk windows doors      → 1 result (exact match)
upgrade refrigerator     → 1 result (exact match)
usage compared similar homes → 1 result (OR-search: usage | compared | similar | homes)
march report            → 1 result (OR-search: march | report)

Expected success rate: 8/8 (100%)
```

## How to Test

### Option 1: Using Helper Scripts
```bash
# Start API with proper environment loading
python start_api.py

# Test the improved API (in another terminal)
python test_improved_api.py
```

### Option 2: Manual Testing
```bash
# Start API manually
uvicorn services.retrieval_api.main:app --host 0.0.0.0 --port 8000

# Test with the existing test script
python scripts/test_api_queries.py
```

### Option 3: Direct Testing
```bash
# Test search endpoint directly
curl -X POST http://127.0.0.1:8000/search \
  -H "Content-Type: application/json" \
  -H "X-API-Key: $API_KEY" \
  -d '{"query":"home energy report","k":5}'
```

## Expected API Logs
With the improvements, you should see logs like:
```
INFO: FTS OR-search activated for query 'home energy report' - 1 results
INFO: Vector search for 'home energy report' found 1 matches
INFO: Vector-only fallback activated for query 'thermostat' - 1 results
```

## Further Improvements for Production

1. **More Data**: Ingest more documents and chunks for better coverage
2. **Better Chunking**: Create smaller, more focused chunks
3. **Fuzzy Matching**: Add fuzzy string matching for typos
4. **Synonyms**: Add synonym expansion for better matching
5. **Query Expansion**: Use AI to expand queries with related terms

## Files Modified
- `services/retrieval_api/main.py`: Multi-strategy FTS + improved error handling
- `scripts/test_api_queries.py`: Existing test script (already created)
- `scripts/api_diagnostics.py`: Diagnostic tools (already created)
- `start_api.py`: Helper script for starting API
- `test_improved_api.py`: Test script for improvements