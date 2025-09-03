#!/usr/bin/env python3
"""
Test script for Retrieval API queries.

Usage:
  set -a && source .env && set +a
  python scripts/test_api_queries.py

Or with custom parameters:
  python scripts/test_api_queries.py --api-host http://localhost:8000 --api-key YOUR_KEY
"""

import os
import sys
import argparse
import requests
import json
from typing import List, Dict, Any


def test_health(api_host: str, api_key: str) -> bool:
    """Test the /health endpoint."""
    try:
        response = requests.get(f"{api_host}/health", timeout=10)
        print(f"Health: {response.status_code}")
        return response.status_code == 200
    except Exception as e:
        print(f"Health: ERROR - {e}")
        return False


def test_search_query(api_host: str, api_key: str, query: str, k: int = 5) -> Dict[str, Any]:
    """Test a single search query."""
    try:
        headers = {
            "Content-Type": "application/json",
            "X-API-Key": api_key
        }
        payload = {
            "query": query,
            "k": k
        }
        
        response = requests.post(
            f"{api_host}/search",
            headers=headers,
            json=payload,
            timeout=30
        )
        
        if response.status_code == 200:
            result = response.json()
            results = result.get("results", [])
            return {
                "status_code": response.status_code,
                "count": len(results),
                "results": results
            }
        else:
            return {
                "status_code": response.status_code,
                "count": 0,
                "error": response.text
            }
            
    except Exception as e:
        return {
            "status_code": 0,
            "count": 0,
            "error": str(e)
        }


def main():
    parser = argparse.ArgumentParser(description="Test Retrieval API queries")
    parser.add_argument(
        "--api-host", 
        default=os.environ.get("API_HOST", "http://127.0.0.1:8000"),
        help="API host (default: http://127.0.0.1:8000)"
    )
    parser.add_argument(
        "--api-key",
        default=os.environ.get("API_KEY"),
        help="API key for authentication"
    )
    
    args = parser.parse_args()
    
    if not args.api_key:
        print("ERROR: API_KEY not provided. Set environment variable or use --api-key")
        sys.exit(1)
    
    # Test queries covering different aspects of the energy report
    queries = [
        "home energy report",
        "energy savings tips", 
        "monthly savings tip",
        "thermostat settings advice",
        "caulk windows doors",
        "upgrade refrigerator",
        "usage compared similar homes",
        "march report"
    ]
    
    print("Testing Retrieval API")
    print("=" * 50)
    print(f"API Host: {args.api_host}")
    print(f"API Key: {'***' + args.api_key[-4:] if len(args.api_key) > 4 else '***'}")
    print()
    
    # Test health endpoint
    health_ok = test_health(args.api_host, args.api_key)
    if not health_ok:
        print("ERROR: Health check failed. Is the API running?")
        sys.exit(1)
    
    print("\nTesting search queries:")
    print("-" * 30)
    
    # Track results
    total_queries = len(queries)
    non_empty_queries = 0
    total_results = 0
    query_results = []
    
    # Test each query
    for i, query in enumerate(queries, 1):
        print(f"{i:2d}. '{query}'", end=" ... ")
        result = test_search_query(args.api_host, args.api_key, query)
        
        if result["status_code"] == 200:
            count = result["count"]
            total_results += count
            if count > 0:
                non_empty_queries += 1
            print(f"Status: {result['status_code']}, Results: {count}")
            query_results.append((query, count, result.get("results", [])))
        else:
            print(f"Status: {result['status_code']}, Error: {result.get('error', 'Unknown')}")
            query_results.append((query, 0, []))
    
    # Print summary
    print("\n" + "=" * 50)
    print("SUMMARY")
    print("=" * 50)
    print(f"Total queries: {total_queries}")
    print(f"Non-empty results: {non_empty_queries}")
    print(f"Total result count: {total_results}")
    print(f"Success rate: {non_empty_queries}/{total_queries} ({100*non_empty_queries/total_queries:.1f}%)")
    
    # Show top results for successful queries
    if non_empty_queries > 0:
        print(f"\nTop results from successful queries:")
        print("-" * 40)
        for query, count, results in query_results:
            if count > 0:
                print(f"\n'{query}' ({count} results):")
                for j, res in enumerate(results[:2], 1):  # Show top 2
                    chunk_id = res.get("chunk_id", "N/A")
                    score = res.get("score", 0)
                    print(f"  {j}. {chunk_id} (score: {score:.3f})")
    
    # Exit with appropriate code
    if non_empty_queries == 0:
        print(f"\n❌ NO QUERIES RETURNED RESULTS - API may need diagnosis")
        sys.exit(1)
    elif non_empty_queries < total_queries // 2:
        print(f"\n⚠️  LOW SUCCESS RATE - API may need optimization")
        sys.exit(2)
    else:
        print(f"\n✅ API TESTS PASSED - {non_empty_queries} queries returned results")
        sys.exit(0)


if __name__ == "__main__":
    main()