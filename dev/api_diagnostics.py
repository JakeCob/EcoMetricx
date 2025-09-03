#!/usr/bin/env python3
"""
Diagnostic script for Retrieval API configuration and health.

Usage:
  set -a && source .env && set +a
  python scripts/api_diagnostics.py

Or with custom parameters:
  python scripts/api_diagnostics.py --api-host http://localhost:8000 --api-key YOUR_KEY
"""

import os
import sys
import argparse
import requests
import json
from typing import Dict, Any


def check_health(api_host: str, api_key: str) -> Dict[str, Any]:
    """Check API health status."""
    try:
        response = requests.get(f"{api_host}/health", timeout=10)
        return {
            "status": "OK" if response.status_code == 200 else f"ERROR {response.status_code}",
            "response_time": response.elapsed.total_seconds()
        }
    except Exception as e:
        return {
            "status": f"ERROR: {e}",
            "response_time": None
        }


def check_debug_config(api_host: str, api_key: str) -> Dict[str, Any]:
    """Check API debug configuration."""
    try:
        headers = {"X-API-Key": api_key} if api_key else {}
        response = requests.get(f"{api_host}/debug/config", headers=headers, timeout=15)
        
        if response.status_code == 200:
            return {
                "status": "OK",
                "config": response.json()
            }
        else:
            return {
                "status": f"ERROR {response.status_code}",
                "error": response.text
            }
    except Exception as e:
        return {
            "status": f"ERROR: {e}",
            "error": str(e)
        }


def print_health_status(health: Dict[str, Any]):
    """Print health status in a readable format."""
    print("🏥 Health Check")
    print("-" * 20)
    print(f"Status: {health['status']}")
    if health.get('response_time'):
        print(f"Response Time: {health['response_time']:.3f}s")
    print()


def print_config_status(debug: Dict[str, Any]):
    """Print debug configuration in a readable format."""
    print("🔧 Configuration Status")
    print("-" * 30)
    
    if debug['status'] != 'OK':
        print(f"❌ Debug endpoint failed: {debug.get('error', debug['status'])}")
        print("   Note: /debug/config endpoint may not be available yet")
        return
    
    config = debug.get('config', {})
    
    # Core dependencies
    print("Dependencies:")
    fastembed_ok = config.get('fastembed_ok', False)
    print(f"  FastEmbed: {'✅ OK' if fastembed_ok else '❌ Failed'}")
    
    # Qdrant configuration
    print("\nQdrant:")
    qdrant_configured = config.get('qdrant_configured', False)
    qdrant_warm = config.get('qdrant_warm', False)
    qdrant_points = config.get('qdrant_points', 0)
    
    print(f"  Configured: {'✅ Yes' if qdrant_configured else '❌ No'}")
    print(f"  Warm/Connected: {'✅ Yes' if qdrant_warm else '❌ No'}")
    print(f"  Points in Collection: {qdrant_points:,}")
    
    # Postgres data
    print("\nPostgres:")
    documents = config.get('documents', 0)
    chunks = config.get('chunks', 0)
    tsv_ready = config.get('tsv_ready', 0)
    
    print(f"  Documents: {documents:,}")
    print(f"  Chunks: {chunks:,}")
    print(f"  TSV Ready: {tsv_ready:,}")
    
    # Overall assessment
    print("\nOverall Assessment:")
    issues = []
    
    if not fastembed_ok:
        issues.append("FastEmbed import failed")
    if not qdrant_configured:
        issues.append("Qdrant not configured (check QDRANT_URL/API_KEY)")
    if not qdrant_warm:
        issues.append("Qdrant connection failed")
    if qdrant_points == 0:
        issues.append("No vectors in Qdrant collection")
    if chunks == 0:
        issues.append("No chunks in Postgres")
    if tsv_ready == 0:
        issues.append("No FTS vectors ready")
    
    if not issues:
        print("✅ All systems operational")
    else:
        print("❌ Issues found:")
        for issue in issues:
            print(f"   • {issue}")


def print_recommendations(health: Dict[str, Any], debug: Dict[str, Any]):
    """Print recommendations based on diagnostic results."""
    print("\n💡 Recommendations")
    print("-" * 20)
    
    if health['status'] != 'OK':
        print("1. Start the API server:")
        print("   uvicorn services.retrieval_api.main:app --host 0.0.0.0 --port 8000")
        return
    
    if debug['status'] != 'OK':
        print("1. Update API to include /debug/config endpoint")
        return
    
    config = debug.get('config', {})
    
    if not config.get('fastembed_ok'):
        print("1. Install FastEmbed: pip install fastembed")
    
    if not config.get('qdrant_configured'):
        print("2. Set Qdrant environment variables:")
        print("   export QDRANT_URL=https://your-qdrant.railway.app")
        print("   export QDRANT_API_KEY=your-api-key")
    
    if not config.get('qdrant_warm'):
        print("3. Check Qdrant connection and API key")
    
    if config.get('qdrant_points', 0) == 0:
        print("4. Ingest vectors to Qdrant:")
        print("   python scripts/ingest_to_qdrant.py")
    
    if config.get('chunks', 0) == 0:
        print("5. Ingest data to Postgres:")
        print("   python scripts/ingest_to_postgres.py")


def main():
    parser = argparse.ArgumentParser(description="Diagnose Retrieval API configuration")
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
    
    print("🔍 Retrieval API Diagnostics")
    print("=" * 40)
    print(f"API Host: {args.api_host}")
    if args.api_key:
        print(f"API Key: {'***' + args.api_key[-4:] if len(args.api_key) > 4 else '***'}")
    else:
        print("API Key: Not provided")
    print()
    
    # Run diagnostics
    health = check_health(args.api_host, args.api_key)
    print_health_status(health)
    
    debug = check_debug_config(args.api_host, args.api_key)
    print_config_status(debug)
    
    print_recommendations(health, debug)
    
    # Exit with appropriate code
    if health['status'] != 'OK':
        sys.exit(1)
    elif debug['status'] != 'OK':
        sys.exit(2)
    else:
        config = debug.get('config', {})
        critical_issues = (
            not config.get('fastembed_ok') or
            not config.get('qdrant_configured') or
            config.get('chunks', 0) == 0
        )
        sys.exit(3 if critical_issues else 0)


if __name__ == "__main__":
    main()