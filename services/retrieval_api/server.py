#!/usr/bin/env python3
"""
Railway-compatible server startup script for EcoMetricx API
"""
import os
import sys

def main():
    # Get port from environment, default to 8000
    port = int(os.environ.get('PORT', 8000))
    host = '0.0.0.0'
    
    print(f"🚀 Starting EcoMetricx API on {host}:{port}")
    
    # Import uvicorn and run the app
    import uvicorn
    uvicorn.run("main:app", host=host, port=port, log_level="info")

if __name__ == "__main__":
    main()