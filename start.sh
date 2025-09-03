#!/bin/bash

# Set default port if PORT environment variable is not set
if [ -z "$PORT" ]; then
    PORT=8000
fi

echo "Starting EcoMetricx API on port $PORT"

# Start the application
uvicorn main:app --host 0.0.0.0 --port $PORT