#!/bin/bash
set -e

# Activate the CPU environment for the main gateway
source .venv-cpu/bin/activate

# Run the server
# The --port 8080 matches the previous default configuration
# We append "$@" to allow overriding args, e.g. ./run.sh --port 9000
echo "Starting Zephyr Gateway..."
fastapi run server/main.py --port 8080 --host 0.0.0.0 "$@"
