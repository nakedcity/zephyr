#!/bin/bash
set -e

# Ensure uv is installed
if ! command -v uv &> /dev/null; then
    echo "uv is not installed. Please install it first: curl -LsSf https://astral.sh/uv/install.sh | sh"
    exit 1
fi

echo "Creating and syncing environments..."

# Function to setup env
setup_env() {
    ENV_NAME=$1
    EXTRA_NAME=$2
    
    echo "Setting up $ENV_NAME with extra: $EXTRA_NAME"
    
    # Create venv if not exists
    if [ ! -d "$ENV_NAME" ]; then
        uv venv "$ENV_NAME"
    fi
    
    # Install dependencies using uv pip api targetting the environment
    # We install dependencies defined in pyproject.toml with the specific extra
    # We do NOT install the project itself, as it is run as a script/app.
    uv pip install -p "$ENV_NAME" -r pyproject.toml --extra "$EXTRA_NAME"
}

# CPU
setup_env ".venv-cpu" "cpu"

# CUDA
setup_env ".venv-cuda" "cuda"

# 3. MIGraphX (AMD) - Requires Python 3.12 for AMD wheels
if [ ! -d ".venv-migraphx" ]; then
    echo "Creating virtual environment for MIGraphX (AMD)..."
    uv venv --python 3.12 .venv-migraphx
    source .venv-migraphx/bin/activate
fi
setup_env ".venv-migraphx" "migraphx"

echo "All environments ready."
