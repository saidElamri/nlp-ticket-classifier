#!/bin/bash
set -e

echo "Installing missing dependencies (CPU Only)..."
.venv/bin/pip install scikit-learn transformers
.venv/bin/pip install torch --index-url https://download.pytorch.org/whl/cpu
echo "All dependencies installed successfully."
