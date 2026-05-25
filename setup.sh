#!/usr/bin/env bash
# QuantumGPT one-line setup script
# Usage: bash setup.sh
set -e

ENVDIR=".venv"

echo "╔══════════════════════════════════════════╗"
echo "║   QuantumGPT Environment Setup          ║"
echo "╚══════════════════════════════════════════╝"

# Check conda available
if ! command -v conda &>/dev/null; then
    echo "ERROR: conda not found. Install miniforge/miniconda first."
    exit 1
fi

# Create env if missing
if [ ! -d "$ENVDIR" ]; then
    echo "→ Creating conda env at $ENVDIR (Python 3.11)..."
    conda create -p "$ENVDIR" python=3.11 -y -q
else
    echo "→ Environment $ENVDIR already exists."
fi

# Install deps
echo "→ Installing dependencies..."
conda run -p "$ENVDIR" pip install -r requirements.txt -q

# Verify
echo "→ Verifying installation..."
conda run -p "$ENVDIR" python -c "
import qiskit, numpy, scipy, matplotlib, duckdb, sklearn
print(f'  qiskit {qiskit.__version__}')
print(f'  numpy {numpy.__version__}')
print(f'  scipy {scipy.__version__}')
print(f'  duckdb {duckdb.__version__}')
print('  All OK ✓')
"

# Quick smoke test
echo "→ Running smoke test (tuneup demo)..."
conda run -p "$ENVDIR" python demos/tuneup_demo.py --quiet

echo ""
echo "╔══════════════════════════════════════════╗"
echo "║   Setup complete!                       ║"
echo "║   Activate: conda activate ./.venv      ║"
echo "║   Demo:     python demos/tuneup_demo.py ║"
echo "╚══════════════════════════════════════════╝"
