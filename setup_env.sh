#!/usr/bin/env bash
# Create a pip-only environment in .venv and download the matrix.
#   ./setup_env.sh          CPU methods (scipy + MKL PARDISO)
#   ./setup_env.sh gpu      + NVIDIA cuDSS (CUDA 12; use gpu13 for CUDA 13)
set -e
cd "$(dirname "$0")"
python3 -m venv .venv
. .venv/bin/activate
python -m pip install --upgrade pip
extras="pardiso"
if [ -n "$1" ]; then extras="$extras,$1"; fi
pip install -e ".[$extras]"
python get_data.py
python -m factorbench list
