#!/usr/bin/env bash
set -euo pipefail

if [ ! -d ".venv" ]; then
  echo "Virtual environment .venv not found."
  echo "Create one with:"
  echo "  python -m venv .venv"
  echo "  source .venv/bin/activate"
  echo "  pip install -r backend_server/requirements.txt"
  exit 1
fi

source ".venv/bin/activate"
python -m uvicorn backend_server.main_server:app --host 0.0.0.0 --port 8000
