#!/usr/bin/env bash
set -euo pipefail

WITH_E2E=0
if [[ "${1:-}" == "--with-e2e" ]]; then
  WITH_E2E=1
elif [[ -n "${1:-}" ]]; then
  echo "Usage: bash scripts/bootstrap_venv.sh [--with-e2e]" >&2
  exit 1
fi

python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip

if [[ "$WITH_E2E" -eq 1 ]]; then
  python -m pip install -e ".[dev,e2e]"
else
  python -m pip install -e ".[dev]"
fi

echo "Environment ready. Activate with: source .venv/bin/activate"
