#!/usr/bin/env bash
# Create the environment and fetch what the repo needs.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VENV="${KIRK_VENV:-$HOME/kirk-env}"
if [ -z "${PYTHON:-}" ]; then
  if command -v python3.12 >/dev/null 2>&1; then
    PYTHON=python3.12
  elif command -v uv >/dev/null 2>&1; then
    PYTHON="$(uv python find 3.12)"
  else
    PYTHON=python3
  fi
fi

if ! "$PYTHON" -c 'import sys; sys.exit(0 if sys.version_info[:2] == (3, 12) else 1)'; then
  echo "Need Python 3.12; set PYTHON=/path/to/python3.12" >&2
  exit 1
fi

if [ -d "$VENV" ]; then
  if ! "$VENV/bin/python" -c 'import sys; sys.exit(0 if sys.version_info[:2] == (3, 12) else 1)'; then
    echo "$VENV exists but is not Python 3.12; set KIRK_VENV or remove it" >&2
    exit 1
  fi
else
  "$PYTHON" -m venv "$VENV"
fi

cd "$ROOT"

# A plain `pip install torch` pulls the CUDA wheels on Linux, which is
# gigabytes. The default here is the CPU wheel; set KIRK_TORCH_INDEX to a CUDA
# index if you want GPU torch.
TORCH_INDEX="${KIRK_TORCH_INDEX:-https://download.pytorch.org/whl/cpu}"
"$VENV/bin/pip" install --index-url "$TORCH_INDEX" torch
"$VENV/bin/pip" install "rebound==4.6.0" pandas matplotlib pytest
"$VENV/bin/pip" install -e . --no-deps

"$VENV/bin/python" scripts/download_assets.py
"$VENV/bin/python" scripts/prepare_data.py

echo
echo "done. Try: make test && make quick"
