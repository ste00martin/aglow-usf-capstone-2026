#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ASSETS_DIR="$SCRIPT_DIR/../expo-pytorch/assets/models"
VENV_PYTHON="$SCRIPT_DIR/../.venv-executorch-export/bin/python"

if [[ -x "$VENV_PYTHON" ]]; then
  PYTHON_BIN="$VENV_PYTHON"
elif command -v python3 >/dev/null 2>&1; then
  PYTHON_BIN="$(command -v python3)"
else
  echo "python3 not found. Run ./scripts/setup_executorch_export_env.sh first." >&2
  exit 1
fi

mkdir -p "$ASSETS_DIR"

cd "$SCRIPT_DIR"

"$PYTHON_BIN" export_blazeface.py --backend xnnpack --output "$ASSETS_DIR/blazeface.pte"
"$PYTHON_BIN" export_age.py --backend xnnpack --output "$ASSETS_DIR/age_model.pte"
"$PYTHON_BIN" export_gender.py --backend xnnpack --output "$ASSETS_DIR/gender_model.pte"
"$PYTHON_BIN" export_nsfw.py --backend xnnpack --output "$ASSETS_DIR/nsfw_model.pte"
