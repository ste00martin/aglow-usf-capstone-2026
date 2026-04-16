#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ASSETS_DIR="$SCRIPT_DIR/../expo-pytorch/assets/models"
VENV_PYTHON="$SCRIPT_DIR/../.venv-executorch-export/bin/python"

if [[ ! -x "$VENV_PYTHON" ]]; then
  echo "ExecuTorch export environment not found. Bootstrapping it first..."
  "$SCRIPT_DIR/setup_executorch_export_env.sh"
fi

if [[ ! -x "$VENV_PYTHON" ]]; then
  echo "Missing export Python environment at $VENV_PYTHON" >&2
  exit 1
fi

PYTHON_BIN="$VENV_PYTHON"

mkdir -p "$ASSETS_DIR"

"$SCRIPT_DIR/fetch_blazeface_assets.sh"

cd "$SCRIPT_DIR"

"$PYTHON_BIN" export_blazeface.py --backend xnnpack --output "$ASSETS_DIR/blazeface.pte"
"$PYTHON_BIN" export_age.py --backend xnnpack --output "$ASSETS_DIR/age_model.pte"
"$PYTHON_BIN" export_gender.py --backend xnnpack --output "$ASSETS_DIR/gender_model.pte"
"$PYTHON_BIN" export_nsfw.py --backend xnnpack --output "$ASSETS_DIR/nsfw_model.pte"
"$PYTHON_BIN" export_text.py --backend xnnpack --output "$ASSETS_DIR/nsfw_model.pte"
