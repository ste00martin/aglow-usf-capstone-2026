#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
VENV_DIR="$REPO_ROOT/.venv-executorch-export"
REQUIREMENTS_FILE="$SCRIPT_DIR/requirements-executorch-export.txt"
PYTHON_VERSION="${PYTHON_VERSION:-3.12}"

python_matches_requested_version() {
  local python_bin="$1"

  "$python_bin" - "$PYTHON_VERSION" <<'PY' >/dev/null 2>&1
import sys

requested = tuple(int(part) for part in sys.argv[1].split(".") if part)
sys.exit(0 if requested and sys.version_info[: len(requested)] == requested else 1)
PY
}

if [[ ! -f "$REQUIREMENTS_FILE" ]]; then
  echo "Missing requirements file: $REQUIREMENTS_FILE" >&2
  exit 1
fi

if command -v uv >/dev/null 2>&1; then
  echo "Using uv to provision Python $PYTHON_VERSION and create $VENV_DIR"
  uv python install "$PYTHON_VERSION"
  uv venv --python "$PYTHON_VERSION" --clear "$VENV_DIR"
  "$VENV_DIR/bin/python" -m ensurepip --upgrade
  "$VENV_DIR/bin/python" -m pip install --upgrade pip
  "$VENV_DIR/bin/python" -m pip install -r "$REQUIREMENTS_FILE"
else
  if command -v "python$PYTHON_VERSION" >/dev/null 2>&1; then
    PYTHON_BIN="$(command -v "python$PYTHON_VERSION")"
  elif command -v python3 >/dev/null 2>&1 && python_matches_requested_version "$(command -v python3)"
  then
    PYTHON_BIN="$(command -v python3)"
  else
    echo "Need Python $PYTHON_VERSION or uv to create the export environment." >&2
    echo "Install uv or Python $PYTHON_VERSION and rerun this script." >&2
    exit 1
  fi

  echo "Using $PYTHON_BIN to create $VENV_DIR"
  "$PYTHON_BIN" -m venv "$VENV_DIR"
  "$VENV_DIR/bin/python" -m ensurepip --upgrade
  "$VENV_DIR/bin/python" -m pip install --upgrade pip
  "$VENV_DIR/bin/python" -m pip install -r "$REQUIREMENTS_FILE"
fi

echo
echo "ExecuTorch export environment is ready:"
echo "  $VENV_DIR"
echo
echo "Next steps:"
echo "  source $VENV_DIR/bin/activate"
echo "  ./scripts/export_models_xnnpack.sh      # Android / CPU-safe default assets"
echo "  ./scripts/export_models_ios_coreml.sh   # iOS CoreML companion assets"
