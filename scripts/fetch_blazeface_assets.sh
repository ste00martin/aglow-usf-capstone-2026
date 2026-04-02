#!/usr/bin/env bash
# Download BlazeFace model sources from hollance/BlazeFace-PyTorch (not vendored in git; see scripts/.gitignore).
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BASE_URL="https://raw.githubusercontent.com/hollance/BlazeFace-PyTorch/master"

fetch_one() {
  local name="$1"
  local dest="$SCRIPT_DIR/$name"
  if [[ -f "$dest" ]]; then
    echo "Already present: $dest"
    return 0
  fi
  echo "Downloading $name ..."
  curl -fsSL -o "$dest" "$BASE_URL/$name"
  echo "Saved: $dest"
}

fetch_one blazeface.py
fetch_one blazeface.pth
fetch_one anchors.npy
