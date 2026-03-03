#!/usr/bin/env bash

set -e

SCRIPT_DIR="${POLYGEIST_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)}"
echo "Running cgeist pipeline (root: $SCRIPT_DIR)" >&2
BUILD_BIN="${SCRIPT_DIR}/build/bin"
CLANG_VER="${CLANG_VER:-18}"
RESOURCE_DIR="${RESOURCE_DIR:-${SCRIPT_DIR}/llvm-project/build/lib/clang/${CLANG_VER}}"

CGEIST="${BUILD_BIN}/cgeist"
if [[ ! -x "$CGEIST" ]]; then
  echo "cgeist not found at $CGEIST (build Polygeist first)" >&2
  exit 1
fi

if [[ $# -ge 1 && "$1" != -* ]]; then
  INPUT="$1"
  shift
else
  INPUT="${SCRIPT_DIR}/examples/vecAdd.cu"
fi

if [[ ! -f "$INPUT" ]]; then
  echo "Input file not found: $INPUT" >&2
  exit 1
fi

OUT_DIR="$SCRIPT_DIR"
OUT_CUBIN="${OUT_DIR}/vecAdd.cubin"
LOG="${OUT_DIR}/log.txt"

cd "$SCRIPT_DIR"

set +e
"$CGEIST" "$INPUT" \
  --resource-dir "$RESOURCE_DIR" \
  --cuda-path="${CUDA_PATH:-}" \
  -lstdc++ -ldl -lpthread -lrt -lcudart_static -lcuda \
  -o "$OUT_CUBIN" \
  --cuda-gpu-arch=sm_80 \
  -emit-cuda -cuda-lower --pm-enable-printing \
  "$@" \
  &> "$LOG"
CGEIST_EXIT=$?
set -e

echo "Pipeline done. Log: $LOG  Output: $OUT_CUBIN" >&2
if [[ $CGEIST_EXIT -ne 0 ]]; then echo "Warning: cgeist exited with $CGEIST_EXIT (see $LOG)" >&2; fi

python3 "${SCRIPT_DIR}/generate_viewer.py"
exit $CGEIST_EXIT
