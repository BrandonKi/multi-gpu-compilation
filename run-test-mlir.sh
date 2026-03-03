#!/usr/bin/env bash
# run-test-mlir.sh split-kernel-mgpu.mlir
# run-test-mlir.sh split-kernel-mgpu

set -e

SCRIPT_DIR="${POLYGEIST_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)}"
TEST_DIR="${SCRIPT_DIR}/test/polygeist-opt"

NAME="$1"
INPUT_FILE=
if [[ -f "${TEST_DIR}/${NAME}" ]]; then
  INPUT_FILE="${TEST_DIR}/${NAME}"
elif [[ -f "${TEST_DIR}/${NAME}.mlir" ]]; then
  INPUT_FILE="${TEST_DIR}/${NAME}.mlir"
else
  echo "Test file not found: ${TEST_DIR}/${NAME} or ${TEST_DIR}/${NAME}.mlir" >&2
  exit 1
fi

POLYGEIST_OPT="${POLYGEIST_OPT:-}"
BUILD_BIN="${SCRIPT_DIR}/build/bin"
if [[ -z "${POLYGEIST_OPT}" && -x "${BUILD_BIN}/polygeist-opt" ]]; then
  POLYGEIST_OPT="${BUILD_BIN}/polygeist-opt"
elif [[ -z "${POLYGEIST_OPT}" ]]; then
  POLYGEIST_OPT="polygeist-opt"
fi
if [[ "$POLYGEIST_OPT" == */* ]]; then
  export PATH="$(dirname "$POLYGEIST_OPT"):$PATH"
fi

RUN_LINE=$(grep -m1 '^// RUN:' "$INPUT_FILE" | sed 's|^// RUN: *||')
if [[ -z "$RUN_LINE" ]]; then
  echo "No // RUN: line found in $INPUT_FILE" >&2
  exit 1
fi

CMD=$(echo "$RUN_LINE" | sed 's/ *| *FileCheck.*//')
CMD="${CMD//%s/$INPUT_FILE}"

eval "$CMD"
