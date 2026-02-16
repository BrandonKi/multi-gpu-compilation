SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

export CUDA_PATH="/usr/local/cuda-12.1"

export LIBRARY_PATH=/usr/local/cuda-12.1/targets/x86_64-linux/lib/:/usr/lib/wsl/lib/:$LIBRARY_PATH

if [ -z "$LLVM_REPO" ]; then
    #export LLVM_REPO=$(realpath "${SCRIPT_DIR}/../fnn2/llvm-project")
    export LLVM_REPO=$(realpath "${SCRIPT_DIR}/llvm-project")
fi

if [ -z "$MLIR_DIR" ]; then
    export MLIR_DIR=$LLVM_REPO/build2/lib/cmake/mlir
fi

if [ -z "$LLVM_SYMBOLIZER_PATH" ]; then
    export LLVM_SYMBOLIZER_PATH="$LLVM_REPO/build2/bin/llvm-symbolizer"
fi
