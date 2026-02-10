#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Dialect.h"
#include "mlir/IR/Matchers.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "mlir/Interfaces/ViewLikeInterface.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/ADT/TypeSwitch.h"

#define GET_TYPEDEF_CLASSES
#include "multigpu/MultiGpuOpsTypes.h.inc"

#include "multigpu/MultiGpuEnums.h.inc"

#define GET_ATTRDEF_CLASSES
#include "multigpu/MultiGpuAttrDefs.h.inc"

#define GET_OP_CLASSES
#include "multigpu/MultiGpuOps.h.inc"
