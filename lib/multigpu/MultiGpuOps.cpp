
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/LLVMIR/LLVMTypes.h"
#include "mlir/IR/AffineExpr.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"

#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Arith/Utils/Utils.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/OpenMP/OpenMPDialect.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/Dominance.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/IR/IntegerSet.h"

#include "llvm/ADT/SetVector.h"
#include "llvm/Support/Debug.h"

#include <set>
#include <memory>

#include "multigpu/MultiGpuOps.h"

#define GET_TYPEDEF_CLASSES
#include "multigpu/MultiGpuOpsTypes.cpp.inc"

#define GET_OP_CLASSES
#include "multigpu/MultiGpuOps.cpp.inc"

using namespace mlir;
using namespace multigpu;

void LaunchOp::print(OpAsmPrinter &p) {
    p << " " << getDevice();
    p << " grid (";
    llvm::interleaveComma(getGrid(), p, [&](Value v) { p << v; });
    p << ") block (";
    llvm::interleaveComma(getBlock(), p, [&](Value v) { p << v; });
    p << ")";
    if (auto s = getStream())
        p << " stream " << s;
    p << " ";
    p.printRegion(getKernelRegion());
    p.printOptionalAttrDict(ArrayRef<mlir::NamedAttribute>{}, ArrayRef<llvm::StringRef>{getOperandSegmentSizesAttrName().getValue()});
}

mlir::ParseResult LaunchOp::parse(OpAsmParser &parser, OperationState &result) {
    OpAsmParser::UnresolvedOperand devOperand;
    if (parser.parseOperand(devOperand))
        return failure();

    SmallVector<OpAsmParser::UnresolvedOperand> gridOperands, blockOperands;
    if (parser.parseKeyword("grid") ||
        parser.parseCommaSeparatedList(OpAsmParser::Delimiter::Paren, [&]() {
            return parser.parseOperand(gridOperands.emplace_back()); 
        }) ||
        parser.parseKeyword("block") || 
        parser.parseCommaSeparatedList(OpAsmParser::Delimiter::Paren, [&]() {
            return parser.parseOperand(blockOperands.emplace_back()); 
        }))
        return failure();

    OpAsmParser::UnresolvedOperand streamOperand;
    bool hasStream = succeeded(parser.parseOptionalKeyword("stream")) && succeeded(parser.parseOperand(streamOperand));

    result.regions.emplace_back(std::make_unique<Region>());
    if (parser.parseRegion(*result.regions.back(), {}, /*enableImplicitTerminator=*/false))
        return failure();
    if (result.regions.back()->empty())
        result.regions.back()->emplaceBlock();
    Block *block = &result.regions.back()->front();
    bool needsTerminator = block->empty();
    if (!needsTerminator) {
        Operation &backOp = block->back();
        needsTerminator = !backOp.mightHaveTrait<OpTrait::IsTerminator>();
    }
    if (needsTerminator) {
        OpBuilder builder(parser.getContext());
        builder.setInsertionPointToEnd(block);
        Location loc = parser.getBuilder().getUnknownLoc();
        builder.create<TerminatorOp>(loc);
    }

    IndexType idxType = parser.getBuilder().getIndexType();
    StreamType streamType = StreamType::get(parser.getContext());
    DeviceType devType   = DeviceType::get(parser.getContext());

    if (parser.resolveOperand(devOperand, devType, result.operands) ||
        parser.resolveOperands(gridOperands, idxType, parser.getNameLoc(),
                                result.operands) ||
        parser.resolveOperands(blockOperands, idxType, parser.getNameLoc(),
                                result.operands))
        return failure();
    if (hasStream &&
        parser.resolveOperand(streamOperand, streamType, result.operands))
        return failure();

    result.addAttribute(
        LaunchOp::getOperandSegmentSizesAttrName(result.name),
        parser.getBuilder().getDenseI32ArrayAttr({1, (int32_t)gridOperands.size(), (int32_t)blockOperands.size(), hasStream ? 1 : 0}));
    return success();
}
