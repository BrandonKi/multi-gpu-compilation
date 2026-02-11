
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

#include "mlir/IR/DialectImplementation.h"
#include "multigpu/MultiGpuOps.h"

#define GET_TYPEDEF_CLASSES
#include "multigpu/MultiGpuOpsTypes.cpp.inc"

#include "multigpu/MultiGpuEnums.cpp.inc"

// Note: MultiGpuAttrDefs.cpp.inc is included in MultiGpuDialect.cpp before initialize()
// to ensure storage types are complete for addAttributes. The verify implementation
// below can still access the attribute class since it's declared in the header.

#define GET_OP_CLASSES
#include "multigpu/MultiGpuOps.cpp.inc"

using namespace mlir;
using namespace multigpu;

void DeviceConfigOp::print(OpAsmPrinter &p) {
  p << ' ';
  p.printSymbolName(getSymNameAttr().getValue());
  p << ' ';
  // Print attribute with type to ensure dialect prefix is included
  p.printAttribute(getConfigAttr());
  p.printOptionalAttrDict((*this)->getAttrs(), {"sym_name", "config"});
}

ParseResult DeviceConfigOp::parse(OpAsmParser &parser, OperationState &result) {
  StringAttr symName;
  if (parser.parseSymbolName(symName, "sym_name", result.attributes))
    return failure();
  
  Attribute configAttr;
  if (parser.parseAttribute(configAttr, "config", result.attributes))
    return failure();
  
  if (parser.parseOptionalAttrDict(result.attributes))
    return failure();
  
  return success();
}

void AllReduceOp::print(OpAsmPrinter &p) {
  p << ' ';
  p << getSendBuf();
  p << ",";
  p << ' ';
  p << getRecvBuf();
  p << ",";
  p << ' ';
  p << getComm();
  p << ",";
  p << ' ';
  // Print enum attribute without quotes
  switch (getReductionKind()) {
  case ReductionKind::Sum:
    p << "sum";
    break;
  case ReductionKind::Prod:
    p << "prod";
    break;
  case ReductionKind::Min:
    p << "min";
    break;
  case ReductionKind::Max:
    p << "max";
    break;
  }
  if (getStream()) {
    p << ' ' << "stream";
    p << ' ';
    p << getStream();
    p << ' ' << ":";
    p << ' ';
    p << getStream().getType();
  }
  p.printOptionalAttrDict((*this)->getAttrs(), {"reductionKind", "stream"});
  p << ' ' << ":";
  p << ' ';
  p << getSendBuf().getType();
  p << ",";
  p << ' ';
  p << getRecvBuf().getType();
  p << ",";
  p << ' ';
  p << getComm().getType();
}

ParseResult AllReduceOp::parse(OpAsmParser &parser, OperationState &result) {
  OpAsmParser::UnresolvedOperand sendBuf, recvBuf, comm;
  if (parser.parseOperand(sendBuf) || parser.parseComma() ||
      parser.parseOperand(recvBuf) || parser.parseComma() ||
      parser.parseOperand(comm) || parser.parseComma())
    return failure();
  
  // Parse enum attribute (quoted or unquoted)
  StringRef reductionStr;
  std::string quotedStr;
  if (succeeded(parser.parseOptionalString(&quotedStr))) {
    // Quoted string
    reductionStr = quotedStr;
  } else if (parser.parseKeyword(&reductionStr)) {
    return failure();
  }
  
  ReductionKind reductionKind;
  if (reductionStr == "sum")
    reductionKind = ReductionKind::Sum;
  else if (reductionStr == "prod")
    reductionKind = ReductionKind::Prod;
  else if (reductionStr == "min")
    reductionKind = ReductionKind::Min;
  else if (reductionStr == "max")
    reductionKind = ReductionKind::Max;
  else
    return parser.emitError(parser.getNameLoc(), "unknown reduction kind: ") << reductionStr;
  
  result.addAttribute("reductionKind", ReductionKindAttr::get(parser.getContext(), reductionKind));
  
  OpAsmParser::UnresolvedOperand stream;
  Type streamType;
  bool hasStream = succeeded(parser.parseOptionalKeyword("stream")) && 
                   succeeded(parser.parseOperand(stream)) &&
                   succeeded(parser.parseColonType(streamType));
  
  if (parser.parseOptionalAttrDict(result.attributes))
    return failure();
  
  SmallVector<Type> types;
  if (parser.parseColonTypeList(types) || types.size() != 3)
    return failure();
  
  if (parser.resolveOperand(sendBuf, types[0], result.operands) ||
      parser.resolveOperand(recvBuf, types[1], result.operands) ||
      parser.resolveOperand(comm, types[2], result.operands))
    return failure();
  
  if (hasStream && parser.resolveOperand(stream, streamType, result.operands))
    return failure();
  
  return success();
}

// void ReduceScatterOp::print(OpAsmPrinter &p) {
//   p << ' ';
//   p << getSendBuf();
//   p << ",";
//   p << ' ';
//   p << getRecvBuf();
//   p << ",";
//   p << ' ';
//   p << getComm();
//   p << ",";
//   p << ' ';
//   // Print enum attribute without quotes
//   switch (getReductionKind()) {
//   case ReductionKind::Sum:
//     p << "sum";
//     break;
//   case ReductionKind::Prod:
//     p << "prod";
//     break;
//   case ReductionKind::Min:
//     p << "min";
//     break;
//   case ReductionKind::Max:
//     p << "max";
//     break;
//   }
//   if (getStream()) {
//     p << ' ' << "stream";
//     p << ' ';
//     p << getStream();
//     p << ' ' << ":";
//     p << ' ';
//     p << getStream().getType();
//   }
//   p.printOptionalAttrDict((*this)->getAttrs(), {"reductionKind", "stream"});
//   p << ' ' << ":";
//   p << ' ';
//   p << getSendBuf().getType();
//   p << ",";
//   p << ' ';
//   p << getRecvBuf().getType();
//   p << ",";
//   p << ' ';
//   p << getComm().getType();
// }

// ParseResult ReduceScatterOp::parse(OpAsmParser &parser, OperationState &result) {
//   OpAsmParser::UnresolvedOperand sendBuf, recvBuf, comm;
//   if (parser.parseOperand(sendBuf) || parser.parseComma() ||
//       parser.parseOperand(recvBuf) || parser.parseComma() ||
//       parser.parseOperand(comm) || parser.parseComma())
//     return failure();
  
//   // Parse enum attribute (quoted or unquoted)
//   StringRef reductionStr;
//   std::string quotedStr;
//   if (succeeded(parser.parseOptionalString(&quotedStr))) {
//     // Quoted string
//     reductionStr = quotedStr;
//   } else if (parser.parseKeyword(&reductionStr)) {
//     return failure();
//   }
  
//   ReductionKind reductionKind;
//   if (reductionStr == "sum")
//     reductionKind = ReductionKind::Sum;
//   else if (reductionStr == "prod")
//     reductionKind = ReductionKind::Prod;
//   else if (reductionStr == "min")
//     reductionKind = ReductionKind::Min;
//   else if (reductionStr == "max")
//     reductionKind = ReductionKind::Max;
//   else
//     return parser.emitError(parser.getNameLoc(), "unknown reduction kind: ") << reductionStr;
  
//   result.addAttribute("reductionKind", ReductionKindAttr::get(parser.getContext(), reductionKind));
  
//   OpAsmParser::UnresolvedOperand stream;
//   Type streamType;
//   bool hasStream = succeeded(parser.parseOptionalKeyword("stream")) && 
//                    succeeded(parser.parseOperand(stream)) &&
//                    succeeded(parser.parseColonType(streamType));
  
//   if (parser.parseOptionalAttrDict(result.attributes))
//     return failure();
  
//   SmallVector<Type> types;
//   if (parser.parseColonTypeList(types) || types.size() != 3)
//     return failure();
  
//   if (parser.resolveOperand(sendBuf, types[0], result.operands) ||
//       parser.resolveOperand(recvBuf, types[1], result.operands) ||
//       parser.resolveOperand(comm, types[2], result.operands))
//     return failure();
  
//   if (hasStream && parser.resolveOperand(stream, streamType, result.operands))
//     return failure();
  
//   return success();
// }

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

namespace mlir::multigpu {
bool DeviceConfigAttr::hasDefaultDeviceIds() const {
  return !getDeviceIds(); // Returns true if the OptionalParameter is null
}

int32_t DeviceConfigAttr::getSystemDeviceId(uint32_t r) const {
  if (hasDefaultDeviceIds())
    return static_cast<int32_t>(r);
  return getDeviceIds()[r];
}

LogicalResult DeviceConfigAttr::verify(
    function_ref<InFlightDiagnostic()> emitError, uint32_t numDevices,
    DenseI32ArrayAttr deviceIds) {
  // Verified: numDevices >= 1; len(deviceIds) == numDevices when present.
  if (numDevices < 1) {
    return emitError() << "numDevices must be >= 1, got " << numDevices;
  }
  if (deviceIds) {
    if (static_cast<uint32_t>(deviceIds.size()) != numDevices) {
      return emitError() << "deviceIds length (" << deviceIds.size()
                         << ") must equal numDevices (" << numDevices << ")";
    }
  }
  return success();
}
} // namespace mlir::multigpu