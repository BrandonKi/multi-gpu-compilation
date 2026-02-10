#include "multigpu/MultiGpuOps.h"
#include "mlir/IR/DialectImplementation.h"

using namespace mlir;
using namespace mlir::multigpu;

#include "multigpu/MultiGpuOpsDialect.h.inc"

#define GET_ATTRDEF_CLASSES
#include "multigpu/MultiGpuAttrDefs.cpp.inc"

void MultiGpuDialect::initialize() {
  addTypes<DeviceType, StreamType, CommunicatorType>();
  addAttributes<DeviceConfigAttr>();
  addOperations<
#define GET_OP_LIST
#include "multigpu/MultiGpuOps.cpp.inc"
      >();
}

Type MultiGpuDialect::parseType(DialectAsmParser &parser) const {
    StringRef keyword;
    if (parser.parseKeyword(&keyword))
        return Type();
    if (keyword == "device")
        return DeviceType::get(getContext());
    if (keyword == "stream")
        return StreamType::get(getContext());
    if (keyword == "communicator")
        return CommunicatorType::get(getContext());
    return Type();
}

void MultiGpuDialect::printType(Type type, DialectAsmPrinter &printer) const {
    if (type.isa<DeviceType>()) {
        printer << "device";
    } else if (type.isa<StreamType>()) {
        printer << "stream";
    } else if (type.isa<CommunicatorType>()) {
        printer << "communicator";
    } else {
        assert(false && "unknown type");
    }
}

#include "multigpu/MultiGpuOpsDialect.cpp.inc"

Attribute MultiGpuDialect::parseAttribute(DialectAsmParser &parser, 
                                          Type type) const {
  StringRef keyword;
  if (parser.parseKeyword(&keyword))
    return Attribute();
  
  if (keyword == "device_config") {
    Attribute attr = DeviceConfigAttr::parse(parser, type);
    if (!attr)
      return Attribute();
    return attr;
  }
  
  if (keyword == "sum")
    return ReductionKindAttr::get(parser.getContext(), ReductionKind::Sum);
  if (keyword == "prod")
    return ReductionKindAttr::get(parser.getContext(), ReductionKind::Prod);
  if (keyword == "min")
    return ReductionKindAttr::get(parser.getContext(), ReductionKind::Min);
  if (keyword == "max")
    return ReductionKindAttr::get(parser.getContext(), ReductionKind::Max);
  
  if (keyword == "uniform")
    return ShardKindAttr::get(parser.getContext(), ShardKind::Uniform);
  if (keyword == "uneven")
    return ShardKindAttr::get(parser.getContext(), ShardKind::Uneven);
  if (keyword == "replicate")
    return ShardKindAttr::get(parser.getContext(), ShardKind::Replicate);
  
  parser.emitError(parser.getNameLoc(), "unknown attribute: ") << keyword;
  return Attribute();
}

void MultiGpuDialect::printAttribute(Attribute attr, 
                                     DialectAsmPrinter &printer) const {
  if (auto deviceConfig = attr.dyn_cast<DeviceConfigAttr>()) {
    printer << "device_config";
    deviceConfig.print(printer);
    return;
  }
  if (auto reductionKind = attr.dyn_cast<ReductionKindAttr>()) {
    printer << "\"";
    switch (reductionKind.getValue()) {
    case ReductionKind::Sum:
      printer << "sum";
      break;
    case ReductionKind::Prod:
      printer << "prod";
      break;
    case ReductionKind::Min:
      printer << "min";
      break;
    case ReductionKind::Max:
      printer << "max";
      break;
    }
    printer << "\"";
    return;
  }
  if (auto shardKind = attr.dyn_cast<ShardKindAttr>()) {
    printer << "\"";
    switch (shardKind.getValue()) {
    case ShardKind::Uniform:
      printer << "uniform";
      break;
    case ShardKind::Uneven:
      printer << "uneven";
      break;
    case ShardKind::Replicate:
      printer << "replicate";
      break;
    }
    printer << "\"";
    return;
  }
}