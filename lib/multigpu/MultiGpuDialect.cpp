#include "multigpu/MultiGpuOps.h"
#include "mlir/IR/DialectImplementation.h"

using namespace mlir;
using namespace mlir::multigpu;

#include "multigpu/MultiGpuOpsDialect.h.inc"

void MultiGpuDialect::initialize() {
  addTypes<DeviceType, StreamType, CommunicatorType>();
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
