#ifndef MLIR_MULTIGPU_PASSES_H
#define MLIR_MULTIGPU_PASSES_H

#include "mlir/Pass/Pass.h"

namespace mlir {
namespace multigpu {

std::unique_ptr<Pass> createIdentityPass();
void registerIdentityPass();

std::unique_ptr<Pass> createGpuToMultiGpuConversionPass();
void registerGpuToMultiGpuConversionPass();

std::unique_ptr<Pass> createMultiGpuToLLVMConversionPass();
void registerMultiGpuToLLVMConversionPass();

std::unique_ptr<Pass> createDeviceAllocationPass();
void registerDeviceAllocationPass();

}
}

#endif
