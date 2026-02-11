#ifndef MLIR_MULTIGPU_PASSES_H
#define MLIR_MULTIGPU_PASSES_H

#include "mlir/Pass/Pass.h"

namespace mlir {
namespace multigpu {

std::unique_ptr<Pass> createIdentityPass();
void registerIdentityPass();

std::unique_ptr<Pass> createMultiGpuToGpuConversionPass();
void registerMultiGpuToGpuConversionPass();

std::unique_ptr<Pass> createMultiGpuToCudaConversionPass();
void registerMultiGpuToCudaConversionPass();

std::unique_ptr<Pass> createGpuToMultiGpuConversionPass();
void registerGpuToMultiGpuConversionPass();

}
}

#endif
