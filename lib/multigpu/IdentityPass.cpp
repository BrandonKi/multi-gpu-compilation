#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassRegistry.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

#include "multigpu/MultiGpuOps.h"

#define DEBUG_TYPE "identity-pass"

using namespace mlir;
using namespace multigpu;

struct IdentityPass
    : public PassWrapper<IdentityPass, OperationPass<ModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(IdentityPass)

    void runOnOperation() override {
        ModuleOp module = getOperation();
        MLIRContext *context = &getContext();

        // patterns.add<LowerDestroyStreamOp>(context);

        // if (failed(applyPatternsAndFoldGreedily(module, std::move(patterns))))
        //   signalPassFailure();
    }

    StringRef getArgument() const final {
        return "identity-pass";
    }

    StringRef getDescription() const final {
        return "Does nothing...";
    }
};

namespace mlir {
namespace multigpu {

std::unique_ptr<Pass> createIdentityPass() {
    return std::make_unique<IdentityPass>();
}

void registerIdentityPass() {
    PassRegistration<IdentityPass>();
}

}
}
