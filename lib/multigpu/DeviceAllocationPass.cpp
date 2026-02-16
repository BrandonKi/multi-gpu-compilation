#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"

#include "llvm/ADT/SmallVector.h"

#include "multigpu/MultiGpuDialect.h"
#include "multigpu/MultiGpuOps.h"

#define DEBUG_TYPE "mgpu-device-allocation"

using namespace mlir;
using namespace multigpu;

namespace {

struct DeviceAllocationPass : public PassWrapper<DeviceAllocationPass, OperationPass<ModuleOp>> {
    MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(DeviceAllocationPass)

    void getDependentDialects(DialectRegistry &registry) const override {
        registry.insert<multigpu::MultiGpuDialect>();
        registry.insert<arith::ArithDialect>();
    }

    void runOnOperation() override {
        ModuleOp module = getOperation();
        MLIRContext *context = &getContext();

        uint32_t numDevices = 1;
        module.walk([&](DeviceConfigOp op) {
            numDevices = op.getNumDevices();
            return WalkResult::interrupt();
        });

        if (numDevices == 0) {
            module.emitError("invalid device config found");
            return;
        }

        // collect GetDeviceOp and assign device indices (just dumb way for now).
        SmallVector<std::pair<GetDeviceOp, uint32_t>, 8> toReplace;
        uint32_t nextDeviceIndex = 0;
        module.walk([&](GetDeviceOp op) {
            uint32_t assigned = nextDeviceIndex % numDevices;
            nextDeviceIndex++;

            Value indexOperand = op.getDeviceIndex();
            auto constantOp = indexOperand.getDefiningOp<arith::ConstantIndexOp>();
            if (!constantOp) {
                op.emitRemark("device allocation skips get_device with non-constant index");
                return;
            }
            int64_t currentIndex = constantOp.value();
            if (currentIndex != static_cast<int64_t>(assigned))
                toReplace.push_back({op, assigned});
        });

        OpBuilder builder(context);
        for (auto [op, assigned] : toReplace) {
            builder.setInsertionPoint(op);
            auto newCst = builder.create<arith::ConstantIndexOp>(op.getLoc(), static_cast<int64_t>(assigned));
            auto deviceType = DeviceType::get(context);
            auto newGetDevice = builder.create<GetDeviceOp>(op.getLoc(), deviceType, newCst);
            op.replaceAllUsesWith(newGetDevice.getResult());
            op.erase();
        }
    }

    StringRef getArgument() const final { return "mgpu-device-allocation"; }

    StringRef getDescription() const final {
        return "Assign device indices to mgpu.get_device ops(depends on "
               "mgpu.device_config module op)";
    }
};

} // namespace

namespace mlir {
namespace multigpu {

std::unique_ptr<Pass> createDeviceAllocationPass() { return std::make_unique<DeviceAllocationPass>(); }

void registerDeviceAllocationPass() { PassRegistration<DeviceAllocationPass>(); }

} // namespace multigpu
} // namespace mlir
