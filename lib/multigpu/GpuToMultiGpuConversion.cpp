#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/SymbolTable.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

#include "multigpu/MultiGpuDialect.h"
#include "multigpu/MultiGpuOps.h"

#define DEBUG_TYPE "gpu-to-multigpu"

using namespace mlir;
using namespace multigpu;
using namespace gpu;

namespace {

// convert gpu.launch to mgpu.launch
struct ConvertLaunchOp : public OpRewritePattern<gpu::LaunchOp> {
    using OpRewritePattern<gpu::LaunchOp>::OpRewritePattern;

    LogicalResult matchAndRewrite(gpu::LaunchOp op, PatternRewriter &rewriter) const override {
        Location loc = op.getLoc();
        MLIRContext *context = rewriter.getContext();

        auto c0 = rewriter.create<arith::ConstantIndexOp>(loc, 0);
        auto deviceType = multigpu::DeviceType::get(context);
        auto device = rewriter.create<multigpu::GetDeviceOp>(loc, deviceType, c0);

        SmallVector<Value> gridDims;
        SmallVector<Value> blockDims;
        
        gridDims.push_back(op.getGridSizeX());
        gridDims.push_back(op.getGridSizeY());
        gridDims.push_back(op.getGridSizeZ());

        blockDims.push_back(op.getBlockSizeX());
        blockDims.push_back(op.getBlockSizeY());
        blockDims.push_back(op.getBlockSizeZ());

        auto mgpuLaunch = rewriter.create<multigpu::LaunchOp>(
            loc, device, gridDims, blockDims, /*stream=*/Value());

        Region &gpuRegion = op.getRegion();
        Region &mgpuRegion = mgpuLaunch.getKernelRegion();
        
        rewriter.inlineRegionBefore(gpuRegion, mgpuRegion, mgpuRegion.end());

        // replace gpu.terminator with mgpu.terminator
        Block &block = mgpuRegion.front();
        for (auto &innerOp : block) {
            if (isa<gpu::TerminatorOp>(innerOp)) {
                rewriter.setInsertionPoint(&innerOp);
                rewriter.create<multigpu::TerminatorOp>(innerOp.getLoc());
                rewriter.eraseOp(&innerOp);
                break;
            }
        }

        rewriter.eraseOp(op);
        return success();
    }
};

struct GpuToMultiGpuConversionPass
    : public PassWrapper<GpuToMultiGpuConversionPass, OperationPass<ModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(GpuToMultiGpuConversionPass)

    void getDependentDialects(DialectRegistry &registry) const override {
        registry.insert<multigpu::MultiGpuDialect>();
        registry.insert<gpu::GPUDialect>();
        registry.insert<arith::ArithDialect>();
    }

    void insertDeviceConfigOp(ModuleOp& mod, MLIRContext *ctx) {
        bool present = false;
        mod.walk([&present](multigpu::DeviceConfigOp op) {
            present = true;
            return WalkResult::interrupt();
        });
        
        if(!present) {
            OpBuilder b(ctx);
            b.setInsertionPointToStart(mod.getBody());
            auto attr = multigpu::DeviceConfigAttr::get(ctx, /*numDevices=*/1, /*deviceIds=*/DenseI32ArrayAttr());
            b.create<multigpu::DeviceConfigOp>(mod.getLoc(), b.getStringAttr("mgpu_device_config"), attr);
        }
    }

    void runOnOperation() override {
        ModuleOp module = getOperation();
        MLIRContext *context = &getContext();

        ConversionTarget target(*context);

        target.addLegalDialect<multigpu::MultiGpuDialect>();
        target.addLegalDialect<arith::ArithDialect>();
        target.addLegalDialect<func::FuncDialect>();
        target.addLegalDialect<memref::MemRefDialect>();
        target.addLegalDialect<scf::SCFDialect>();
        target.addLegalDialect<affine::AffineDialect>();

        target.addIllegalOp<gpu::LaunchOp>();

        target.addLegalOp<gpu::ThreadIdOp>();
        target.addLegalOp<gpu::BlockIdOp>();
        target.addLegalOp<gpu::BlockDimOp>();
        target.addLegalOp<gpu::GridDimOp>();
        
        // I guess this should be a rewrite pattern?
        insertDeviceConfigOp(module, context);

        RewritePatternSet patterns(context);
        patterns.add<ConvertLaunchOp>(context);

        if (failed(applyPartialConversion(module, target, std::move(patterns))))
            signalPassFailure();
    }

    StringRef getArgument() const final {
        return "gpu-to-multigpu";
    }

    StringRef getDescription() const final {
        return "Convert GPU dialect operations to MultiGPU dialect";
    }
};

}

namespace mlir {
namespace multigpu {

std::unique_ptr<Pass> createGpuToMultiGpuConversionPass() {
    return std::make_unique<GpuToMultiGpuConversionPass>();
}

void registerGpuToMultiGpuConversionPass() {
    PassRegistration<GpuToMultiGpuConversionPass>();
}

}
}
