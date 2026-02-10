#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

#include "multigpu/MultiGpuDialect.h"
#include "multigpu/MultiGpuOps.h"

#define DEBUG_TYPE "multigpu-to-gpu"

using namespace mlir;
using namespace multigpu;
using namespace gpu;

namespace {

class MultiGpuToGpuTypeConverter : public TypeConverter {
public:
    MultiGpuToGpuTypeConverter() {
        addConversion([](multigpu::DeviceType type) {
            return IndexType::get(type.getContext());
        });

        addConversion([](multigpu::StreamType type) {
            return IndexType::get(type.getContext());
        });

        addConversion([](Type type) {
            return type;
        });

        addTargetMaterialization([](OpBuilder &builder, Type resultType, ValueRange inputs, Location loc) -> Value {
            if (inputs.size() != 1)
                return nullptr;
            return inputs[0];
        });

        addSourceMaterialization([](OpBuilder &builder, Type resultType, ValueRange inputs, Location loc) -> Value {
            if (inputs.size() != 1)
                return nullptr;
            return inputs[0];
        });
    }
};

struct ConvertGetDeviceOp : public OpConversionPattern<GetDeviceOp> {
    using OpConversionPattern<GetDeviceOp>::OpConversionPattern;

    LogicalResult
    matchAndRewrite(GetDeviceOp op, OpAdaptor adaptor, ConversionPatternRewriter &rewriter) const override {
        auto deviceId = adaptor.getIndex().getDefiningOp<mlir::arith::ConstantOp>();
        if (!deviceId) {
            return rewriter.notifyMatchFailure(op, "expected a constant operand");
        }
        auto indexType = rewriter.getIndexType();
        auto constantOp = rewriter.create<arith::ConstantIndexOp>(op.getLoc(), deviceId.getValue().cast<IntegerAttr>().getInt());
        rewriter.replaceOp(op, constantOp.getResult());
        return success();
    }
};

struct ConvertCreateStreamOp : public OpConversionPattern<CreateStreamOp> {
    using OpConversionPattern<CreateStreamOp>::OpConversionPattern;

    LogicalResult
    matchAndRewrite(CreateStreamOp op, OpAdaptor adaptor, ConversionPatternRewriter &rewriter) const override {
        auto indexType = rewriter.getIndexType();
        auto constantOp = rewriter.create<arith::ConstantIndexOp>(op.getLoc(), 0);
        rewriter.replaceOp(op, constantOp.getResult());
        return success();
    }
};

// mgpu.free to gpu.dealloc
struct ConvertFreeOp : public OpConversionPattern<FreeOp> {
    using OpConversionPattern<FreeOp>::OpConversionPattern;

    LogicalResult
    matchAndRewrite(FreeOp op, OpAdaptor adaptor, ConversionPatternRewriter &rewriter) const override {
        rewriter.replaceOpWithNewOp<gpu::DeallocOp>(op, /*asyncTokenType=*/Type(), /*asyncDependencies=*/ValueRange(), adaptor.getPtr());
        return success();
    }
};

// mgpu.sync_device to gpu.wait
struct ConvertSyncDeviceOp : public OpConversionPattern<SyncDeviceOp> {
    using OpConversionPattern<SyncDeviceOp>::OpConversionPattern;

    LogicalResult
    matchAndRewrite(SyncDeviceOp op, OpAdaptor adaptor, ConversionPatternRewriter &rewriter) const override {
        rewriter.eraseOp(op);
        return success();
    }
};

// mgpu.sync_stream to gpu.wait
struct ConvertSyncStreamOp : public OpConversionPattern<SyncStreamOp> {
    using OpConversionPattern<SyncStreamOp>::OpConversionPattern;

    LogicalResult
    matchAndRewrite(SyncStreamOp op, OpAdaptor adaptor, ConversionPatternRewriter &rewriter) const override {
        rewriter.eraseOp(op);
        return success();
    }
};

// mgpu.destroy_stream to noop
struct ConvertDestroyStreamOp : public OpConversionPattern<DestroyStreamOp> {
    using OpConversionPattern<DestroyStreamOp>::OpConversionPattern;

    LogicalResult matchAndRewrite(DestroyStreamOp op, OpAdaptor adaptor, ConversionPatternRewriter &rewriter) const override {
        rewriter.eraseOp(op);
        return success();
    }
};

struct MultiGpuToGpuConversionPass
    : public PassWrapper<MultiGpuToGpuConversionPass, OperationPass<ModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(MultiGpuToGpuConversionPass)

    void getDependentDialects(DialectRegistry &registry) const override {
        registry.insert<gpu::GPUDialect>();
    }

    void runOnOperation() override {
        ModuleOp module = getOperation();
        MLIRContext *context = &getContext();

        MultiGpuToGpuTypeConverter typeConverter;

        ConversionTarget target(*context);

        target.addLegalDialect<gpu::GPUDialect>();
        target.addLegalDialect<arith::ArithDialect>();
        target.addLegalDialect<func::FuncDialect>();

        // target.addIllegalDialect<multigpu::MultiGpuDialect>();

        RewritePatternSet patterns(context);
        patterns.add<ConvertGetDeviceOp>(typeConverter, context);
        patterns.add<ConvertCreateStreamOp>(typeConverter, context);
        patterns.add<ConvertFreeOp>(typeConverter, context);
        patterns.add<ConvertSyncDeviceOp>(typeConverter, context);
        patterns.add<ConvertSyncStreamOp>(typeConverter, context);
        patterns.add<ConvertDestroyStreamOp>(typeConverter, context);

        if (failed(applyPartialConversion(module, target, std::move(patterns))))
            signalPassFailure();
    }

    StringRef getArgument() const final {
        return "multigpu-to-gpu";
    }

    StringRef getDescription() const final {
        return "Convert MultiGPU operations to GPU dialect operations";
    }
};

}

namespace mlir {
namespace multigpu {

std::unique_ptr<Pass> createMultiGpuToGpuConversionPass() {
    return std::make_unique<MultiGpuToGpuConversionPass>();
}

void registerMultiGpuToGpuConversionPass() {
    PassRegistration<MultiGpuToGpuConversionPass>();
}

}
}
