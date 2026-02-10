#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"

#include "multigpu/MultiGpuDialect.h"
#include "multigpu/MultiGpuOps.h"

#define DEBUG_TYPE "multigpu-to-cuda"

using namespace mlir;
using namespace multigpu;

namespace {

static LLVM::LLVMFuncOp getOrCreateCudaFunction(
    ModuleOp module, StringRef name, LLVM::LLVMFunctionType funcType) {
    if (auto existingFunc = module.lookupSymbol<LLVM::LLVMFuncOp>(name))
        return existingFunc;

    OpBuilder::InsertionGuard guard(OpBuilder(module));
    OpBuilder builder(module.getBodyRegion());
    auto func = builder.create<LLVM::LLVMFuncOp>(module.getLoc(), name, funcType);
    return func;
}

// cudaError_t cudaSetDevice(int device)
static LLVM::LLVMFuncOp getCudaSetDeviceFunc(ModuleOp module) {
    auto *context = module.getContext();
    auto i32Type = IntegerType::get(context, 32);
    auto funcType = LLVM::LLVMFunctionType::get(i32Type, {i32Type});
    return getOrCreateCudaFunction(module, "cudaSetDevice", funcType);
}

// cudaError_t cudaStreamCreate(cudaStream_t* stream)
static LLVM::LLVMFuncOp getCudaStreamCreateFunc(ModuleOp module) {
    auto *context = module.getContext();
    auto i32Type = IntegerType::get(context, 32);
    auto ptrType = LLVM::LLVMPointerType::get(context);
    auto funcType = LLVM::LLVMFunctionType::get(i32Type, {ptrType});
    return getOrCreateCudaFunction(module, "cudaStreamCreate", funcType);
}

// cudaError_t cudaStreamDestroy(cudaStream_t stream)
static LLVM::LLVMFuncOp getCudaStreamDestroyFunc(ModuleOp module) {
    auto *context = module.getContext();
    auto i32Type = IntegerType::get(context, 32);
    auto ptrType = LLVM::LLVMPointerType::get(context);
    auto funcType = LLVM::LLVMFunctionType::get(i32Type, {ptrType});
    return getOrCreateCudaFunction(module, "cudaStreamDestroy", funcType);
}

// cudaError_t cudaStreamSynchronize(cudaStream_t stream)
static LLVM::LLVMFuncOp getCudaStreamSynchronizeFunc(ModuleOp module) {
    auto *context = module.getContext();
    auto i32Type = IntegerType::get(context, 32);
    auto ptrType = LLVM::LLVMPointerType::get(context);
    auto funcType = LLVM::LLVMFunctionType::get(i32Type, {ptrType});
    return getOrCreateCudaFunction(module, "cudaStreamSynchronize", funcType);
}

// cudaError_t cudaDeviceSynchronize()
static LLVM::LLVMFuncOp getCudaDeviceSynchronizeFunc(ModuleOp module) {
    auto *context = module.getContext();
    auto i32Type = IntegerType::get(context, 32);
    auto funcType = LLVM::LLVMFunctionType::get(i32Type, {});
    return getOrCreateCudaFunction(module, "cudaDeviceSynchronize", funcType);
}

// cudaError_t cudaFree(void* devPtr)
static LLVM::LLVMFuncOp getCudaFreeFunc(ModuleOp module) {
    auto *context = module.getContext();
    auto i32Type = IntegerType::get(context, 32);
    auto ptrType = LLVM::LLVMPointerType::get(context);
    auto funcType = LLVM::LLVMFunctionType::get(i32Type, {ptrType});
    return getOrCreateCudaFunction(module, "cudaFree", funcType);
}

//----------------------------------------------------------------------------

class MultiGpuToCudaTypeConverter : public TypeConverter {
public:
    MultiGpuToCudaTypeConverter() {
        // Device type -> i32 (device ID)
        addConversion([](multigpu::DeviceType type) { 
            return IntegerType::get(type.getContext(), 32); 
        });

        // Stream type -> !llvm.ptr (cudaStream_t)
        addConversion([](multigpu::StreamType type) { 
            return LLVM::LLVMPointerType::get(type.getContext()); 
        });

        addConversion([](Type type) { return type; });

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

//----------------------------------------------------------------------------

// mgpu.get_device to cudaSetDevice
struct ConvertGetDeviceOp : public OpConversionPattern<GetDeviceOp> {
    using OpConversionPattern<GetDeviceOp>::OpConversionPattern;

    LogicalResult
    matchAndRewrite(GetDeviceOp op, OpAdaptor adaptor, ConversionPatternRewriter &rewriter) const override {
        auto module = op->getParentOfType<ModuleOp>();
        auto cudaSetDevice = getCudaSetDeviceFunc(module);
        
        auto i32Type = rewriter.getI32Type();
        auto deviceId = adaptor.getIndex().getDefiningOp<mlir::arith::ConstantOp>();
        if (!deviceId) {
            return rewriter.notifyMatchFailure(op, "expected a constant operand");
        }
        auto deviceIdConst = rewriter.create<arith::ConstantIndexOp>(
            op.getLoc(), deviceId.getValue().cast<IntegerAttr>().getInt());
        // auto deviceIdAttr = rewriter.getI32IntegerAttr(op.getIndex());
        // auto deviceIdConst = rewriter.create<LLVM::ConstantOp>(
        //     op.getLoc(), i32Type, deviceIdAttr);
        
        // cudaSetDevice(deviceId)
        auto callOp = rewriter.create<LLVM::CallOp>(
            op.getLoc(), cudaSetDevice, ValueRange{deviceIdConst});
        
        rewriter.replaceOp(op, deviceIdConst.getResult());
        return success();
    }
};

// mgpu.create_stream to cudaStreamCreate
struct ConvertCreateStreamOp : public OpConversionPattern<CreateStreamOp> {
    using OpConversionPattern<CreateStreamOp>::OpConversionPattern;

    LogicalResult
    matchAndRewrite(CreateStreamOp op, OpAdaptor adaptor, ConversionPatternRewriter &rewriter) const override {
        auto module = op->getParentOfType<ModuleOp>();
        auto cudaStreamCreate = getCudaStreamCreateFunc(module);
        auto loc = op.getLoc();
        auto *context = rewriter.getContext();

        auto ptrType = LLVM::LLVMPointerType::get(context);
        auto i32Type = rewriter.getI32Type();
        auto one = rewriter.create<LLVM::ConstantOp>(loc, i32Type, rewriter.getI32IntegerAttr(1));
        auto streamAlloca = rewriter.create<LLVM::AllocaOp>(loc, ptrType, ptrType, one, 0);

        // cudaStreamCreate(&stream)
        rewriter.create<LLVM::CallOp>(loc, cudaStreamCreate, ValueRange{streamAlloca});
        
        auto streamHandle = rewriter.create<LLVM::LoadOp>(loc, ptrType, streamAlloca);
        
        rewriter.replaceOp(op, streamHandle.getResult());
        return success();
  }
};

// mgpu.destroy_stream to cudaStreamDestroy
struct ConvertDestroyStreamOp : public OpConversionPattern<DestroyStreamOp> {
    using OpConversionPattern<DestroyStreamOp>::OpConversionPattern;

    LogicalResult
    matchAndRewrite(DestroyStreamOp op, OpAdaptor adaptor, ConversionPatternRewriter &rewriter) const override {
        auto module = op->getParentOfType<ModuleOp>();
        auto cudaStreamDestroy = getCudaStreamDestroyFunc(module);

        // cudaStreamDestroy(stream)
        rewriter.replaceOpWithNewOp<LLVM::CallOp>(op, cudaStreamDestroy, ValueRange{adaptor.getStream()});

        return success();
    }
};

// mgpu.sync_stream to cudaStreamSynchronize
struct ConvertSyncStreamOp : public OpConversionPattern<SyncStreamOp> {
    using OpConversionPattern<SyncStreamOp>::OpConversionPattern;

    LogicalResult
    matchAndRewrite(SyncStreamOp op, OpAdaptor adaptor, ConversionPatternRewriter &rewriter) const override {
        auto module = op->getParentOfType<ModuleOp>();
        auto cudaStreamSync = getCudaStreamSynchronizeFunc(module);

        // cudaStreamSynchronize(stream)
        rewriter.replaceOpWithNewOp<LLVM::CallOp>(
            op, cudaStreamSync, ValueRange{adaptor.getStream()});

        return success();
    }
};

// mgpu.sync_device to cudaDeviceSynchronize
struct ConvertSyncDeviceOp : public OpConversionPattern<SyncDeviceOp> {
    using OpConversionPattern<SyncDeviceOp>::OpConversionPattern;

    LogicalResult
    matchAndRewrite(SyncDeviceOp op, OpAdaptor adaptor, ConversionPatternRewriter &rewriter) const override {
        auto module = op->getParentOfType<ModuleOp>();
        auto cudaDeviceSync = getCudaDeviceSynchronizeFunc(module);

        // cudaDeviceSynchronize()
        rewriter.replaceOpWithNewOp<LLVM::CallOp>(op, cudaDeviceSync, ValueRange{});

        return success();
    }
};

// mgpu.free to cudaFree
struct ConvertFreeOp : public OpConversionPattern<FreeOp> {
    using OpConversionPattern<FreeOp>::OpConversionPattern;

    LogicalResult
    matchAndRewrite(FreeOp op, OpAdaptor adaptor, ConversionPatternRewriter &rewriter) const override {
        auto module = op->getParentOfType<ModuleOp>();
        auto cudaFree = getCudaFreeFunc(module);
        auto loc = op.getLoc();
        auto *context = rewriter.getContext();

        auto memref = adaptor.getPtr();
        auto ptrType = LLVM::LLVMPointerType::get(context);

        auto extractedPtr = rewriter.create<LLVM::ExtractValueOp>(loc, memref, ArrayRef<int64_t>{0});

        // cudaFree(ptr)
        rewriter.replaceOpWithNewOp<LLVM::CallOp>(op, cudaFree, ValueRange{extractedPtr});
        
        return success();
  }
};

//----------------------------------------------------------------------------

struct MultiGpuToCudaConversionPass
    : public PassWrapper<MultiGpuToCudaConversionPass, OperationPass<ModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(MultiGpuToCudaConversionPass)

    void getDependentDialects(DialectRegistry &registry) const override {
        registry.insert<LLVM::LLVMDialect>();
        registry.insert<arith::ArithDialect>();
        registry.insert<func::FuncDialect>();
    }

    void runOnOperation() override {
        ModuleOp module = getOperation();
        MLIRContext *context = &getContext();

        MultiGpuToCudaTypeConverter typeConverter;

        ConversionTarget target(*context);

        target.addLegalDialect<LLVM::LLVMDialect>();
        target.addLegalDialect<arith::ArithDialect>();
        target.addLegalDialect<func::FuncDialect>();

        target.addIllegalDialect<multigpu::MultiGpuDialect>();

        RewritePatternSet patterns(context);
        patterns.add<ConvertGetDeviceOp>(typeConverter, context);
        patterns.add<ConvertCreateStreamOp>(typeConverter, context);
        patterns.add<ConvertDestroyStreamOp>(typeConverter, context);
        patterns.add<ConvertSyncStreamOp>(typeConverter, context);
        patterns.add<ConvertSyncDeviceOp>(typeConverter, context);
        patterns.add<ConvertFreeOp>(typeConverter, context);

        if (failed(applyPartialConversion(module, target, std::move(patterns))))
            signalPassFailure();
    }

    StringRef getArgument() const {
        return "multigpu-to-cuda";
    }
  
    StringRef getDescription() const {
        return "Convert MultiGPU operations to CUDA runtime calls";
    }
};

}

//----------------------------------------------------------------------------

namespace mlir {
namespace multigpu {

std::unique_ptr<Pass> createMultiGpuToCudaConversionPass() {
    return std::make_unique<MultiGpuToCudaConversionPass>();
}

void registerMultiGpuToCudaConversionPass() {
    PassRegistration<MultiGpuToCudaConversionPass>();
}

}
}
