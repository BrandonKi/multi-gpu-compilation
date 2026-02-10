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

struct ConvertFuncOp : public OpConversionPattern<func::FuncOp> {
    using OpConversionPattern<func::FuncOp>::OpConversionPattern;

    LogicalResult
    matchAndRewrite(func::FuncOp funcOp, OpAdaptor adaptor, ConversionPatternRewriter &rewriter) const override {
        TypeConverter::SignatureConversion signatureConversion(funcOp.getNumArguments());
        for (const auto &[index, type] : llvm::enumerate(funcOp.getArgumentTypes())) {
            Type converted = getTypeConverter()->convertType(type);
            if (!converted)
                return failure();
            signatureConversion.addInputs(index, converted);
        }
        SmallVector<Type> convertedResultTypes;
        if (getTypeConverter()->convertTypes(funcOp.getFunctionType().getResults(), convertedResultTypes).failed())
            return failure();
        auto convertedType = FunctionType::get(rewriter.getContext(), 
                                               signatureConversion.getConvertedTypes(),
                                               convertedResultTypes);

        auto newFuncOp = rewriter.create<func::FuncOp>(
            funcOp.getLoc(), funcOp.getName(), convertedType,
            funcOp.getSymVisibilityAttr(), funcOp.getArgAttrsAttr(),
            funcOp.getResAttrsAttr());
        newFuncOp->setDiscardableAttrs(funcOp->getDiscardableAttrs());

        rewriter.inlineRegionBefore(funcOp.getBody(), newFuncOp.getBody(), newFuncOp.end());
        if (failed(rewriter.convertRegionTypes(&newFuncOp.getBody(), *getTypeConverter(), &signatureConversion))) {
            return rewriter.notifyMatchFailure(funcOp->getLoc(), "failed to apply signature conversion");
        }

        rewriter.eraseOp(funcOp);
        return success();
    }
};

// Return op conversion
struct ConvertReturnOp : public OpConversionPattern<func::ReturnOp> {
    using OpConversionPattern<func::ReturnOp>::OpConversionPattern;

    LogicalResult
    matchAndRewrite(func::ReturnOp op, OpAdaptor adaptor, ConversionPatternRewriter &rewriter) const override {
        rewriter.replaceOpWithNewOp<func::ReturnOp>(op, adaptor.getOperands());
        return success();
    }
};

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
        auto deviceIdAttr = rewriter.getI32IntegerAttr(deviceId.getValue().cast<IntegerAttr>().getInt());
        auto deviceIdConst = rewriter.create<LLVM::ConstantOp>(
            op.getLoc(), i32Type, deviceIdAttr);
        
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

        // Get the converted stream operand from adaptor
        Value stream = adaptor.getStream();
        if (!stream) {
            // Fallback to operands array
            if (adaptor.getOperands().empty() || !adaptor.getOperands()[0]) {
                return rewriter.notifyMatchFailure(op, "stream operand is null");
            }
            stream = adaptor.getOperands()[0];
        }

        // cudaStreamDestroy(stream)
        rewriter.create<LLVM::CallOp>(op.getLoc(), cudaStreamDestroy, ValueRange{stream});
        rewriter.replaceOp(op, ValueRange{});

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

        // Get the converted stream value from adaptor
        Value stream = adaptor.getStream();
        if (!stream) {
            // Fallback to operands array
            if (adaptor.getOperands().empty() || !adaptor.getOperands()[0]) {
                return rewriter.notifyMatchFailure(op, "stream operand is null");
            }
            stream = adaptor.getOperands()[0];
        }

        // cudaStreamSynchronize(stream)
        rewriter.create<LLVM::CallOp>(op.getLoc(), cudaStreamSync, ValueRange{stream});
        rewriter.replaceOp(op, ValueRange{});

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
        rewriter.create<LLVM::CallOp>(op.getLoc(), cudaDeviceSync, ValueRange{});
        rewriter.replaceOp(op, ValueRange{});

        return success();
    }
};

// mgpu.free to cudaFree
struct ConvertFreeOp : public OpConversionPattern<FreeOp> {
    using OpConversionPattern<FreeOp>::OpConversionPattern;


    LogicalResult
    matchAndRewrite(FreeOp op, OpAdaptor adaptor, ConversionPatternRewriter &rewriter) const override {
        auto loc = op.getLoc();
        Value memref = adaptor.getPtr();
        Type memrefType = memref.getType();

        Value extractedPtr;

        if (memrefType.isa<LLVM::LLVMPointerType>()) {
            extractedPtr = memref;
        } else if (memrefType.isa<LLVM::LLVMStructType>()) {
            extractedPtr = rewriter.create<LLVM::ExtractValueOp>(loc, memref, ArrayRef<int64_t>{0});
        } else {
            return rewriter.notifyMatchFailure(op, "Expected LLVM pointer or struct (memref descriptor).");
        }

        auto module = op->getParentOfType<ModuleOp>();
        auto cudaFree = getCudaFreeFunc(module);

        rewriter.replaceOpWithNewOp<LLVM::CallOp>(op, cudaFree, ValueRange{extractedPtr});
        return success();
    }

};

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
        
        target.addLegalOp<UnrealizedConversionCastOp>();

        target.addIllegalDialect<multigpu::MultiGpuDialect>();

        RewritePatternSet patterns(context);
        patterns.add<ConvertFuncOp>(typeConverter, context);
        patterns.add<ConvertReturnOp>(typeConverter, context);
        patterns.add<ConvertGetDeviceOp>(typeConverter, context);
        patterns.add<ConvertCreateStreamOp>(typeConverter, context);
        patterns.add<ConvertDestroyStreamOp>(typeConverter, context);
        patterns.add<ConvertSyncStreamOp>(typeConverter, context);
        patterns.add<ConvertSyncDeviceOp>(typeConverter, context);
        patterns.add<ConvertFreeOp>(typeConverter, context);

        // Mark functions as legal if their signatures are converted
        target.addDynamicallyLegalOp<func::FuncOp>([&](func::FuncOp op) {
            return typeConverter.isLegal(op.getFunctionType());
        });
        target.addDynamicallyLegalOp<func::ReturnOp>([&](func::ReturnOp op) {
            return llvm::all_of(op.getOperandTypes(),
                               [&](Type type) { return typeConverter.isLegal(type); });
        });

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

