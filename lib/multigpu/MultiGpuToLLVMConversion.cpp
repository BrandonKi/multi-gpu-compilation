//===- MultiGpuToLLVMConversion.cpp - MultiGpu dialect to LLVM -----------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Lower mgpu dialect to LLVM dialect and CUDA runtime calls.
// (Uses mgpurt* wrappers to make lowering easier)
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

#include "multigpu/MultiGpuDialect.h"
#include "multigpu/MultiGpuOps.h"

#define DEBUG_TYPE "mgpu-to-llvm"

using namespace mlir;
using namespace mlir::multigpu;

//
// removes consecutive set_device call if same operand as previous
//
// ...
// %4 = llvm.call @mgpurtSetDevice(%1) : (i32) -> i32
// ...
// %5 = llvm.call @mgpurtSetDevice(%1) : (i32) -> i32
// ...
//
struct RemoveRedundantSetDeviceOp : public OpRewritePattern<LLVM::CallOp> {
    using OpRewritePattern::OpRewritePattern;

    LogicalResult matchAndRewrite(LLVM::CallOp op, PatternRewriter &rewriter) const override {
        auto callee = op.getCallee();
        if (!callee || *callee != "mgpurtSetDevice")
            return failure();

        if (op.getNumOperands() != 1)
            return failure();

        Value deviceArg = op.getOperand(0);
        for (Operation *prevOp = op->getPrevNode(); prevOp; prevOp = prevOp->getPrevNode()) {
            auto prevCall = dyn_cast<LLVM::CallOp>(prevOp);
            if (!prevCall)
                continue;
            auto prevCallee = prevCall.getCallee();
            if (!prevCallee || *prevCallee != "mgpurtSetDevice" || prevCall.getNumOperands() != 1)
                continue;
            if (prevCall.getOperand(0) != deviceArg)
                return failure();
            rewriter.eraseOp(op);
            return success();
        }
        return failure();
    }
};

// namespace mlir::multigpu {

static LLVM::LLVMFuncOp getOrCreateFunc(ModuleOp module, StringRef name, LLVM::LLVMFunctionType type) {
    if (auto f = module.lookupSymbol<LLVM::LLVMFuncOp>(name))
        return f;
    OpBuilder b(module.getBodyRegion());
    return b.create<LLVM::LLVMFuncOp>(module.getLoc(), name, type);
}

// inserts setDevice(device) directly before each mgpurt* wrapper call
// canonicalization removes redundant setDevice calls later
static void insertSetDeviceBeforeCudaCalls(ModuleOp module) {
    MLIRContext *ctx = module.getContext();
    Type i32Type = IntegerType::get(ctx, 32);
    LLVM::LLVMFuncOp setDeviceFunc =
        getOrCreateFunc(module, "mgpurtSetDevice", LLVM::LLVMFunctionType::get(i32Type, {i32Type}));

    module.walk([&](LLVM::CallOp call) {
        auto callee = call.getCallee();
        if (!callee)
            return;
        StringRef name = *callee;
        unsigned deviceOperandIndex = 0;
        if (name == "mgpurtMemAllocOnDevice" || name == "mgpurtMemFree" ||
            name == "mgpurtStreamCreate" || name == "mgpurtStreamDestroy" ||
            name == "mgpurtStreamSynchronize" || name == "mgpurtDeviceSynchronizeErr") {
            deviceOperandIndex = 0;
        // } else if (name == "mgpurtMemcpyPeer") {
            // deviceOperandIndex = 1; // destination device
        } else {
            return;
        }
        if (deviceOperandIndex >= call.getNumOperands())
            return;
        Value device = call.getOperand(deviceOperandIndex);
        OpBuilder b(call);
        b.create<LLVM::CallOp>(call.getLoc(), setDeviceFunc, device);
    });
}

// Device -> i32, Stream -> !llvm.ptr
class MultiGpuToLLVMTypeConverter : public TypeConverter {
  public:
    MultiGpuToLLVMTypeConverter(MLIRContext *ctx) {
        addConversion([ctx](DeviceType) { return IntegerType::get(ctx, 32); });

        addConversion([ctx](StreamType) { return LLVM::LLVMPointerType::get(ctx); });

        addConversion([](Type t) { return t; });

        addTargetMaterialization([](OpBuilder &b, Type resultType, ValueRange inputs, Location loc) -> Value {
            if (inputs.size() != 1)
                return nullptr;
            return b.create<UnrealizedConversionCastOp>(loc, resultType, inputs[0]).getResult(0);
        });

        addSourceMaterialization([](OpBuilder &b, Type resultType, ValueRange inputs, Location loc) -> Value {
            if (inputs.size() != 1)
                return nullptr;
            return inputs[0];
        });
    }
};

static Value getPointerFromValue(ConversionPatternRewriter &rewriter, Location loc, Value v) {
    if (!v)
        return nullptr;
    Type t = v.getType();
    if (t.isa<LLVM::LLVMPointerType>())
        return v;
    if (auto castOp = v.getDefiningOp<UnrealizedConversionCastOp>()) {
        Value input = castOp.getOperand(0);
        if (input.getType().isa<LLVM::LLVMPointerType>())
            return input;
    }
    if (auto st = t.dyn_cast<LLVM::LLVMStructType>())
        return rewriter.create<LLVM::ExtractValueOp>(loc, v, ArrayRef<int64_t>{0});
    return nullptr;
}

// can be mostly removed now, the canonicalization handles this
static Value getOrCreateI32Constant(ConversionPatternRewriter &rewriter, Location loc, int32_t value) {
    // Block *block = rewriter.getInsertionBlock();
    Type i32Type = rewriter.getI32Type();
    // for (Operation &op : *block) {
    //     if (auto cst = dyn_cast<LLVM::ConstantOp>(&op)) {
    //         if (cst.getType() == i32Type && cst.getValue().isa<IntegerAttr>()) {
    //             if (cst.getValue().cast<IntegerAttr>().getInt() == value)
    //                 return cst.getResult();
    //         }
    //     }
    // }
    // just keep this
    return rewriter.create<LLVM::ConstantOp>(loc, i32Type, rewriter.getI32IntegerAttr(value));
}

static Value getDeviceI32(ConversionPatternRewriter &rewriter, Location loc, Value origDev, Value adaptedDev) {
    if (adaptedDev && adaptedDev.getType().isa<IntegerType>())
        return adaptedDev;
    if (auto getDev = origDev.getDefiningOp<GetDeviceOp>()) {
        Value idx = getDev.getDeviceIndex();
        if (idx) {
            if (auto cst = idx.getDefiningOp<arith::ConstantOp>()) {
                int64_t v = cst.getValue().cast<IntegerAttr>().getInt();
                return getOrCreateI32Constant(rewriter, loc, static_cast<int32_t>(v));
            }
            if (auto llvmCst = idx.getDefiningOp<LLVM::ConstantOp>()) {
                int64_t v = llvmCst.getValue().cast<IntegerAttr>().getInt();
                return getOrCreateI32Constant(rewriter, loc, static_cast<int32_t>(v));
            }
            return rewriter.create<arith::IndexCastOp>(loc, rewriter.getI32Type(), idx);
        }
        return getOrCreateI32Constant(rewriter, loc, 0);
    }
    return nullptr;
}

static Value getMemRefSizeBytes(ConversionPatternRewriter &rewriter, Location loc, MemRefType memrefType) {
    if (!memrefType.hasStaticShape())
        return nullptr;
    int64_t numElements = memrefType.getNumElements();
    Type elemType = memrefType.getElementType();
    unsigned elemBits = 0;
    if (elemType.isIntOrFloat())
        elemBits = elemType.getIntOrFloatBitWidth();
    else
        return nullptr;
    int64_t sizeBytes = numElements * (elemBits / 8);
    auto i64Type = rewriter.getI64Type();
    return rewriter.create<LLVM::ConstantOp>(loc, i64Type, rewriter.getI64IntegerAttr(sizeBytes));
}

struct ConvertGetDeviceOp : public OpConversionPattern<GetDeviceOp> {
    using OpConversionPattern::OpConversionPattern;

    LogicalResult matchAndRewrite(GetDeviceOp op, OpAdaptor adaptor,
                                  ConversionPatternRewriter &rewriter) const override {
        auto i32Type = rewriter.getI32Type();
        Value indexVal = adaptor.getDeviceIndex();
        Value i32Val;
        if (auto cst = indexVal.getDefiningOp<arith::ConstantOp>()) {
            int64_t id = cst.getValue().cast<IntegerAttr>().getInt();
            i32Val = getOrCreateI32Constant(rewriter, op.getLoc(), static_cast<int32_t>(id));
        } else {
            i32Val = rewriter.create<arith::IndexCastOp>(op.getLoc(), i32Type, indexVal);
        }
        rewriter.replaceOp(op, i32Val);
        return success();
    }
};

struct ConvertAllocOp : public OpConversionPattern<AllocOp> {
    using OpConversionPattern::OpConversionPattern;

    LogicalResult matchAndRewrite(AllocOp op, OpAdaptor adaptor, ConversionPatternRewriter &rewriter) const override {
        Location loc = op.getLoc();
        MemRefType memrefType = op.getPtr().getType().cast<MemRefType>();
        Value sizeBytes = getMemRefSizeBytes(rewriter, loc, memrefType);
        if (!sizeBytes)
            return rewriter.notifyMatchFailure(op, "only static shapes supported");

        Value deviceI32 = adaptor.getDevice();
        bool fromAdaptor = deviceI32 && deviceI32.getType().isa<IntegerType>();
        if (!fromAdaptor) {
            Value origDev = op.getDevice();
            if (auto getDev = origDev.getDefiningOp<GetDeviceOp>()) {
                Value idx = getDev.getDeviceIndex();
                if (idx) {
                    if (auto cst = idx.getDefiningOp<arith::ConstantOp>()) {
                        int64_t v = cst.getValue().cast<IntegerAttr>().getInt();
                        deviceI32 = getOrCreateI32Constant(rewriter, loc, static_cast<int32_t>(v));
                    } else if (auto llvmCst = idx.getDefiningOp<LLVM::ConstantOp>()) {
                        int64_t v = llvmCst.getValue().cast<IntegerAttr>().getInt();
                        deviceI32 = getOrCreateI32Constant(rewriter, loc, static_cast<int32_t>(v));
                    } else {
                        deviceI32 = rewriter.create<arith::IndexCastOp>(loc, rewriter.getI32Type(), idx);
                    }
                }
                if (!deviceI32)
                    deviceI32 = getOrCreateI32Constant(rewriter, loc, 0);
            }
        }
        if (!deviceI32)
            return rewriter.notifyMatchFailure(op, "missing device");
        if (!deviceI32.getType().isa<IntegerType>())
            return rewriter.notifyMatchFailure(op, "missing device (convert get_device first)");

        ModuleOp module = op->getParentOfType<ModuleOp>();
        auto ptrType = LLVM::LLVMPointerType::get(rewriter.getContext());
        auto i64Type = rewriter.getI64Type();
        auto i32Type = rewriter.getI32Type();
        auto allocFunc =
            getOrCreateFunc(module, "mgpurtMemAllocOnDevice", LLVM::LLVMFunctionType::get(ptrType, {i32Type, i64Type}));

        Value sizeI64 = sizeBytes.getType().isa<IntegerType>()
                            ? sizeBytes
                            : rewriter.create<arith::IndexCastOp>(loc, i64Type, sizeBytes);
        Value ptr = rewriter.create<LLVM::CallOp>(loc, allocFunc, ValueRange{deviceI32, sizeI64}).getResult();

        Type elemPtrType = LLVM::LLVMPointerType::get(rewriter.getContext());
        ptr = rewriter.create<LLVM::BitcastOp>(loc, elemPtrType, ptr);
        Value cast = rewriter.create<UnrealizedConversionCastOp>(loc, TypeRange{memrefType}, ptr).getResult(0);
        rewriter.replaceOp(op, cast);
        return success();
    }
};

struct ConvertFreeOp : public OpConversionPattern<FreeOp> {
    using OpConversionPattern::OpConversionPattern;

    LogicalResult matchAndRewrite(FreeOp op, OpAdaptor adaptor, ConversionPatternRewriter &rewriter) const override {
        Location loc = op.getLoc();
        Value ptrVal = getPointerFromValue(rewriter, loc, adaptor.getPtr());
        if (!ptrVal)
            return rewriter.notifyMatchFailure(op, "could not get pointer from memref");

        Value deviceI32 = adaptor.getDevice();
        bool fromAdaptor = deviceI32 && deviceI32.getType().isa<IntegerType>();
        if (!fromAdaptor) {
            if (auto getDev = op.getDevice().getDefiningOp<GetDeviceOp>()) {
                Value idx = getDev.getDeviceIndex();
                if (!idx)
                    return rewriter.notifyMatchFailure(op,
                                                       "get_device has no index operand (convert get_device first)");
                if (auto cst = idx.getDefiningOp<arith::ConstantOp>())
                    deviceI32 = getOrCreateI32Constant(
                        rewriter, loc, static_cast<int32_t>(cst.getValue().cast<IntegerAttr>().getInt()));
                else if (auto llvmCst = idx.getDefiningOp<LLVM::ConstantOp>())
                    deviceI32 = getOrCreateI32Constant(
                        rewriter, loc, static_cast<int32_t>(llvmCst.getValue().cast<IntegerAttr>().getInt()));
                else
                    deviceI32 = rewriter.create<arith::IndexCastOp>(loc, rewriter.getI32Type(), idx);
            }
        }
        if (!deviceI32)
            return rewriter.notifyMatchFailure(op, "missing device");
        if (!deviceI32.getType().isa<IntegerType>())
            return rewriter.notifyMatchFailure(op, "missing device (convert get_device first)");

        ModuleOp module = op->getParentOfType<ModuleOp>();
        auto voidType = LLVM::LLVMVoidType::get(rewriter.getContext());
        auto ptrType = LLVM::LLVMPointerType::get(rewriter.getContext());
        auto i32Type = rewriter.getI32Type();
        auto freeFunc = getOrCreateFunc(
            module, "mgpurtMemFree",
            LLVM::LLVMFunctionType::get(voidType, {i32Type, ptrType, ptrType}));
        Value nullStream = rewriter.create<LLVM::ZeroOp>(loc, ptrType);
        rewriter.create<LLVM::CallOp>(loc, freeFunc, ValueRange{deviceI32, ptrVal, nullStream});
        rewriter.eraseOp(op);
        return success();
    }
};

struct ConvertMemcpyOp : public OpConversionPattern<MemcpyOp> {
    using OpConversionPattern::OpConversionPattern;

    LogicalResult matchAndRewrite(MemcpyOp op, OpAdaptor adaptor, ConversionPatternRewriter &rewriter) const override {
        Location loc = op.getLoc();
        Value dstPtr = getPointerFromValue(rewriter, loc, adaptor.getDst());
        Value srcPtr = getPointerFromValue(rewriter, loc, adaptor.getSrc());
        if (!dstPtr || !srcPtr)
            return rewriter.notifyMatchFailure(op, "could not get pointers from memrefs");

        MemRefType dstType = op.getDst().getType().cast<MemRefType>();
        Value sizeBytes = getMemRefSizeBytes(rewriter, loc, dstType);
        if (!sizeBytes)
            return rewriter.notifyMatchFailure(op, "only static shapes supported");

        Value dstDeviceI32 = getDeviceI32(rewriter, loc, op.getDstDevice(), adaptor.getDstDevice());
        Value srcDeviceI32 = getDeviceI32(rewriter, loc, op.getSrcDevice(), adaptor.getSrcDevice());
        if (!dstDeviceI32 || !srcDeviceI32)
            return rewriter.notifyMatchFailure(op, "missing device operands");

        ModuleOp module = op->getParentOfType<ModuleOp>();
        auto ptrType = LLVM::LLVMPointerType::get(rewriter.getContext());
        auto i32Type = rewriter.getI32Type();
        auto i64Type = rewriter.getI64Type();
        auto memcpyPeerFunc =
            getOrCreateFunc(module, "mgpurtMemcpyPeer",
                            LLVM::LLVMFunctionType::get(i32Type, {ptrType, i32Type, ptrType, i32Type, i64Type}));

        Value sizeI64 = sizeBytes.getType().isa<IntegerType>()
                            ? sizeBytes
                            : rewriter.create<arith::IndexCastOp>(loc, i64Type, sizeBytes);
        rewriter.create<LLVM::CallOp>(loc, memcpyPeerFunc,
                                      ValueRange{dstPtr, dstDeviceI32, srcPtr, srcDeviceI32, sizeI64});
        rewriter.eraseOp(op);
        return success();
    }
};

struct ConvertCreateStreamOp : public OpConversionPattern<CreateStreamOp> {
    using OpConversionPattern::OpConversionPattern;

    LogicalResult matchAndRewrite(CreateStreamOp op, OpAdaptor adaptor,
                                  ConversionPatternRewriter &rewriter) const override {
        Location loc = op.getLoc();
        Value deviceI32 = adaptor.getDevice();
        if (!deviceI32) {
            if (auto getDev = op.getDevice().getDefiningOp<GetDeviceOp>()) {
                Value idx = getDev.getDeviceIndex();
                if (!idx)
                    return rewriter.notifyMatchFailure(op,
                                                       "get_device has no index operand (convert get_device first)");
                if (auto cst = idx.getDefiningOp<arith::ConstantOp>())
                    deviceI32 = getOrCreateI32Constant(
                        rewriter, loc, static_cast<int32_t>(cst.getValue().cast<IntegerAttr>().getInt()));
                else if (auto llvmCst = idx.getDefiningOp<LLVM::ConstantOp>())
                    deviceI32 = getOrCreateI32Constant(
                        rewriter, loc, static_cast<int32_t>(llvmCst.getValue().cast<IntegerAttr>().getInt()));
                else
                    deviceI32 = rewriter.create<arith::IndexCastOp>(loc, rewriter.getI32Type(), idx);
            }
        }
        if (!deviceI32)
            return rewriter.notifyMatchFailure(op, "missing device");
        if (!deviceI32.getType().isa<IntegerType>())
            deviceI32 = rewriter.create<arith::IndexCastOp>(loc, rewriter.getI32Type(), deviceI32);

        ModuleOp module = op->getParentOfType<ModuleOp>();
        auto ptrType = LLVM::LLVMPointerType::get(rewriter.getContext());
        auto i32Type = rewriter.getI32Type();
        auto streamCreateFunc = getOrCreateFunc(
            module, "mgpurtStreamCreate",
            LLVM::LLVMFunctionType::get(i32Type, {i32Type, ptrType}));
        Value one = rewriter.create<LLVM::ConstantOp>(loc, rewriter.getI32Type(), rewriter.getI32IntegerAttr(1));
        Value alloca = rewriter.create<LLVM::AllocaOp>(loc, ptrType, ptrType, one, 0);
        rewriter.create<LLVM::CallOp>(loc, streamCreateFunc, ValueRange{deviceI32, alloca});
        Value stream = rewriter.create<LLVM::LoadOp>(loc, ptrType, alloca);
        rewriter.replaceOp(op, stream);
        return success();
    }
};

struct ConvertDestroyStreamOp : public OpConversionPattern<DestroyStreamOp> {
    using OpConversionPattern::OpConversionPattern;

    LogicalResult matchAndRewrite(DestroyStreamOp op, OpAdaptor adaptor,
                                  ConversionPatternRewriter &rewriter) const override {
        Location loc = op.getLoc();
        Value deviceI32 = getDeviceI32(rewriter, loc, op.getDevice(), adaptor.getDevice());
        Value stream = adaptor.getStream();
        if (!deviceI32)
            return rewriter.notifyMatchFailure(op, "missing device");
        if (!stream)
            return rewriter.notifyMatchFailure(op, "missing stream");

        ModuleOp module = op->getParentOfType<ModuleOp>();
        auto i32Type = rewriter.getI32Type();
        auto ptrType = LLVM::LLVMPointerType::get(rewriter.getContext());
        auto destroyFunc = getOrCreateFunc(
            module, "mgpurtStreamDestroy",
            LLVM::LLVMFunctionType::get(i32Type, {i32Type, ptrType}));
        rewriter.create<LLVM::CallOp>(loc, destroyFunc, ValueRange{deviceI32, stream});
        rewriter.eraseOp(op);
        return success();
    }
};

struct ConvertSyncStreamOp : public OpConversionPattern<SyncStreamOp> {
    using OpConversionPattern::OpConversionPattern;

    LogicalResult matchAndRewrite(SyncStreamOp op, OpAdaptor adaptor,
                                  ConversionPatternRewriter &rewriter) const override {
        Location loc = op.getLoc();
        Value deviceI32 = getDeviceI32(rewriter, loc, op.getDevice(), adaptor.getDevice());
        Value stream = adaptor.getStream();
        if (!deviceI32)
            return rewriter.notifyMatchFailure(op, "missing device");
        if (!stream)
            return rewriter.notifyMatchFailure(op, "missing stream");

        ModuleOp module = op->getParentOfType<ModuleOp>();
        auto i32Type = rewriter.getI32Type();
        auto ptrType = LLVM::LLVMPointerType::get(rewriter.getContext());
        auto syncFunc = getOrCreateFunc(
            module, "mgpurtStreamSynchronize",
            LLVM::LLVMFunctionType::get(i32Type, {i32Type, ptrType}));
        rewriter.create<LLVM::CallOp>(loc, syncFunc, ValueRange{deviceI32, stream});
        rewriter.eraseOp(op);
        return success();
    }
};

struct ConvertSyncDeviceOp : public OpConversionPattern<SyncDeviceOp> {
    using OpConversionPattern::OpConversionPattern;

    LogicalResult matchAndRewrite(SyncDeviceOp op, OpAdaptor adaptor,
                                  ConversionPatternRewriter &rewriter) const override {
        Location loc = op.getLoc();
        Value deviceI32 = getDeviceI32(rewriter, loc, op.getDevice(), adaptor.getDevice());
        if (!deviceI32)
            return rewriter.notifyMatchFailure(op, "missing device");

        ModuleOp module = op->getParentOfType<ModuleOp>();
        auto i32Type = rewriter.getI32Type();
        auto syncFunc = getOrCreateFunc(module, "mgpurtDeviceSynchronizeErr",
                                        LLVM::LLVMFunctionType::get(i32Type, {i32Type}));
        rewriter.create<LLVM::CallOp>(loc, syncFunc, deviceI32);
        rewriter.eraseOp(op);
        return success();
    }
};

struct MultiGpuToLLVMConversionPass : public PassWrapper<MultiGpuToLLVMConversionPass, OperationPass<ModuleOp>> {
    MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(MultiGpuToLLVMConversionPass)

    void getDependentDialects(DialectRegistry &registry) const override {
        registry.insert<LLVM::LLVMDialect>();
        registry.insert<arith::ArithDialect>();
        registry.insert<memref::MemRefDialect>();
        registry.insert<func::FuncDialect>();
    }

    void runOnOperation() override {
        ModuleOp module = getOperation();
        MLIRContext *ctx = &getContext();
        MultiGpuToLLVMTypeConverter typeConverter(ctx);

        ConversionTarget target(*ctx);
        target.addLegalDialect<LLVM::LLVMDialect>();
        target.addLegalDialect<arith::ArithDialect>();
        target.addLegalDialect<memref::MemRefDialect>();
        target.addLegalDialect<func::FuncDialect>();
        target.addLegalOp<UnrealizedConversionCastOp>();
        target.addLegalDialect<MultiGpuDialect>();
        target.addIllegalOp<GetDeviceOp, AllocOp, FreeOp, MemcpyOp, SyncDeviceOp, CreateStreamOp,
                            DestroyStreamOp, SyncStreamOp>();

        RewritePatternSet patterns(ctx);
        patterns.add<ConvertGetDeviceOp>(typeConverter, ctx, PatternBenefit(10));
        patterns.add<ConvertAllocOp>(typeConverter, ctx);
        patterns.add<ConvertFreeOp>(typeConverter, ctx);
        patterns.add<ConvertMemcpyOp>(typeConverter, ctx);
        patterns.add<ConvertSyncDeviceOp>(typeConverter, ctx);
        patterns.add<ConvertCreateStreamOp>(typeConverter, ctx);
        patterns.add<ConvertDestroyStreamOp>(typeConverter, ctx);
        patterns.add<ConvertSyncStreamOp>(typeConverter, ctx);

        if (failed(applyPartialConversion(module, target, std::move(patterns))))
            signalPassFailure();

        insertSetDeviceBeforeCudaCalls(module);
        // RewritePatternSet canonicalizationPatterns(ctx);
        // canonicalizationPatterns.add<RemoveRedundantSetDeviceOp>(ctx);
        // GreedyRewriteConfig config;
        // applyPatternsAndFoldGreedily(module, std::move(canonicalizationPatterns), config);
    }

    StringRef getArgument() const final { return "mgpu-to-llvm"; }
    StringRef getDescription() const final { return "Convert MultiGpu dialect to LLVM and CUDA runtime calls"; }
};

// } // namespace

namespace mlir {
namespace multigpu {

std::unique_ptr<Pass> createMultiGpuToLLVMConversionPass() { return std::make_unique<MultiGpuToLLVMConversionPass>(); }

void registerMultiGpuToLLVMConversionPass() { PassRegistration<MultiGpuToLLVMConversionPass>(); }

void populateMultiGpuCanonicalizationPatterns(RewritePatternSet &patterns) {
    patterns.add<RemoveRedundantSetDeviceOp>(patterns.getContext());
}

} // namespace multigpu
} // namespace mlir
