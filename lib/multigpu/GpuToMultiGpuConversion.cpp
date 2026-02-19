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

#include "polygeist/Dialect.h"
#include "multigpu/MultiGpuDialect.h"
#include "multigpu/MultiGpuOps.h"

#define DEBUG_TYPE "gpu-to-mgpu"

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

        rewriter.setInsertionPoint(op);
        Value c0 = rewriter.create<arith::ConstantIndexOp>(loc, 0);
        multigpu::GetDeviceOp getDeviceOp = rewriter.create<multigpu::GetDeviceOp>(
            loc, multigpu::DeviceType::get(context), c0);
        Value device = getDeviceOp.getResult();

        ValueRange gridDims = ValueRange{op.getGridSizeX(), op.getGridSizeY(), op.getGridSizeZ()};
        ValueRange blockDims = ValueRange{op.getBlockSizeX(), op.getBlockSizeY(), op.getBlockSizeZ()};

        OperationState state(loc, multigpu::LaunchOp::getOperationName());
        multigpu::LaunchOp::build(rewriter, state, device, gridDims, blockDims,
                                  /*stream=*/Value());

        multigpu::LaunchOp mgpuLaunch =
            cast<multigpu::LaunchOp>(rewriter.create(state));

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

// element type from memref<1xmemref<?xElemTy>> or memref<?xmemref<?xElemTy>>
static Type getCudaMallocWrapperElemType(Type type) {
    auto mt = type.dyn_cast<MemRefType>();
    if (!mt || mt.getRank() != 1)
        return nullptr;
    Type elem = mt.getElementType();
    auto inner = elem.dyn_cast<MemRefType>();
    if (!inner || inner.getRank() != 1 || !inner.isDynamicDim(0))
        return nullptr;
    return inner.getElementType();
}

// _ZL10cudaMallocIfE9cudaErrorPPT_m) and (memref<?xmemref<?xT>>, i64) -> i32.
static bool isCudaMallocWrapperCall(func::CallOp call) {
    StringRef callee = call.getCallee();
    if (callee.empty() || !callee.contains("cudaMalloc"))
        return false;
    if (call.getNumOperands() != 2 || call.getNumResults() != 1)
        return false;
    if (!call.getResult(0).getType().isSignlessInteger(32))
        return false;
    if (!call.getOperand(1).getType().isSignlessInteger(64))
        return false;
    Type firstArgTy = call.getOperand(0).getType();
    if (!getCudaMallocWrapperElemType(firstArgTy))
        return false;
    return true;
}

static LogicalResult tryConvertOneCudaAlloc(func::CallOp call, OpBuilder &b,
    SmallVector<affine::AffineLoadOp> &loadsToErase) {
    if (!isCudaMallocWrapperCall(call))
        return failure();
    Value castOp = call.getOperand(0);
    auto cast = dyn_cast<memref::CastOp>(castOp.getDefiningOp());
    if (!cast)
        return failure();
    Value allocaVal = cast.getOperand();
    auto alloca = dyn_cast<memref::AllocaOp>(allocaVal.getDefiningOp());
    if (!alloca)
        return failure();
    MemRefType allocaType = alloca.getType().dyn_cast<MemRefType>();
    if (!allocaType || allocaType.getRank() != 1 || allocaType.getNumElements() != 1)
        return failure();
    Type innerMemref = allocaType.getElementType();
    auto innerType = innerMemref.dyn_cast<MemRefType>();
    if (!innerType || innerType.getRank() != 1 || !innerType.isDynamicDim(0))
        return failure();
    Type elemTy = innerType.getElementType();
    unsigned elemSize = 0;
    if (elemTy.isIntOrFloat())
        elemSize = elemTy.getIntOrFloatBitWidth() / 8;
    else
        return failure();
    Value sizeOp = call.getOperand(1);
    std::optional<int64_t> sizeBytesOpt;
    if (auto sizeConst = sizeOp.getDefiningOp<arith::ConstantOp>()) {
        if (auto sizeAttr = sizeConst.getValue().dyn_cast<IntegerAttr>())
            sizeBytesOpt = sizeAttr.getInt();
    }
    if (!sizeBytesOpt)
        return failure();
    int64_t sizeBytes = *sizeBytesOpt;
    if (sizeBytes <= 0 || sizeBytes % elemSize != 0)
        return failure();
    int64_t numElements = sizeBytes / elemSize;
    func::FuncOp func = call->getParentOfType<func::FuncOp>();
    if (!func)
        return failure();
    Value device;
    for (Operation &op : func.getBody().front()) {
        if (auto getDev = dyn_cast<multigpu::GetDeviceOp>(&op)) {
            Value idx = getDev.getDeviceIndex();
            if (auto cst = idx.getDefiningOp<arith::ConstantIndexOp>()) {
                if (cst.value() == 0) {
                    device = getDev.getResult();
                    break;
                }
            }
        }
    }
    if (!device) {
        Block *entry = &func.getBody().front();
        OpBuilder entryBuilder(entry, entry->begin());
        device = entryBuilder.create<multigpu::GetDeviceOp>(
            call.getLoc(), multigpu::DeviceType::get(call.getContext()),
            entryBuilder.create<arith::ConstantIndexOp>(call.getLoc(), 0));
    }
    Location loc = call.getLoc();
    auto resultType = MemRefType::get({numElements}, elemTy);
    Value alloc = b.create<multigpu::AllocOp>(loc, resultType, device);
    auto dynamicType = MemRefType::get({ShapedType::kDynamic}, elemTy);
    Value allocCast = b.create<memref::CastOp>(loc, dynamicType, alloc);
    loadsToErase.clear();
    for (OpOperand &use : alloca.getResult().getUses()) {
        auto load = dyn_cast<affine::AffineLoadOp>(use.getOwner());
        if (!load || load.getMemRef() != alloca.getResult())
            continue;
        for (OpOperand &resUse : llvm::make_early_inc_range(load.getResult().getUses()))
            resUse.set(allocCast);
        loadsToErase.push_back(load);
    }
    if (loadsToErase.empty())
        return failure();
    for (affine::AffineLoadOp load : loadsToErase)
        load.erase();
    call.erase();
    if (cast.getResult().use_empty())
        cast.erase();
    if (alloca.getResult().use_empty())
        alloca.erase();
    return success();
}

// cudaMalloc wrapper + alloca + loads to mgpu.get_device + mgpu.alloc.
struct ConvertCudaAllocToMgpu : public OpRewritePattern<func::CallOp> {
    using OpRewritePattern<func::CallOp>::OpRewritePattern;
    LogicalResult matchAndRewrite(func::CallOp call, PatternRewriter &rewriter) const override {
        if (!isCudaMallocWrapperCall(call))
            return failure();
        Value castOp = call.getOperand(0);
        auto cast = dyn_cast<memref::CastOp>(castOp.getDefiningOp());
        if (!cast)
            return failure();
        Value allocaVal = cast.getOperand();
        auto alloca = dyn_cast<memref::AllocaOp>(allocaVal.getDefiningOp());
        if (!alloca)
            return failure();
        MemRefType allocaType = alloca.getType().dyn_cast<MemRefType>();
        if (!allocaType || allocaType.getRank() != 1 || allocaType.getNumElements() != 1)
            return failure();
        Type innerMemref = allocaType.getElementType();
        auto innerType = innerMemref.dyn_cast<MemRefType>();
        if (!innerType || innerType.getRank() != 1 || !innerType.isDynamicDim(0))
            return failure();
        Type elemTy = innerType.getElementType();
        unsigned elemSize = 0;
        if (elemTy.isIntOrFloat())
            elemSize = elemTy.getIntOrFloatBitWidth() / 8;
        else
            return failure();
        Value sizeOp = call.getOperand(1);
        std::optional<int64_t> sizeBytesOpt;
        if (auto sizeConst = sizeOp.getDefiningOp<arith::ConstantOp>()) {
            if (auto sizeAttr = sizeConst.getValue().dyn_cast<IntegerAttr>())
                sizeBytesOpt = sizeAttr.getInt();
        }
        if (!sizeBytesOpt)
            return failure();
        int64_t sizeBytes = *sizeBytesOpt;
        if (sizeBytes <= 0 || sizeBytes % elemSize != 0)
            return failure();
        int64_t numElements = sizeBytes / elemSize;
        func::FuncOp func = call->getParentOfType<func::FuncOp>();
        if (!func)
            return failure();
        Value device;
        for (Operation &op : func.getBody().front()) {
            if (auto getDev = dyn_cast<multigpu::GetDeviceOp>(&op)) {
                Value idx = getDev.getDeviceIndex();
                if (auto cst = idx.getDefiningOp<arith::ConstantIndexOp>()) {
                    if (cst.value() == 0) {
                        device = getDev.getResult();
                        break;
                    }
                }
            }
        }
        if (!device) {
            Block *entry = &func.getBody().front();
            OpBuilder b(entry, entry->begin());
            device = b.create<multigpu::GetDeviceOp>(
                call.getLoc(), multigpu::DeviceType::get(getContext()),
                b.create<arith::ConstantIndexOp>(call.getLoc(), 0));
        }
        Location loc = call.getLoc();
        auto resultType = MemRefType::get({numElements}, elemTy);
        Value alloc = rewriter.create<multigpu::AllocOp>(loc, resultType, device);
        auto dynamicType = MemRefType::get({ShapedType::kDynamic}, elemTy);
        Value allocCast = rewriter.create<memref::CastOp>(loc, dynamicType, alloc);
        SmallVector<affine::AffineLoadOp> loadsToErase;
        for (OpOperand &use : alloca.getResult().getUses()) {
            auto load = dyn_cast<affine::AffineLoadOp>(use.getOwner());
            if (!load || load.getMemRef() != alloca.getResult())
                continue;
            rewriter.replaceAllUsesWith(load.getResult(), allocCast);
            loadsToErase.push_back(load);
        }
        if (loadsToErase.empty())
            return failure();
        for (affine::AffineLoadOp load : loadsToErase)
            rewriter.eraseOp(load);
        rewriter.eraseOp(call);
        if (cast.getResult().use_empty())
            rewriter.eraseOp(cast);
        if (alloca.getResult().use_empty())
            rewriter.eraseOp(alloca);
        return success();
    }
};

static Value resolveFreeOperandToAlloc(Value arg, Operation **ptr2memrefOp,
                                       Operation **memref2ptrOp) {
    *ptr2memrefOp = nullptr;
    *memref2ptrOp = nullptr;
    Value v = arg;
    while (auto cast = v.getDefiningOp<memref::CastOp>())
        v = cast.getOperand();
    if (Operation *op = v.getDefiningOp()) {
        if (op->getName().getStringRef() == "polygeist.pointer2memref") {
            *ptr2memrefOp = op;
            v = op->getOperand(0);
            if (Operation *op2 = v.getDefiningOp()) {
                if (op2->getName().getStringRef() == "polygeist.memref2pointer") {
                    *memref2ptrOp = op2;
                    v = op2->getOperand(0);
                }
            }
        }
    }
    while (auto cast = v.getDefiningOp<memref::CastOp>())
        v = cast.getOperand();
    return v;
}

// cudaFree to mgpu.free.
struct ConvertCudaFreeToMgpu : public OpRewritePattern<func::CallOp> {
    using OpRewritePattern<func::CallOp>::OpRewritePattern;

    LogicalResult matchAndRewrite(func::CallOp call, PatternRewriter &rewriter) const override {
        if (call.getCallee() != "cudaFree")
            return failure();
        if (call.getNumOperands() != 1)
            return failure();

        Operation *ptr2memrefOp = nullptr;
        Operation *memref2ptrOp = nullptr;
        Value buffer = resolveFreeOperandToAlloc(call.getOperand(0), &ptr2memrefOp, &memref2ptrOp);
        if (!buffer.getDefiningOp() || !isa<multigpu::AllocOp>(buffer.getDefiningOp()))
            return failure();

        Value device = getOrCreateDevice(rewriter, call);
        Value freeArg = call.getOperand(0);
        memref::CastOp maybeCast = freeArg.getDefiningOp<memref::CastOp>();
        rewriter.setInsertionPoint(call);
        rewriter.create<multigpu::FreeOp>(call.getLoc(), buffer, device);
        rewriter.eraseOp(call);
        if (ptr2memrefOp && ptr2memrefOp->use_empty())
            rewriter.eraseOp(ptr2memrefOp);
        if (memref2ptrOp && memref2ptrOp->use_empty())
            rewriter.eraseOp(memref2ptrOp);
        if (maybeCast && maybeCast.getResult().use_empty())
            rewriter.eraseOp(maybeCast);
        return success();
    }

    Value getOrCreateDevice(PatternRewriter &rewriter, func::CallOp call) const {
        func::FuncOp func = call->getParentOfType<func::FuncOp>();
        if (!func)
            return nullptr;
        for (Operation &op : func.getBody().front()) {
            if (auto getDev = dyn_cast<multigpu::GetDeviceOp>(&op)) {
                Value idx = getDev.getDeviceIndex();
                if (auto cst = idx.getDefiningOp<arith::ConstantIndexOp>()) {
                    if (cst.value() == 0)
                        return getDev.getResult();
                }
            }
        }
        Block *entry = &func.getBody().front();
        OpBuilder b(entry, entry->begin());
        return b.create<multigpu::GetDeviceOp>(
            call.getLoc(), multigpu::DeviceType::get(getContext()),
            b.create<arith::ConstantIndexOp>(call.getLoc(), 0));
    }
};

struct GpuToMultiGpuConversionPass
    : public PassWrapper<GpuToMultiGpuConversionPass, OperationPass<ModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(GpuToMultiGpuConversionPass)

    void getDependentDialects(DialectRegistry &registry) const override {
        registry.insert<multigpu::MultiGpuDialect>();
        registry.insert<gpu::GPUDialect>();
        registry.insert<arith::ArithDialect>();
        registry.insert<mlir::polygeist::PolygeistDialect>();
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

        insertDeviceConfigOp(module, context);

        // Phase 1a: Convert cudaMalloc to mgpu.alloc
        SmallVector<affine::AffineLoadOp> loadsToErase;
        bool allocChanged = true;
        while (allocChanged) {
            allocChanged = false;
            module.walk([&](func::CallOp call) {
                if (!isCudaMallocWrapperCall(call))
                    return WalkResult::advance();
                OpBuilder b(call);
                if (succeeded(tryConvertOneCudaAlloc(call, b, loadsToErase))) {
                    allocChanged = true;
                    return WalkResult::interrupt();
                }
                return WalkResult::advance();
            });
        }

        // Phase 1b: Convert cudaFree to mgpu.free
        SmallVector<func::CallOp> cudaFreeCalls;
        module.walk([&](func::CallOp call) {
            if (call.getCallee() == "cudaFree" && call.getNumOperands() == 1)
                cudaFreeCalls.push_back(call);
        });
        for (func::CallOp call : cudaFreeCalls) {
            if (!call || call->getBlock() == nullptr)
                continue;
            Operation *ptr2memrefOp = nullptr;
            Operation *memref2ptrOp = nullptr;
            Value buffer = resolveFreeOperandToAlloc(call.getOperand(0), &ptr2memrefOp, &memref2ptrOp);
            if (!buffer.getDefiningOp() || !isa<multigpu::AllocOp>(buffer.getDefiningOp()))
                continue;
            func::FuncOp func = call->getParentOfType<func::FuncOp>();
            if (!func)
                continue;
            Value device;
            for (Operation &op : func.getBody().front()) {
                if (auto getDev = dyn_cast<multigpu::GetDeviceOp>(&op)) {
                    Value idx = getDev.getDeviceIndex();
                    if (auto cst = idx.getDefiningOp<arith::ConstantIndexOp>()) {
                        if (cst.value() == 0) {
                            device = getDev.getResult();
                            break;
                        }
                    }
                }
            }
            if (!device) {
                Block *entry = &func.getBody().front();
                OpBuilder b(entry, entry->begin());
                device = b.create<multigpu::GetDeviceOp>(
                    call.getLoc(), multigpu::DeviceType::get(context),
                    b.create<arith::ConstantIndexOp>(call.getLoc(), 0));
            }
            Value freeArg = call.getOperand(0);
            memref::CastOp maybeCast = freeArg.getDefiningOp<memref::CastOp>();
            OpBuilder b(call);
            b.create<multigpu::FreeOp>(call.getLoc(), buffer, device);
            call.erase();
            if (ptr2memrefOp && ptr2memrefOp->use_empty())
                ptr2memrefOp->erase();
            if (memref2ptrOp && memref2ptrOp->use_empty())
                memref2ptrOp->erase();
            if (maybeCast && maybeCast.getResult().use_empty())
                maybeCast.erase();
        }

        // Phase 2: Convert gpu.launch to mgpu.launch
        SmallVector<Operation *> launchOps;
        module.walk([&](gpu::LaunchOp launch) { launchOps.push_back(launch); });
        if (!launchOps.empty()) {
            RewritePatternSet launchPatterns(context);
            launchPatterns.add<ConvertLaunchOp>(context);
            GreedyRewriteConfig config;
            config.strictMode = GreedyRewriteStrictness::ExistingOps;
            FrozenRewritePatternSet frozenLaunch(std::move(launchPatterns));
            if (failed(applyOpPatternsAndFold(launchOps, frozenLaunch, config)))
                signalPassFailure();
        }
    }

    StringRef getArgument() const final {
        return "gpu-to-mgpu";
    }

    StringRef getDescription() const final {
        return "Convert GPU dialect and CUDA runtime calls to mgpu dialect";
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
