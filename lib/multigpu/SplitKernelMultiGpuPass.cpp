#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/Pass/Pass.h"
#include "llvm/ADT/Twine.h"

#include "multigpu/MultiGpuDialect.h"
#include "multigpu/MultiGpuOps.h"

using namespace mlir;
using namespace multigpu;

namespace {

static std::optional<int64_t> getConstantIndexValue(Value v) {
    if (auto cst = v.getDefiningOp<arith::ConstantIndexOp>())
        return cst.value();
    if (auto cst = v.getDefiningOp<arith::ConstantOp>()) {
        if (auto intAttr = llvm::dyn_cast<IntegerAttr>(cst.getValue()))
            return intAttr.getInt();
    }
    return std::nullopt;
}

static Value lookThroughIndexCast(Value v) {
    while (Operation *def = v.getDefiningOp()) {
        if (auto cast = dyn_cast<arith::IndexCastOp>(def))
            v = cast.getIn();
        else if (auto cast = dyn_cast<arith::IndexCastUIOp>(def))
            v = cast.getIn();
        else
            break;
    }
    return v;
}

static Value findLinearIndexValue(Block &block) {
    for (Operation &op : block) {
        auto addi = dyn_cast<arith::AddIOp>(op);
        if (!addi)
            continue;
        Value lhs = addi.getLhs(), rhs = addi.getRhs();
        Value muliResult;
        Value threadIdVal;
        if (auto muli = lhs.getDefiningOp<arith::MulIOp>()) {
            muliResult = lhs;
            threadIdVal = rhs;
        } else if (auto muli = rhs.getDefiningOp<arith::MulIOp>()) {
            muliResult = rhs;
            threadIdVal = lhs;
        } else {
            continue;
        }
        auto muliOp = muliResult.getDefiningOp<arith::MulIOp>();
        if (!muliOp)
            continue;
        Value mulL = lookThroughIndexCast(muliOp.getLhs());
        Value mulR = lookThroughIndexCast(muliOp.getRhs());
        Value threadRoot = lookThroughIndexCast(threadIdVal);
        bool mulHasBlockId = (isa<gpu::BlockIdOp>(mulL.getDefiningOp()) || isa<gpu::BlockIdOp>(mulR.getDefiningOp()));
        bool mulHasBlockDim =
            (isa<gpu::BlockDimOp>(mulL.getDefiningOp()) || isa<gpu::BlockDimOp>(mulR.getDefiningOp()));
        bool threadIdIsThreadId = isa<gpu::ThreadIdOp>(threadRoot.getDefiningOp());
        if (mulHasBlockId && mulHasBlockDim && threadIdIsThreadId)
            return addi.getResult();
    }
    return Value();
}

static Value findLinearIndexValueInRegion(Region &region) {
    for (Block &block : region)
        if (Value v = findLinearIndexValue(block))
            return v;
    return Value();
}

static void injectOffsetInBlock(Block &block, Value linearIndex, int64_t offsetVal, IRRewriter &rewriter) {
    if (!linearIndex)
        return;
    Location loc = linearIndex.getLoc();
    rewriter.setInsertionPointAfter(linearIndex.getDefiningOp());
    Type idxTy = linearIndex.getType();
    Value offsetConst =
        idxTy.isIndex()
            ? Value(rewriter.create<arith::ConstantIndexOp>(loc, offsetVal))
            : Value(rewriter.create<arith::ConstantOp>(loc, idxTy, rewriter.getIntegerAttr(idxTy, offsetVal)));
    Value globalIdx = rewriter.create<arith::AddIOp>(loc, linearIndex, offsetConst);
    OpOperand &offsetAddLhs = globalIdx.getDefiningOp()->getOpOperand(0);
    SmallVector<OpOperand *> toReplace;
    for (OpOperand &use : linearIndex.getUses()) {
        if (&use == &offsetAddLhs)
            continue;
        toReplace.push_back(&use);
    }
    for (OpOperand *use : toReplace)
        use->set(globalIdx);
}

static void injectOffsetInBlockWithValue(Block &block, Value linearIndex, Value offsetVal, IRRewriter &rewriter) {
    if (!linearIndex || !offsetVal)
        return;
    Location loc = linearIndex.getLoc();
    rewriter.setInsertionPointAfter(linearIndex.getDefiningOp());
    Type idxType = linearIndex.getType();
    Value offsetForAdd = offsetVal;
    if (offsetVal.getType() != idxType) {
        offsetForAdd = rewriter.create<arith::IndexCastOp>(loc, idxType, offsetVal);
        rewriter.setInsertionPointAfter(offsetForAdd.getDefiningOp());
    }
    Value globalIdx = rewriter.create<arith::AddIOp>(loc, linearIndex, offsetForAdd);
    OpOperand &offsetAddLhs = globalIdx.getDefiningOp()->getOpOperand(0);
    SmallVector<OpOperand *> toReplace;
    for (OpOperand &use : linearIndex.getUses()) {
        if (&use == &offsetAddLhs)
            continue;
        toReplace.push_back(&use);
    }
    for (OpOperand *use : toReplace)
        use->set(globalIdx);
}

struct SplitKernelMultiGpuPass : public PassWrapper<SplitKernelMultiGpuPass, OperationPass<ModuleOp>> {
    MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(SplitKernelMultiGpuPass)

    void getDependentDialects(DialectRegistry &registry) const override {
        registry.insert<multigpu::MultiGpuDialect>();
        registry.insert<gpu::GPUDialect>();
        registry.insert<arith::ArithDialect>();
        registry.insert<memref::MemRefDialect>();
        registry.insert<func::FuncDialect>();
    }

    StringRef getArgument() const final { return "split-kernel-mgpu"; }
    StringRef getDescription() const final { return "Split mgpu.launch iteration space across N GPUs"; }

    void runOnOperation() override {
        ModuleOp module = getOperation();
        uint32_t numDevices = 1;
        module.walk([&](DeviceConfigOp configOp) {
            numDevices = configOp.getNumDevices();
            return WalkResult::interrupt();
        });

        if (numDevices <= 1)
            return;

        SmallVector<LaunchOp> launches;
        module.walk([&](LaunchOp launch) { launches.push_back(launch); });

        for (LaunchOp op : launches) {
            if (failed(splitLaunch(op, numDevices)))
                return signalPassFailure();
        }
    }

    LogicalResult splitLaunch(LaunchOp op, uint32_t numDevices) {
        MLIRContext *ctx = op.getContext();
        Location loc = op.getLoc();
        IRRewriter rewriter(ctx);
        rewriter.setInsertionPoint(op);

        ValueRange grid = op.getGrid();
        ValueRange block = op.getBlock();
        Value one = rewriter.create<arith::ConstantIndexOp>(loc, 1);
        Value gridX = grid.size() > 0 ? grid[0] : one;
        Value gridY = grid.size() > 1 ? grid[1] : one;
        Value gridZ = grid.size() > 2 ? grid[2] : one;
        Value blockX = block.size() > 0 ? block[0] : one;
        Value blockY = block.size() > 1 ? block[1] : one;
        Value blockZ = block.size() > 2 ? block[2] : one;

        // totalThreads = grid * block
        Value totalThreads = rewriter.create<arith::MulIOp>(loc, gridX, gridY);
        totalThreads = rewriter.create<arith::MulIOp>(loc, totalThreads, gridZ);
        totalThreads = rewriter.create<arith::MulIOp>(loc, totalThreads, blockX);
        totalThreads = rewriter.create<arith::MulIOp>(loc, totalThreads, blockY);
        totalThreads = rewriter.create<arith::MulIOp>(loc, totalThreads, blockZ);

        std::optional<int64_t> totalConst = getConstantIndexValue(totalThreads);
        std::optional<int64_t> blockXConst = getConstantIndexValue(blockX);
        if (!totalConst || !blockXConst) {
            auto gx = getConstantIndexValue(gridX), gy = getConstantIndexValue(gridY),
                 gz = getConstantIndexValue(gridZ), bx = getConstantIndexValue(blockX),
                 by = getConstantIndexValue(blockY), bz = getConstantIndexValue(blockZ);
            if (gx && gy && gz && bx && by && bz) {
                totalConst = *gx * *gy * *gz * *bx * *by * *bz;
                blockXConst = *bx;
            }
        }

        Block &kernelBlock = op.getKernelRegion().front();
        ModuleOp module = op->getParentOfType<ModuleOp>();

        // one func.call + one terminator.
        bool launchBodyIsSingleCall = false;
        func::CallOp kernelCall;
        func::FuncOp kernelCallee;
        Value linearIndexVal;

        Operation *nonTerminator = nullptr;
        unsigned numNonTerminators = 0;
        for (Operation &o : kernelBlock) {
            if (o.hasTrait<OpTrait::IsTerminator>())
                continue;
            numNonTerminators++;
            if (numNonTerminators > 1) {
                nonTerminator = nullptr;
                break;
            }
            nonTerminator = &o;
        }
        if (nonTerminator) {
            kernelCall = dyn_cast<func::CallOp>(nonTerminator);
            if (kernelCall && kernelCall.getNumResults() == 0) {
                StringRef calleeName = kernelCall.getCalleeAttr().getValue();
                kernelCallee = module.lookupSymbol<func::FuncOp>(calleeName);
                if (kernelCallee && !kernelCallee.getBody().empty()) {
                    linearIndexVal = findLinearIndexValueInRegion(kernelCallee.getBody());
                    if (linearIndexVal)
                        launchBodyIsSingleCall = true;
                } else if (!kernelCallee) {
                    op.emitRemark("split-kernel-mgpu: launch body is single call to '" + calleeName +
                                  "' but callee is not a func.func");
                }
            }
        }
        if (!launchBodyIsSingleCall)
            linearIndexVal = findLinearIndexValue(kernelBlock);

        if (!linearIndexVal) {
            op.emitRemark(
                "split-kernel-mgpu: could not find linear index (blockIdx*blockDim+threadIdx)");
            return success();
        }

        if (launchBodyIsSingleCall && (!totalConst || !blockXConst)) {
            op.emitRemark("split-kernel-mgpu: constant grid/block required for per-device offset");
            rewriter.eraseOp(op);
            return success();
        }
        func::FuncOp bodySourceForInline;
        if (launchBodyIsSingleCall) {
            bodySourceForInline = kernelCallee;
            Block &kernelBody = kernelCallee.getBody().front();
            for (Operation &o : kernelBody) {
                if (o.hasTrait<OpTrait::IsTerminator>())
                    break;
                if (auto call = dyn_cast<func::CallOp>(o)) {
                    if (func::FuncOp inner = module.lookupSymbol<func::FuncOp>(call.getCalleeAttr().getValue()))
                        bodySourceForInline = inner;
                    break;
                }
            }
        }
        launchBodyIsSingleCall = false;

        Value nVal = rewriter.create<arith::ConstantIndexOp>(loc, numDevices);
        Value zero = rewriter.create<arith::ConstantIndexOp>(loc, 0);

        SmallVector<std::optional<int64_t>> countConst(numDevices);
        SmallVector<std::optional<int64_t>> offsetConst(numDevices);
        SmallVector<Value> countGVal(numDevices);
        SmallVector<Value> offsetGVal(numDevices);

        if (totalConst && blockXConst) {
            int64_t total = *totalConst;
            int64_t n = numDevices;
            int64_t remainder = total % n;
            int64_t baseCount = total / n;
            int64_t offset = 0;
            for (uint32_t g = 0; g < numDevices; ++g) {
                int64_t c = baseCount + (static_cast<int64_t>(g) < remainder ? 1 : 0);
                countConst[g] = c;
                offsetConst[g] = offset;
                offset += c;
            }
        } else {
            Value baseCount = rewriter.create<arith::DivSIOp>(loc, totalThreads, nVal);
            Value remainderVal = rewriter.create<arith::RemSIOp>(loc, totalThreads, nVal);
            for (uint32_t g = 0; g < numDevices; ++g) {
                Value gVal = rewriter.create<arith::ConstantIndexOp>(loc, g);
                Value inRemainder = rewriter.create<arith::CmpIOp>(loc, arith::CmpIPredicate::ult, gVal, remainderVal);
                Value oneIdx = rewriter.create<arith::ConstantIndexOp>(loc, 1);
                countGVal[g] = rewriter.create<arith::SelectOp>(
                    loc, inRemainder, rewriter.create<arith::AddIOp>(loc, baseCount, oneIdx), baseCount);
                if (g == 0) {
                    offsetGVal[g] = zero;
                } else {
                    offsetGVal[g] = zero;
                    for (uint32_t i = 0; i < g; ++i) {
                        Value iVal = rewriter.create<arith::ConstantIndexOp>(loc, i);
                        Value inRem =
                            rewriter.create<arith::CmpIOp>(loc, arith::CmpIPredicate::ult, iVal, remainderVal);
                        Value cnt = rewriter.create<arith::SelectOp>(
                            loc, inRem, rewriter.create<arith::AddIOp>(loc, baseCount, oneIdx), baseCount);
                        offsetGVal[g] = rewriter.create<arith::AddIOp>(loc, offsetGVal[g], cnt);
                    }
                }
            }
        }

        if (launchBodyIsSingleCall) {
            // create N clones by inlining the inner kernel body
            func::FuncOp bodySource = kernelCallee;
            Block &kernelBody = kernelCallee.getBody().front();
            for (Operation &o : kernelBody) {
                if (o.hasTrait<OpTrait::IsTerminator>())
                    break;
                if (auto call = dyn_cast<func::CallOp>(o)) {
                    if (func::FuncOp inner = module.lookupSymbol<func::FuncOp>(call.getCalleeAttr().getValue()))
                        bodySource = inner;
                    break;
                }
            }
            StringRef calleeName = kernelCallee.getName();
            SmallVector<func::FuncOp> clones(numDevices);
            rewriter.setInsertionPointAfter(kernelCallee);
            for (uint32_t g = 0; g < numDevices; ++g) {
                std::string newName = (calleeName + "_mgpu_" + Twine(g)).str();
                FunctionType fnType = kernelCallee.getFunctionType();
                func::FuncOp clone = rewriter.create<func::FuncOp>(loc, newName, fnType);
                SmallVector<Location> argLocs(fnType.getNumInputs(), loc);
                Block *cloneBlock = rewriter.createBlock(&clone.getBody(), {}, fnType.getInputs(), argLocs);
                IRMapping map;
                for (unsigned i = 0; i < bodySource.getNumArguments(); ++i)
                    map.map(bodySource.getArgument(i), cloneBlock->getArgument(i));
                rewriter.setInsertionPointToStart(cloneBlock);
                for (Operation &o : bodySource.getBody().front()) {
                    if (o.hasTrait<OpTrait::IsTerminator>()) {
                        rewriter.create<func::ReturnOp>(loc);
                        break;
                    }
                    Operation *cloned = rewriter.clone(o, map);
                    for (unsigned i = 0; i < o.getNumResults(); ++i)
                        map.map(o.getResult(i), cloned->getResult(i));
                }
                Value linearInClone = findLinearIndexValue(*cloneBlock);
                if (linearInClone)
                    injectOffsetInBlock(*cloneBlock, linearInClone, *offsetConst[g], rewriter);
                clones[g] = clone;
            }
            rewriter.setInsertionPoint(op);

            for (uint32_t g = 0; g < numDevices; ++g) {
                Value devIdx = rewriter.create<arith::ConstantIndexOp>(loc, g);
                Value device = rewriter.create<GetDeviceOp>(loc, DeviceType::get(ctx), devIdx).getResult();

                int64_t countG = *countConst[g];
                int64_t bx = *blockXConst;
                int64_t gridXNew = (countG + bx - 1) / bx;
                Value gridXNewVal = rewriter.create<arith::ConstantIndexOp>(loc, gridXNew);
                Value gridYNew = one;
                Value gridZNew = one;

                OperationState state(loc, LaunchOp::getOperationName());
                LaunchOp::build(rewriter, state, device, ValueRange{gridXNewVal, gridYNew, gridZNew},
                                ValueRange{blockX, blockY, blockZ}, Value());
                LaunchOp newLaunch = cast<LaunchOp>(rewriter.create(state));
                newLaunch->setAttr("polygeist.mgpu.device_index", rewriter.getI32IntegerAttr(g));
                if (offsetConst[g] && countConst[g]) {
                    newLaunch->setAttr("polygeist.mgpu.linear_offset", rewriter.getI64IntegerAttr(*offsetConst[g]));
                    newLaunch->setAttr("polygeist.mgpu.linear_count", rewriter.getI64IntegerAttr(*countConst[g]));
                }
                Region &newRegion = newLaunch.getKernelRegion();
                Block *newBlock = newRegion.empty() ? rewriter.createBlock(&newRegion) : &newRegion.front();
                if (!newBlock->empty()) {
                    for (Operation &term : llvm::make_early_inc_range(llvm::reverse(newBlock->getOperations())))
                        rewriter.eraseOp(&term);
                }
                rewriter.setInsertionPointToStart(newBlock);
                rewriter.create<func::CallOp>(loc, TypeRange(), clones[g].getSymName(), kernelCall.getOperands());
                rewriter.setInsertionPointToEnd(newBlock);
                rewriter.create<TerminatorOp>(loc);
                rewriter.setInsertionPointAfter(newLaunch);
            }
        } else {
            // create N launches with cloned kernel block
            for (uint32_t g = 0; g < numDevices; ++g) {
                Value devIdx = rewriter.create<arith::ConstantIndexOp>(loc, g);
                Value device = rewriter.create<GetDeviceOp>(loc, DeviceType::get(ctx), devIdx).getResult();

                Value countG;
                Value offsetG;
                if (countConst[g]) {
                    countG = rewriter.create<arith::ConstantIndexOp>(loc, *countConst[g]);
                    offsetG = rewriter.create<arith::ConstantIndexOp>(loc, *offsetConst[g]);
                } else {
                    countG = countGVal[g];
                    offsetG = offsetGVal[g];
                }

                Value blockXMinus1 = rewriter.create<arith::SubIOp>(loc, blockX, one);
                Value sum = rewriter.create<arith::AddIOp>(loc, countG, blockXMinus1);
                Value gridXNew = rewriter.create<arith::DivSIOp>(loc, sum, blockX);
                Value gridYNew = one;
                Value gridZNew = one;

                OperationState state(loc, LaunchOp::getOperationName());
                LaunchOp::build(rewriter, state, device, ValueRange{gridXNew, gridYNew, gridZNew},
                                ValueRange{blockX, blockY, blockZ}, Value());
                LaunchOp newLaunch = cast<LaunchOp>(rewriter.create(state));
                newLaunch->setAttr("polygeist.mgpu.device_index", rewriter.getI32IntegerAttr(g));
                if (offsetConst[g] && countConst[g]) {
                    newLaunch->setAttr("polygeist.mgpu.linear_offset", rewriter.getI64IntegerAttr(*offsetConst[g]));
                    newLaunch->setAttr("polygeist.mgpu.linear_count", rewriter.getI64IntegerAttr(*countConst[g]));
                }
                Region &newRegion = newLaunch.getKernelRegion();
                Block *newBlock = newRegion.empty() ? rewriter.createBlock(&newRegion) : &newRegion.front();
                if (!newBlock->empty()) {
                    for (Operation &term : llvm::make_early_inc_range(llvm::reverse(newBlock->getOperations())))
                        rewriter.eraseOp(&term);
                }
                rewriter.setInsertionPointToStart(newBlock);

                // clone the inner kernel body and map its args to the call operands
                Block *sourceBlock = bodySourceForInline ? &bodySourceForInline.getBody().front() : &kernelBlock;
                IRMapping mapping;
                if (bodySourceForInline) {
                    for (unsigned i = 0; i < bodySourceForInline.getNumArguments(); ++i)
                        mapping.map(bodySourceForInline.getArgument(i), kernelCall.getOperand(i));
                }
                for (Operation &srcOp : *sourceBlock) {
                    if (srcOp.hasTrait<OpTrait::IsTerminator>())
                        continue;
                    Operation *cloned = rewriter.clone(srcOp, mapping);
                    for (unsigned i = 0; i < srcOp.getNumResults(); ++i)
                        mapping.map(srcOp.getResult(i), cloned->getResult(i));
                }
                Value newLinearIndex = findLinearIndexValue(*newBlock);
                if (newLinearIndex) {
                    if (offsetConst[g])
                        injectOffsetInBlock(*newBlock, newLinearIndex, *offsetConst[g], rewriter);
                    else
                        injectOffsetInBlockWithValue(*newBlock, newLinearIndex, offsetG, rewriter);
                }
                rewriter.setInsertionPointToEnd(newBlock);
                rewriter.create<TerminatorOp>(loc);
                rewriter.setInsertionPointAfter(newLaunch);
            }
        }

        rewriter.eraseOp(op);
        return success();
    }
};

} // namespace

namespace mlir {
namespace multigpu {

std::unique_ptr<Pass> createSplitKernelMultiGpuPass() { return std::make_unique<SplitKernelMultiGpuPass>(); }

void registerSplitKernelMultiGpuPass() { PassRegistration<SplitKernelMultiGpuPass>(); }

} // namespace multigpu
} // namespace mlir
