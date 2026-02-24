#include "polygeist/PolygeistToLLVMIRTranslation.h"
#include "polygeist/Dialect.h"
#include "polygeist/Ops.h"
#include "mlir/Target/LLVMIR/LLVMTranslationInterface.h"

using namespace mlir;
using namespace mlir::polygeist;

// This pass really should not be here
// TODO figure out why polygeist isn't lowering the op correctly in the first place
namespace {
class PolygeistDialectLLVMIRTranslationInterface
    : public LLVMTranslationDialectInterface {
public:
  using LLVMTranslationDialectInterface::LLVMTranslationDialectInterface;

  LogicalResult
  convertOperation(Operation *operation, llvm::IRBuilderBase &builder,
                   LLVM::ModuleTranslation &moduleTranslation) const override {
    if (auto alt = dyn_cast<polygeist::AlternativesOp>(operation)) {
      operation->erase();
      return success();
    }

    return operation->emitError("unsupported Polygeist operation during "
                                 "LLVM translation: ")
           << operation->getName();
  }
};

} // end namespace

void mlir::registerPolygeistDialectTranslation(DialectRegistry &registry) {
  registry.insert<polygeist::PolygeistDialect>();
  registry.addExtension(+[](MLIRContext *ctx, PolygeistDialect *dialect) {
    dialect->addInterfaces<PolygeistDialectLLVMIRTranslationInterface>();
  });
}

void mlir::registerPolygeistDialectTranslation(MLIRContext &context) {
  DialectRegistry registry;
  registerPolygeistDialectTranslation(registry);
  context.appendDialectRegistry(registry);
}
