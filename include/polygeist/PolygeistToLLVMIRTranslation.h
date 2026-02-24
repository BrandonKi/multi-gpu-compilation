#ifndef POLYGEIST_POLYGEISTTOLLVMIRTRANSLATION_H
#define POLYGEIST_POLYGEISTTOLLVMIRTRANSLATION_H

#include "mlir/IR/DialectRegistry.h"

namespace mlir {
void registerPolygeistDialectTranslation(DialectRegistry &registry);

void registerPolygeistDialectTranslation(MLIRContext &context);
} // namespace mlir

#endif // POLYGEIST_POLYGEISTTOLLVMIRTRANSLATION_H
