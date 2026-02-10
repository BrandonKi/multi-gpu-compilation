# set(MULTIGPU_TD_FILE include/multigpu/MultiGpuOps.td)
# set(MGPU_SOURCE MultiGpuOps.td)
set(LLVM_TARGET_DEFINITIONS MultiGpuOps.td)

mlir_tablegen(MultiGpuAttrDefs.h.inc -gen-attrdef-decls)
mlir_tablegen(MultiGpuAttrDefs.cpp.inc -gen-attrdef-defs)

mlir_tablegen(MultiGpuTypeDefs.h.inc -gen-typedef-decls)
mlir_tablegen(MultiGpuTypeDefs.cpp.inc -gen-typedef-defs)

mlir_tablegen(MultiGpuEnums.h.inc -gen-enum-decls)
mlir_tablegen(MultiGpuEnums.cpp.inc -gen-enum-defs)

add_mlir_dialect(MultiGpuOps mgpu)
# add_mlir_doc(MultiGpuDialect -gen-dialect-doc MultiGpuDialect MultiGpu/)
# add_mlir_doc(MultiGpuOps -gen-op-doc MultiGpuOps /home/kirincichb/Polygeist/build/MultiGpu/)
add_mlir_doc(MultiGpuOps -gen-op-doc MultiGpuOps ./MultiGpu/)

add_public_tablegen_target(MultiGpuAttributesAndTypesGen)