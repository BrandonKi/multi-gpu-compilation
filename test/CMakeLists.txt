configure_lit_site_cfg(
  ${CMAKE_CURRENT_SOURCE_DIR}/lit.site.cfg.py.in
  ${CMAKE_CURRENT_BINARY_DIR}/lit.site.cfg.py
  MAIN_CONFIG
  ${CMAKE_CURRENT_SOURCE_DIR}/lit.cfg.py
)

add_lit_testsuite(check-polygeist-opt "Verify Polygeist passes perform correctly"
    ${CMAKE_CURRENT_BINARY_DIR}
    DEPENDS polygeist-opt FileCheck not
    ARGS -v
)

set_target_properties(check-polygeist-opt PROPERTIES FOLDER "Tests")

# check-mgpu: mgpu tests only
file(MAKE_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/mgpu)
configure_file(
  ${CMAKE_CURRENT_SOURCE_DIR}/mgpu/lit.cfg.py
  ${CMAKE_CURRENT_BINARY_DIR}/mgpu/lit.cfg.py
  COPYONLY
)
set(MGPU_TEST_FILES
  gpu-to-multigpu-vecadd.mlir
  gpu-to-multigpu-vecadd-example.mlir
  mgpu-device-allocation.mlir
  mgpu-to-llvm.mlir
  mgpu-to-llvm-launch.mlir
  mgpu-to-llvm-memory-ops.mlir
  multi_gpu.mlir
  multi_gpu_config.mlir
  multi_gpu_reduction.mlir
  multi_gpu_sharding.mlir
)
foreach(_mgpu_file ${MGPU_TEST_FILES})
  configure_file(
    ${CMAKE_CURRENT_SOURCE_DIR}/polygeist-opt/${_mgpu_file}
    ${CMAKE_CURRENT_BINARY_DIR}/mgpu/${_mgpu_file}
    COPYONLY
  )
endforeach()

add_lit_testsuite(check-mgpu "Multigpu/mgpu dialect tests"
    ${CMAKE_CURRENT_BINARY_DIR}/mgpu
    DEPENDS polygeist-opt FileCheck not
    ARGS -v
)

set_target_properties(check-mgpu PROPERTIES FOLDER "Tests")
