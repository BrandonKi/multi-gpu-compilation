// RUN: polygeist-opt %s -mgpu-to-llvm | FileCheck %s
//
// Test that mgpu.launch is converted to gpu.launch by the mgpu-to-llvm pass.
// Grid/block are padded to 3D with 1 for unused dimensions.

// CHECK-NOT: mgpu.launch
// CHECK-LABEL: func @launch_1d
func.func @launch_1d() {
  %c0 = arith.constant 0 : index
  %c256 = arith.constant 256 : index
  %c32 = arith.constant 32 : index
  %dev = mgpu.get_device %c0 : !mgpu.device

  // CHECK: gpu.launch blocks
  // CHECK: gpu.terminator
  mgpu.launch %dev grid (%c256) block (%c32) {
    mgpu.terminator
  }

  return
}

// CHECK-LABEL: func @launch_3d
func.func @launch_3d() {
  %c0 = arith.constant 0 : index
  %c4 = arith.constant 4 : index
  %c8 = arith.constant 8 : index
  %c2 = arith.constant 2 : index
  %c16 = arith.constant 16 : index
  %c32 = arith.constant 32 : index
  %c1 = arith.constant 1 : index
  %dev = mgpu.get_device %c0 : !mgpu.device

  // CHECK: gpu.launch blocks
  // CHECK: gpu.terminator
  mgpu.launch %dev grid (%c4, %c8, %c2) block (%c16, %c32, %c1) {
    mgpu.terminator
  }

  return
}
