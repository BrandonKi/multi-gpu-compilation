// RUN: polygeist-opt %s -mgpu-to-llvm | FileCheck %s
//
// Simple test for mgpu-to-llvm: get_device and sync_device become runtime calls.

// CHECK-LABEL: func @simple_sync
func.func @simple_sync() {
  %c0 = arith.constant 0 : index
  %dev = mgpu.get_device %c0 : !mgpu.device

  // CHECK-NOT: mgpu.get_device
  // CHECK: llvm.call @mgpurtSetDevice
  // CHECK-NOT: mgpu.sync_device
  // CHECK: llvm.call @mgpurtDeviceSynchronizeErr
  mgpu.sync_device %dev : !mgpu.device

  return
}
