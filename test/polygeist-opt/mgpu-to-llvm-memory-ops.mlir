// RUN: polygeist-opt %s -mgpu-to-llvm | FileCheck %s
//
// Test lowering of mgpu memory operations by the mgpu-to-llvm (CUDA runtime) pass:
// - mgpu.alloc -> llvm.call @mgpurtMemAllocOnDevice
// - mgpu.free -> llvm.call @mgpurtSetDevice, llvm.call @mgpurtMemFree
// - mgpu.memcpy -> llvm.call @mgpurtMemcpyPeer

// alloc: mgpu.alloc becomes mgpurtMemAllocOnDevice call

// CHECK-LABEL: func @test_alloc
func.func @test_alloc() {
  %c0 = arith.constant 0 : index
  %dev = mgpu.get_device %c0 : !mgpu.device

  // CHECK-NOT: mgpu.alloc
  // CHECK: llvm.call @mgpurtMemAllocOnDevice
  // CHECK-SAME: (i32, i64) -> !llvm.ptr
  %ptr = mgpu.alloc %dev : !mgpu.device -> memref<1024xf32>

  // CHECK-NOT: mgpu.free
  // CHECK: llvm.call @mgpurtSetDevice
  // CHECK: llvm.call @mgpurtMemFree
  mgpu.free %ptr, %dev : memref<1024xf32>, !mgpu.device

  return
}

// alloc + free on same device: both become runtime calls

// CHECK-LABEL: func @test_alloc_free_same_device
func.func @test_alloc_free_same_device() {
  %c0 = arith.constant 0 : index
  %dev = mgpu.get_device %c0 : !mgpu.device

  // CHECK: llvm.call @mgpurtMemAllocOnDevice
  %ptr = mgpu.alloc %dev : !mgpu.device -> memref<256xf32>

  // CHECK: llvm.call @mgpurtMemFree
  mgpu.free %ptr, %dev : memref<256xf32>, !mgpu.device

  return
}

// alloc on two devices + memcpy: all become CUDA runtime calls
// Use distinct constants so we get device 0 and device 1 (no folding to same).

// CHECK-LABEL: func @test_alloc_memcpy_free
func.func @test_alloc_memcpy_free() {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %dev0 = mgpu.get_device %c0 : !mgpu.device
  %dev1 = mgpu.get_device %c1 : !mgpu.device

  // CHECK: llvm.call @mgpurtMemAllocOnDevice
  %src = mgpu.alloc %dev0 : !mgpu.device -> memref<128xf32>
  // CHECK: llvm.call @mgpurtMemAllocOnDevice
  %dst = mgpu.alloc %dev1 : !mgpu.device -> memref<128xf32>

  // CHECK-NOT: mgpu.memcpy
  // CHECK: llvm.call @mgpurtMemcpyPeer
  // CHECK-SAME: i32, i64) -> i32
  mgpu.memcpy %dst, %src, %dev1, %dev0 : memref<128xf32>, memref<128xf32>, !mgpu.device, !mgpu.device

  // CHECK: llvm.call @mgpurtMemFree
  mgpu.free %dst, %dev1 : memref<128xf32>, !mgpu.device
  // CHECK: llvm.call @mgpurtMemFree
  mgpu.free %src, %dev0 : memref<128xf32>, !mgpu.device

  return
}
