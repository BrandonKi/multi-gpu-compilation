// RUN: polygeist-opt %s -multigpu-to-gpu | FileCheck %s
//
// Test lowering of mgpu memory operations:
// - mgpu.alloc -> gpu.alloc
// - mgpu.free -> gpu.dealloc
// - mgpu.memcpy is not lowered by this pass (lowered to CUDA elsewhere)

// alloc: mgpu.alloc becomes gpu.alloc (device operand dropped)

// CHECK-LABEL: func @test_alloc_to_gpu
func.func @test_alloc_to_gpu() {
  %c0 = arith.constant 0 : index
  %dev = mgpu.get_device %c0 : !mgpu.device

  // CHECK: gpu.alloc
  // CHECK-SAME: memref<1024xf32>
  %ptr = mgpu.alloc %dev : !mgpu.device -> memref<1024xf32>

  // CHECK: gpu.dealloc %{{.*}} : memref<1024xf32>
  mgpu.free %ptr, %dev : memref<1024xf32>, !mgpu.device

  return
}

// alloc + free on same device: both converted

// CHECK-LABEL: func @test_alloc_free_same_device
func.func @test_alloc_free_same_device() {
  %c0 = arith.constant 0 : index
  %dev = mgpu.get_device %c0 : !mgpu.device

  // CHECK-NOT: mgpu.alloc
  // CHECK: %[[ptr:.*]] = gpu.alloc() : memref<256xf32>
  %ptr = mgpu.alloc %dev : !mgpu.device -> memref<256xf32>

  // CHECK-NOT: mgpu.free
  // CHECK: gpu.dealloc %[[ptr]] : memref<256xf32>
  mgpu.free %ptr, %dev : memref<256xf32>, !mgpu.device

  return
}

// alloc on two devices + memcpy: alloc/free become gpu.*, memcpy stays

// CHECK-LABEL: func @test_alloc_memcpy_free
func.func @test_alloc_memcpy_free() {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %dev0 = mgpu.get_device %c0 : !mgpu.device
  %dev1 = mgpu.get_device %c1 : !mgpu.device

  // CHECK: %[[src:.*]] = gpu.alloc() : memref<128xf32>
  %src = mgpu.alloc %dev0 : !mgpu.device -> memref<128xf32>
  // CHECK: %[[dst:.*]] = gpu.alloc() : memref<128xf32>
  %dst = mgpu.alloc %dev1 : !mgpu.device -> memref<128xf32>

  // memcpy is not lowered by multigpu-to-gpu (no gpu.memcpy); stays as mgpu.memcpy
  // CHECK: mgpu.memcpy %[[dst]], %[[src]], %{{.*}}, %{{.*}} : memref<128xf32>, memref<128xf32>, !mgpu.device, !mgpu.device
  mgpu.memcpy %dst, %src, %dev1, %dev0 : memref<128xf32>, memref<128xf32>, !mgpu.device, !mgpu.device

  // CHECK: gpu.dealloc %[[dst]] : memref<128xf32>
  mgpu.free %dst, %dev1 : memref<128xf32>, !mgpu.device
  // CHECK: gpu.dealloc %[[src]] : memref<128xf32>
  mgpu.free %src, %dev0 : memref<128xf32>, !mgpu.device

  return
}
