// RUN: polygeist-opt %s -multigpu-to-gpu | FileCheck %s
//
// Test the MultiGPU to GPU conversion pass.

// CHECK-LABEL: func @test_free_conversion
func.func @test_free_conversion(%ptr: memref<1024xf32>, %dev: !mgpu.device) {
  // CHECK: gpu.dealloc %{{.*}} : memref<1024xf32>
  // CHECK-NOT: mgpu.free
  mgpu.free %ptr, %dev : memref<1024xf32>, !mgpu.device
  
  return
}

// CHECK-LABEL: func @test_sync_conversion
func.func @test_sync_conversion() {
  %c0 = arith.constant 0 : index
  %dev = mgpu.get_device %c0 : !mgpu.device
  
  // CHECK-NOT: mgpu.sync_device
  mgpu.sync_device %dev : !mgpu.device
  
  %stream = mgpu.create_stream %dev : !mgpu.device -> !mgpu.stream
  
  // CHECK-NOT: mgpu.sync_stream
  mgpu.sync_stream %stream : !mgpu.stream
  
  // CHECK-NOT: mgpu.destroy_stream
  mgpu.destroy_stream %stream : !mgpu.stream
  
  return
}

// CHECK-LABEL: func @test_multiple_ops
func.func @test_multiple_ops(%ptr0: memref<256xf32>, %ptr1: memref<256xf32>, 
                              %dev0: !mgpu.device, %dev1: !mgpu.device) {
  // CHECK: gpu.dealloc %{{.*}} : memref<256xf32>
  // CHECK: gpu.dealloc %{{.*}} : memref<256xf32>
  mgpu.free %ptr0, %dev0 : memref<256xf32>, !mgpu.device
  mgpu.free %ptr1, %dev1 : memref<256xf32>, !mgpu.device
  
  // CHECK-NOT: mgpu.sync_device
  mgpu.sync_device %dev0 : !mgpu.device
  mgpu.sync_device %dev1 : !mgpu.device
  
  return
}
