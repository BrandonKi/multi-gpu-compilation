// RUN: polygeist-opt %s -canonicalize-polygeist -gpu-to-mgpu | FileCheck %s
//
// Test converting two vecAdd GPU kernels to MultiGPU dialect.

// CHECK: module {
// CHECK: mgpu.device_config @mgpu_device_config #mgpu.device_config<numDevices = 1>
module {
  // CHECK-LABEL: func @vecAdd1
  func.func @vecAdd1(%a: memref<1024xf32>, %b: memref<1024xf32>, %c: memref<1024xf32>) {
    %c256 = arith.constant 256 : index
    %c32 = arith.constant 32 : index
    
    // CHECK: %[[dev0:.*]] = mgpu.get_device %{{.*}} : !mgpu.device
    // CHECK: mgpu.launch %[[dev0]] grid (%{{.*}}) block (%{{.*}}) {
    gpu.launch blocks(%bx, %by, %bz) in (%gx = %c256, %gy = %c256, %gz = %c256) 
             threads(%tx, %ty, %tz) in (%tx_size = %c32, %ty_size = %c32, %tz_size = %c32) {
      %bidx = gpu.block_id x
      %tidx = gpu.thread_id x
      %bdim = gpu.block_dim x
      %idx = arith.muli %bidx, %bdim : index
      %idx2 = arith.addi %idx, %tidx : index
      %a_val = memref.load %a[%idx2] : memref<1024xf32>
      %b_val = memref.load %b[%idx2] : memref<1024xf32>
      %sum = arith.addf %a_val, %b_val : f32
      memref.store %sum, %c[%idx2] : memref<1024xf32>
      gpu.terminator
    }
    return
  }

  // CHECK-LABEL: func @vecAdd2
  func.func @vecAdd2(%a: memref<2048xf32>, %b: memref<2048xf32>, %c: memref<2048xf32>) {
    %c512 = arith.constant 512 : index
    %c64 = arith.constant 64 : index
    
    // CHECK: %[[dev1:.*]] = mgpu.get_device %{{.*}} : !mgpu.device
    // CHECK: mgpu.launch %[[dev1]] grid (%{{.*}}) block (%{{.*}}) {
    gpu.launch blocks(%bx, %by, %bz) in (%gx = %c512, %gy = %c512, %gz = %c512) 
             threads(%tx, %ty, %tz) in (%tx_size = %c64, %ty_size = %c64, %tz_size = %c64) {
      %bidx = gpu.block_id x
      %tidx = gpu.thread_id x
      %bdim = gpu.block_dim x
      %idx = arith.muli %bidx, %bdim : index
      %idx2 = arith.addi %idx, %tidx : index
      %a_val = memref.load %a[%idx2] : memref<2048xf32>
      %b_val = memref.load %b[%idx2] : memref<2048xf32>
      %sum = arith.addf %a_val, %b_val : f32
      memref.store %sum, %c[%idx2] : memref<2048xf32>
      gpu.terminator
    }
    return
  }
}
