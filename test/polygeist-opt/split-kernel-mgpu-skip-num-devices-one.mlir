// RUN: polygeist-opt %s -split-kernel-mgpu | FileCheck %s
//
// When numDevices is 1, the pass does nothing

// CHECK: module {
// CHECK: mgpu.device_config @mgpu_device_config #mgpu.device_config<numDevices = 1>
module {
  mgpu.device_config @mgpu_device_config #mgpu.device_config<numDevices = 1>

  // CHECK-LABEL: func @vecAdd
  // Single launch unchanged (numDevices=1)
  // CHECK: mgpu.launch {{.*}} grid ({{.*}}, {{.*}}, {{.*}}) block ({{.*}}, {{.*}}, {{.*}})
  // CHECK: mgpu.terminator
  func.func @vecAdd(%a: memref<1024xf32>, %b: memref<1024xf32>, %c: memref<1024xf32>) {
    %c0 = arith.constant 0 : index
    %c32 = arith.constant 32 : index
    %c1 = arith.constant 1 : index
    %dev = mgpu.get_device %c0 : !mgpu.device
    mgpu.launch %dev grid (%c32, %c1, %c1) block (%c32, %c1, %c1) {
      %bidx = gpu.block_id x
      %tidx = gpu.thread_id x
      %bdim = gpu.block_dim x
      %idx = arith.muli %bidx, %bdim : index
      %idx2 = arith.addi %idx, %tidx : index
      %a_val = memref.load %a[%idx2] : memref<1024xf32>
      %b_val = memref.load %b[%idx2] : memref<1024xf32>
      %sum = arith.addf %a_val, %b_val : f32
      memref.store %sum, %c[%idx2] : memref<1024xf32>
      mgpu.terminator
    }
    return
  }
}
