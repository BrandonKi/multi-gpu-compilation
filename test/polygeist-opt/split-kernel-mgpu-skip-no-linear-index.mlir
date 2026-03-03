// RUN: polygeist-opt %s -split-kernel-mgpu | FileCheck %s
//
// cannot findlinear index pattern so launch unchanged
// 

// CHECK: module {
// CHECK: mgpu.device_config @mgpu_device_config #mgpu.device_config<numDevices = 2>
module {
  mgpu.device_config @mgpu_device_config #mgpu.device_config<numDevices = 2>

  // CHECK-LABEL: func @noLinearIndex
  // Single launch unchanged (no linear index pattern in body)
  // CHECK: mgpu.launch {{.*}} grid ({{.*}}, {{.*}}, {{.*}}) block ({{.*}}, {{.*}}, {{.*}})
  // CHECK: memref.store {{.*}} : memref<1024xf32>
  // CHECK: mgpu.terminator
  func.func @noLinearIndex(%c: memref<1024xf32>) {
    %c0 = arith.constant 0 : index
    %c32 = arith.constant 32 : index
    %c1 = arith.constant 1 : index
    %dev = mgpu.get_device %c0 : !mgpu.device
    mgpu.launch %dev grid (%c32, %c1, %c1) block (%c32, %c1, %c1) {
      %zero = arith.constant 0.0 : f32
      memref.store %zero, %c[%c0] : memref<1024xf32>
      mgpu.terminator
    }
    return
  }
}
