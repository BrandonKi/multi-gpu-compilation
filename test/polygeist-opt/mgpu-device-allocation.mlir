// RUN: polygeist-opt %s -mgpu-device-allocation | FileCheck %s
//
// Test that mgpu-device-allocation assigns device indices using numDevices from mgpu.device_config.

module {
  mgpu.device_config @config #mgpu.device_config<numDevices = 2>

  // CHECK-LABEL: func @two_launches
  func.func @two_launches() {
    %c0 = arith.constant 0 : index
    // First get_device stays at device 0
    // CHECK: mgpu.get_device %c0
    %dev0 = mgpu.get_device %c0 : !mgpu.device
    // Second get_device is reassigned to device 1
    // CHECK: arith.constant 1
    // CHECK: mgpu.get_device {{.*}} : !mgpu.device
    %dev1 = mgpu.get_device %c0 : !mgpu.device
    return
  }
}
