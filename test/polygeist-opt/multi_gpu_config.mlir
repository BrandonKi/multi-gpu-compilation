// RUN: polygeist-opt %s | polygeist-opt | FileCheck %s

module {
  // Define a 4-GPU setup using non-contiguous system indices
  // CHECK: mgpu.device_config @gpu_cluster #mgpu.device_config<numDevices = 4, deviceIds = [0, 2, 4, 6]>
  mgpu.device_config @gpu_cluster #mgpu.device_config<numDevices = 4, deviceIds = [0, 2, 4, 6]>

  func.func @test_device_init(%idx: index) {
    // Get device handle
    // CHECK: %[[DEV:.*]] = mgpu.get_device %{{.*}} : !mgpu.device
    %dev = mgpu.get_device %idx : !mgpu.device

    // Create a stream
    // CHECK: %[[STREAM:.*]] = mgpu.create_stream %[[DEV]] : !mgpu.device -> !mgpu.stream
    %stream = mgpu.create_stream %dev : !mgpu.device -> !mgpu.stream

    // Sync the device
    // CHECK: mgpu.sync_device %[[DEV]] : !mgpu.device
    mgpu.sync_device %dev : !mgpu.device

    // Cleanup stream
    mgpu.destroy_stream %dev %stream : !mgpu.device, !mgpu.stream

    return
  }
}