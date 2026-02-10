// RUN: polygeist-opt %s | polygeist-opt | FileCheck %s

module {
  func.func @test_sharding(%idx0: index, %idx1: index) {
    %dev0 = mgpu.get_device %idx0 : !mgpu.device
    %dev1 = mgpu.get_device %idx1 : !mgpu.device

    // Allocate sharded buffers for a 1024x512 tensor split across 2 devices
    // CHECK: mgpu.sharded_alloc %{{.*}}, %{{.*}} logicalShape = [1024, 512] shardAxis = 0 shardKind = uniform
    %s0, %s1 = mgpu.sharded_alloc %dev0, %dev1 
        logicalShape = [1024, 512] shardAxis = 0 shardKind = uniform
        : !mgpu.device, !mgpu.device -> memref<512x512xf32>, memref<512x512xf32>

    // Scatter a host buffer to the allocated shards
    %host_buf = memref.alloc() : memref<1024x512xf32>
    %host_dev = mgpu.get_device %idx0 : !mgpu.device // conceptual host dev

    // CHECK: mgpu.scatter %{{.*}}, %{{.*}}, [%{{.*}}, %{{.*}}], [%{{.*}}, %{{.*}}] axis = 0
    mgpu.scatter %host_buf, %host_dev, [%s0, %s1], [%dev0, %dev1]
        axis = 0
        : memref<1024x512xf32>, !mgpu.device, 
          [memref<512x512xf32>, memref<512x512xf32>], 
          [!mgpu.device, !mgpu.device]

    // Clean up
    mgpu.sharded_free [%s0, %s1], [%dev0, %dev1]
        : [memref<512x512xf32>, memref<512x512xf32>], [!mgpu.device, !mgpu.device]

    return
  }
}