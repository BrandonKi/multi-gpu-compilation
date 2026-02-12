// RUN: polygeist-opt %s | polygeist-opt | FileCheck %s

module {
  func.func @test_reductions(%comm: !mgpu.communicator, %stream: !mgpu.stream) {
    %send = memref.alloc() : memref<1024xf32>
    %recv = memref.alloc() : memref<1024xf32>
    // %scatter_recv = memref.alloc() : memref<256xf32>

    // All-Reduce using SUM
    // CHECK: mgpu.all_reduce %{{.*}}, %{{.*}}, %{{.*}}, sum stream %{{.*}} : !mgpu.stream
    mgpu.all_reduce %send, %recv, %comm, sum stream %stream : !mgpu.stream
        : memref<1024xf32>, memref<1024xf32>, !mgpu.communicator

    // All-Reduce using MAX
    mgpu.all_reduce %send, %recv, %comm, max 
        : memref<1024xf32>, memref<1024xf32>, !mgpu.communicator

    // Reduce-Scatter using PRODUCT
    // CH // ECK: mgpu.reduce_scatter %{{.*}}, %{{.*}}, %{{.*}}, prod
    // mgpu.reduce_scatter %send, %scatter_recv, %comm, prod
    //     : memref<1024xf32>, memref<256xf32>, !mgpu.communicator

    return
  }
}