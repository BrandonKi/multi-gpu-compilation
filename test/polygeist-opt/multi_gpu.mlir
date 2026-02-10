// RUN: polygeist-opt %s | polygeist-opt | FileCheck %s
//
// Basic round-trip and smoke tests for the mgpu dialect.
// All syntax matches the assemblyFormat strings in MultiGpuOps.td exactly.

// ===------------------------------------------------------------------- ===//
// Types and device/stream/communicator creation
// ===------------------------------------------------------------------- ===//

// CHECK-LABEL: func @test_device_and_stream_creation
func.func @test_device_and_stream_creation() {
  %c0 = arith.constant 0 : index

  // CHECK:       %[[dev:.*]] = mgpu.get_device %{{.*}} : !mgpu.device
  %dev = mgpu.get_device %c0 : !mgpu.device

  // CHECK:       %[[s0:.*]] = mgpu.create_stream %[[dev]] : !mgpu.device -> !mgpu.stream
  %s0 = mgpu.create_stream %dev : !mgpu.device -> !mgpu.stream

  // CHECK:       %[[s1:.*]] = mgpu.create_stream %[[dev]] : !mgpu.device -> !mgpu.stream
  %s1 = mgpu.create_stream %dev : !mgpu.device -> !mgpu.stream

  // CHECK:       mgpu.destroy_stream %[[s1]] : !mgpu.stream
  mgpu.destroy_stream %s1 : !mgpu.stream

  // CHECK:       mgpu.destroy_stream %[[s0]] : !mgpu.stream
  mgpu.destroy_stream %s0 : !mgpu.stream

  return
}

// CHECK-LABEL: func @test_communicator_creation
func.func @test_communicator_creation() {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c2 = arith.constant 2 : index
  %dev0 = mgpu.get_device %c0 : !mgpu.device
  %dev1 = mgpu.get_device %c1 : !mgpu.device
  %dev2 = mgpu.get_device %c2 : !mgpu.device

  // CHECK:       %[[comm:.*]] = mgpu.create_communicator %[[d0:.*]], %[[d1:.*]], %[[d2:.*]] : !mgpu.device, !mgpu.device, !mgpu.device -> !mgpu.communicator
  %comm = mgpu.create_communicator %dev0, %dev1, %dev2 : !mgpu.device, !mgpu.device, !mgpu.device -> !mgpu.communicator

  return
}

// ===------------------------------------------------------------------- ===//
// Memory management
// ===------------------------------------------------------------------- ===//

// CHECK-LABEL: func @test_alloc_free
func.func @test_alloc_free() {
  %c0 = arith.constant 0 : index
  %dev = mgpu.get_device %c0 : !mgpu.device

  // CHECK:       %[[ptr:.*]] = mgpu.alloc %[[dev:.*]] : !mgpu.device -> memref<1024xf32>
  %ptr = mgpu.alloc %dev : !mgpu.device -> memref<1024xf32>

  // CHECK:       mgpu.free %[[ptr]], %[[dev]] : memref<1024xf32>, !mgpu.device
  mgpu.free %ptr, %dev : memref<1024xf32>, !mgpu.device

  return
}

// CHECK-LABEL: func @test_memcpy_sync
func.func @test_memcpy_sync() {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %dev0 = mgpu.get_device %c0 : !mgpu.device
  %dev1 = mgpu.get_device %c1 : !mgpu.device
  %src  = mgpu.alloc %dev0 : !mgpu.device -> memref<256xf32>
  %dst  = mgpu.alloc %dev1 : !mgpu.device -> memref<256xf32>

  // Synchronous — no stream group printed.
  // CHECK:       mgpu.memcpy %[[dst:.*]], %[[src:.*]], %[[d1:.*]], %[[d0:.*]] : memref<256xf32>, memref<256xf32>, !mgpu.device, !mgpu.device
  mgpu.memcpy %dst, %src, %dev1, %dev0 : memref<256xf32>, memref<256xf32>, !mgpu.device, !mgpu.device

  return
}

// CHECK-LABEL: func @test_memcpy_async
func.func @test_memcpy_async() {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %dev0   = mgpu.get_device %c0 : !mgpu.device
  %dev1   = mgpu.get_device %c1 : !mgpu.device
  %stream = mgpu.create_stream %dev1 : !mgpu.device -> !mgpu.stream
  %src    = mgpu.alloc %dev0 : !mgpu.device -> memref<256xf32>
  %dst    = mgpu.alloc %dev1 : !mgpu.device -> memref<256xf32>

  // Async — stream group with its type is printed.
  // CHECK:       mgpu.memcpy %[[dst:.*]], %[[src:.*]], %[[d1:.*]], %[[d0:.*]] stream %[[st:.*]] : !mgpu.stream : memref<256xf32>, memref<256xf32>, !mgpu.device, !mgpu.device
  mgpu.memcpy %dst, %src, %dev1, %dev0 stream %stream : !mgpu.stream : memref<256xf32>, memref<256xf32>, !mgpu.device, !mgpu.device

  mgpu.destroy_stream %stream : !mgpu.stream
  return
}

// ===------------------------------------------------------------------- ===//
// Kernel launch
// ===------------------------------------------------------------------- ===//

// CHECK-LABEL: func @test_launch_sync
func.func @test_launch_sync() {
  %c0   = arith.constant 0   : index
  %c256 = arith.constant 256 : index
  %c32  = arith.constant 32  : index
  %dev  = mgpu.get_device %c0 : !mgpu.device

  // 1D grid, 1D block, synchronous.
  // CHECK:       mgpu.launch %[[dev:.*]] grid (%{{.*}}) block (%{{.*}}) {
  mgpu.launch %dev grid (%c256) block (%c32) {
  }

  return
}

// CHECK-LABEL: func @test_launch_3d_async
func.func @test_launch_3d_async() {
  %c0  = arith.constant 0  : index
  %dev = mgpu.get_device %c0 : !mgpu.device
  %stream = mgpu.create_stream %dev : !mgpu.device -> !mgpu.stream

  %c4  = arith.constant 4  : index
  %c8  = arith.constant 8  : index
  %c2  = arith.constant 2  : index
  %c16 = arith.constant 16 : index
  %c32 = arith.constant 32 : index
  %c1  = arith.constant 1  : index

  // 3D grid, 3D block, async.
  // CHECK:       mgpu.launch %[[dev:.*]] grid (%{{.*}}, %{{.*}}, %{{.*}}) block (%{{.*}}, %{{.*}}, %{{.*}}) stream %[[st:.*]] {
  mgpu.launch %dev grid (%c4, %c8, %c2) block (%c16, %c32, %c1) stream %stream {
  }

  mgpu.destroy_stream %stream : !mgpu.stream
  return
}

// ===------------------------------------------------------------------- ===//
// Synchronization
// ===------------------------------------------------------------------- ===//

// CHECK-LABEL: func @test_sync_ops
func.func @test_sync_ops() {
  %c0  = arith.constant 0 : index
  %dev = mgpu.get_device %c0 : !mgpu.device
  %s0  = mgpu.create_stream %dev : !mgpu.device -> !mgpu.stream
  %s1  = mgpu.create_stream %dev : !mgpu.device -> !mgpu.stream

  // CHECK:       mgpu.sync_device %[[dev:.*]] : !mgpu.device
  mgpu.sync_device %dev : !mgpu.device

  // CHECK:       mgpu.sync_stream %[[s0:.*]] : !mgpu.stream
  mgpu.sync_stream %s0 : !mgpu.stream

  // CHECK:       mgpu.stream_wait %[[s1:.*]], %[[s0:.*]] : !mgpu.stream, !mgpu.stream
  mgpu.stream_wait %s1, %s0 : !mgpu.stream, !mgpu.stream

  mgpu.destroy_stream %s0 : !mgpu.stream
  mgpu.destroy_stream %s1 : !mgpu.stream
  return
}

// CHECK-LABEL: func @test_stream_wait_multiple
func.func @test_stream_wait_multiple() {
  %c0  = arith.constant 0 : index
  %dev = mgpu.get_device %c0 : !mgpu.device
  %s0  = mgpu.create_stream %dev : !mgpu.device -> !mgpu.stream
  %s1  = mgpu.create_stream %dev : !mgpu.device -> !mgpu.stream
  %s2  = mgpu.create_stream %dev : !mgpu.device -> !mgpu.stream

  // CHECK:       mgpu.stream_wait %[[s2:.*]], %[[s0:.*]], %[[s1:.*]] : !mgpu.stream, !mgpu.stream, !mgpu.stream
  mgpu.stream_wait %s2, %s0, %s1 : !mgpu.stream, !mgpu.stream, !mgpu.stream

  mgpu.destroy_stream %s0 : !mgpu.stream
  mgpu.destroy_stream %s1 : !mgpu.stream
  mgpu.destroy_stream %s2 : !mgpu.stream
  return
}

// ===------------------------------------------------------------------- ===//
// Collective communication
// ===------------------------------------------------------------------- ===//

// CHECK-LABEL: func @test_all_reduce
func.func @test_all_reduce() {
  %c0   = arith.constant 0 : index
  %c1   = arith.constant 1 : index
  %dev0 = mgpu.get_device %c0 : !mgpu.device
  %dev1 = mgpu.get_device %c1 : !mgpu.device
  %comm = mgpu.create_communicator %dev0, %dev1 : !mgpu.device, !mgpu.device -> !mgpu.communicator
  %send = mgpu.alloc %dev0 : !mgpu.device -> memref<128xf32>
  %recv = mgpu.alloc %dev0 : !mgpu.device -> memref<128xf32>

  // CHECK:       mgpu.all_reduce %[[send:.*]], %[[recv:.*]], %[[comm:.*]], sum : memref<128xf32>, memref<128xf32>, !mgpu.communicator
  mgpu.all_reduce %send, %recv, %comm, sum : memref<128xf32>, memref<128xf32>, !mgpu.communicator

  return
}

// CHECK-LABEL: func @test_all_reduce_async
func.func @test_all_reduce_async() {
  %c0     = arith.constant 0 : index
  %c1     = arith.constant 1 : index
  %dev0   = mgpu.get_device %c0 : !mgpu.device
  %dev1   = mgpu.get_device %c1 : !mgpu.device
  %comm   = mgpu.create_communicator %dev0, %dev1 : !mgpu.device, !mgpu.device -> !mgpu.communicator
  %stream = mgpu.create_stream %dev0 : !mgpu.device -> !mgpu.stream
  %send   = mgpu.alloc %dev0 : !mgpu.device -> memref<128xf32>
  %recv   = mgpu.alloc %dev0 : !mgpu.device -> memref<128xf32>

  // CHECK:       mgpu.all_reduce %[[send:.*]], %[[recv:.*]], %[[comm:.*]], max stream %[[st:.*]] : !mgpu.stream : memref<128xf32>, memref<128xf32>, !mgpu.communicator
  mgpu.all_reduce %send, %recv, %comm, max stream %stream : !mgpu.stream : memref<128xf32>, memref<128xf32>, !mgpu.communicator

  mgpu.destroy_stream %stream : !mgpu.stream
  return
}

// CHECK-LABEL: func @test_all_gather
func.func @test_all_gather() {
  %c0   = arith.constant 0 : index
  %c1   = arith.constant 1 : index
  %dev0 = mgpu.get_device %c0 : !mgpu.device
  %dev1 = mgpu.get_device %c1 : !mgpu.device
  %comm = mgpu.create_communicator %dev0, %dev1 : !mgpu.device, !mgpu.device -> !mgpu.communicator
  %send = mgpu.alloc %dev0 : !mgpu.device -> memref<64xf32>
  %recv = mgpu.alloc %dev0 : !mgpu.device -> memref<128xf32>

  // CHECK:       mgpu.all_gather %[[send:.*]], %[[recv:.*]], %[[comm:.*]] : memref<64xf32>, memref<128xf32>, !mgpu.communicator
  mgpu.all_gather %send, %recv, %comm : memref<64xf32>, memref<128xf32>, !mgpu.communicator

  return
}

// CHECK-LABEL: func @test_reduce_scatter
func.func @test_reduce_scatter() {
  %c0   = arith.constant 0 : index
  %c1   = arith.constant 1 : index
  %dev0 = mgpu.get_device %c0 : !mgpu.device
  %dev1 = mgpu.get_device %c1 : !mgpu.device
  %comm = mgpu.create_communicator %dev0, %dev1 : !mgpu.device, !mgpu.device -> !mgpu.communicator
  %send = mgpu.alloc %dev0 : !mgpu.device -> memref<128xf32>
  %recv = mgpu.alloc %dev0 : !mgpu.device -> memref<64xf32>

  // CHECK:       mgpu.reduce_scatter %[[send:.*]], %[[recv:.*]], %[[comm:.*]], sum : memref<128xf32>, memref<64xf32>, !mgpu.communicator
  mgpu.reduce_scatter %send, %recv, %comm, sum : memref<128xf32>, memref<64xf32>, !mgpu.communicator

  return
}

// CHECK-LABEL: func @test_broadcast
func.func @test_broadcast() {
  %c0   = arith.constant 0 : index
  %c1   = arith.constant 1 : index
  %dev0 = mgpu.get_device %c0 : !mgpu.device
  %dev1 = mgpu.get_device %c1 : !mgpu.device
  %comm = mgpu.create_communicator %dev0, %dev1 : !mgpu.device, !mgpu.device -> !mgpu.communicator
  %buf  = mgpu.alloc %dev0 : !mgpu.device -> memref<256xf32>

  // CHECK:       mgpu.broadcast %[[buf:.*]], %[[comm:.*]], %[[root:.*]] : memref<256xf32>, !mgpu.communicator
  mgpu.broadcast %buf, %comm, %c0 : memref<256xf32>, !mgpu.communicator

  return
}

// CHECK-LABEL: func @test_send_recv
func.func @test_send_recv() {
  %c0   = arith.constant 0 : index
  %c1   = arith.constant 1 : index
  %dev0 = mgpu.get_device %c0 : !mgpu.device
  %dev1 = mgpu.get_device %c1 : !mgpu.device
  %comm = mgpu.create_communicator %dev0, %dev1 : !mgpu.device, !mgpu.device -> !mgpu.communicator
  %buf  = mgpu.alloc %dev0 : !mgpu.device -> memref<64xf32>

  // CHECK:       mgpu.send %[[buf:.*]], %[[comm:.*]], %[[dst:.*]] : memref<64xf32>, !mgpu.communicator
  mgpu.send %buf, %comm, %c1 : memref<64xf32>, !mgpu.communicator

  // CHECK:       mgpu.recv %[[buf:.*]], %[[comm:.*]], %[[src:.*]] : memref<64xf32>, !mgpu.communicator
  mgpu.recv %buf, %comm, %c0 : memref<64xf32>, !mgpu.communicator

  return
}
