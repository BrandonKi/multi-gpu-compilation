// RUN: polygeist-opt %s -multigpu-to-gpu | FileCheck %s
//
// Test the MultiGPU lowering pass that removes simple no-op operations.

// CHECK-LABEL: func @test_sync_ops
func.func @test_sync_ops() {
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

// CHECK-LABEL: func @test_multiple_syncs
func.func @test_multiple_syncs() {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %dev0 = mgpu.get_device %c0 : !mgpu.device
  %dev1 = mgpu.get_device %c1 : !mgpu.device
  
  // CHECK-NOT: mgpu.sync_device
  mgpu.sync_device %dev0 : !mgpu.device
  mgpu.sync_device %dev1 : !mgpu.device
  
  %stream0 = mgpu.create_stream %dev0 : !mgpu.device -> !mgpu.stream
  %stream1 = mgpu.create_stream %dev1 : !mgpu.device -> !mgpu.stream
  
  // CHECK-NOT: mgpu.sync_stream
  mgpu.sync_stream %stream0 : !mgpu.stream
  mgpu.sync_stream %stream1 : !mgpu.stream
  
  // CHECK-NOT: mgpu.destroy_stream
  mgpu.destroy_stream %stream0 : !mgpu.stream
  mgpu.destroy_stream %stream1 : !mgpu.stream
  
  return
}
