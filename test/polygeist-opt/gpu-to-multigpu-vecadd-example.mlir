// RUN: polygeist-opt %s -canonicalize-polygeist -gpu-to-multigpu | FileCheck %s
//
// Test converting the vecAdd example from examples/vecAdd.mlir to MultiGPU dialect

module attributes {dlti.dl_spec = #dlti.dl_spec<#dlti.dl_entry<f64, dense<64> : vector<2xi32>>, #dlti.dl_entry<f16, dense<16> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr<270>, dense<32> : vector<4xi32>>, #dlti.dl_entry<f128, dense<128> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr<272>, dense<64> : vector<4xi32>>, #dlti.dl_entry<!llvm.ptr<271>, dense<32> : vector<4xi32>>, #dlti.dl_entry<i64, dense<64> : vector<2xi32>>, #dlti.dl_entry<f80, dense<128> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr, dense<64> : vector<4xi32>>, #dlti.dl_entry<i1, dense<8> : vector<2xi32>>, #dlti.dl_entry<i8, dense<8> : vector<2xi32>>, #dlti.dl_entry<i16, dense<16> : vector<2xi32>>, #dlti.dl_entry<i32, dense<32> : vector<2xi32>>, #dlti.dl_entry<"dlti.endianness", "little">, #dlti.dl_entry<"dlti.stack_alignment", 128 : i32>>, llvm.data_layout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128", llvm.target_triple = "x86_64-unknown-linux-gnu", polygeist.gpu_module.llvm.data_layout = "e-i64:64-i128:128-v16:16-v32:32-n16:32:64", polygeist.gpu_module.llvm.target_triple = "nvptx64-nvidia-cuda", "polygeist.target-cpu" = "x86-64", "polygeist.target-features" = "+cmov,+cx8,+fxsr,+mmx,+sse,+sse2,+x87", "polygeist.tune-cpu" = "generic"} {
  func.func private @_Z24__device_stub__vectorAddPfS_S_i(%arg0: memref<?xf32>, %arg1: memref<?xf32>, %arg2: memref<?xf32>, %arg3: i32) attributes {llvm.linkage = #llvm.linkage<external>, polygeist.device_only_func = "1"} {
    %0 = gpu.block_dim  x
    %1 = arith.index_cast %0 : index to i32
    %2 = gpu.block_id  x
    %3 = arith.index_cast %2 : index to i32
    %4 = arith.muli %1, %3 : i32
    %5 = gpu.thread_id  x
    %6 = arith.index_cast %5 : index to i32
    %7 = arith.addi %4, %6 : i32
    %8 = arith.index_cast %7 : i32 to index
    %9 = arith.cmpi slt, %7, %arg3 : i32
    scf.if %9 {
      %10 = affine.load %arg0[symbol(%8)] : memref<?xf32>
      %11 = affine.load %arg1[symbol(%8)] : memref<?xf32>
      %12 = arith.addf %10, %11 : f32
      affine.store %12, %arg2[symbol(%8)] : memref<?xf32>
    }
    return
  }
  func.func @main() -> i32 attributes {llvm.linkage = #llvm.linkage<external>} {
    %c1 = arith.constant 1 : index
    %c32 = arith.constant 32 : index
    %c4096_i64 = arith.constant 4096 : i64
    %c0_i32 = arith.constant 0 : i32
    %c1024_i32 = arith.constant 1024 : i32
    %alloca = memref.alloca() : memref<1xmemref<?xf32>>
    %alloca_0 = memref.alloca() : memref<1xmemref<?xf32>>
    %alloca_1 = memref.alloca() : memref<1xmemref<?xf32>>
    %cast = memref.cast %alloca_1 : memref<1xmemref<?xf32>> to memref<?xmemref<?xf32>>
    %0 = call @_ZL10cudaMallocIfE9cudaErrorPPT_m(%cast, %c4096_i64) : (memref<?xmemref<?xf32>>, i64) -> i32
    %cast_2 = memref.cast %alloca_0 : memref<1xmemref<?xf32>> to memref<?xmemref<?xf32>>
    %1 = call @_ZL10cudaMallocIfE9cudaErrorPPT_m(%cast_2, %c4096_i64) : (memref<?xmemref<?xf32>>, i64) -> i32
    %cast_3 = memref.cast %alloca : memref<1xmemref<?xf32>> to memref<?xmemref<?xf32>>
    %2 = call @_ZL10cudaMallocIfE9cudaErrorPPT_m(%cast_3, %c4096_i64) : (memref<?xmemref<?xf32>>, i64) -> i32
    %3 = affine.load %alloca_1[0] : memref<1xmemref<?xf32>>
    %4 = affine.load %alloca_0[0] : memref<1xmemref<?xf32>>
    %5 = affine.load %alloca[0] : memref<1xmemref<?xf32>>
    
    // CHECK: %[[dev:.*]] = mgpu.get_device %{{.*}} : !mgpu.device
    // CHECK: mgpu.launch %[[dev]] grid (%{{.*}}, %{{.*}}, %{{.*}}) block (%{{.*}}, %{{.*}}, %{{.*}}) {
    gpu.launch blocks(%arg0, %arg1, %arg2) in (%arg6 = %c32, %arg7 = %c1, %arg8 = %c1) threads(%arg3, %arg4, %arg5) in (%arg9 = %c32, %arg10 = %c1, %arg11 = %c1) {
      func.call @_Z24__device_stub__vectorAddPfS_S_i(%3, %4, %5, %c1024_i32) : (memref<?xf32>, memref<?xf32>, memref<?xf32>, i32) -> ()
      gpu.terminator
    }
    // CHECK: mgpu.terminator
    // CHECK-NEXT: }
    
    %6 = affine.load %alloca_1[0] : memref<1xmemref<?xf32>>
    %7 = "polygeist.memref2pointer"(%6) : (memref<?xf32>) -> !llvm.ptr
    %8 = "polygeist.pointer2memref"(%7) : (!llvm.ptr) -> memref<?xi8>
    %9 = call @cudaFree(%8) : (memref<?xi8>) -> i32
    %10 = affine.load %alloca_0[0] : memref<1xmemref<?xf32>>
    %11 = "polygeist.memref2pointer"(%10) : (memref<?xf32>) -> !llvm.ptr
    %12 = "polygeist.pointer2memref"(%11) : (!llvm.ptr) -> memref<?xi8>
    %13 = call @cudaFree(%12) : (memref<?xi8>) -> i32
    %14 = affine.load %alloca[0] : memref<1xmemref<?xf32>>
    %15 = "polygeist.memref2pointer"(%14) : (memref<?xf32>) -> !llvm.ptr
    %16 = "polygeist.pointer2memref"(%15) : (!llvm.ptr) -> memref<?xi8>
    %17 = call @cudaFree(%16) : (memref<?xi8>) -> i32
    return %c0_i32 : i32
  }
  func.func private @_ZL10cudaMallocIfE9cudaErrorPPT_m(%arg0: memref<?xmemref<?xf32>>, %arg1: i64) -> i32 attributes {llvm.linkage = #llvm.linkage<internal>} {
    %0 = "polygeist.memref2pointer"(%arg0) : (memref<?xmemref<?xf32>>) -> !llvm.ptr
    %1 = "polygeist.pointer2memref"(%0) : (!llvm.ptr) -> memref<?xmemref<?xi8>>
    %2 = call @cudaMalloc(%1, %arg1) : (memref<?xmemref<?xi8>>, i64) -> i32
    return %2 : i32
  }
  func.func private @cudaFree(memref<?xi8>) -> i32 attributes {llvm.linkage = #llvm.linkage<external>}
  func.func private @cudaMalloc(memref<?xmemref<?xi8>>, i64) -> i32 attributes {llvm.linkage = #llvm.linkage<external>}
}
