// Minimal test: just get_device and alloc
func.func @test_alloc_only() {
  %c0 = arith.constant 0 : index
  %dev = mgpu.get_device %c0 : !mgpu.device
  %ptr = mgpu.alloc %dev : !mgpu.device -> memref<1024xf32>
  return
}
