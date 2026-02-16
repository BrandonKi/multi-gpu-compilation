#ifndef MULTIGPU_OPS
#define MULTIGPU_OPS

include "mlir/IR/OpBase.td"
include "mlir/IR/AttrTypeBase.td"
include "mlir/IR/BuiltinAttributeInterfaces.td"
include "mlir/IR/EnumAttr.td"
include "mlir/IR/SymbolInterfaces.td"
include "mlir/Interfaces/SideEffectInterfaces.td"

def MultiGpu_Dialect : Dialect {
  let name = "mgpu";
  let description = [{
    The `mgpu` dialect models high-level multi-GPU computation.
    This includes memory management, kernel launches, device synchronization, and collective communication primitives.
    It uses a stream-based async execution model for parallelism.
  }];
  let cppNamespace = "::mlir::multigpu";
  let extraClassDeclaration = [{
    Type parseType(DialectAsmParser &parser) const override;
    void printType(Type type, DialectAsmPrinter &printer) const override;
    Attribute parseAttribute(DialectAsmParser &parser, Type type) const override;
    void printAttribute(Attribute attr, DialectAsmPrinter &printer) const override;
  }];
}

class MultiGpu_Op<string mnemonic, list<Trait> traits = []>
    : Op<MultiGpu_Dialect, mnemonic, traits>;

class MultiGpu_Attr<string name, string attrMnemonic, list<Trait> traits = []>
    : AttrDef<MultiGpu_Dialect, name, traits> {
  let mnemonic = attrMnemonic;
}


def DeviceType : TypeDef<MultiGpu_Dialect, "Device"> {
  let mnemonic = "device";
  let summary = "An opaque handle to a GPU device";
}

def StreamType : TypeDef<MultiGpu_Dialect, "Stream"> {
  let mnemonic = "stream";
  let summary = "An opaque handle to a device stream";
}

def CommunicatorType : TypeDef<MultiGpu_Dialect, "Communicator"> {
  let mnemonic = "communicator";
  let summary = "A group of devices that can be used for collective communication operations";
}

def Device       : Type<Or<[DeviceType.predicate]>,       "device handle">;
def Stream       : Type<Or<[StreamType.predicate]>,       "stream handle">;
def Communicator : Type<Or<[CommunicatorType.predicate]>, "communicator handle">;

def ReductionKind_Sum  : I32EnumAttrCase<"Sum",  0, "sum">;
def ReductionKind_Prod : I32EnumAttrCase<"Prod", 1, "prod">;
def ReductionKind_Min  : I32EnumAttrCase<"Min",  2, "min">;
def ReductionKind_Max  : I32EnumAttrCase<"Max",  3, "max">;

def ReductionKindAttr : I32EnumAttr<"ReductionKind",
    "Reduction operation for collective operations",
    [ReductionKind_Sum, ReductionKind_Prod,
     ReductionKind_Min, ReductionKind_Max]> {
  let cppNamespace = "::mlir::multigpu";
  let genSpecializedAttr = 1;
}

def ShardKind_Uniform  : I32EnumAttrCase<"Uniform",   0, "uniform">;
def ShardKind_Uneven   : I32EnumAttrCase<"Uneven",    1, "uneven">;
def ShardKind_Replicate: I32EnumAttrCase<"Replicate", 2, "replicate">;

def ShardKindAttr : I32EnumAttr<"ShardKind",
    "Describes how a buffer is distributed(sharded) across GPUs",
    [ShardKind_Uniform, ShardKind_Uneven, ShardKind_Replicate]> {
  let cppNamespace = "::mlir::multigpu";
  let genSpecializedAttr = 1;
}

def DeviceConfigAttr : MultiGpu_Attr<"DeviceConfig", "device_config"> {
  let summary = "Compile-time GPU configuration for the module (at the moment, just the number of GPUs and their IDs)";
  let description = [{
    A structured attribute recording the GPU cluster configuration known at 
    compile time. 

    Fields:
    - `numDevices`: The total count of GPUs available.
    - `deviceIds`: Optional explicit physical device indices for non-contiguous 
      sets (e.g., [0, 2, 4, 6]).

    Passes retrieve this info through the `mgpu.device_config` symbol op.

    Example — 4 contiguous GPUs:
    ```mlir
    #mgpu.device_config<numDevices = 4>
    ```
    Example — 4 GPUs with specific indices:
    ```mlir
    #mgpu.device_config<numDevices = 4, deviceIds = [0, 2, 4, 6]>
    ```
  }];

  let parameters = (ins
    "uint32_t":$numDevices,
    OptionalParameter<"DenseI32ArrayAttr">:$deviceIds
  );

  let assemblyFormat = [{
    `<` `numDevices` `=` $numDevices
    (`,` `deviceIds` `=` $deviceIds^)?
    `>`
  }];

  let genVerifyDecl = 1;
  // numDevices >= 1; len(deviceIds) == numDevices
  let extraClassDeclaration = [{
    /// True when deviceIds is absent (devices are [0, numDevices)).
    bool hasDefaultDeviceIds() const;
    /// Returns the system device index for device `r` (0-based).
    /// Equivalent to `r` when deviceIds is absent.
    int32_t getSystemDeviceId(uint32_t r) const;
  }];
}

// class DeviceConfig_Attr : Attr<CPred<"llvm::isa<DeviceConfigAttr>($_self)">, "device config attribute">;
def DeviceConfig_Attr : Attr<CPred<"$_self.isa<::mlir::multigpu::DeviceConfigAttr>()">, 
                             "device config attribute"> {
  let storageType = "::mlir::multigpu::DeviceConfigAttr";
  let returnType = "::mlir::multigpu::DeviceConfigAttr";
  let convertFromStorage = "$_self";
}


def GetDeviceOp : MultiGpu_Op<"get_device", [Pure]> {
  let summary = "Get a handle to a GPU device by index";
  let arguments = (ins Index:$device_index);
  let results = (outs Device:$device);
  let description = [{
    Returns a handle to the device at the index.
    The index must be in the range [0, numDevices) specified by the `mgpu.device_config` attribute in the module.
  }];
  // mgpu.get_device %idx : !mgpu.device
  let assemblyFormat = "$device_index attr-dict `:` type($device)";
}

def CreateStreamOp : MultiGpu_Op<"create_stream", [RecursiveMemoryEffects]> {
  let summary = "Create a new stream for a given device";
  let arguments = (ins Device:$device);
  let results = (outs Stream:$stream);
  let description = [{
    Creates an execution stream on a device.
    Streams are the mechanism provided to enable parallel execution.
    Each stream maintains its own queue of operations and executes them in order, but operations on 
    different streams may run in parallel.
    In other words, each stream can be thought of as a separate thread of execution.
    The returned stream handle must be destroyed with `mgpu.destroy_stream` when no longer needed.
  }];
  let assemblyFormat = "$device attr-dict `:` type($device) `->` type($stream)";
}

def DestroyStreamOp : MultiGpu_Op<"destroy_stream", [RecursiveMemoryEffects]> {
  let summary = "Destroy a stream";
  let arguments = (ins Stream:$stream);
  let description = [{
    Releases the resources associated with a stream.
    All work enqueued on the stream should complete before destroying it.
    Destroying a stream with pending work is invalid.
  }];
  // mgpu.destroy_stream %stream : !mgpu.stream
  let assemblyFormat = "$stream attr-dict `:` type($stream)";
}

def CreateCommunicatorOp
    : MultiGpu_Op<"create_communicator", [RecursiveMemoryEffects]> {
  let summary = "Create a communicator group for collective operations over a list of devices.";
  let arguments = (ins Variadic<Device>:$devices);
  let results = (outs Communicator:$comm);
  let description = [{
    Establishes a communication group over a set of devices for collective 
    operations (all_reduce, all_gather, broadcast, etc.). The ordering of 
    devices determines each device's index within the communicator (first device 
    is index 0, second is index 1, etc.).
    All collective operations using this communicator must be called by all GPUs included in the operation.
  }];
  // mgpu.create_communicator %d0, %d1 : !mgpu.device, !mgpu.device -> !mgpu.communicator
  let assemblyFormat = "$devices attr-dict `:` type($devices) `->` type($comm)";
}

def DeviceConfigOp : MultiGpu_Op<"device_config",
    [Pure, Symbol, HasParent<"::mlir::ModuleOp">]> {
  let summary = "Symbolic declaration of the cluster configuration";
  let description = [{
    Wraps `DeviceConfigAttr` to allow passes to do 
    `SymbolTable` lookups. This op is just metadata.

    Only one `mgpu.device_config` op should appear per module.

    Example — 4 contiguous GPUs:
    ```mlir
    mgpu.device_config @config #mgpu.device_config<numDevices = 4>
    ```
    Example — 4 GPUs with specific indices:
    ```mlir
    mgpu.device_config @config
        #mgpu.device_config<numDevices = 4, deviceIds = [0, 2, 4, 6]>
    ```
  }];

  let arguments = (ins
    SymbolNameAttr:$sym_name,
    DeviceConfig_Attr:$config
  );
  let results = (outs);
  // mgpu.device_config @sym #mgpu.device_config<...>
  let hasCustomAssemblyFormat = 1;

  let extraClassDeclaration = [{
    /// Convenience accessor — the primary field most passes need.
    uint32_t getNumDevices() { return getConfig().getNumDevices(); }
  }];
}


def AllocOp : MultiGpu_Op<"alloc", [RecursiveMemoryEffects]> {
  let summary = "Allocate device memory";
  let arguments = (ins Device:$device);
  let results = (outs AnyMemRef:$ptr);
  let description = [{
    Allocates memory for a device.
  }];
  // mgpu.alloc %dev : !mgpu.device -> memref<1024xf32>
  let assemblyFormat = "$device attr-dict `:` type($device) `->` type($ptr)";
}

def FreeOp : MultiGpu_Op<"free", [RecursiveMemoryEffects]> {
  let summary = "Free device memory";
  let arguments = (ins AnyMemRef:$ptr, Device:$device);
  let description = [{
    Releases memory allocated for a device. The pointer must 
    have been allocated on the same device.
  }];
  // mgpu.free %ptr, %dev : memref<1024xf32>, !mgpu.device
  let assemblyFormat = "$ptr `,` $device attr-dict `:` type($ptr) `,` type($device)";
}

def MemcpyOp : MultiGpu_Op<"memcpy", [RecursiveMemoryEffects]> {
  let summary = "Copy memory between devices";
  let arguments = (ins AnyMemRef:$dst, AnyMemRef:$src, Device:$dstDevice,
                       Device:$srcDevice, Optional<Stream>:$stream);
  let description = [{
    Copies the contents of `src` to `dst`.
    The source and destination memrefs should have the same element type and number of elements.
    Supports copying between different devices (peer-to-peer) or 
    between host and device memory.

    If `stream` is provided the copy is enqueued on that stream (async);
    otherwise it executes synchronously (blocking the host until the copy completes).
  }];
  let assemblyFormat = [{
    $dst `,` $src `,` $dstDevice `,` $srcDevice
    (`stream` $stream^ `:` type($stream))?
    attr-dict `:` type($dst) `,` type($src) `,` type($dstDevice) `,` type($srcDevice)
  }];
}

// Lowering sketch:
//   sliceSize = src.shape[axis] / len(devices) 
//   for i, dev in enumerate(devices):
//     offset = i * sliceSize
//     slice  = subview src [..., offset, ...][..., sliceSize, ...]
//     mgpu.memcpy dstBufs[i], slice, dev, srcDevice [stream streams[i]]
//
//   - src.shape[axis] % len(devices) == 0.
//   - dstBufs[i].shape[axis] == src.shape[axis] / len(devices) for all i.
//   - All dstBufs share the same element type as src.
//   - len(dstBufs) == len(devices).
//   - len(streams) == len(devices) when streams are provided.
def ScatterOp : MultiGpu_Op<"scatter",
    [RecursiveMemoryEffects, AttrSizedOperandSegments]> {
  let summary = "Distribute a buffer evenly across multiple GPUs";
  let description = [{
    Splits a buffer along `axis` into equal slices and copies them to 
    the destination buffers on target GPUs. Requires the axis 
    dimension to be divisible by the number of GPUs. Copies may be 
    issued concurrently on the optional `streams`.

    Example — split [1024xf32] from the host across 4 GPUs along axis 0:
    ```mlir
    mgpu.scatter %hostBuf, %hostDev,
                 %dBuf0, %dBuf1, %dBuf2, %dBuf3,
                 %dev0, %dev1, %dev2, %dev3
        axis = 0
        : memref<1024xf32>, !mgpu.device,
          memref<256xf32>, memref<256xf32>, memref<256xf32>, memref<256xf32>,
          !mgpu.device, !mgpu.device, !mgpu.device, !mgpu.device
    ```
  }];

  let arguments = (ins
    AnyMemRef:$src,
    Device:$srcDevice,
    Variadic<AnyMemRef>:$dstBufs,
    Variadic<Device>:$devices,
    Variadic<Stream>:$streams,
    IndexAttr:$axis
  );
  let results = (outs);

  let assemblyFormat = [{
    $src `,` $srcDevice `,` `[` $dstBufs `]` `,` `[` $devices `]`
    (`streams` `[` $streams^ `]` `:` type($streams))?
    `axis` `=` $axis
    attr-dict `:` type($src) `,` type($srcDevice) `,`
    `[` type($dstBufs) `]` `,` `[` type($devices) `]`
  }];

  let extraClassDeclaration = [{
    /// Returns the per-device slice size along the scatter axis.
    int64_t getSliceSize() const;
  }];
}

//   - sum(sliceSizes) == src.shape[axis].
//   - len(sliceSizes) == len(devices) == len(dstBufs).
//   - dstBufs[i].shape[axis] == sliceSizes[i].
//   - All dstBufs share the same element type as src.
// ---------------------------------------------------------------------------
def UnevenScatterOp : MultiGpu_Op<"uneven_scatter",
    [RecursiveMemoryEffects, AttrSizedOperandSegments]> {
  let summary = "Distribute a buffer across devices with asymmetric slice sizes";
  let description = [{
    Similar to `mgpu.scatter`, but the size of data to be distributed to each GPU is specified. 
    Copies may be issued concurrently on the optional `streams`.

    Example — partition [100xf32] across 3 GPUs as [34, 33, 33]:
    ```mlir
    mgpu.uneven_scatter %src, %srcDev,
                        %dBuf0, %dBuf1, %dBuf2,
                        %dev0, %dev1, %dev2
        sliceSizes = [34, 33, 33] axis = 0
        : memref<100xf32>, !mgpu.device,
          memref<34xf32>, memref<33xf32>, memref<33xf32>,
          !mgpu.device, !mgpu.device, !mgpu.device
    ```
  }];

  let arguments = (ins
    AnyMemRef:$src,
    Device:$srcDevice,
    Variadic<AnyMemRef>:$dstBufs,
    Variadic<Device>:$devices,
    Variadic<Stream>:$streams,
    DenseI64ArrayAttr:$sliceSizes,
    IndexAttr:$axis
  );
  let results = (outs);

  let assemblyFormat = [{
    $src `,` $srcDevice `,` `[` $dstBufs `]` `,` `[` $devices `]`
    (`streams` `[` $streams^ `]` `:` type($streams))?
    `sliceSizes` `=` $sliceSizes `axis` `=` $axis
    attr-dict `:` type($src) `,` type($srcDevice) `,`
    `[` type($dstBufs) `]` `,` `[` type($devices) `]`
  }];
}

// Lowering sketch:
//   offset = 0
//   for i, dev in enumerate(devices):
//     sliceSize = srcBufs[i].shape[axis]
//     slice = subview dst [..., offset, ...][..., sliceSize, ...]
//     mgpu.memcpy slice, srcBufs[i], dstDevice, dev [stream streams[i]]
//     offset += sliceSize
//
//   - sum(srcBufs[i].shape[axis]) == dst.shape[axis].
//   - All srcBufs share the same element type as dst.
//   - len(srcBufs) == len(devices).
//   - len(streams) == len(devices) when streams are provided.
// ---------------------------------------------------------------------------
def GatherOp : MultiGpu_Op<"gather",
    [RecursiveMemoryEffects, AttrSizedOperandSegments]> {
  let summary = "Concatenate device-local shards into a single destination";
  let description = [{
    The opposite of `mgpu.scatter`. Copies shard buffers from multiple 
    devices and combines them along `axis`. Supports uneven shards. Copies may be issued 
    concurrently on the optional `streams`.

    Example — gather 4x[256xf32] back into [1024xf32] on the host:
    ```mlir
    mgpu.gather %hostBuf, %hostDev,
                %dBuf0, %dBuf1, %dBuf2, %dBuf3,
                %dev0, %dev1, %dev2, %dev3
        axis = 0
        : memref<1024xf32>, !mgpu.device,
          memref<256xf32>, memref<256xf32>, memref<256xf32>, memref<256xf32>,
          !mgpu.device, !mgpu.device, !mgpu.device, !mgpu.device
    ```
  }];

  let arguments = (ins
    AnyMemRef:$dst,
    Device:$dstDevice,
    Variadic<AnyMemRef>:$srcBufs,
    Variadic<Device>:$devices,
    Variadic<Stream>:$streams,
    IndexAttr:$axis
  );
  let results = (outs);

  let assemblyFormat = [{
    $dst `,` $dstDevice `,` `[` $srcBufs `]` `,` `[` $devices `]`
    (`streams` `[` $streams^ `]` `:` type($streams))?
    `axis` `=` $axis
    attr-dict `:` type($dst) `,` type($dstDevice) `,`
    `[` type($srcBufs) `]` `,` `[` type($devices) `]`
  }];
}

// Lowering sketch:
//   for i, dev in enumerate(devices):
//     mgpu.memcpy dstBufs[i], src, dev, srcDevice [stream streams[i]]
//
//   - All dstBufs have the same type as src.
//   - len(dstBufs) == len(devices).
//   - len(streams) == len(devices) when streams are provided.
// ---------------------------------------------------------------------------
def ReplicateOp : MultiGpu_Op<"replicate",
    [RecursiveMemoryEffects, AttrSizedOperandSegments]> {
  let summary = "The CPU copies a buffer to every GPU";
  let description = [{
    The CPU copies a buffer to all target GPUs. 
    All destination buffers should have the same type as `src`. 
    Copies may be issued concurrently on the optional `streams`.

    Example — the CPU copies a [512xf32] weight buffer to 4 GPUs:
    ```mlir
    mgpu.replicate %weights, %hostDev,
                   %dev0, %dev1, %dev2, %dev3,
                   %wBuf0, %wBuf1, %wBuf2, %wBuf3
        : memref<512xf32>, !mgpu.device,
          !mgpu.device, !mgpu.device, !mgpu.device, !mgpu.device,
          memref<512xf32>, memref<512xf32>, memref<512xf32>, memref<512xf32>
    ```
  }];

  let arguments = (ins
    AnyMemRef:$src,
    Device:$srcDevice,
    Variadic<Device>:$devices,
    Variadic<AnyMemRef>:$dstBufs,
    Variadic<Stream>:$streams
  );
  let results = (outs);

  let assemblyFormat = [{
    $src `,` $srcDevice `,` `[` $devices `]` `,` `[` $dstBufs `]`
    (`streams` `[` $streams^ `]` `:` type($streams))?
    attr-dict `:` type($src) `,` type($srcDevice) `,`
    `[` type($devices) `]` `,` `[` type($dstBufs) `]`
  }];
}

//   - len(devices) == number of result values.
//   - For uniform sharding: result[i].shape[shardAxis] ==
//     logicalShape[shardAxis] / len(devices).
// ---------------------------------------------------------------------------
def ShardedAllocOp : MultiGpu_Op<"sharded_alloc",
    [RecursiveMemoryEffects]> {
  let summary = "Grouped allocation for memory allocation across multiple GPUs";
  let description = [{
    Allocates one shard buffer per GPU.

    Example — 4 shards of [1024x512xf32] along axis 0:
    ```mlir
    %s0, %s1, %s2, %s3 = mgpu.sharded_alloc %dev0, %dev1, %dev2, %dev3
        logicalShape = [1024, 512] shardAxis = 0 shardKind = uniform
        : !mgpu.device, !mgpu.device, !mgpu.device, !mgpu.device
        -> memref<256x512xf32>, memref<256x512xf32>,
           memref<256x512xf32>, memref<256x512xf32>
    ```
  }];

  let arguments = (ins
    Variadic<Device>:$devices,
    DenseI64ArrayAttr:$logicalShape,
    IndexAttr:$shardAxis,
    ShardKindAttr:$shardKind
  );
  let results = (outs Variadic<AnyMemRef>:$shards);

  let assemblyFormat = [{
    $devices
    `logicalShape` `=` $logicalShape
    `shardAxis` `=` $shardAxis
    `shardKind` `=` $shardKind
    attr-dict `:` type($devices) `->` type($shards)
  }];
}

// len(shards) == len(devices).
// ---------------------------------------------------------------------------
def ShardedFreeOp : MultiGpu_Op<"sharded_free",
    [RecursiveMemoryEffects, AttrSizedOperandSegments]> {
  let summary = "Grouped free of GPU shard buffers";
  let description = [{
    Free the buffers allocated on the group of GPUs. Each buffer in 
    `shards` is freed on the corresponding GPU in `devices`.

    Example:
    ```mlir
    mgpu.sharded_free %s0, %s1, %s2, %s3, %dev0, %dev1, %dev2, %dev3
        : memref<256x512xf32>, memref<256x512xf32>,
          memref<256x512xf32>, memref<256x512xf32>,
          !mgpu.device, !mgpu.device, !mgpu.device, !mgpu.device
    ```
  }];

  let arguments = (ins
    Variadic<AnyMemRef>:$shards,
    Variadic<Device>:$devices
  );
  let results = (outs);

  let assemblyFormat = [{
    `[` $shards `]` `,` `[` $devices `]`
    attr-dict `:` `[` type($shards) `]` `,` `[` type($devices) `]`
  }];
}


def LaunchOp : MultiGpu_Op<"launch", [RecursiveMemoryEffects]> {
  let summary = "Launch a kernel on a device";
  let description = [{
    Launches a region as a kernel on the given device. The region 
    body contains the kernel code that will execute on the GPU.
    Directly lowered to a GPU kernel.

    `grid` is the number of blocks in each dimension .
    `block` is the number of threads per block.
    The total number of 
    threads launched is the product of grid and block.

    If `stream` is provided the kernel is enqueued on that stream (async);
    otherwise it executes synchronously (blocking the host until completion).
  }];

  let arguments = (ins Device:$device, Variadic<Index>:$grid,
                       Variadic<Index>:$block, Optional<Stream>:$stream);
  let results = (outs);
  let traits = [RecursiveMemoryEffects, AttrSizedOperandSegments];
  let regions = (region SizedRegion<1>:$kernelRegion);
  let hasCustomAssemblyFormat = 1;

  let extraClassDeclaration = [{
    unsigned getGridDimensions() const;
    unsigned getBlockDimensions() const;
  }];
}

def TerminatorOp : MultiGpu_Op<"terminator", [HasParent<"LaunchOp">,
                                               Pure, Terminator]>,
    Arguments<(ins)>, Results<(outs)> {
  let summary = "Terminator for mgpu.launch regions";
  let description = [{
    A terminator for the region of a `mgpu.launch`.
    The region returns no values.
  }];
  let assemblyFormat = "attr-dict";
}


def SyncDeviceOp : MultiGpu_Op<"sync_device", [RecursiveMemoryEffects]> {
  let summary = "Synchronize on a GPU";
  let arguments = (ins Device:$device);
  let description = [{
    Blocks the host until all work submitted to a GPU
    has completed.
    This operation waits for all work submitted on a GPU to complete.
    Use `mgpu.sync_stream` to synchronize individual streams.
  }];
  // mgpu.sync_device %dev : !mgpu.device
  let assemblyFormat = "$device attr-dict `:` type($device)";
}

def SyncStreamOp : MultiGpu_Op<"sync_stream", [RecursiveMemoryEffects]> {
  let summary = "Synchronize a single stream";
  let arguments = (ins Stream:$stream);
  let description = [{
    Blocks the host until all work submitted to a stream has completed.
  }];
  // mgpu.sync_stream %stream : !mgpu.stream
  let assemblyFormat = "$stream attr-dict `:` type($stream)";
}

def StreamWaitOp : MultiGpu_Op<"stream_wait", [RecursiveMemoryEffects]> {
  let summary = "Make one stream wait on another";
  let arguments = (ins Stream:$waitingStream, Variadic<Stream>:$waitOnStreams);
  let description = [{
    Wait for all streams in `waitOnStreams` to complete.
    Work submitted to `waitingStream`
    after this op will not begin until all recorded work on the waited-on
    streams has completed.
    This enables more fine-grained synchronization rather than waiting for the entire GPU to complete.
    Streams may reside on different GPUs(and it's almost expected for this dialect).
  }];
  // mgpu.stream_wait %target, %src0, %src1 : !mgpu.stream, ...
  let assemblyFormat = [{
    $waitingStream `,` $waitOnStreams attr-dict `:`
    type($waitingStream) `,` type($waitOnStreams)
  }];
}


def AllReduceOp : MultiGpu_Op<"all_reduce", [RecursiveMemoryEffects]> {
  let summary = "All-reduce across all GPUs in a communication group";
  let arguments = (ins AnyMemRef:$sendBuf, AnyMemRef:$recvBuf,
                       Communicator:$comm, ReductionKindAttr:$reductionKind,
                       Optional<Stream>:$stream);
  let description = [{
    Each GPU sends `sendBuf` and receives the result
    in `recvBuf`.
    `reductionKind` is the reduction operation (for example,sum, prod, 
    min, max).
    Send and receive buffers must have the same type, so a reduction with  (sendBuf == recvBuf) is valid.
    All GPUs must  
    call this operation.

    Example:
    ```mlir
    mgpu.all_reduce %send, %recv, %comm, sum
        : memref<256xf32>, memref<256xf32>, !mgpu.communicator
    ```
  }];
  let hasCustomAssemblyFormat = 1;
}

def AllGatherOp : MultiGpu_Op<"all_gather", [RecursiveMemoryEffects]> {
  let summary = "All-gather across all GPUs in a communicator";
  let arguments = (ins AnyMemRef:$sendBuf, AnyMemRef:$recvBuf,
                       Communicator:$comm, Optional<Stream>:$stream);
  let description = [{
    Each GPU sends `sendBuf` and every GPU receives the result in `recvBuf`.
    The types of the buffers must match.
    GPU i's contribution appears at 
    position i in the result. 
  }];
  let assemblyFormat = [{
    $sendBuf `,` $recvBuf `,` $comm
    (`stream` $stream^ `:` type($stream))?
    attr-dict `:` type($sendBuf) `,` type($recvBuf) `,` type($comm)
  }];
}

// def ReduceScatterOp : MultiGpu_Op<"reduce_scatter", [RecursiveMemoryEffects]> {
//   let summary = "Reduce-scatter across all GPUs in a communicator";
//   let arguments = (ins AnyMemRef:$sendBuf, AnyMemRef:$recvBuf,
//                        Communicator:$comm, ReductionKindAttr:$reductionKind,
//                        Optional<Stream>:$stream);
//   let description = [{
//     The inverse of `mgpu.all_gather`. Each GPU contributes `sendBuf`; the
//     element-wise reduction is computed and the result is scattered so that
//     GPU i receives the i-th slice into `recvBuf`. `reductionKind` names
//     the reduction operation. The leading dimension of `sendBuf` must equal 
//     N x the corresponding dimension of `recvBuf`, where N is the number of 
//     GPUs. This operation combines reduction and scattering in a single 
//     communication step, often more efficient than separate reduce and scatter 
//     operations.

//     Example:
//     ```mlir
//     mgpu.reduce_scatter %send, %recv, %comm, sum
//         : memref<1024xf32>, memref<256xf32>, !mgpu.communicator
//     ```
//   }];
//   let hasCustomAssemblyFormat = 1;
// }

def BroadcastOp : MultiGpu_Op<"broadcast", [RecursiveMemoryEffects]> {
  let summary = "Broadcast from a GPU to all GPUs in a communicator";
  let arguments = (ins AnyMemRef:$buf, Communicator:$comm, Index:$rootRank,
                       Optional<Stream>:$stream);
  let description = [{
    The `buf` is copied to every other GPU's `buf`. All GPUs
    must call this op with the same root and buffer shape.
    This would usually be used to distribute model parameters, configuration data, or other 
    data from a single source to all devices.
    The GPU's buffer serves as the source, and all other GPUs 
    receive the same data.
  }];
  let assemblyFormat = [{
    $buf `,` $comm `,` $rootRank
    (`stream` $stream^ `:` type($stream))?
    attr-dict `:` type($buf) `,` type($comm)
  }];
}

def SendOp : MultiGpu_Op<"send", [RecursiveMemoryEffects]> {
  let summary = "Point-to-point send";
  let arguments = (ins AnyMemRef:$buf, Communicator:$comm, Index:$dstRank,
                       Optional<Stream>:$stream);
  let description = [{
    Sends the contents of `buf` to the specified destination GPU within the 
    communication group.
    The destination GPU must have a matching `mgpu.recv` operation.
    If the send and receive operations are not paired correctly then it will cause deadlocks.
  }];
  let assemblyFormat = [{
    $buf `,` $comm `,` $dstRank
    (`stream` $stream^ `:` type($stream))?
    attr-dict `:` type($buf) `,` type($comm)
  }];
}

def RecvOp : MultiGpu_Op<"recv", [RecursiveMemoryEffects]> {
  let summary = "Point-to-point receive";
  let arguments = (ins AnyMemRef:$buf, Communicator:$comm, Index:$srcRank,
                       Optional<Stream>:$stream);
  let description = [{
    Receives data from the specified source GPU into `buf`.
    The buffer must have the same shape and element type as the buffer sent by the source GPU.
    If the send and receive operations are not paired correctly then it will cause deadlocks.
  }];
  let assemblyFormat = [{
    $buf `,` $comm `,` $srcRank
    (`stream` $stream^ `:` type($stream))?
    attr-dict `:` type($buf) `,` type($comm)
  }];
}

#endif


