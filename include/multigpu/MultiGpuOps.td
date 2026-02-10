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
    The `mgpu` dialect models multi-GPU computation including memory management,
    kernel launches with stream-based async execution, device synchronization,
    and collective communication primitives.
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
  let summary = "A group of devices for collective communication";
}

def Device       : Type<Or<[DeviceType.predicate]>,       "device handle">;
def Stream       : Type<Or<[StreamType.predicate]>,       "stream handle">;
def Communicator : Type<Or<[CommunicatorType.predicate]>, "communicator handle">;

def ReductionKind_Sum  : I32EnumAttrCase<"Sum",  0, "sum">;
def ReductionKind_Prod : I32EnumAttrCase<"Prod", 1, "prod">;
def ReductionKind_Min  : I32EnumAttrCase<"Min",  2, "min">;
def ReductionKind_Max  : I32EnumAttrCase<"Max",  3, "max">;

def ReductionKindAttr : I32EnumAttr<"ReductionKind",
    "Element-wise reduction operation for collective ops",
    [ReductionKind_Sum, ReductionKind_Prod,
     ReductionKind_Min, ReductionKind_Max]> {
  let cppNamespace = "::mlir::multigpu";
  let genSpecializedAttr = 1;
}

def ShardKind_Uniform  : I32EnumAttrCase<"Uniform",   0, "uniform">;
def ShardKind_Uneven   : I32EnumAttrCase<"Uneven",    1, "uneven">;
def ShardKind_Replicate: I32EnumAttrCase<"Replicate", 2, "replicate">;

def ShardKindAttr : I32EnumAttr<"ShardKind",
    "Describes how a buffer is distributed across devices",
    [ShardKind_Uniform, ShardKind_Uneven, ShardKind_Replicate]> {
  let cppNamespace = "::mlir::multigpu";
  let genSpecializedAttr = 1;
}

def DeviceConfigAttr : MultiGpu_Attr<"DeviceConfig", "device_config"> {
  let summary = "Static device count for the compilation unit";
  let description = [{
    Records the number of GPUs in the target row, known at compile time.
    Passes retrieve this via the `mgpu.device_config` symbol op.

    Example — 4 contiguous GPUs:
    ```mlir
    #mgpu.device_config<numDevices = 4>
    ```
    Example — 4 GPUs at non-contiguous system indices:
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
  // Verified: numDevices >= 1; len(deviceIds) == numDevices when present.
  let extraClassDeclaration = [{
    /// True when deviceIds is absent (devices are [0, numDevices)).
    bool hasDefaultDeviceIds() const;
    /// Returns the system device index for logical rank `r` (0-based).
    /// Equivalent to `r` when deviceIds is absent.
    int32_t getSystemDeviceId(uint32_t r) const;
  }];
}

// Type constraint for DeviceConfigAttr
// class DeviceConfig_Attr : Attr<CPred<"llvm::isa<DeviceConfigAttr>($_self)">, "device config attribute">;
def DeviceConfig_Attr : Attr<CPred<"$_self.isa<::mlir::multigpu::DeviceConfigAttr>()">, 
                             "device config attribute"> {
  let storageType = "::mlir::multigpu::DeviceConfigAttr";
  let returnType = "::mlir::multigpu::DeviceConfigAttr";
  let convertFromStorage = "$_self";
}


def GetDeviceOp : MultiGpu_Op<"get_device", [Pure]> {
  let summary = "Get a handle to a GPU device by index";
  let arguments = (ins Index:$index);
  let results = (outs Device:$device);
  let description = [{
    Returns a handle to the device at the given index. The index must be in
    range [0, number_of_devices). Behavior is undefined otherwise.
  }];
  // mgpu.get_device %idx : !mgpu.device
  let assemblyFormat = "$index attr-dict `:` type($device)";
}

def CreateStreamOp : MultiGpu_Op<"create_stream", [RecursiveMemoryEffects]> {
  let summary = "Create a new stream on a device";
  let arguments = (ins Device:$device);
  let results = (outs Stream:$stream);
  let assemblyFormat = "$device attr-dict `:` type($device) `->` type($stream)";
}

def DestroyStreamOp : MultiGpu_Op<"destroy_stream", [RecursiveMemoryEffects]> {
  let summary = "Destroy a stream";
  let arguments = (ins Stream:$stream);
  // mgpu.destroy_stream %stream : !mgpu.stream
  let assemblyFormat = "$stream attr-dict `:` type($stream)";
}

def CreateCommunicatorOp
    : MultiGpu_Op<"create_communicator", [RecursiveMemoryEffects]> {
  let summary = "Create a communicator over a set of devices";
  let arguments = (ins Variadic<Device>:$devices);
  let results = (outs Communicator:$comm);
  let description = [{
    The device list must contain at least one device and must not contain
    duplicates. The ordering of devices determines each device's rank within
    the communicator.
  }];
  // mgpu.create_communicator %d0, %d1 : !mgpu.device, !mgpu.device -> !mgpu.communicator
  let assemblyFormat = "$devices attr-dict `:` type($devices) `->` type($comm)";
}

def DeviceConfigOp : MultiGpu_Op<"device_config",
    [Pure, Symbol, HasParent<"::mlir::ModuleOp">]> {
  let summary = "Declare the static device count for this compilation unit";
  let description = [{
    Attaches a compile-time device configuration to the module. Exactly one
    `mgpu.device_config` op should appear per module; a verifier enforces
    this. Passes query it via `SymbolTable::lookupSymbolIn`.

    The op is purely metadata: it produces no values and has no lowering.

    Example — 4 contiguous GPUs:
    ```mlir
    mgpu.device_config @config #mgpu.device_config<numDevices = 4>
    ```
    Example — 4 GPUs at non-contiguous system indices:
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
    Allocates memory on the given device. The shape and element type of the
    result memref fully determine the allocation size.
  }];
  // mgpu.alloc %dev : !mgpu.device -> memref<1024xf32>
  let assemblyFormat = "$device attr-dict `:` type($device) `->` type($ptr)";
}

def FreeOp : MultiGpu_Op<"free", [RecursiveMemoryEffects]> {
  let summary = "Free device memory";
  let arguments = (ins AnyMemRef:$ptr, Device:$device);
  // mgpu.free %ptr, %dev : memref<1024xf32>, !mgpu.device
  let assemblyFormat = "$ptr `,` $device attr-dict `:` type($ptr) `,` type($device)";
}

def MemcpyOp : MultiGpu_Op<"memcpy", [RecursiveMemoryEffects]> {
  let summary = "Copy memory between devices";
  let arguments = (ins AnyMemRef:$dst, AnyMemRef:$src, Device:$dstDevice,
                       Device:$srcDevice, Optional<Stream>:$stream);
  let description = [{
    Copies the contents of `src` to `dst`. The source and destination memrefs
    must have the same element type and total number of elements; this is
    verified.

    If `stream` is provided the copy is enqueued on that stream (async);
    otherwise it executes synchronously.
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
  let summary = "Scatter-distribute a buffer evenly across devices";
  let description = [{
    Splits `src` along `axis` into `len(devices)` equal slices and copies
    slice `i` to `dstBufs[i]` on `devices[i]`. Copies may be issued
    concurrently on the optional per-device `streams`.

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
  let summary = "Scatter a buffer across devices with explicit per-device slice sizes";
  let description = [{
    Like `mgpu.scatter` but `sliceSizes` specifies each slice's extent along
    `axis`. Slices need not be equal.

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
  let summary = "Gather device-local buffers into a single destination buffer";
  let description = [{
    Copies `srcBufs[i]` from `devices[i]` and writes each into the
    corresponding slice of `dst` on `dstDevice`, concatenating along `axis`.
    The inverse of `mgpu.scatter`. Naturally handles uneven shards since each
    source buffer encodes its own size.

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
  let summary = "Copy a buffer identically to every device";
  let description = [{
    Replicates `src` from `srcDevice` to each of `devices[i]`, writing into
    `dstBufs[i]`. All destination buffers must have the same type as `src`.

    Example — replicate a [512xf32] weight buffer to 4 GPUs:
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
  let summary = "Allocate sharded device buffers for a distributed tensor";
  let description = [{
    Allocates one shard buffer per device. `logicalShape`, `shardAxis`, and
    `shardKind` record the sharding intent for analysis passes and do not
    affect the runtime allocation size.

    Example — 4 uniform shards of [1024x512xf32] along axis 0:
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
  let summary = "Free a set of per-device shard buffers";
  let description = [{
    Releases each buffer in `shards` on the corresponding device in `devices`.

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
    Launches the single-block region as a kernel on the given device.

    `grid` specifies the number of blocks in each dimension (up to 3D).
    `block` specifies the number of threads per block in each dimension
    (up to 3D). Both must have the same length (1-3). Operand segment sizes
    are recorded by the `AttrSizedOperandSegments` trait.

    If `stream` is provided the kernel is enqueued on that stream (async);
    otherwise it executes synchronously.
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
    A terminator for regions that appear in the body of `mgpu.launch`.
    These regions return no values so the terminator takes no operands.
  }];
  let assemblyFormat = "attr-dict";
}


def SyncDeviceOp : MultiGpu_Op<"sync_device", [RecursiveMemoryEffects]> {
  let summary = "Synchronize all streams on a device";
  let arguments = (ins Device:$device);
  let description = [{
    Blocks the host until all work submitted to any stream on the given device
    has completed.
  }];
  // mgpu.sync_device %dev : !mgpu.device
  let assemblyFormat = "$device attr-dict `:` type($device)";
}

def SyncStreamOp : MultiGpu_Op<"sync_stream", [RecursiveMemoryEffects]> {
  let summary = "Synchronize a single stream";
  let arguments = (ins Stream:$stream);
  let description = [{
    Blocks the host until all work submitted to the given stream has completed.
  }];
  // mgpu.sync_stream %stream : !mgpu.stream
  let assemblyFormat = "$stream attr-dict `:` type($stream)";
}

def StreamWaitOp : MultiGpu_Op<"stream_wait", [RecursiveMemoryEffects]> {
  let summary = "Make one stream wait on another";
  let arguments = (ins Stream:$waitingStream, Variadic<Stream>:$waitOnStreams);
  let description = [{
    Records the current state of each stream in `waitOnStreams` and makes
    `waitingStream` wait on all of them. Work submitted to `waitingStream`
    after this op will not begin until all recorded work on the waited-on
    streams has completed. Streams may reside on different devices.
  }];
  // mgpu.stream_wait %target, %src0, %src1 : !mgpu.stream, ...
  let assemblyFormat = [{
    $waitingStream `,` $waitOnStreams attr-dict `:`
    type($waitingStream) `,` type($waitOnStreams)
  }];
}


def AllReduceOp : MultiGpu_Op<"all_reduce", [RecursiveMemoryEffects]> {
  let summary = "All-reduce across all ranks in a communicator";
  let arguments = (ins AnyMemRef:$sendBuf, AnyMemRef:$recvBuf,
                       Communicator:$comm, ReductionKindAttr:$reductionKind,
                       Optional<Stream>:$stream);
  let description = [{
    Each rank contributes `sendBuf` and receives the element-wise reduction
    into `recvBuf`. `reductionKind` names the reduction (sum, prod, min, max,
    and, or, xor). Send and receive buffers must have the same type; in-place
    reduction (sendBuf == recvBuf) is permitted.

    Example:
    ```mlir
    mgpu.all_reduce %send, %recv, %comm, sum
        : memref<256xf32>, memref<256xf32>, !mgpu.communicator
    ```
  }];
  let hasCustomAssemblyFormat = 1;
}

def AllGatherOp : MultiGpu_Op<"all_gather", [RecursiveMemoryEffects]> {
  let summary = "All-gather across all ranks in a communicator";
  let arguments = (ins AnyMemRef:$sendBuf, AnyMemRef:$recvBuf,
                       Communicator:$comm, Optional<Stream>:$stream);
  let description = [{
    Each rank contributes `sendBuf`; every rank receives the concatenation of
    all contributions into `recvBuf`. The leading dimension of `recvBuf` must
    equal N x the corresponding dimension of `sendBuf`, where N is the number
    of ranks. Element types must match.
  }];
  let assemblyFormat = [{
    $sendBuf `,` $recvBuf `,` $comm
    (`stream` $stream^ `:` type($stream))?
    attr-dict `:` type($sendBuf) `,` type($recvBuf) `,` type($comm)
  }];
}

def ReduceScatterOp : MultiGpu_Op<"reduce_scatter", [RecursiveMemoryEffects]> {
  let summary = "Reduce-scatter across all ranks in a communicator";
  let arguments = (ins AnyMemRef:$sendBuf, AnyMemRef:$recvBuf,
                       Communicator:$comm, ReductionKindAttr:$reductionKind,
                       Optional<Stream>:$stream);
  let description = [{
    The inverse of AllGatherOp. Each rank contributes `sendBuf`; the
    element-wise reduction is computed and the result is scattered so that
    rank i receives the i-th slice into `recvBuf`. `reductionKind` names
    the reduction operation.

    Example:
    ```mlir
    mgpu.reduce_scatter %send, %recv, %comm, sum
        : memref<1024xf32>, memref<256xf32>, !mgpu.communicator
    ```
  }];
  let hasCustomAssemblyFormat = 1;
}

def BroadcastOp : MultiGpu_Op<"broadcast", [RecursiveMemoryEffects]> {
  let summary = "Broadcast from a root rank to all ranks";
  let arguments = (ins AnyMemRef:$buf, Communicator:$comm, Index:$rootRank,
                       Optional<Stream>:$stream);
  let description = [{
    The root rank's `buf` is copied to every other rank's `buf`. All ranks
    must call this op with the same root and buffer shape.
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
  let assemblyFormat = [{
    $buf `,` $comm `,` $srcRank
    (`stream` $stream^ `:` type($stream))?
    attr-dict `:` type($buf) `,` type($comm)
  }];
}

#endif


