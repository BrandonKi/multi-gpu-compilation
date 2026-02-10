#ifndef MULTIGPU_OPS
#define MULTIGPU_OPS

include "mlir/IR/OpBase.td"
include "mlir/Interfaces/SideEffectInterfaces.td"

def MultiGpu_Dialect : Dialect {
  let name = "mgpu";
  let description = [{
    The `mgpu` dialect models multi-GPU computation including memory management,
    kernel launches with stream-based async execution, device
    synchronization, and collective communication primitives.
  }];
  let cppNamespace = "::mlir::multigpu";
  let extraClassDeclaration = [{
    Type parseType(DialectAsmParser &parser) const;
    void printType(Type type, DialectAsmPrinter &printer) const;
  }];
}

class MultiGpu_Op<string mnemonic, list<Trait> traits = []>
    : Op<MultiGpu_Dialect, mnemonic, traits>;

//----------------------------------------------------------------------------

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

//----------------------------------------------------------------------------

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

//----------------------------------------------------------------------------

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
  // mgpu.memcpy %dst, %src, %dstDev, %srcDev [stream %s] : memref<...>, memref<...>
  let assemblyFormat = [{
    $dst `,` $src `,` $dstDevice `,` $srcDevice
    (`stream` $stream^ `:` type($stream))?
    attr-dict `:` type($dst) `,` type($src) `,` type($dstDevice) `,` type($srcDevice)
  }];
}

//----------------------------------------------------------------------------

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
  let summary = "Terminator for mgpu launch regions.";
  let description = [{
    A terminator operation for regions that appear in the body of `mgpu.launch`
    operation. These regions are not expected to return any value so the
    terminator takes no operands.
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
    `waitingStream` wait on all of them. Work submitted to `waitingStream` after
    this op will not begin until all recorded work on the waited-on streams has
    completed. The waiting stream and the waited-on streams may reside on
    different devices (cross-device stream synchronization).
  }];
  // mgpu.stream_wait %target, %src0, %src1 : !mgpu.stream, ...
  let assemblyFormat = "$waitingStream `,` $waitOnStreams attr-dict `:` type($waitingStream) `,` type($waitOnStreams)";
}

//----------------------------------------------------------------------------

def AllReduceOp : MultiGpu_Op<"all_reduce", [RecursiveMemoryEffects]> {
  let summary = "All-reduce across all ranks in a communicator";
  let arguments = (ins AnyMemRef:$sendBuf, AnyMemRef:$recvBuf,
                       Communicator:$comm, StrAttr:$op,
                       Optional<Stream>:$stream);
  let description = [{
    Each rank contributes `sendBuf` and receives the element-wise reduction
    into `recvBuf`. `op` names the reduction operation (e.g. "sum", "max",
    "min", "prod"). Send and receive buffers must have the same type; in-place
    reduction (sendBuf == recvBuf) is permitted.
  }];
  let assemblyFormat = [{
    $sendBuf `,` $recvBuf `,` $comm `,` $op
    (`stream` $stream^ `:` type($stream))?
    attr-dict `:` type($sendBuf) `,` type($recvBuf) `,` type($comm)
  }];
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
                       Communicator:$comm, StrAttr:$op,
                       Optional<Stream>:$stream);
  let description = [{
    The inverse of AllGatherOp. Each rank contributes `sendBuf`; the
    element-wise reduction is computed and the result is scattered so that
    rank i receives the i-th slice into `recvBuf`. `op` names the reduction.
  }];
  let assemblyFormat = [{
    $sendBuf `,` $recvBuf `,` $comm `,` $op
    (`stream` $stream^ `:` type($stream))?
    attr-dict `:` type($sendBuf) `,` type($recvBuf) `,` type($comm)
  }];
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
