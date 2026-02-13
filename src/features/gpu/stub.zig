//! GPU Stub Module
//!
//! This module provides API-compatible no-op implementations for all public GPU
//! functions when the GPU feature is disabled at compile time. All functions
//! return `error.GpuDisabled` or empty/default values as appropriate.
//!
//! The GPU module encompasses:
//! - Multi-backend GPU acceleration (CUDA, Vulkan, Metal, WebGPU, OpenGL, FPGA)
//! - Device discovery and management
//! - Memory allocation and buffer operations
//! - Kernel compilation and execution
//! - Stream and event synchronization
//! - Multi-GPU coordination and load balancing
//! - Lock-free memory pools for LLM workloads
//! - Performance profiling and metrics
//!
//! To enable the real implementation, build with `-Denable-gpu=true`.
//! To select specific backends, use `-Dgpu-backend=vulkan,cuda` (comma-separated).

const std = @import("std");
const config_module = @import("../../core/config/mod.zig");
const stub_common = @import("../../services/shared/stub_common.zig");

// ============================================================================
// Errors
// ============================================================================

pub const Error = error{
    GpuDisabled,
    NoDeviceAvailable,
    InitializationFailed,
    InvalidConfig,
    OutOfMemory,
    KernelCompilationFailed,
    KernelExecutionFailed,
} || stub_common.CommonError;

pub const GpuError = Error;
pub const MemoryError = Error;
pub const KernelError = Error;
pub const AcceleratorError = Error;
pub const CodegenError = Error;
pub const CompileError = Error;

// ============================================================================
// Local Stubs Imports
// ============================================================================

pub const backend = @import("stubs/backend.zig");
pub const device = @import("stubs/device.zig");
pub const memory = @import("stubs/memory.zig");
pub const stream = @import("stubs/stream.zig");
pub const kernel = @import("stubs/kernel.zig");
pub const dsl_mod = @import("stubs/dsl.zig");
pub const execution = @import("stubs/execution.zig");
pub const profiler = @import("stubs/profiler.zig");
pub const recovery_mod = @import("stubs/recovery.zig");
pub const multi_gpu = @import("stubs/multi_gpu.zig");
pub const config = @import("stubs/config.zig");
pub const platform_mod = @import("stubs/platform.zig");
pub const backend_factory_mod = @import("stubs/backend_factory.zig");
pub const dispatcher_mod = @import("stubs/dispatcher.zig");
pub const diagnostics_mod = @import("stubs/diagnostics.zig");
pub const execution_coordinator_mod = @import("stubs/execution_coordinator.zig");
pub const std_gpu_mod = @import("stubs/std_gpu.zig");
pub const misc = @import("stubs/misc.zig");

// ============================================================================
// Backend Re-exports
// ============================================================================

pub const Backend = backend.Backend;
pub const BackendInfo = backend.BackendInfo;
pub const DetectionLevel = backend.DetectionLevel;
pub const BackendAvailability = backend.BackendAvailability;
pub const Summary = backend.Summary;

// ============================================================================
// Device Re-exports
// ============================================================================

pub const Device = device.Device;
pub const DeviceType = device.DeviceType;
pub const DeviceInfo = device.DeviceInfo;
pub const DeviceCapability = device.DeviceCapability;
pub const DeviceFeature = device.DeviceFeature;
pub const DeviceSelector = device.DeviceSelector;
pub const DeviceManager = device.DeviceManager;
pub const Vendor = device.Vendor;

// ============================================================================
// Memory Re-exports
// ============================================================================

pub const Buffer = memory.Buffer;
pub const GpuBuffer = Buffer;
pub const UnifiedBuffer = memory.UnifiedBuffer;
pub const BufferFlags = memory.BufferFlags;
pub const BufferOptions = memory.BufferOptions;
pub const BufferView = memory.BufferView;
pub const BufferStats = memory.BufferStats;
pub const MappedBuffer = memory.MappedBuffer;
pub const MemoryPool = memory.MemoryPool;
pub const GpuMemoryPool = MemoryPool;
pub const MemoryStats = memory.MemoryStats;
pub const MemoryInfo = memory.MemoryInfo;
pub const MemoryMode = memory.MemoryMode;
pub const MemoryLocation = memory.MemoryLocation;
pub const AccessHint = memory.AccessHint;
pub const ElementType = memory.ElementType;
pub const AsyncTransfer = memory.AsyncTransfer;

// ============================================================================
// Stream Re-exports
// ============================================================================

pub const Stream = stream.Stream;
pub const GpuStream = Stream;
pub const StreamOptions = stream.StreamOptions;
pub const StreamPriority = stream.StreamPriority;
pub const StreamFlags = stream.StreamFlags;
pub const StreamState = stream.StreamState;
pub const StreamManager = stream.StreamManager;
pub const Event = stream.Event;
pub const EventOptions = stream.EventOptions;
pub const EventFlags = stream.EventFlags;
pub const EventState = stream.EventState;

// ============================================================================
// Kernel Re-exports
// ============================================================================

pub const KernelBuilder = kernel.KernelBuilder;
pub const KernelIR = kernel.KernelIR;
pub const KernelSource = kernel.KernelSource;
pub const KernelConfig = kernel.KernelConfig;
pub const CompiledKernel = kernel.CompiledKernel;
pub const KernelCache = kernel.KernelCache;
pub const KernelCacheConfig = kernel.KernelCacheConfig;
pub const CacheStats = kernel.CacheStats;
pub const PortableKernelSource = kernel.PortableKernelSource;

// ============================================================================
// DSL Re-exports
// ============================================================================

pub const dsl = dsl_mod.dsl;
pub const ScalarType = dsl_mod.ScalarType;
pub const VectorType = dsl_mod.VectorType;
pub const MatrixType = dsl_mod.MatrixType;
pub const AddressSpace = dsl_mod.AddressSpace;
pub const DslType = dsl_mod.DslType;
pub const AccessMode = dsl_mod.AccessMode;
pub const Expr = dsl_mod.Expr;
pub const BinaryOp = dsl_mod.BinaryOp;
pub const UnaryOp = dsl_mod.UnaryOp;
pub const BuiltinFn = dsl_mod.BuiltinFn;
pub const BuiltinVar = dsl_mod.BuiltinVar;
pub const Stmt = dsl_mod.Stmt;
pub const GeneratedSource = dsl_mod.GeneratedSource;
pub const CompileOptions = dsl_mod.CompileOptions;

pub fn compile(_: std.mem.Allocator, _: *const KernelIR, _: Backend, _: CompileOptions) Error!GeneratedSource {
    return error.GpuDisabled;
}

pub fn compileToKernelSource(_: std.mem.Allocator, _: *const KernelIR, _: Backend, _: CompileOptions) Error!KernelSource {
    return error.GpuDisabled;
}

pub fn compileAll(_: std.mem.Allocator, _: *const KernelIR, _: CompileOptions) Error![]GeneratedSource {
    return error.GpuDisabled;
}

// ============================================================================
// Execution Re-exports
// ============================================================================

pub const LaunchConfig = execution.LaunchConfig;
pub const ExecutionResult = execution.ExecutionResult;
pub const ExecutionStats = execution.ExecutionStats;
pub const HealthStatus = execution.HealthStatus;
pub const GpuStats = execution.GpuStats;
pub const MatrixDims = execution.MatrixDims;
pub const MultiGpuConfig = execution.MultiGpuConfig;
pub const LoadBalanceStrategy = execution.LoadBalanceStrategy;
pub const ReduceResult = execution.ReduceResult;
pub const DotProductResult = execution.DotProductResult;

// ============================================================================
// Profiler Re-exports
// ============================================================================

pub const Profiler = profiler.Profiler;
pub const TimingResult = profiler.TimingResult;
pub const OccupancyResult = profiler.OccupancyResult;
pub const MemoryBandwidth = profiler.MemoryBandwidth;
pub const MetricsSummary = profiler.MetricsSummary;
pub const KernelMetrics = profiler.KernelMetrics;
pub const MetricsCollector = profiler.MetricsCollector;

// ============================================================================
// Recovery Re-exports
// ============================================================================

pub const recovery = recovery_mod.recovery;
pub const failover = recovery_mod.failover;
pub const RecoveryManager = recovery_mod.RecoveryManager;
pub const FailoverManager = recovery_mod.FailoverManager;

// ============================================================================
// Multi-GPU Re-exports
// ============================================================================

pub const DeviceGroup = multi_gpu.DeviceGroup;
pub const WorkDistribution = multi_gpu.WorkDistribution;
pub const GroupStats = multi_gpu.GroupStats;
pub const GPUCluster = multi_gpu.GPUCluster;
pub const GPUClusterConfig = multi_gpu.GPUClusterConfig;
pub const ReduceOp = multi_gpu.ReduceOp;
pub const AllReduceAlgorithm = multi_gpu.AllReduceAlgorithm;
pub const ParallelismStrategy = multi_gpu.ParallelismStrategy;
pub const ModelPartition = multi_gpu.ModelPartition;
pub const DeviceBarrier = multi_gpu.DeviceBarrier;
pub const GradientBucket = multi_gpu.GradientBucket;
pub const GradientBucketManager = multi_gpu.GradientBucketManager;

// ============================================================================
// Config Re-exports
// ============================================================================

pub const GpuConfig = config.GpuConfig;

// ============================================================================
// Platform Re-exports
// ============================================================================

pub const PlatformCapabilities = platform_mod.PlatformCapabilities;
pub const BackendSupport = platform_mod.BackendSupport;
pub const GpuVendor = platform_mod.GpuVendor;
pub const isCudaSupported = platform_mod.isCudaSupported;
pub const isMetalSupported = platform_mod.isMetalSupported;
pub const isVulkanSupported = platform_mod.isVulkanSupported;
pub const isWebGpuSupported = platform_mod.isWebGpuSupported;
pub const platformDescription = platform_mod.platformDescription;

// ============================================================================
// Backend Factory Re-exports
// ============================================================================

pub const BackendFactory = backend_factory_mod.BackendFactory;
pub const BackendInstance = backend_factory_mod.BackendInstance;
pub const BackendFeature = backend_factory_mod.BackendFeature;
pub const createBackend = backend_factory_mod.createBackend;
pub const createBestBackend = backend_factory_mod.createBestBackend;
pub const destroyBackend = backend_factory_mod.destroyBackend;

// ============================================================================
// Dispatcher Re-exports
// ============================================================================

pub const KernelDispatcher = dispatcher_mod.KernelDispatcher;
pub const DispatchError = dispatcher_mod.DispatchError;
pub const CompiledKernelHandle = dispatcher_mod.CompiledKernelHandle;
pub const KernelArgs = dispatcher_mod.KernelArgs;

// ============================================================================
// Diagnostics Re-exports
// ============================================================================

pub const DiagnosticsInfo = diagnostics_mod.DiagnosticsInfo;
pub const ErrorContext = diagnostics_mod.ErrorContext;
pub const GpuErrorCode = diagnostics_mod.GpuErrorCode;
pub const GpuErrorType = diagnostics_mod.GpuErrorType;

// ============================================================================
// Execution Coordinator Re-exports
// ============================================================================

pub const ExecutionCoordinator = execution_coordinator_mod.ExecutionCoordinator;
pub const ExecutionMethod = execution_coordinator_mod.ExecutionMethod;

// ============================================================================
// std.gpu Re-exports (Zig native GPU address spaces and shader built-ins)
// ============================================================================

pub const GlobalPtr = std_gpu_mod.GlobalPtr;
pub const SharedPtr = std_gpu_mod.SharedPtr;
pub const StoragePtr = std_gpu_mod.StoragePtr;
pub const UniformPtr = std_gpu_mod.UniformPtr;
pub const ConstantPtr = std_gpu_mod.ConstantPtr;
pub const globalInvocationId = std_gpu_mod.globalInvocationId;
pub const workgroupId = std_gpu_mod.workgroupId;
pub const localInvocationId = std_gpu_mod.localInvocationId;
pub const workgroupBarrier = std_gpu_mod.workgroupBarrier;
pub const setLocalSize = std_gpu_mod.setLocalSize;

// ============================================================================
// Peer Transfer Re-exports
// ============================================================================

pub const PeerTransferManager = misc.peer_transfer.PeerTransferManager;
pub const TransferCapability = misc.peer_transfer.TransferCapability;
pub const TransferHandle = misc.peer_transfer.TransferHandle;
pub const TransferStatus = misc.peer_transfer.TransferStatus;
pub const TransferOptions = misc.peer_transfer.TransferOptions;
pub const TransferError = misc.peer_transfer.TransferError;
pub const TransferStats = misc.peer_transfer.TransferStats;
pub const DeviceBuffer = misc.peer_transfer.DeviceBuffer;
pub const RecoveryStrategy = misc.peer_transfer.RecoveryStrategy;

// ============================================================================
// Mega GPU Orchestration Re-exports
// ============================================================================

pub const MegaCoordinator = misc.mega.Coordinator;
pub const MegaBackendInstance = misc.mega.BackendInstance;
pub const MegaWorkloadProfile = misc.mega.WorkloadProfile;
pub const MegaWorkloadCategory = misc.mega.WorkloadCategory;
pub const MegaScheduleDecision = misc.mega.ScheduleDecision;
pub const MegaPrecision = misc.mega.Precision;

// ============================================================================
// Sync/Performance Re-exports
// ============================================================================

pub const SyncEvent = misc.sync_event.SyncEvent;
pub const KernelRing = misc.kernel_ring.KernelRing;
pub const AdaptiveTiling = misc.adaptive_tiling.AdaptiveTiling;
pub const TileConfig = misc.adaptive_tiling.AdaptiveTiling.TileConfig;

// ============================================================================
// Sub-module Namespace Stubs
// ============================================================================

pub const profiling = misc.profiling;
pub const occupancy = misc.occupancy;
pub const fusion = misc.fusion;
pub const execution_coordinator = misc.execution_coordinator;
pub const memory_pool_advanced = misc.memory_pool_advanced;
pub const memory_pool_lockfree = misc.memory_pool_lockfree;
pub const sync_event = misc.sync_event;
pub const kernel_ring = misc.kernel_ring;
pub const adaptive_tiling = misc.adaptive_tiling;
pub const std_gpu = misc.std_gpu;
pub const std_gpu_kernels = misc.std_gpu_kernels;
pub const unified = misc.unified;
pub const unified_buffer = misc.unified_buffer;
pub const interface = misc.interface;
pub const cuda_loader = misc.cuda_loader;
pub const builtin_kernels = misc.builtin_kernels;
pub const diagnostics = misc.diagnostics_ns;
pub const error_handling = misc.error_handling;
pub const multi_device = misc.multi_device;
pub const peer_transfer = misc.peer_transfer;
pub const mega = misc.mega;
pub const platform = misc.platform_ns;
pub const backend_factory = misc.backend_factory_ns;
pub const dispatcher = misc.dispatcher_ns;

// ============================================================================
// Lock-free memory pool stubs
// ============================================================================

pub const CACHE_LINE_SIZE: usize = 64;
pub const INVALID_HANDLE: ResourceHandle = .{ .value = std.math.maxInt(u64) };

pub const ResourceHandle = struct {
    value: u64,

    pub fn init(idx: u32, gen: u32) ResourceHandle {
        return .{ .value = (@as(u64, gen) << 32) | @as(u64, idx) };
    }

    pub fn index(self: ResourceHandle) u32 {
        return @truncate(self.value);
    }

    pub fn generation(self: ResourceHandle) u32 {
        return @truncate(self.value >> 32);
    }

    pub fn isValid(self: ResourceHandle) bool {
        return self.value != INVALID_HANDLE.value;
    }
};

pub const LockFreePoolConfig = struct {
    max_slots: u32 = 1024,
    slot_size: usize = 65536,
    enable_thread_local_cache: bool = true,
    thread_local_cache_size: usize = 8,
    preallocate: bool = false,
};

pub const LockFreePoolStats = struct {
    total_allocations: u64 = 0,
    total_deallocations: u64 = 0,
    active_allocations: u64 = 0,
    peak_allocations: u64 = 0,
    failed_allocations: u64 = 0,
    invalid_accesses: u64 = 0,

    pub fn utilizationRatio(self: LockFreePoolStats, max_slots: usize) f64 {
        if (max_slots == 0) return 0.0;
        return @as(f64, @floatFromInt(self.active_allocations)) / @as(f64, @floatFromInt(max_slots));
    }

    pub fn allocationSuccessRate(self: LockFreePoolStats) f64 {
        const total = self.total_allocations + self.failed_allocations;
        if (total == 0) return 1.0;
        return @as(f64, @floatFromInt(self.total_allocations)) / @as(f64, @floatFromInt(total));
    }
};

pub const LockFreeResourcePool = struct {
    pub fn init(_: std.mem.Allocator, _: LockFreePoolConfig) Error!LockFreeResourcePool {
        return error.GpuDisabled;
    }

    pub fn deinit(_: *LockFreeResourcePool) void {}

    pub fn allocate(_: *LockFreeResourcePool) Error!ResourceHandle {
        return error.GpuDisabled;
    }

    pub fn free(_: *LockFreeResourcePool, _: ResourceHandle) bool {
        return false;
    }

    pub fn get(_: *LockFreeResourcePool, _: ResourceHandle) ?*Buffer {
        return null;
    }

    pub fn validateHandle(_: *const LockFreeResourcePool, _: ResourceHandle) bool {
        return false;
    }

    pub fn getStats(_: *const LockFreeResourcePool) LockFreePoolStats {
        return .{};
    }

    pub fn freeSlotCount(_: *const LockFreeResourcePool) u64 {
        return 0;
    }
};

pub const ConcurrentCommandPool = struct {
    pub const CommandBuffer = struct {
        pub fn reset(_: *CommandBuffer) void {}
        pub fn write(_: *CommandBuffer, _: []const u8) Error!void {
            return error.GpuDisabled;
        }
        pub fn getWritten(_: *const CommandBuffer) []const u8 {
            return &.{};
        }
    };

    pub fn init(_: std.mem.Allocator, _: usize, _: usize) Error!ConcurrentCommandPool {
        return error.GpuDisabled;
    }

    pub fn deinit(_: *ConcurrentCommandPool) void {}

    pub fn acquire(_: *ConcurrentCommandPool) ?*CommandBuffer {
        return null;
    }

    pub fn release(_: *ConcurrentCommandPool, _: *CommandBuffer) void {}
};

// ============================================================================
// Gpu struct
// ============================================================================

pub const Gpu = struct {
    pub fn init(_: std.mem.Allocator, _: GpuConfig) Error!Gpu {
        return error.GpuDisabled;
    }
    pub fn deinit(_: *Gpu) void {}
    pub fn isAvailable(_: *const Gpu) bool {
        return false;
    }
    pub fn getActiveDevice(_: *const Gpu) ?*const Device {
        return null;
    }
    pub fn createBuffer(_: *Gpu, _: usize, _: BufferOptions) Error!*Buffer {
        return error.GpuDisabled;
    }
    pub fn createBufferFromSlice(_: *Gpu, comptime _: type, _: anytype, _: BufferOptions) Error!*Buffer {
        return error.GpuDisabled;
    }
    pub fn destroyBuffer(_: *Gpu, _: *Buffer) void {}
    pub fn vectorAdd(_: *Gpu, _: *Buffer, _: *Buffer, _: *Buffer) Error!ExecutionResult {
        return error.GpuDisabled;
    }
    pub fn reduceSum(_: *Gpu, _: *Buffer) Error!ReduceResult {
        return error.GpuDisabled;
    }
    pub fn dotProduct(_: *Gpu, _: *Buffer, _: *Buffer) Error!DotProductResult {
        return error.GpuDisabled;
    }
    pub fn getStats(_: *const Gpu) GpuStats {
        return .{};
    }
    pub fn getMetricsSummary(_: *const Gpu) ?MetricsSummary {
        return null;
    }
    pub fn getMemoryInfo(_: *const Gpu) MemoryInfo {
        return .{};
    }
    pub fn selectDevice(_: *Gpu, _: DeviceSelector) Error!void {
        return error.GpuDisabled;
    }
    pub fn enableMultiGpu(_: *Gpu, _: MultiGpuConfig) Error!void {
        return error.GpuDisabled;
    }
    pub fn getDeviceGroup(_: *Gpu) ?*DeviceGroup {
        return null;
    }
    pub fn distributeWork(_: *Gpu, _: usize) Error![]WorkDistribution {
        return error.GpuDisabled;
    }
    pub fn compileKernel(_: *Gpu, _: PortableKernelSource) Error!CompiledKernel {
        return error.GpuDisabled;
    }
    pub fn launchKernel(_: *Gpu, _: *const CompiledKernel, _: LaunchConfig, _: anytype) Error!ExecutionResult {
        return error.GpuDisabled;
    }
    pub fn synchronize(_: *Gpu) Error!void {
        return error.GpuDisabled;
    }
    pub fn createStream(_: *Gpu, _: StreamOptions) Error!*Stream {
        return error.GpuDisabled;
    }
    pub fn createEvent(_: *Gpu, _: EventOptions) Error!*Event {
        return error.GpuDisabled;
    }
    pub fn isProfilingEnabled(_: *const Gpu) bool {
        return false;
    }
    pub fn enableProfiling(_: *Gpu) void {}
    pub fn disableProfiling(_: *Gpu) void {}
    pub fn getKernelMetrics(_: *Gpu, _: []const u8) ?KernelMetrics {
        return null;
    }
    pub fn getMetricsCollector(_: *Gpu) ?*MetricsCollector {
        return null;
    }
    pub fn resetMetrics(_: *Gpu) void {}
    pub fn isMultiGpuEnabled(_: *const Gpu) bool {
        return false;
    }
    pub fn getMultiGpuStats(_: *const Gpu) ?GroupStats {
        return null;
    }
    pub fn activeDeviceCount(_: *const Gpu) usize {
        return 0;
    }
    pub fn softmax(_: *Gpu, _: *Buffer, _: *Buffer) Error!ExecutionResult {
        return error.GpuDisabled;
    }
    pub fn checkHealth(_: *const Gpu) HealthStatus {
        return .unhealthy;
    }
};

// ============================================================================
// Context - Stub implementation
// ============================================================================

pub const Context = struct {
    pub fn init(_: std.mem.Allocator, _: config_module.GpuConfig) !*Context {
        return error.GpuDisabled;
    }

    pub fn deinit(_: *Context) void {}

    pub fn getGpu(_: *Context) Error!*Gpu {
        return error.GpuDisabled;
    }

    pub fn createBuffer(_: *Context, comptime _: type, _: usize, _: BufferOptions) Error!UnifiedBuffer {
        return error.GpuDisabled;
    }

    pub fn createBufferFromSlice(_: *Context, comptime T: type, _: []const T, _: BufferOptions) Error!UnifiedBuffer {
        return error.GpuDisabled;
    }

    pub fn destroyBuffer(_: *Context, _: *UnifiedBuffer) void {}

    pub fn vectorAdd(_: *Context, _: *UnifiedBuffer, _: *UnifiedBuffer, _: *UnifiedBuffer) Error!ExecutionResult {
        return error.GpuDisabled;
    }

    pub fn matrixMultiply(_: *Context, _: *UnifiedBuffer, _: *UnifiedBuffer, _: *UnifiedBuffer, _: MatrixDims) Error!ExecutionResult {
        return error.GpuDisabled;
    }

    pub fn getHealth(_: *Context) Error!HealthStatus {
        return error.GpuDisabled;
    }
};

// ============================================================================
// Module-level stub functions
// ============================================================================

pub fn isEnabled(_: Backend) bool {
    return false;
}

pub fn isInitialized() bool {
    return false;
}

pub fn init(_: std.mem.Allocator) Error!void {
    return error.GpuDisabled;
}

pub fn deinit() void {}

pub fn ensureInitialized(_: std.mem.Allocator) Error!void {
    return error.GpuDisabled;
}

pub fn isGpuAvailable() bool {
    return false;
}

pub fn getAvailableBackends() []const Backend {
    return &.{};
}

pub fn availableBackends(allocator: std.mem.Allocator) Error![]Backend {
    _ = allocator;
    return error.GpuDisabled;
}

pub fn getBestBackend() Backend {
    return .stdgpu;
}

pub fn listBackendInfo(_: std.mem.Allocator) Error![]BackendInfo {
    return error.GpuDisabled;
}

pub fn listDevices(_: std.mem.Allocator) Error![]DeviceInfo {
    return error.GpuDisabled;
}

pub fn defaultDevice(_: std.mem.Allocator) !?DeviceInfo {
    return null;
}

pub fn discoverDevices(_: std.mem.Allocator) Error![]Device {
    return error.GpuDisabled;
}

pub fn backendName(_: Backend) []const u8 {
    return "disabled";
}

pub fn backendDisplayName(_: Backend) []const u8 {
    return "GPU Disabled";
}

pub fn backendDescription(_: Backend) []const u8 {
    return "GPU feature is disabled at compile time";
}

pub fn backendFromString(_: []const u8) ?Backend {
    return null;
}

pub fn backendSupportsKernels(_: Backend) bool {
    return false;
}

pub fn backendFlag(_: Backend) []const u8 {
    return "disabled";
}

pub fn defaultDeviceLabel() []const u8 {
    return "disabled";
}

pub fn getBestKernelBackend() Backend {
    return .stdgpu;
}

pub fn moduleEnabled() bool {
    return false;
}

pub fn createDefaultKernels(_: std.mem.Allocator) Error!void {
    return error.GpuDisabled;
}

pub fn compileKernel(_: KernelSource, _: KernelConfig) Error!CompiledKernel {
    return error.GpuDisabled;
}

pub fn backendAvailability(_: Backend) BackendAvailability {
    return .{
        .enabled = false,
        .available = false,
        .reason = "gpu module disabled",
        .device_count = 0,
        .level = .none,
    };
}

pub fn summary() Summary {
    return .{
        .module_enabled = false,
        .enabled_backend_count = 0,
        .available_backend_count = 0,
        .device_count = 0,
        .emulated_devices = 0,
    };
}
