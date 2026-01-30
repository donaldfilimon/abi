//! GPU Module - Hardware Acceleration API
//!
//! This module provides a unified interface for GPU compute operations across
//! multiple backends including CUDA, Vulkan, Metal, WebGPU, OpenGL, and std.gpu.
//!
//! ## Overview
//!
//! The GPU module abstracts away backend differences, allowing you to write
//! portable GPU code that runs on any supported hardware. Key features include:
//!
//! - **Backend Auto-detection**: Automatically selects the best available backend
//! - **Unified Buffer API**: Cross-platform memory management
//! - **Kernel DSL**: Write portable kernels that compile to any backend
//! - **Execution Coordinator**: Automatic fallback from GPU to SIMD to scalar
//! - **Multi-device Support**: Manage multiple GPUs with peer-to-peer transfers
//! - **Profiling**: Built-in timing and occupancy analysis
//!
//! ## Available Backends
//!
//! | Backend | Platform | Build Flag |
//! |---------|----------|------------|
//! | CUDA | NVIDIA GPUs | `-Dgpu-backend=cuda` |
//! | Vulkan | Cross-platform | `-Dgpu-backend=vulkan` |
//! | Metal | Apple devices | `-Dgpu-backend=metal` |
//! | WebGPU | Web/Native | `-Dgpu-backend=webgpu` |
//! | OpenGL | Legacy support | `-Dgpu-backend=opengl` |
//! | std.gpu | Zig native | `-Dgpu-backend=stdgpu` |
//!
//! ## Public API
//!
//! These exports form the stable interface:
//! - `Gpu` - Main unified GPU context
//! - `GpuConfig` - Configuration for GPU initialization
//! - `UnifiedBuffer` - Cross-backend buffer type
//! - `Device`, `DeviceType` - Device discovery and selection
//! - `KernelBuilder`, `KernelIR` - DSL for custom kernels
//! - `Backend`, `BackendAvailability` - Backend detection
//!
//! ## Quick Start
//!
//! ```zig
//! const abi = @import("abi");
//!
//! // Initialize framework with GPU
//! var fw = try abi.Framework.init(allocator, .{
//!     .gpu = .{ .backend = .auto },  // Auto-detect best backend
//! });
//! defer fw.deinit();
//!
//! // Get GPU context
//! const gpu_ctx = try fw.getGpu();
//! const gpu = gpu_ctx.getGpu();
//!
//! // Create buffers
//! var a = try gpu.createBufferFromSlice(f32, &[_]f32{ 1, 2, 3, 4 }, .{});
//! var b = try gpu.createBufferFromSlice(f32, &[_]f32{ 5, 6, 7, 8 }, .{});
//! var result = try gpu.createBuffer(f32, 4, .{});
//! defer {
//!     gpu.destroyBuffer(&a);
//!     gpu.destroyBuffer(&b);
//!     gpu.destroyBuffer(&result);
//! }
//!
//! // Perform vector addition
//! _ = try gpu.vectorAdd(&a, &b, &result);
//! ```
//!
//! ## Standalone Usage
//!
//! ```zig
//! const gpu = abi.gpu;
//!
//! var g = try gpu.Gpu.init(allocator, .{
//!     .preferred_backend = .vulkan,
//!     .allow_fallback = true,
//! });
//! defer g.deinit();
//!
//! // Check device capabilities
//! const health = try g.getHealth();
//! std.debug.print("Backend: {t}\n", .{health.backend});
//! std.debug.print("Memory: {} MB\n", .{health.memory_total / (1024 * 1024)});
//! ```
//!
//! ## Custom Kernels
//!
//! ```zig
//! const kernel = gpu.KernelBuilder.init()
//!     .name("my_kernel")
//!     .addParam(.{ .name = "input", .type = .buffer_f32 })
//!     .addParam(.{ .name = "output", .type = .buffer_f32 })
//!     .setBody(
//!         \\output[gid] = input[gid] * 2.0;
//!     )
//!     .build();
//!
//! // Compile for all backends
//! const sources = try gpu.compileAll(kernel);
//! ```
//!
//! ## Internal (do not depend on)
//!
//! These may change without notice:
//! - Direct backend module imports (cuda_loader, vulkan_*, etc.)
//! - Lifecycle management internals (gpu_lifecycle, cuda_backend_init_lock)
//! - Backend-specific initialization functions (initCudaComponents, etc.)
const std = @import("std");
const backend = @import("backend.zig");
const kernels = @import("runtime_kernels.zig");
const memory = @import("memory.zig");
const kernel_cache = @import("kernel_cache.zig");
pub const profiling = @import("profiling.zig");

// Performance optimization modules
pub const occupancy = @import("occupancy.zig");
pub const fusion = @import("fusion.zig");
pub const execution_coordinator = @import("execution_coordinator.zig");
pub const memory_pool_advanced = @import("memory_pool_advanced.zig");
pub const sync_event = @import("sync_event.zig");
pub const kernel_ring = @import("kernel_ring.zig");
pub const adaptive_tiling = @import("adaptive_tiling.zig");

// std.gpu integration (Zig 0.16+ native GPU support)
pub const std_gpu = @import("std_gpu.zig");
pub const std_gpu_kernels = @import("std_gpu_kernels.zig");

// Performance types
pub const SyncEvent = sync_event.SyncEvent;
pub const KernelRing = kernel_ring.KernelRing;
pub const AdaptiveTiling = adaptive_tiling.AdaptiveTiling;
pub const TileConfig = adaptive_tiling.AdaptiveTiling.TileConfig;

// std.gpu types (Zig native GPU address spaces and shader built-ins)
pub const GlobalPtr = std_gpu.GlobalPtr;
pub const SharedPtr = std_gpu.SharedPtr;
pub const StoragePtr = std_gpu.StoragePtr;
pub const UniformPtr = std_gpu.UniformPtr;
pub const ConstantPtr = std_gpu.ConstantPtr;
pub const globalInvocationId = std_gpu.globalInvocationId;
pub const workgroupId = std_gpu.workgroupId;
pub const localInvocationId = std_gpu.localInvocationId;
pub const workgroupBarrier = std_gpu.workgroupBarrier;
pub const setLocalSize = std_gpu.setLocalSize;

// Unified API modules
pub const unified = @import("unified.zig");
pub const unified_buffer = @import("unified_buffer.zig");
pub const device = @import("device.zig");
pub const stream = @import("stream.zig");
pub const dsl = @import("dsl/mod.zig");

// Backend interface and loaders
pub const interface = @import("interface.zig");
pub const cuda_loader = @import("backends/cuda/loader.zig");

// Platform detection
pub const platform = @import("platform.zig");
pub const PlatformCapabilities = platform.PlatformCapabilities;
pub const BackendSupport = platform.BackendSupport;
pub const GpuVendor = platform.GpuVendor;
pub const isCudaSupported = platform.isCudaSupported;
pub const isMetalSupported = platform.isMetalSupported;
pub const isVulkanSupported = platform.isVulkanSupported;
pub const isWebGpuSupported = platform.isWebGpuSupported;
pub const platformDescription = platform.platformDescription;

// Modular backend abstraction layer
pub const backend_factory = @import("backend_factory.zig");
pub const dispatcher = @import("dispatcher.zig");
pub const builtin_kernels = @import("builtin_kernels.zig");

// Factory convenience exports
pub const BackendFactory = backend_factory.BackendFactory;
pub const BackendInstance = backend_factory.BackendInstance;
pub const BackendFeature = backend_factory.BackendFeature;
pub const createBackend = backend_factory.createBackend;
pub const createBestBackend = backend_factory.createBestBackend;
pub const destroyBackend = backend_factory.destroyBackend;

// Dispatcher convenience exports
pub const KernelDispatcher = dispatcher.KernelDispatcher;
pub const DispatchError = dispatcher.DispatchError;
pub const CompiledKernelHandle = dispatcher.CompiledKernelHandle;
pub const KernelArgs = dispatcher.KernelArgs;

// Recovery and failover
pub const recovery = @import("recovery.zig");
pub const failover = @import("failover.zig");
pub const RecoveryManager = recovery.RecoveryManager;
pub const FailoverManager = failover.FailoverManager;

// Diagnostics and error handling
pub const diagnostics = @import("diagnostics.zig");
pub const error_handling = @import("error_handling.zig");
pub const DiagnosticsInfo = diagnostics.DiagnosticsInfo;
pub const ErrorContext = error_handling.ErrorContext;
pub const GpuErrorCode = error_handling.GpuErrorCode;
pub const GpuErrorType = error_handling.GpuErrorType;

// Execution coordinator convenience exports (GPU→SIMD→scalar fallback)
pub const ExecutionCoordinator = execution_coordinator.ExecutionCoordinator;
pub const ExecutionMethod = execution_coordinator.ExecutionMethod;

// Multi-device and peer transfer
pub const multi_device = @import("multi_device.zig");
pub const peer_transfer = @import("peer_transfer/mod.zig");

// Multi-device types
pub const DeviceGroup = multi_device.DeviceGroup;
pub const GPUCluster = multi_device.GPUCluster;
pub const GPUClusterConfig = multi_device.GPUClusterConfig;
pub const ReduceOp = multi_device.ReduceOp;
pub const AllReduceAlgorithm = multi_device.AllReduceAlgorithm;
pub const ParallelismStrategy = multi_device.ParallelismStrategy;
pub const WorkDistribution = multi_device.WorkDistribution;
pub const ModelPartition = multi_device.ModelPartition;
pub const DeviceBarrier = multi_device.DeviceBarrier;
pub const GradientBucket = multi_device.GradientBucket;
pub const GradientBucketManager = multi_device.GradientBucketManager;

// Peer transfer types
pub const PeerTransferManager = peer_transfer.PeerTransferManager;
pub const TransferCapability = peer_transfer.TransferCapability;
pub const TransferHandle = peer_transfer.TransferHandle;
pub const TransferStatus = peer_transfer.TransferStatus;
pub const TransferOptions = peer_transfer.TransferOptions;
pub const TransferError = peer_transfer.TransferError;
pub const TransferStats = peer_transfer.TransferStats;
pub const DeviceBuffer = peer_transfer.DeviceBuffer;
pub const RecoveryStrategy = peer_transfer.RecoveryStrategy;

// Mega GPU orchestration (cross-backend coordinator)
pub const mega = @import("mega/mod.zig");

// Mega types for cross-backend coordination
pub const MegaCoordinator = mega.Coordinator;
pub const MegaBackendInstance = mega.BackendInstance;
pub const MegaWorkloadProfile = mega.WorkloadProfile;
pub const MegaWorkloadCategory = mega.WorkloadCategory;
pub const MegaScheduleDecision = mega.ScheduleDecision;
pub const MegaPrecision = mega.Precision;

// Include test modules in test builds
comptime {
    if (@import("builtin").is_test) {
        _ = @import("tests/device_enumeration_test.zig");
        _ = @import("tests/backend_detection_test.zig");
        _ = @import("tests/std_gpu_test.zig");
        _ = @import("tests/execution_fallback_test.zig");
        _ = @import("tests/integration_test.zig");
        _ = @import("tests/all_backends_test.zig");
        _ = @import("peer_transfer/tests.zig");
        // Performance optimization module tests
        _ = @import("sync_event.zig");
        _ = @import("kernel_ring.zig");
        _ = @import("adaptive_tiling.zig");
        // std.gpu integration tests
        _ = @import("std_gpu.zig");
        _ = @import("std_gpu_kernels.zig");
        // Mega GPU orchestration tests
        _ = @import("mega/mod.zig");
        // Platform detection tests
        _ = @import("platform.zig");
    }
}

const build_options = @import("build_options");

// Import lifecycle management from shared utils
const lifecycle = @import("../shared/utils.zig");
const SimpleModuleLifecycle = lifecycle.SimpleModuleLifecycle;
const LifecycleError = lifecycle.LifecycleError;

var gpu_lifecycle = SimpleModuleLifecycle{};

var cuda_backend_init_lock = std.Thread.Mutex{};
var cuda_backend_initialized = false;
var cached_gpu_allocator: ?std.mem.Allocator = null;

pub const MemoryError = memory.MemoryError;
pub const BufferFlags = memory.BufferFlags;
pub const GpuBuffer = memory.GpuBuffer;
pub const Buffer = GpuBuffer; // Alias for convenience
pub const GpuMemoryPool = memory.GpuMemoryPool;
pub const MemoryPool = GpuMemoryPool; // Alias for convenience
pub const MemoryStats = memory.MemoryStats;
pub const AsyncTransfer = memory.AsyncTransfer;
pub const GpuError = memory.MemoryError || error{GpuDisabled};

pub const KernelSource = kernels.KernelSource;
pub const KernelConfig = kernels.KernelConfig;
pub const CompiledKernel = kernels.CompiledKernel;
pub const Stream = kernels.Stream;
pub const KernelError = kernels.KernelError;
pub const compileKernel = kernels.compileKernel;
pub const createDefaultKernels = kernels.createDefaultKernels;

// Kernel Cache exports
pub const KernelCache = kernel_cache.KernelCache;
pub const KernelCacheConfig = kernel_cache.KernelCacheConfig;
pub const CacheStats = kernel_cache.CacheStats;

pub const Backend = backend.Backend;
pub const DetectionLevel = backend.DetectionLevel;
pub const BackendAvailability = backend.BackendAvailability;
pub const BackendInfo = backend.BackendInfo;
pub const DeviceCapability = backend.DeviceCapability;
pub const DeviceInfo = backend.DeviceInfo;
pub const Summary = backend.Summary;
pub const backendName = backend.backendName;
pub const backendDisplayName = backend.backendDisplayName;
pub const backendDescription = backend.backendDescription;
pub const backendFlag = backend.backendFlag;
pub const backendFromString = backend.backendFromString;
pub const backendSupportsKernels = backend.backendSupportsKernels;
pub const backendAvailability = backend.backendAvailability;
pub const availableBackends = backend.availableBackends;
pub const listBackendInfo = backend.listBackendInfo;
pub const listDevices = backend.listDevices;
pub const defaultDevice = backend.defaultDevice;
pub const defaultDeviceLabel = backend.defaultDeviceLabel;
pub const summary = backend.summary;
pub const moduleEnabled = backend.moduleEnabled;
pub const isEnabled = backend.isEnabled;

// Profiling exports
pub const Profiler = profiling.Profiler;
pub const TimingResult = profiling.TimingResult;
pub const OccupancyResult = profiling.OccupancyResult;
pub const MemoryBandwidth = profiling.MemoryBandwidth;

// ============================================================================
// Unified API Exports
// ============================================================================

// Main Gpu struct and config
pub const Gpu = unified.Gpu;
pub const GpuConfig = unified.GpuConfig;
pub const ExecutionResult = unified.ExecutionResult;
pub const MatrixDims = unified.MatrixDims;
pub const LaunchConfig = unified.LaunchConfig;
pub const HealthStatus = unified.HealthStatus;
pub const GpuStats = unified.GpuStats;
pub const MemoryInfo = unified.MemoryInfo;
pub const MultiGpuConfig = unified.MultiGpuConfig;
pub const LoadBalanceStrategy = unified.LoadBalanceStrategy;

// Unified buffer types
pub const UnifiedBuffer = unified_buffer.Buffer;
pub const BufferOptions = unified_buffer.BufferOptions;
pub const MemoryMode = unified_buffer.MemoryMode;
pub const MemoryLocation = unified_buffer.MemoryLocation;
pub const AccessHint = unified_buffer.AccessHint;
pub const ElementType = unified_buffer.ElementType;
pub const BufferView = unified_buffer.BufferView;
pub const MappedBuffer = unified_buffer.MappedBuffer;
pub const BufferStats = unified_buffer.BufferStats;

// Device types
pub const Device = device.Device;
pub const DeviceType = device.DeviceType;
pub const DeviceFeature = device.DeviceFeature;
pub const DeviceSelector = device.DeviceSelector;
pub const DeviceManager = device.DeviceManager;
pub const discoverDevices = device.discoverDevices;
pub const Vendor = device.Vendor;
pub const getBestKernelBackend = device.getBestKernelBackend;

// Stream and event types
pub const GpuStream = stream.Stream;
pub const StreamOptions = stream.StreamOptions;
pub const StreamPriority = stream.StreamPriority;
pub const StreamFlags = stream.StreamFlags;
pub const StreamState = stream.StreamState;
pub const StreamManager = stream.StreamManager;
pub const Event = stream.Event;
pub const EventOptions = stream.EventOptions;
pub const EventFlags = stream.EventFlags;
pub const EventState = stream.EventState;

// DSL types for custom kernels
pub const KernelBuilder = dsl.KernelBuilder;
pub const KernelIR = dsl.KernelIR;
pub const PortableKernelSource = dsl.PortableKernelSource;
pub const compile = dsl.compile;
pub const compileToKernelSource = dsl.compileToKernelSource;
pub const compileAll = dsl.compileAll;
pub const CompileOptions = dsl.CompileOptions;
pub const CompileError = dsl.CompileError;

// DSL type system
pub const ScalarType = dsl.ScalarType;
pub const VectorType = dsl.VectorType;
pub const MatrixType = dsl.MatrixType;
pub const AddressSpace = dsl.AddressSpace;
pub const DslType = dsl.Type;
pub const AccessMode = dsl.AccessMode;

// DSL expression types
pub const Expr = dsl.Expr;
pub const BinaryOp = dsl.BinaryOp;
pub const UnaryOp = dsl.UnaryOp;
pub const BuiltinFn = dsl.BuiltinFn;
pub const BuiltinVar = dsl.BuiltinVar;

// DSL statement types
pub const Stmt = dsl.Stmt;

// DSL code generation
pub const CodegenError = dsl.CodegenError;
pub const GeneratedSource = dsl.GeneratedSource;

pub fn init(allocator: std.mem.Allocator) GpuError!void {
    if (!moduleEnabled()) return error.GpuDisabled;

    cached_gpu_allocator = allocator;
    gpu_lifecycle.init(initCudaComponents) catch {
        return error.GpuDisabled;
    };
}

fn initCudaComponents() !void {
    if (comptime build_options.gpu_cuda) {
        cuda_backend_init_lock.lock();
        defer cuda_backend_init_lock.unlock();

        if (!cuda_backend_initialized) {
            const cuda_module = @import("backends/cuda/mod.zig");
            const allocator = cached_gpu_allocator orelse return error.OutOfMemory;

            cuda_module.init(allocator) catch |err| {
                std.log.warn("CUDA backend initialization failed: {t}. Using fallback mode.", .{err});
            };

            if (comptime build_options.enable_gpu) {
                const cuda_stream = @import("backends/cuda/stream.zig");
                cuda_stream.init() catch |err| {
                    std.log.warn("CUDA stream initialization failed: {t}", .{err});
                };

                const cuda_memory = @import("backends/cuda/memory.zig");
                cuda_memory.init(allocator) catch |err| {
                    std.log.warn("CUDA memory initialization failed: {t}", .{err});
                };
            }

            cuda_backend_initialized = true;
        }
    }
}

fn deinitCudaComponents() void {
    if (cuda_backend_initialized) {
        if (comptime build_options.gpu_cuda) {
            const cuda_module = @import("backends/cuda/mod.zig");
            cuda_module.deinit();

            if (comptime build_options.enable_gpu) {
                const cuda_stream = @import("backends/cuda/stream.zig");
                cuda_stream.deinit();

                const cuda_memory = @import("backends/cuda/memory.zig");
                cuda_memory.deinit();
            }
        }
        cuda_backend_initialized = false;
    }
}

pub fn ensureInitialized(allocator: std.mem.Allocator) GpuError!void {
    if (!isInitialized()) {
        try init(allocator);
    }
}

pub fn deinit() void {
    deinitCudaComponents();
    gpu_lifecycle.deinit(null);
}

pub fn isInitialized() bool {
    return gpu_lifecycle.isInitialized();
}

// ============================================================================
// Context - Framework Integration
// ============================================================================

const config_module = @import("../config/mod.zig");

/// GPU Context for Framework integration.
///
/// The Context struct wraps the `Gpu` struct to provide a consistent interface
/// with other framework modules. It handles configuration translation and
/// provides convenient access to GPU operations.
///
/// ## Thread Safety
///
/// The Context itself is not thread-safe. For concurrent GPU operations,
/// use the underlying Gpu's stream-based operations or external synchronization.
///
/// ## Example
///
/// ```zig
/// var ctx = try Context.init(allocator, .{ .backend = .vulkan });
/// defer ctx.deinit();
///
/// // Get the underlying Gpu instance
/// const gpu = ctx.getGpu();
///
/// // Create and use buffers
/// var buffer = try ctx.createBuffer(f32, 1024, .{});
/// defer ctx.destroyBuffer(&buffer);
/// ```
pub const Context = struct {
    /// Memory allocator for GPU operations.
    allocator: std.mem.Allocator,
    /// The underlying unified GPU instance.
    gpu: Gpu,

    /// Initialize the GPU context with the given configuration.
    ///
    /// ## Parameters
    ///
    /// - `allocator`: Memory allocator for GPU resources
    /// - `cfg`: GPU configuration (backend selection, memory limits, etc.)
    ///
    /// ## Returns
    ///
    /// A pointer to the initialized Context.
    ///
    /// ## Errors
    ///
    /// - `error.GpuDisabled`: GPU feature is disabled at compile time
    /// - `error.NoDeviceAvailable`: No compatible GPU device found
    /// - `error.OutOfMemory`: Memory allocation failed
    pub fn init(allocator: std.mem.Allocator, cfg: config_module.GpuConfig) !*Context {
        if (!moduleEnabled()) return error.GpuDisabled;

        // Convert config_module.GpuConfig to unified.GpuConfig
        const preferred_backend: ?Backend = switch (cfg.backend) {
            .auto => null,
            .vulkan => .vulkan,
            .cuda => .cuda,
            .metal => .metal,
            .webgpu => .webgpu,
            .opengl => .opengl,
            .fpga => .fpga,
            .cpu => .stdgpu, // CPU fallback uses stdgpu backend
        };

        const gpu_config = GpuConfig{
            .preferred_backend = preferred_backend,
            .allow_fallback = true,
            .max_memory_bytes = cfg.memory_limit orelse 0,
            .enable_profiling = false,
        };

        const ctx = try allocator.create(Context);
        errdefer allocator.destroy(ctx);

        ctx.* = .{
            .allocator = allocator,
            .gpu = try Gpu.init(allocator, gpu_config),
        };
        return ctx;
    }

    pub fn deinit(self: *Context) void {
        self.gpu.deinit();
        self.allocator.destroy(self);
    }

    /// Get the underlying Gpu instance.
    pub fn getGpu(self: *Context) *Gpu {
        return &self.gpu;
    }

    /// Create a buffer.
    pub fn createBuffer(self: *Context, comptime T: type, count: usize, options: BufferOptions) !UnifiedBuffer {
        return self.gpu.createBuffer(T, count, options);
    }

    /// Create a buffer from a slice.
    pub fn createBufferFromSlice(self: *Context, comptime T: type, data: []const T, options: BufferOptions) !UnifiedBuffer {
        return self.gpu.createBufferFromSlice(T, data, options);
    }

    /// Destroy a buffer.
    pub fn destroyBuffer(self: *Context, buffer: *UnifiedBuffer) void {
        self.gpu.destroyBuffer(buffer);
    }

    /// Vector addition.
    pub fn vectorAdd(self: *Context, a: *UnifiedBuffer, b: *UnifiedBuffer, result: *UnifiedBuffer) !ExecutionResult {
        return self.gpu.vectorAdd(a, b, result);
    }

    /// Matrix multiplication.
    pub fn matrixMultiply(self: *Context, a: *UnifiedBuffer, b: *UnifiedBuffer, result: *UnifiedBuffer, dims: MatrixDims) !ExecutionResult {
        return self.gpu.matrixMultiply(a, b, result, dims);
    }

    /// Get GPU health status.
    pub fn getHealth(self: *Context) !HealthStatus {
        return self.gpu.getHealth();
    }
};
