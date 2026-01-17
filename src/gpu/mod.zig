//! GPU Acceleration Module
//!
//! Unified GPU compute interface supporting multiple backends:
//! - CUDA (NVIDIA)
//! - Vulkan (cross-platform)
//! - Metal (Apple)
//! - WebGPU (browser/wasm)
//! - OpenGL (legacy)
//! - CPU fallback (stdgpu)
//!
//! ## Usage
//!
//! ```zig
//! const gpu = @import("gpu/mod.zig");
//!
//! // Initialize GPU context
//! var ctx = try gpu.Context.init(allocator, .{});
//! defer ctx.deinit();
//!
//! // Create buffers
//! var a = try ctx.createBuffer(f32, 1024, .{});
//! var b = try ctx.createBuffer(f32, 1024, .{});
//! var result = try ctx.createBuffer(f32, 1024, .{});
//! defer { ctx.destroyBuffer(&a); ctx.destroyBuffer(&b); ctx.destroyBuffer(&result); }
//!
//! // Execute kernel
//! _ = try ctx.vectorAdd(&a, &b, &result);
//! ```

const std = @import("std");
const build_options = @import("build_options");
const config_module = @import("../config.zig");

// Re-export from compute/gpu for now (gradual migration)
const compute_gpu = @import("../compute/gpu/mod.zig");

// ============================================================================
// Public Types
// ============================================================================

// Core GPU context
pub const Gpu = compute_gpu.Gpu;
pub const GpuConfig = compute_gpu.GpuConfig;

// Backend detection
pub const Backend = compute_gpu.Backend;
pub const BackendInfo = compute_gpu.BackendInfo;
pub const BackendAvailability = compute_gpu.BackendAvailability;
pub const DetectionLevel = compute_gpu.DetectionLevel;

// Device management
pub const Device = compute_gpu.Device;
pub const DeviceType = compute_gpu.DeviceType;
pub const DeviceInfo = compute_gpu.DeviceInfo;
pub const DeviceCapability = compute_gpu.DeviceCapability;
pub const DeviceFeature = compute_gpu.DeviceFeature;
pub const DeviceSelector = compute_gpu.DeviceSelector;
pub const DeviceManager = compute_gpu.DeviceManager;

// Buffers
pub const Buffer = compute_gpu.GpuBuffer;
pub const UnifiedBuffer = compute_gpu.UnifiedBuffer;
pub const BufferFlags = compute_gpu.BufferFlags;
pub const BufferOptions = compute_gpu.BufferOptions;
pub const BufferView = compute_gpu.BufferView;
pub const BufferStats = compute_gpu.BufferStats;
pub const MappedBuffer = compute_gpu.MappedBuffer;

// Memory
pub const MemoryPool = compute_gpu.GpuMemoryPool;
pub const MemoryStats = compute_gpu.MemoryStats;
pub const MemoryInfo = compute_gpu.MemoryInfo;
pub const MemoryMode = compute_gpu.MemoryMode;
pub const MemoryLocation = compute_gpu.MemoryLocation;
pub const MemoryError = compute_gpu.MemoryError;

// Streams and events
pub const Stream = compute_gpu.GpuStream;
pub const StreamOptions = compute_gpu.StreamOptions;
pub const StreamPriority = compute_gpu.StreamPriority;
pub const StreamFlags = compute_gpu.StreamFlags;
pub const StreamState = compute_gpu.StreamState;
pub const StreamManager = compute_gpu.StreamManager;
pub const Event = compute_gpu.Event;
pub const EventOptions = compute_gpu.EventOptions;
pub const EventFlags = compute_gpu.EventFlags;
pub const EventState = compute_gpu.EventState;

// Kernels
pub const KernelBuilder = compute_gpu.KernelBuilder;
pub const KernelIR = compute_gpu.KernelIR;
pub const KernelSource = compute_gpu.KernelSource;
pub const KernelConfig = compute_gpu.KernelConfig;
pub const CompiledKernel = compute_gpu.CompiledKernel;
pub const KernelError = compute_gpu.KernelError;
pub const KernelCache = compute_gpu.KernelCache;
pub const KernelCacheConfig = compute_gpu.KernelCacheConfig;
pub const CacheStats = compute_gpu.CacheStats;
pub const PortableKernelSource = compute_gpu.PortableKernelSource;

// DSL for custom kernels
pub const dsl = compute_gpu.dsl;
pub const ScalarType = compute_gpu.ScalarType;
pub const VectorType = compute_gpu.VectorType;
pub const MatrixType = compute_gpu.MatrixType;
pub const AddressSpace = compute_gpu.AddressSpace;
pub const DslType = compute_gpu.DslType;
pub const AccessMode = compute_gpu.AccessMode;
pub const Expr = compute_gpu.Expr;
pub const BinaryOp = compute_gpu.BinaryOp;
pub const UnaryOp = compute_gpu.UnaryOp;
pub const BuiltinFn = compute_gpu.BuiltinFn;
pub const BuiltinVar = compute_gpu.BuiltinVar;
pub const Stmt = compute_gpu.Stmt;
pub const CodegenError = compute_gpu.CodegenError;
pub const GeneratedSource = compute_gpu.GeneratedSource;
pub const CompileOptions = compute_gpu.CompileOptions;
pub const CompileError = compute_gpu.CompileError;

// Execution
pub const LaunchConfig = compute_gpu.LaunchConfig;
pub const ExecutionResult = compute_gpu.ExecutionResult;
pub const ExecutionStats = compute_gpu.ExecutionStats;
pub const HealthStatus = compute_gpu.HealthStatus;
pub const GpuStats = compute_gpu.GpuStats;
pub const MatrixDims = compute_gpu.MatrixDims;
pub const MultiGpuConfig = compute_gpu.MultiGpuConfig;
pub const LoadBalanceStrategy = compute_gpu.LoadBalanceStrategy;

// Profiling
pub const Profiler = compute_gpu.Profiler;
pub const TimingResult = compute_gpu.TimingResult;
pub const OccupancyResult = compute_gpu.OccupancyResult;
pub const MemoryBandwidth = compute_gpu.MemoryBandwidth;

// Acceleration API
pub const Accelerator = compute_gpu.Accelerator;
pub const AcceleratorConfig = compute_gpu.AcceleratorConfig;
pub const AcceleratorError = compute_gpu.AcceleratorError;
pub const ComputeTask = compute_gpu.ComputeTask;

// Recovery and failover
pub const recovery = compute_gpu.recovery;
pub const failover = compute_gpu.failover;
pub const RecoveryManager = compute_gpu.RecoveryManager;
pub const FailoverManager = compute_gpu.FailoverManager;

// ============================================================================
// Errors
// ============================================================================

pub const GpuError = compute_gpu.GpuError;

pub const Error = error{
    /// GPU feature is disabled at compile time
    GpuDisabled,
    /// No GPU device available
    NoDeviceAvailable,
    /// GPU initialization failed
    InitializationFailed,
    /// Invalid configuration
    InvalidConfig,
    /// Memory allocation failed
    OutOfMemory,
    /// Kernel compilation failed
    KernelCompilationFailed,
    /// Kernel execution failed
    KernelExecutionFailed,
};

// ============================================================================
// Context - New unified interface for Framework integration
// ============================================================================

/// GPU context for Framework integration.
/// Wraps the underlying Gpu instance with lifecycle management.
pub const Context = struct {
    allocator: std.mem.Allocator,
    gpu: ?Gpu = null,
    config: config_module.GpuConfig,

    pub fn init(allocator: std.mem.Allocator, cfg: config_module.GpuConfig) !*Context {
        if (!isModuleEnabled()) return error.GpuDisabled;

        const ctx = try allocator.create(Context);
        errdefer allocator.destroy(ctx);

        ctx.* = .{
            .allocator = allocator,
            .config = cfg,
        };

        // Lazily initialize GPU on first use
        return ctx;
    }

    pub fn deinit(self: *Context) void {
        if (self.gpu) |*g| {
            g.deinit();
        }
        self.allocator.destroy(self);
    }

    /// Get or initialize the underlying GPU instance.
    pub fn getGpu(self: *Context) !*Gpu {
        if (self.gpu) |*g| return g;

        // Convert config to GpuConfig
        const gpu_config = GpuConfig{
            .backend = switch (self.config.backend) {
                .auto => .auto,
                .vulkan => .vulkan,
                .cuda => .cuda,
                .metal => .metal,
                .webgpu => .webgpu,
                .opengl => .opengl,
                .cpu => .cpu,
            },
        };

        self.gpu = try Gpu.init(self.allocator, gpu_config);
        return &self.gpu.?;
    }

    /// Create a buffer.
    pub fn createBuffer(self: *Context, comptime T: type, count: usize, options: BufferOptions) !UnifiedBuffer {
        const g = try self.getGpu();
        return g.createBuffer(count * @sizeOf(T), options);
    }

    /// Create a buffer from a slice.
    pub fn createBufferFromSlice(self: *Context, comptime T: type, data: []const T, options: BufferOptions) !UnifiedBuffer {
        const g = try self.getGpu();
        return g.createBufferFromSlice(T, data, options);
    }

    /// Destroy a buffer.
    pub fn destroyBuffer(self: *Context, buffer: *UnifiedBuffer) void {
        if (self.gpu) |*g| {
            g.destroyBuffer(buffer);
        }
    }

    /// Vector addition.
    pub fn vectorAdd(self: *Context, a: *UnifiedBuffer, b: *UnifiedBuffer, result: *UnifiedBuffer) !ExecutionResult {
        const g = try self.getGpu();
        return g.vectorAdd(a, b, result);
    }

    /// Matrix multiplication.
    pub fn matrixMultiply(self: *Context, a: *UnifiedBuffer, b: *UnifiedBuffer, result: *UnifiedBuffer, dims: MatrixDims) !ExecutionResult {
        const g = try self.getGpu();
        return g.matrixMultiply(a, b, result, dims);
    }

    /// Get health status.
    pub fn getHealth(self: *Context) !HealthStatus {
        const g = try self.getGpu();
        return g.getHealth();
    }
};

// ============================================================================
// Module-level functions
// ============================================================================

/// Check if GPU module is enabled at compile time.
pub fn isModuleEnabled() bool {
    return build_options.enable_gpu;
}

/// Check if a specific backend is enabled.
pub fn isEnabled(backend: Backend) bool {
    return compute_gpu.isEnabled(backend);
}

/// Check if GPU module is initialized.
pub fn isInitialized() bool {
    return compute_gpu.isInitialized();
}

/// Initialize the GPU module.
pub fn init(allocator: std.mem.Allocator) Error!void {
    if (!isEnabled()) return error.GpuDisabled;
    compute_gpu.init(allocator) catch return error.InitializationFailed;
}

/// Deinitialize the GPU module.
pub fn deinit() void {
    compute_gpu.deinit();
}

/// Check if any GPU is available.
pub fn isGpuAvailable() bool {
    return compute_gpu.isGpuAvailable();
}

/// Get available backends.
pub fn getAvailableBackends() []const Backend {
    return compute_gpu.getAvailableBackends();
}

/// Get the best available backend.
pub fn getBestBackend() Backend {
    return compute_gpu.getBestBackend();
}

/// List all backend information.
pub fn listBackendInfo(allocator: std.mem.Allocator) ![]BackendInfo {
    return compute_gpu.listBackendInfo(allocator);
}

/// List all devices.
pub fn listDevices(allocator: std.mem.Allocator) ![]DeviceInfo {
    return compute_gpu.listDevices(allocator);
}

/// Get default device.
pub fn defaultDevice(allocator: std.mem.Allocator) !?DeviceInfo {
    return compute_gpu.defaultDevice(allocator);
}

/// Discover devices.
pub fn discoverDevices(allocator: std.mem.Allocator) ![]Device {
    return compute_gpu.discoverDevices(allocator);
}

/// Get backend name.
pub fn backendName(backend: Backend) []const u8 {
    return compute_gpu.backendName(backend);
}

/// Get available backends (returns slice).
pub fn availableBackends(allocator: std.mem.Allocator) ![]Backend {
    return compute_gpu.availableBackends(allocator);
}

/// Get backend display name.
pub fn backendDisplayName(backend: Backend) []const u8 {
    return compute_gpu.backendDisplayName(backend);
}

/// Get backend description.
pub fn backendDescription(backend: Backend) []const u8 {
    return compute_gpu.backendDescription(backend);
}

/// Check if module is enabled.
pub fn moduleEnabled() bool {
    return compute_gpu.moduleEnabled();
}

/// Get GPU summary.
pub fn summary() compute_gpu.Summary {
    return compute_gpu.summary();
}

/// Get backend availability.
pub fn backendAvailability(backend: Backend) BackendAvailability {
    return compute_gpu.backendAvailability(backend);
}

pub const Summary = compute_gpu.Summary;

/// Create default kernels.
pub fn createDefaultKernels(allocator: std.mem.Allocator) !void {
    return compute_gpu.createDefaultKernels(allocator);
}

/// Compile a kernel.
pub fn compileKernel(source: KernelSource, config: KernelConfig) !CompiledKernel {
    return compute_gpu.compileKernel(source, config);
}

/// Ensure GPU is initialized.
pub fn ensureInitialized(allocator: std.mem.Allocator) !void {
    return compute_gpu.ensureInitialized(allocator);
}

// ============================================================================
// Tests
// ============================================================================

test "isEnabled returns build option" {
    try std.testing.expectEqual(build_options.enable_gpu, isEnabled());
}
