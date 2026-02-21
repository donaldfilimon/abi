//! GPU Backend Factory
//!
//! Provides unified instantiation of GPU backends using the interface VTable pattern.
//! This factory abstracts away backend-specific initialization and provides a consistent
//! way to create and manage GPU backends.
//!
//! ## Usage
//!
//! ```zig
//! const factory = @import("backend_factory.zig");
//!
//! // Create a backend for a specific type
//! const backend = try factory.createBackend(allocator, .cuda);
//! defer factory.destroyBackend(backend);
//!
//! // Or auto-detect the best available backend
//! const best = try factory.createBestBackend(allocator);
//!
//! // Get VTable backend interface for dispatcher
//! const vtable_backend = try factory.createVTableBackend(allocator, .stdgpu);
//! defer vtable_backend.deinit();
//! ```
//!
//! ## Supported Backends (Neural Networks: GPU, WebGPU, TPU, CPU)
//!
//! Neural network inference and training can use:
//! - **GPU**: CUDA, Metal, Vulkan, WebGPU (preferred when available).
//! - **TPU**: Tensor Processing Unit slot; use `-Dgpu-backend=tpu` and link a TPU runtime (e.g. libtpu/cloud API) for availability.
//! - **CPU**: Multi-threaded CPU via `abi.runtime.ThreadPool` and `parallelFor`; set `InferenceConfig.num_threads` for LLM CPU inference.
//!
//! | Backend | Platform | Hardware Required |
//! |---------|----------|-------------------|
//! | CUDA    | Windows/Linux | NVIDIA GPU |
//! | Vulkan  | Windows/Linux/macOS | Vulkan 1.2+ driver |
//! | Metal   | macOS/iOS | Apple Silicon or AMD GPU |
//! | WebGPU  | All (browser/native) | WebGPU-capable browser or Dawn/wgpu; first-class for NN |
//! | TPU     | When runtime linked | Cloud TPU / libtpu (stub until linked) |
//! | OpenGL  | All | OpenGL 4.3+ |
//! | OpenGL ES | Mobile/Embedded | OpenGL ES 3.1+ |
//! | stdgpu  | All | None (CPU emulation) |

const std = @import("std");
const interface = @import("interface.zig");
const backend_mod = @import("backend.zig");
const build_options = @import("build_options");
const policy = @import("policy/mod.zig");
const backend_registry = @import("backends/registry.zig");
const backend_shared = @import("backends/shared.zig");
const android_probe = @import("device/android_probe.zig");

pub const Backend = backend_mod.Backend;
pub const BackendInterface = interface.Backend;
pub const LaunchConfig = interface.LaunchConfig;

/// Errors that can occur during backend creation.
pub const FactoryError = error{
    BackendNotAvailable,
    BackendInitializationFailed,
    OutOfMemory,
    UnsupportedBackend,
    NoBackendsAvailable,
};

/// Backend instance with its interface and metadata.
pub const BackendInstance = struct {
    /// The backend interface for kernel operations.
    backend: BackendInterface,
    /// Backend type identifier.
    backend_type: Backend,
    /// Allocator used for this backend.
    allocator: std.mem.Allocator,
    /// Whether this is a CPU emulation backend.
    is_emulated: bool,
    /// Maximum threads per block/workgroup.
    max_threads_per_block: u32,
    /// Total device memory in bytes (null if unknown).
    total_memory: ?u64,

    /// Check if this backend supports a specific feature.
    pub fn supportsFeature(self: *const BackendInstance, feature: BackendFeature) bool {
        return backendSupportsFeature(self.backend_type, feature);
    }
};

/// Stateful wrapper for backend creation APIs.
///
/// Maintained for compatibility with the public `abi.gpu.BackendFactory` export.
pub const BackendFactory = struct {
    allocator: std.mem.Allocator,

    pub fn init(allocator: std.mem.Allocator) BackendFactory {
        return .{ .allocator = allocator };
    }

    pub fn deinit(_: *BackendFactory) void {}

    pub fn create(self: *BackendFactory, backend_type: Backend) FactoryError!*BackendInstance {
        return createBackend(self.allocator, backend_type);
    }

    pub fn createBest(self: *BackendFactory) FactoryError!*BackendInstance {
        return createBestBackend(self.allocator);
    }

    pub fn destroy(_: *BackendFactory, instance: *BackendInstance) void {
        destroyBackend(instance);
    }
};

/// GPU features that may be supported by backends.
pub const BackendFeature = enum {
    fp16,
    fp64,
    atomics,
    shared_memory,
    subgroups,
    cooperative_groups,
    tensor_cores,
    dynamic_parallelism,
    bf16,
    tf32,
    fp8,
    async_copy,
    int8_tensor_cores,
    mesh_shaders,
    ray_tracing,
    neural_engine,
    mps,
};

const PriorityList = struct {
    items: [16]Backend = undefined,
    len: usize = 0,

    fn append(self: *PriorityList, backend: Backend) void {
        for (self.items[0..self.len]) |existing| {
            if (existing == backend) return;
        }
        if (self.len >= self.items.len) return;
        self.items[self.len] = backend;
        self.len += 1;
    }

    fn slice(self: *const PriorityList) []const Backend {
        return self.items[0..self.len];
    }
};

fn priorityList() PriorityList {
    var list = PriorityList{};
    const platform = policy.classifyBuiltin();
    const android_primary_name: ?[]const u8 = if (platform == .android)
        if (android_probe.chooseAndroidPrimary()) |backend| backend.name() else null
    else
        null;

    const names = policy.resolveAutoBackendNames(.{
        .platform = platform,
        .enable_gpu = build_options.enable_gpu,
        .enable_web = build_options.enable_web,
        .can_link_metal = true,
        .warn_if_metal_skipped = false,
        .allow_simulated = true,
        .android_primary = android_primary_name,
    });

    for (names.slice()) |name| {
        if (backend_mod.backendFromString(name)) |backend| {
            list.append(backend);
        }
    }

    if (@hasDecl(build_options, "gpu_tpu") and build_options.gpu_tpu) list.append(.tpu);
    if (@hasDecl(build_options, "gpu_fpga") and build_options.gpu_fpga) list.append(.fpga);

    // Keep deterministic software fallback ordering.
    list.append(.stdgpu);
    list.append(.simulated);
    return list;
}

/// Create a backend instance for the specified type.
/// Uses the VTable backend system internally.
pub fn createBackend(allocator: std.mem.Allocator, backend_type: Backend) FactoryError!*BackendInstance {
    const instance = allocator.create(BackendInstance) catch return FactoryError.OutOfMemory;
    errdefer allocator.destroy(instance);

    instance.allocator = allocator;
    instance.backend_type = backend_type;
    instance.is_emulated = backend_type == .stdgpu or backend_type == .simulated;
    instance.backend = try createVTableBackend(allocator, backend_type);
    instance.total_memory = queryDeviceMemory(backend_type);
    instance.max_threads_per_block = switch (backend_type) {
        .cuda, .vulkan, .metal, .opengl, .opengles, .tpu => 1024,
        .webgpu, .stdgpu, .webgl2, .simulated => 256,
        .fpga => 1,
    };

    return instance;
}

/// Create the best available backend based on hardware detection.
pub fn createBestBackend(allocator: std.mem.Allocator) FactoryError!*BackendInstance {
    const priorities = priorityList();
    for (priorities.slice()) |backend_type| {
        if (isBackendAvailable(backend_type)) {
            return createBackend(allocator, backend_type) catch continue;
        }
    }
    return FactoryError.NoBackendsAvailable;
}

/// Destroy a backend instance and release its resources.
pub fn destroyBackend(instance: *BackendInstance) void {
    instance.backend.deinit();
    instance.allocator.destroy(instance);
}

/// Check if a specific backend is available on this system.
pub fn isBackendAvailable(backend_type: Backend) bool {
    return backend_mod.backendAvailability(backend_type).available;
}

/// List all available backends on this system.
pub fn listAvailableBackends(allocator: std.mem.Allocator) ![]Backend {
    var backends = std.ArrayListUnmanaged(Backend).empty;
    errdefer backends.deinit(allocator);

    const priorities = priorityList();
    for (priorities.slice()) |backend_type| {
        if (isBackendAvailable(backend_type)) {
            backends.append(allocator, backend_type) catch continue;
        }
    }

    return backends.toOwnedSlice(allocator);
}

/// Alias for listAvailableBackends (Task 1.2 compatibility).
pub fn detectAvailableBackends(allocator: std.mem.Allocator) ![]Backend {
    return listAvailableBackends(allocator);
}

// ============================================================================
// Enhanced Backend Selection (Task 1.2)
// ============================================================================

/// Backend selection options.
pub const SelectionOptions = struct {
    preferred: ?Backend = null,
    fallback_chain: []const Backend = &.{ .vulkan, .metal, .stdgpu },
    required_features: []const BackendFeature = &.{},
    fallback_to_cpu: bool = true,
};

/// Select the best backend with fallback chain.
pub fn selectBestBackendWithFallback(
    allocator: std.mem.Allocator,
    options: SelectionOptions,
) !?Backend {
    _ = allocator; // May be needed for future enhancements

    // Try preferred first
    if (options.preferred) |preferred| {
        if (isBackendAvailable(preferred)) {
            if (meetsFeatureRequirements(preferred, options.required_features)) {
                return preferred;
            }
        }
    }

    // Try fallback chain
    for (options.fallback_chain) |backend_type| {
        if (isBackendAvailable(backend_type)) {
            if (meetsFeatureRequirements(backend_type, options.required_features)) {
                return backend_type;
            }
        }
    }

    // Last resort: std.gpu or simulated if allowed
    if (options.fallback_to_cpu) {
        if (isBackendAvailable(.stdgpu)) return .stdgpu;
        if (isBackendAvailable(.simulated)) return .simulated;
    }

    return null;
}

/// Select backend with specific feature requirements.
pub fn selectBackendWithFeatures(
    allocator: std.mem.Allocator,
    options: SelectionOptions,
) !?Backend {
    const available = try detectAvailableBackends(allocator);
    defer allocator.free(available);

    // Try backends in priority order
    const priorities = priorityList();
    for (priorities.slice()) |backend_type| {
        // Check if available
        var is_available = false;
        for (available) |avail| {
            if (avail == backend_type) {
                is_available = true;
                break;
            }
        }
        if (!is_available) continue;

        // Check if meets requirements
        if (meetsFeatureRequirements(backend_type, options.required_features)) {
            return backend_type;
        }
    }

    // Fallback to CPU/simulated if allowed
    if (options.fallback_to_cpu) {
        if (isBackendAvailable(.stdgpu)) return .stdgpu;
        if (isBackendAvailable(.simulated)) return .simulated;
    }

    return null;
}

fn meetsFeatureRequirements(backend_type: Backend, features: []const BackendFeature) bool {
    for (features) |feature| {
        if (!backendSupportsFeature(backend_type, feature)) {
            return false;
        }
    }
    return true;
}

fn backendSupportsFeature(backend_type: Backend, feature: BackendFeature) bool {
    return switch (feature) {
        .fp16 => backend_type == .cuda or backend_type == .metal,
        .fp64 => backend_type == .cuda,
        .atomics => switch (backend_type) {
            .cuda, .vulkan, .metal, .webgpu, .opengl, .opengles, .fpga, .tpu => true,
            .stdgpu, .webgl2, .simulated => false,
        },
        .shared_memory => switch (backend_type) {
            .cuda, .vulkan, .metal, .webgpu, .opengl, .opengles, .fpga, .tpu => true,
            .stdgpu, .webgl2, .simulated => false,
        },
        .subgroups => backend_type == .cuda or backend_type == .vulkan,
        .cooperative_groups => backend_type == .cuda,
        .tensor_cores => backend_type == .cuda,
        .dynamic_parallelism => backend_type == .cuda,
        .bf16 => backend_type == .cuda,
        .tf32 => backend_type == .cuda,
        .fp8 => backend_type == .cuda,
        .async_copy => backend_type == .cuda,
        .int8_tensor_cores => backend_type == .cuda,
        .mesh_shaders => backend_type == .metal,
        .ray_tracing => backend_type == .metal or backend_type == .vulkan,
        .neural_engine => backend_type == .metal,
        .mps => backend_type == .metal,
    };
}

// ============================================================================
// Memory Query Functions
// ============================================================================

fn queryDeviceMemory(backend_type: Backend) ?u64 {
    return switch (backend_type) {
        .cuda, .vulkan, .metal, .webgpu, .opengl, .opengles, .webgl2, .fpga, .tpu => null,
        .stdgpu => null,
        .simulated => 2 * 1024 * 1024 * 1024, // 2 GiB (matches backend_meta)
    };
}

// ============================================================================
// VTable Backend Implementation (interface.Backend)
// ============================================================================

/// Simulated/CPU backend implementing the full VTable interface.
/// This provides a working backend for kernel execution via CPU emulation.
pub const SimulatedBackend = struct {
    allocator: std.mem.Allocator,
    allocations: std.ArrayListUnmanaged(Allocation),
    kernels: std.ArrayListUnmanaged(CompiledKernel),
    device_name: [256]u8 = undefined,
    device_name_len: usize = 0,

    const Allocation = struct {
        ptr: [*]u8,
        size: usize,
    };

    const CompiledKernel = struct {
        source: []const u8,
        name: []const u8,
    };

    pub fn init(allocator: std.mem.Allocator) error{OutOfMemory}!*SimulatedBackend {
        const self = try allocator.create(SimulatedBackend);
        self.* = .{
            .allocator = allocator,
            .allocations = .empty,
            .kernels = .empty,
        };
        const name = "Simulated CPU Backend";
        @memcpy(self.device_name[0..name.len], name);
        self.device_name_len = name.len;
        return self;
    }

    pub fn deinit(self: *SimulatedBackend) void {
        // Free all allocations
        for (self.allocations.items) |alloc| {
            self.allocator.free(alloc.ptr[0..alloc.size]);
        }
        self.allocations.deinit(self.allocator);

        // Free kernel storage
        for (self.kernels.items) |kernel| {
            self.allocator.free(kernel.source);
            self.allocator.free(kernel.name);
        }
        self.kernels.deinit(self.allocator);

        self.allocator.destroy(self);
    }

    pub fn getDeviceCount(_: *SimulatedBackend) u32 {
        return 1; // Always one simulated device
    }

    pub fn getDeviceCaps(self: *SimulatedBackend, device_id: u32) interface.BackendError!interface.DeviceCaps {
        if (device_id != 0) return interface.BackendError.DeviceNotFound;

        var caps = interface.DeviceCaps{
            .total_memory = 8 * 1024 * 1024 * 1024, // 8GB simulated
            .max_threads_per_block = 1024,
            .max_shared_memory = 48 * 1024,
            .warp_size = 32,
            .supports_fp16 = true,
            .supports_fp64 = true,
            .supports_int8 = true,
            .unified_memory = true,
        };
        @memcpy(caps.name[0..self.device_name_len], self.device_name[0..self.device_name_len]);
        caps.name_len = self.device_name_len;
        return caps;
    }

    pub fn allocate(self: *SimulatedBackend, size: usize, _: interface.MemoryFlags) interface.MemoryError!*anyopaque {
        const mem = self.allocator.alloc(u8, size) catch return interface.MemoryError.OutOfMemory;
        self.allocations.append(self.allocator, .{ .ptr = mem.ptr, .size = size }) catch {
            self.allocator.free(mem);
            return interface.MemoryError.OutOfMemory;
        };
        return @ptrCast(mem.ptr);
    }

    pub fn free(self: *SimulatedBackend, ptr: *anyopaque) void {
        const target: [*]u8 = @ptrCast(ptr);
        for (self.allocations.items, 0..) |alloc, i| {
            if (alloc.ptr == target) {
                self.allocator.free(alloc.ptr[0..alloc.size]);
                _ = self.allocations.swapRemove(i);
                return;
            }
        }
    }

    pub fn copyToDevice(_: *SimulatedBackend, dst: *anyopaque, src: []const u8) interface.MemoryError!void {
        const dest_ptr: [*]u8 = @ptrCast(dst);
        @memcpy(dest_ptr[0..src.len], src);
    }

    pub fn copyFromDevice(_: *SimulatedBackend, dst: []u8, src: *anyopaque) interface.MemoryError!void {
        const src_ptr: [*]const u8 = @ptrCast(src);
        @memcpy(dst, src_ptr[0..dst.len]);
    }

    pub fn copyToDeviceAsync(self: *SimulatedBackend, dst: *anyopaque, src: []const u8, stream: ?*anyopaque) interface.MemoryError!void {
        _ = stream;
        return self.copyToDevice(dst, src);
    }

    pub fn copyFromDeviceAsync(self: *SimulatedBackend, dst: []u8, src: *anyopaque, stream: ?*anyopaque) interface.MemoryError!void {
        _ = stream;
        return self.copyFromDevice(dst, src);
    }

    pub fn compileKernel(
        self: *SimulatedBackend,
        allocator: std.mem.Allocator,
        source: []const u8,
        kernel_name: []const u8,
    ) interface.KernelError!*anyopaque {
        // Store the kernel for later reference
        const source_copy = allocator.dupe(u8, source) catch return interface.KernelError.CompileFailed;
        errdefer allocator.free(source_copy);

        const name_copy = allocator.dupe(u8, kernel_name) catch {
            allocator.free(source_copy);
            return interface.KernelError.CompileFailed;
        };

        const kernel = CompiledKernel{ .source = source_copy, .name = name_copy };
        self.kernels.append(self.allocator, kernel) catch {
            allocator.free(source_copy);
            allocator.free(name_copy);
            return interface.KernelError.CompileFailed;
        };

        // Return pointer to the last kernel
        return @ptrCast(&self.kernels.items[self.kernels.items.len - 1]);
    }

    pub fn launchKernel(
        _: *SimulatedBackend,
        _: *anyopaque,
        _: interface.LaunchConfig,
        _: []const *anyopaque,
    ) interface.KernelError!void {
        // Simulated kernel execution - the actual work is done by CPU fallback
        // in the dispatcher. This just validates the call succeeds.
        return;
    }

    pub fn destroyKernel(_: *SimulatedBackend, _: *anyopaque) void {
        // Kernels are freed when backend is deinitialized
    }

    pub fn synchronize(_: *SimulatedBackend) interface.BackendError!void {
        // CPU is always synchronized
    }
};

/// Create a VTable-compatible backend interface for use with the dispatcher.
/// This is the primary way to get a backend that supports actual kernel execution.
pub fn createVTableBackend(allocator: std.mem.Allocator, backend_type: Backend) FactoryError!interface.Backend {
    return backend_registry.createVTable(allocator, backend_type) catch |err| {
        return mapBackendCreateError(err);
    };
}

/// Create the best available VTable backend based on hardware detection.
pub fn createBestVTableBackend(allocator: std.mem.Allocator) FactoryError!interface.Backend {
    const priorities = priorityList();
    for (priorities.slice()) |backend_type| {
        if (isBackendAvailable(backend_type)) {
            return createVTableBackend(allocator, backend_type) catch continue;
        }
    }
    return FactoryError.NoBackendsAvailable;
}

fn createSimulatedVTableBackend(allocator: std.mem.Allocator) FactoryError!interface.Backend {
    const impl = SimulatedBackend.init(allocator) catch return FactoryError.OutOfMemory;
    return interface.createBackend(SimulatedBackend, impl);
}

fn mapBackendCreateError(err: anyerror) FactoryError {
    return switch (interface.normalizeBackendError(err)) {
        error.NotAvailable, error.DeviceNotFound, error.DriverNotFound => FactoryError.BackendNotAvailable,
        error.OutOfMemory => FactoryError.OutOfMemory,
        else => FactoryError.BackendInitializationFailed,
    };
}

fn createFpgaVTableBackend(allocator: std.mem.Allocator) FactoryError!interface.Backend {
    // Check if FPGA is available at comptime
    if (comptime !build_options.gpu_fpga) {
        return FactoryError.BackendNotAvailable;
    }

    const fpga_vtable = @import("backends/fpga/vtable.zig");
    return fpga_vtable.createFpgaVTable(allocator) catch |err| {
        return mapBackendCreateError(err);
    };
}

fn createCudaVTableBackend(allocator: std.mem.Allocator) FactoryError!interface.Backend {
    // Check if CUDA is available at comptime
    if (comptime build_options.gpu_cuda and backend_shared.dynlibSupported) {
        const cuda_vtable = @import("backends/cuda/vtable.zig");
        return cuda_vtable.createCudaVTable(allocator) catch |err| {
            return mapBackendCreateError(err);
        };
    }
    return FactoryError.BackendNotAvailable;
}

fn createVulkanVTableBackend(allocator: std.mem.Allocator) FactoryError!interface.Backend {
    // Check if Vulkan is available at comptime
    if (comptime !build_options.gpu_vulkan) {
        return FactoryError.BackendNotAvailable;
    }

    // Try to create real Vulkan backend
    const vulkan = @import("backends/vulkan.zig");
    return vulkan.createVulkanVTable(allocator) catch |err| {
        return mapBackendCreateError(err);
    };
}

fn createMetalVTableBackend(allocator: std.mem.Allocator) FactoryError!interface.Backend {
    // Check if Metal is available at comptime
    if (comptime !build_options.gpu_metal) {
        return FactoryError.BackendNotAvailable;
    }

    // Try to create real Metal backend
    const metal_vtable = @import("backends/metal_vtable.zig");
    return metal_vtable.createMetalVTable(allocator) catch |err| {
        return mapBackendCreateError(err);
    };
}

fn createWebGPUVTableBackend(allocator: std.mem.Allocator) FactoryError!interface.Backend {
    // Check if WebGPU is available at comptime
    if (comptime !build_options.gpu_webgpu) {
        return FactoryError.BackendNotAvailable;
    }

    // Try to create real WebGPU backend
    const webgpu_vtable = @import("backends/webgpu_vtable.zig");
    return webgpu_vtable.createWebGpuVTable(allocator) catch |err| {
        return mapBackendCreateError(err);
    };
}

fn createOpenGLVTableBackend(allocator: std.mem.Allocator) FactoryError!interface.Backend {
    const gl_backend = @import("backends/gl/backend.zig");
    const gl_profile = @import("backends/gl/profile.zig");
    return gl_backend.createVTableForProfile(allocator, gl_profile.Profile.desktop) catch |err| {
        return mapBackendCreateError(err);
    };
}

fn createOpenGLESVTableBackend(allocator: std.mem.Allocator) FactoryError!interface.Backend {
    const gl_backend = @import("backends/gl/backend.zig");
    const gl_profile = @import("backends/gl/profile.zig");
    return gl_backend.createVTableForProfile(allocator, gl_profile.Profile.es) catch |err| {
        return mapBackendCreateError(err);
    };
}

fn createWebGL2VTableBackend(allocator: std.mem.Allocator) FactoryError!interface.Backend {
    // Check if WebGL2 is available at comptime
    if (comptime !build_options.gpu_webgl2) {
        return FactoryError.BackendNotAvailable;
    }

    // WebGL2 uses the SimulatedBackend since it targets browsers
    // and has no native kernel support
    const impl = SimulatedBackend.init(allocator) catch return FactoryError.OutOfMemory;
    return interface.createBackend(SimulatedBackend, impl);
}

// ============================================================================
// Tests
// ============================================================================

test "factory creates stdgpu backend" {
    const instance = try createBackend(std.testing.allocator, .stdgpu);
    defer destroyBackend(instance);

    try std.testing.expectEqual(Backend.stdgpu, instance.backend_type);
    try std.testing.expect(instance.is_emulated);
}

test "factory lists available backends" {
    const backends = try listAvailableBackends(std.testing.allocator);
    defer std.testing.allocator.free(backends);

    // stdgpu should always be available
    var has_stdgpu = false;
    for (backends) |b| {
        if (b == .stdgpu) has_stdgpu = true;
    }
    try std.testing.expect(has_stdgpu);
}

test "backend feature support" {
    const instance = try createBackend(std.testing.allocator, .stdgpu);
    defer destroyBackend(instance);

    // stdgpu has limited feature support
    try std.testing.expect(!instance.supportsFeature(.atomics));
    try std.testing.expect(!instance.supportsFeature(.shared_memory));
}

test "simulated and webgl2 do not advertise atomics/shared memory" {
    try std.testing.expect(!backendSupportsFeature(.simulated, .atomics));
    try std.testing.expect(!backendSupportsFeature(.simulated, .shared_memory));
    try std.testing.expect(!backendSupportsFeature(.webgl2, .atomics));
    try std.testing.expect(!backendSupportsFeature(.webgl2, .shared_memory));
}
