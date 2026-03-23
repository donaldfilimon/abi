const std = @import("std");
const sync = @import("../../foundation/mod.zig").sync;
const lifecycle = @import("../../foundation/mod.zig").utils;
const backend = @import("backend.zig");
const unified = @import("unified.zig");
const unified_buffer = @import("unified_buffer.zig");
const build_options = @import("build_options");
const backend_shared = @import("backends/shared.zig");
const config_module = @import("../../core/config/mod.zig");

const SimpleModuleLifecycle = lifecycle.SimpleModuleLifecycle;

pub var gpu_lifecycle = SimpleModuleLifecycle{};

pub var cuda_backend_init_lock = sync.Mutex{};
pub var cuda_backend_initialized = false;
pub var cached_gpu_allocator: ?std.mem.Allocator = null;

pub const GpuError = @import("types.zig").GpuError;
pub const Error = @import("types.zig").Error;

pub fn init(allocator: std.mem.Allocator) GpuError!void {
    if (!backend.moduleEnabled()) return error.GpuDisabled;

    cached_gpu_allocator = allocator;
    gpu_lifecycle.init(initCudaComponents) catch {
        return error.GpuDisabled;
    };
}

fn initCudaComponents() !void {
    if (comptime build_options.gpu_cuda and backend_shared.dynlibSupported) {
        cuda_backend_init_lock.lock();
        defer cuda_backend_init_lock.unlock();

        if (!cuda_backend_initialized) {
            const cuda_module = @import("backends/cuda/mod.zig");
            const alloc = cached_gpu_allocator orelse return error.OutOfMemory;

            cuda_module.init(alloc) catch |err| {
                std.log.warn("CUDA backend initialization failed: {t}. Using fallback mode.", .{err});
            };

            if (comptime build_options.feat_gpu) {
                const cuda_stream = @import("backends/cuda/stream.zig");
                cuda_stream.init() catch |err| {
                    std.log.warn("CUDA stream initialization failed: {t}", .{err});
                };

                const cuda_memory = @import("backends/cuda/memory.zig");
                cuda_memory.init(alloc) catch |err| {
                    std.log.warn("CUDA memory initialization failed: {t}", .{err});
                };
            }

            cuda_backend_initialized = true;
        }
    }
}

fn deinitCudaComponents() void {
    if (cuda_backend_initialized) {
        if (comptime build_options.gpu_cuda and backend_shared.dynlibSupported) {
            const cuda_module = @import("backends/cuda/mod.zig");
            cuda_module.deinit();

            if (comptime build_options.feat_gpu) {
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

pub const Context = struct {
    allocator: std.mem.Allocator,
    gpu: unified.Gpu,

    pub fn init(allocator: std.mem.Allocator, cfg: config_module.GpuConfig) !*Context {
        if (!backend.moduleEnabled()) return error.GpuDisabled;

        const preferred_backend: ?backend.Backend = switch (cfg.backend) {
            .auto => null,
            .cuda => .cuda,
            .vulkan => .vulkan,
            .stdgpu => .stdgpu,
            .metal => .metal,
            .webgpu => .webgpu,
            .opengl => .opengl,
            .opengles => .opengles,
            .webgl2 => .webgl2,
            .fpga => .fpga,
            .tpu => .tpu,
            .cpu => .stdgpu,
        };

        const gpu_config = unified.GpuConfig{
            .preferred_backend = preferred_backend,
            .allow_fallback = true,
            .max_memory_bytes = cfg.memory_limit orelse 0,
            .enable_profiling = false,
        };

        const ctx = try allocator.create(Context);
        errdefer allocator.destroy(ctx);

        ctx.* = .{
            .allocator = allocator,
            .gpu = try unified.Gpu.init(allocator, gpu_config),
        };
        return ctx;
    }

    pub fn deinit(self: *Context) void {
        self.gpu.deinit();
        self.allocator.destroy(self);
    }

    pub fn getGpu(self: *Context) Error!*unified.Gpu {
        return &self.gpu;
    }

    pub fn createBuffer(self: *Context, comptime T: type, count: usize, options: unified_buffer.BufferOptions) !unified_buffer.Buffer {
        return self.gpu.createBuffer(T, count, options);
    }

    pub fn createBufferFromSlice(self: *Context, comptime T: type, data: []const T, options: unified_buffer.BufferOptions) !unified_buffer.Buffer {
        return self.gpu.createBufferFromSlice(T, data, options);
    }

    pub fn destroyBuffer(self: *Context, buffer: *unified_buffer.Buffer) void {
        self.gpu.destroyBuffer(buffer);
    }

    pub fn vectorAdd(self: *Context, a: *unified_buffer.Buffer, b: *unified_buffer.Buffer, result: *unified_buffer.Buffer) !unified.ExecutionResult {
        return self.gpu.vectorAdd(a, b, result);
    }

    pub fn matrixMultiply(self: *Context, a: *unified_buffer.Buffer, b: *unified_buffer.Buffer, result: *unified_buffer.Buffer, dims: unified.MatrixDims) !unified.ExecutionResult {
        return self.gpu.matrixMultiply(a, b, result, dims);
    }

    pub fn getHealth(self: *Context) !unified.HealthStatus {
        return self.gpu.getHealth();
    }
};
