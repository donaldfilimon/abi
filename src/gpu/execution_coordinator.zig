//! Unified Execution Coordinator
//!
//! Provides seamless fallback: GPU → SIMD → scalar
//! Automatically selects the best execution method based on:
//! - Hardware availability
//! - Data size
//! - Operation type
//! - User preferences

const std = @import("std");
const backend_factory = @import("backend_factory.zig");
const simd = @import("../shared/simd.zig");

pub const ExecutionMethod = enum {
    gpu,
    simd,
    scalar,
    failed,
};

pub const CoordinatorConfig = struct {
    prefer_gpu: bool = true,
    fallback_chain: []const ExecutionMethod = &.{ .gpu, .simd, .scalar },
    gpu_threshold_size: usize = 1024, // Min elements for GPU
    simd_threshold_size: usize = 4, // Min elements for SIMD
    backend_timeout_ms: u64 = 1000,
};

pub const ExecutionCoordinator = struct {
    allocator: std.mem.Allocator,
    config: CoordinatorConfig,
    gpu_backend: ?*backend_factory.BackendInstance = null,
    gpu_available: bool = false,
    simd_available: bool = false,

    pub fn init(allocator: std.mem.Allocator, config: CoordinatorConfig) !ExecutionCoordinator {
        var coord = ExecutionCoordinator{
            .allocator = allocator,
            .config = config,
            .simd_available = simd.hasSimdSupport(),
        };

        // Try to initialize GPU
        if (config.prefer_gpu) {
            coord.gpu_backend = backend_factory.createBestBackend(allocator) catch null;
            coord.gpu_available = coord.gpu_backend != null;
        }

        return coord;
    }

    pub fn deinit(self: *ExecutionCoordinator) void {
        if (self.gpu_backend) |backend| {
            backend_factory.destroyBackend(backend);
        }
    }

    /// Vector addition with automatic method selection
    pub fn vectorAdd(
        self: *ExecutionCoordinator,
        a: []const f32,
        b: []const f32,
        result: []f32,
    ) !ExecutionMethod {
        const method = self.selectMethod(a.len, .vector_add);
        return self.vectorAddWithMethod(a, b, result, method);
    }

    /// Vector addition with explicit method
    pub fn vectorAddWithMethod(
        self: *ExecutionCoordinator,
        a: []const f32,
        b: []const f32,
        result: []f32,
        method: ExecutionMethod,
    ) !ExecutionMethod {
        return switch (method) {
            .gpu => self.vectorAddGpu(a, b, result) catch |err| blk: {
                // Fallback on GPU failure
                std.log.warn("GPU vector add failed: {}, falling back to SIMD", .{err});
                break :blk try self.vectorAddWithMethod(a, b, result, .simd);
            },
            .simd => blk: {
                if (self.simd_available and a.len >= self.config.simd_threshold_size) {
                    simd.vectorAdd(a, b, result);
                    break :blk .simd;
                } else {
                    // Fall through to scalar
                    break :blk try self.vectorAddWithMethod(a, b, result, .scalar);
                }
            },
            .scalar => blk: {
                for (a, b, 0..) |av, bv, i| {
                    result[i] = av + bv;
                }
                break :blk .scalar;
            },
            .failed => .failed,
        };
    }

    fn vectorAddGpu(
        self: *ExecutionCoordinator,
        a: []const f32,
        b: []const f32,
        result: []f32,
    ) !ExecutionMethod {
        _ = a;
        _ = b;
        _ = result;

        if (self.gpu_backend == null) return error.GpuNotAvailable;

        // TODO: Implement GPU vector add via backend
        // For now, fallback to SIMD
        return error.NotImplemented;
    }

    /// Select best execution method for operation
    fn selectMethod(self: *ExecutionCoordinator, size: usize, op: OperationType) ExecutionMethod {
        _ = op; // Reserved for operation-specific heuristics

        // Try methods in fallback chain order
        for (self.config.fallback_chain) |method| {
            if (self.canUseMethod(method, size)) {
                return method;
            }
        }

        // Last resort: scalar
        return .scalar;
    }

    fn canUseMethod(self: *ExecutionCoordinator, method: ExecutionMethod, size: usize) bool {
        return switch (method) {
            .gpu => self.gpu_available and size >= self.config.gpu_threshold_size,
            .simd => self.simd_available and size >= self.config.simd_threshold_size,
            .scalar => true,
            .failed => false,
        };
    }
};

const OperationType = enum {
    vector_add,
    vector_multiply,
    matrix_multiply,
    dot_product,
};
