//! Unified Accelerator Interface
//!
//! Abstract interface for all compute backends: GPU (CUDA, Vulkan, Metal),
//! AI accelerators (TPU, NPU), and CPU (SIMD, scalar).

const std = @import("std");
const builtin = @import("builtin");
const hardware_detection = @import("hardware_detection.zig");

pub const BackendType = hardware_detection.BackendType;

/// Memory allocation on an accelerator device
pub const DeviceMemory = struct {
    ptr: ?*anyopaque,
    size: usize,
    backend: BackendType,

    pub fn isValid(self: DeviceMemory) bool {
        return self.ptr != null;
    }
};

/// Tensor descriptor for accelerator operations
pub const TensorDesc = struct {
    shape: []const usize,
    dtype: DataType,
    layout: Layout,

    pub const DataType = enum {
        f16,
        bf16,
        f32,
        f64,
        i8,
        i16,
        i32,
        i64,
        u8,
        u16,
        u32,
        u64,
    };

    pub const Layout = enum {
        row_major,
        col_major,
        blocked,
    };

    pub fn elementCount(self: TensorDesc) usize {
        var count: usize = 1;
        for (self.shape) |dim| count *= dim;
        return count;
    }

    pub fn byteSize(self: TensorDesc) usize {
        const elem_size: usize = switch (self.dtype) {
            .f16, .bf16, .i16, .u16 => 2,
            .f32, .i32, .u32 => 4,
            .f64, .i64, .u64 => 8,
            .i8, .u8 => 1,
        };
        return self.elementCount() * elem_size;
    }
};

/// Kernel dispatch parameters
pub const DispatchParams = struct {
    global_size: [3]u32 = .{ 1, 1, 1 },
    local_size: [3]u32 = .{ 1, 1, 1 },
    shared_mem_size: u32 = 0,
};

/// Unified accelerator interface
pub const Accelerator = struct {
    backend: BackendType,
    allocator: std.mem.Allocator,
    device_id: u32,
    name: []const u8,

    // Memory management
    total_memory: u64,
    available_memory: u64,

    /// Allocate memory on the device
    pub fn alloc(self: *Accelerator, size: usize) !DeviceMemory {
        if (self.backend == .cpu_fallback or self.backend == .cpu_simd) {
            const ptr = self.allocator.alloc(u8, size) catch return error.OutOfMemory;
            return DeviceMemory{
                .ptr = @ptrCast(ptr.ptr),
                .size = size,
                .backend = self.backend,
            };
        }
        // For GPU/TPU/NPU - stub for now, would call native APIs
        return DeviceMemory{
            .ptr = null,
            .size = size,
            .backend = self.backend,
        };
    }

    /// Free device memory
    pub fn free(self: *Accelerator, mem: *DeviceMemory) void {
        if (mem.ptr == null) return;
        if (self.backend == .cpu_fallback or self.backend == .cpu_simd) {
            const slice: [*]u8 = @ptrCast(mem.ptr.?);
            self.allocator.free(slice[0..mem.size]);
        }
        mem.ptr = null;
    }

    /// Copy data to device
    pub fn copyToDevice(self: *Accelerator, dst: DeviceMemory, src: []const u8) !void {
        if (dst.ptr == null) return error.InvalidMemory;
        if (src.len > dst.size) return error.BufferTooSmall;

        if (self.backend == .cpu_fallback or self.backend == .cpu_simd) {
            const dst_slice: [*]u8 = @ptrCast(dst.ptr.?);
            @memcpy(dst_slice[0..src.len], src);
        }
    }

    /// Copy data from device
    pub fn copyFromDevice(self: *Accelerator, dst: []u8, src: DeviceMemory) !void {
        if (src.ptr == null) return error.InvalidMemory;
        if (dst.len > src.size) return error.BufferTooSmall;

        if (self.backend == .cpu_fallback or self.backend == .cpu_simd) {
            const src_slice: [*]const u8 = @ptrCast(src.ptr.?);
            @memcpy(dst, src_slice[0..dst.len]);
        }
    }

    /// Synchronize - wait for all operations to complete
    pub fn sync(self: *Accelerator) void {
        _ = self;
        // CPU is synchronous, GPU backends would wait here
    }
};

/// Tensor operations on accelerator
pub const TensorOps = struct {
    accel: *Accelerator,

    pub fn init(accel: *Accelerator) TensorOps {
        return .{ .accel = accel };
    }

    /// Matrix multiply: C = A @ B
    pub fn matmul(self: *TensorOps, c: DeviceMemory, a: DeviceMemory, b: DeviceMemory, m: usize, n: usize, k: usize) void {
        if (self.accel.backend == .cpu_simd or self.accel.backend == .cpu_fallback) {
            self.cpuMatmul(c, a, b, m, n, k);
        }
        // GPU/TPU would dispatch kernels
    }

    fn cpuMatmul(self: *TensorOps, c_mem: DeviceMemory, a_mem: DeviceMemory, b_mem: DeviceMemory, m: usize, n: usize, k: usize) void {
        _ = self;
        if (c_mem.ptr == null or a_mem.ptr == null or b_mem.ptr == null) return;

        const a: [*]const f32 = @ptrCast(@alignCast(a_mem.ptr.?));
        const b: [*]const f32 = @ptrCast(@alignCast(b_mem.ptr.?));
        const c: [*]f32 = @ptrCast(@alignCast(c_mem.ptr.?));

        // Simple matrix multiply (would use SIMD in production)
        for (0..m) |i| {
            for (0..n) |j| {
                var sum: f32 = 0;
                for (0..k) |l| {
                    sum += a[i * k + l] * b[l * n + j];
                }
                c[i * n + j] = sum;
            }
        }
    }

    /// Element-wise ReLU activation
    pub fn relu(self: *TensorOps, dst: DeviceMemory, src: DeviceMemory, count: usize) void {
        if (self.accel.backend == .cpu_simd or self.accel.backend == .cpu_fallback) {
            if (dst.ptr == null or src.ptr == null) return;
            const s: [*]const f32 = @ptrCast(@alignCast(src.ptr.?));
            const d: [*]f32 = @ptrCast(@alignCast(dst.ptr.?));
            for (0..count) |i| {
                d[i] = @max(0, s[i]);
            }
        }
    }

    /// Softmax activation
    pub fn softmax(self: *TensorOps, dst: DeviceMemory, src: DeviceMemory, count: usize) void {
        if (self.accel.backend == .cpu_simd or self.accel.backend == .cpu_fallback) {
            if (dst.ptr == null or src.ptr == null) return;
            const s: [*]const f32 = @ptrCast(@alignCast(src.ptr.?));
            const d: [*]f32 = @ptrCast(@alignCast(dst.ptr.?));

            // Find max for numerical stability
            var max_val: f32 = s[0];
            for (1..count) |i| max_val = @max(max_val, s[i]);

            // Compute exp and sum
            var sum: f32 = 0;
            for (0..count) |i| {
                d[i] = @exp(s[i] - max_val);
                sum += d[i];
            }

            // Normalize
            for (0..count) |i| d[i] /= sum;
        }
    }

    /// Dot product for vector similarity
    pub fn dotProduct(self: *TensorOps, a: DeviceMemory, b: DeviceMemory, count: usize) f32 {
        if (self.accel.backend == .cpu_simd or self.accel.backend == .cpu_fallback) {
            if (a.ptr == null or b.ptr == null) return 0;
            const va: [*]const f32 = @ptrCast(@alignCast(a.ptr.?));
            const vb: [*]const f32 = @ptrCast(@alignCast(b.ptr.?));

            var sum: f32 = 0;
            for (0..count) |i| sum += va[i] * vb[i];
            return sum;
        }
        return 0;
    }
};

/// Create accelerator for the best available backend
pub fn createBestAccelerator(allocator: std.mem.Allocator) Accelerator {
    // Detect and select best backend
    const backends = [_]BackendType{ .tpu, .npu, .cuda, .rocm, .vulkan, .metal, .sycl, .cpu_simd, .cpu_fallback };

    for (backends) |backend| {
        if (backend.isAvailable()) {
            return Accelerator{
                .backend = backend,
                .allocator = allocator,
                .device_id = 0,
                .name = backend.displayName(),
                .total_memory = 0,
                .available_memory = 0,
            };
        }
    }

    return Accelerator{
        .backend = .cpu_fallback,
        .allocator = allocator,
        .device_id = 0,
        .name = "CPU Fallback",
        .total_memory = 0,
        .available_memory = 0,
    };
}

/// Create accelerator for specific backend
pub fn createAccelerator(allocator: std.mem.Allocator, backend: BackendType) !Accelerator {
    if (!backend.isAvailable()) return error.BackendNotAvailable;

    return Accelerator{
        .backend = backend,
        .allocator = allocator,
        .device_id = 0,
        .name = backend.displayName(),
        .total_memory = 0,
        .available_memory = 0,
    };
}

test "accelerator cpu operations" {
    const testing = std.testing;

    var accel = createBestAccelerator(testing.allocator);

    // Test allocation
    var mem = try accel.alloc(256);
    defer accel.free(&mem);

    try testing.expect(mem.isValid());
    try testing.expectEqual(@as(usize, 256), mem.size);
}

test "tensor ops matmul" {
    const testing = std.testing;

    var accel = createBestAccelerator(testing.allocator);
    var ops = TensorOps.init(&accel);

    // 2x2 matrices
    var a_mem = try accel.alloc(4 * @sizeOf(f32));
    defer accel.free(&a_mem);
    var b_mem = try accel.alloc(4 * @sizeOf(f32));
    defer accel.free(&b_mem);
    var c_mem = try accel.alloc(4 * @sizeOf(f32));
    defer accel.free(&c_mem);

    // Initialize: A = [[1,2],[3,4]], B = [[1,0],[0,1]] (identity)
    const a_data = [_]f32{ 1, 2, 3, 4 };
    const b_data = [_]f32{ 1, 0, 0, 1 };
    try accel.copyToDevice(a_mem, std.mem.sliceAsBytes(&a_data));
    try accel.copyToDevice(b_mem, std.mem.sliceAsBytes(&b_data));

    ops.matmul(c_mem, a_mem, b_mem, 2, 2, 2);

    var c_result: [4]f32 = undefined;
    try accel.copyFromDevice(std.mem.sliceAsBytes(&c_result), c_mem);

    // C should equal A (multiplied by identity)
    try testing.expectApproxEqAbs(@as(f32, 1), c_result[0], 0.001);
    try testing.expectApproxEqAbs(@as(f32, 2), c_result[1], 0.001);
    try testing.expectApproxEqAbs(@as(f32, 3), c_result[2], 0.001);
    try testing.expectApproxEqAbs(@as(f32, 4), c_result[3], 0.001);
}
