//! TPU VTable Backend Implementation
//!
//! Implements the Backend interface for Google TPU accelerators.
//! Currently operates in simulation mode only — all memory and kernel
//! operations are backed by host allocations and logged, returning
//! success without actual TPU hardware interaction.
//!
//! When real TPU support is added, this module would integrate with
//! the PJRT (Portable JAX Runtime) C API or libtpu.

const std = @import("std");
const interface = @import("../../interface.zig");
const tpu_types = @import("types.zig");
const PointerCast = @import("../../pointer_cast.zig");

pub const types = tpu_types;

pub const TpuBackend = struct {
    allocator: std.mem.Allocator,

    /// Track allocations for cleanup (simulation only)
    allocations: std.ArrayListUnmanaged(Allocation),
    /// Track compiled kernels
    kernels: std.ArrayListUnmanaged(CompiledKernel),

    /// Simulated device config
    config: tpu_types.TpuConfig,
    /// Runtime statistics
    stats: tpu_types.TpuStats,

    pub const Allocation = struct {
        ptr: [*]u8,
        size: usize,
    };

    pub const CompiledKernel = struct {
        name: []const u8,
        kernel_class: tpu_types.KernelClass,
    };

    const Self = @This();

    pub fn init(allocator: std.mem.Allocator) interface.BackendError!*Self {
        const self = allocator.create(Self) catch return interface.BackendError.OutOfMemory;
        self.* = .{
            .allocator = allocator,
            .allocations = .empty,
            .kernels = .empty,
            .config = .{},
            .stats = .{},
        };

        std.log.info("TPU backend: initialized in simulation mode", .{});
        return self;
    }

    pub fn deinit(self: *Self) void {
        // Clean up tracked allocations
        for (self.allocations.items) |alloc| {
            self.allocator.free(alloc.ptr[0..alloc.size]);
        }
        self.allocations.deinit(self.allocator);

        // Clean up compiled kernels
        for (self.kernels.items) |kernel| {
            self.allocator.free(kernel.name);
        }
        self.kernels.deinit(self.allocator);

        self.allocator.destroy(self);
    }

    pub fn getDeviceCount(self: *Self) u32 {
        // In simulation mode, report the configured chip count
        return @as(u32, self.config.chip_count);
    }

    pub fn getDeviceCaps(self: *Self, device_id: u32) interface.BackendError!interface.DeviceCaps {
        if (device_id >= self.config.chip_count) {
            return interface.BackendError.DeviceNotFound;
        }

        var caps = interface.DeviceCaps{};

        // Set device name
        const gen_name = self.config.generation.name();
        const suffix = " (simulated)";
        const name_len = @min(gen_name.len + suffix.len, caps.name.len);
        const gen_copy = @min(gen_name.len, caps.name.len);
        @memcpy(caps.name[0..gen_copy], gen_name[0..gen_copy]);
        if (gen_copy + suffix.len <= caps.name.len) {
            @memcpy(caps.name[gen_copy .. gen_copy + suffix.len], suffix);
        }
        caps.name_len = name_len;

        // Memory: use generation defaults
        const hbm_gb = self.config.generation.hbmCapacityGb();
        caps.total_memory = @as(usize, hbm_gb) * 1024 * 1024 * 1024;

        // TPU-specific capability mapping
        caps.max_threads_per_block = 1; // TPUs use systolic arrays, not thread blocks
        caps.max_shared_memory = 16 * 1024 * 1024; // Simulated on-chip SRAM
        caps.warp_size = 1; // No warps on TPU
        caps.supports_fp16 = false; // TPUs prefer bf16 over fp16
        caps.supports_bf16 = true;
        caps.supports_fp64 = false;
        caps.supports_int8 = true;
        caps.supports_tensor_cores = true; // MXUs are similar in concept
        caps.unified_memory = false;
        caps.compute_capability_major = @intFromEnum(self.config.generation);
        caps.compute_capability_minor = 0;
        caps.async_engine_count = 1;

        // Architecture name
        const arch = "Systolic Array (MXU)";
        @memcpy(caps.architecture_name[0..arch.len], arch);
        caps.architecture_name_len = arch.len;

        return caps;
    }

    pub fn allocate(self: *Self, size: usize, flags: interface.MemoryFlags) interface.MemoryError!*anyopaque {
        _ = flags;

        // Simulate TPU HBM allocation with host memory
        const buffer = self.allocator.alloc(u8, size) catch return interface.MemoryError.OutOfMemory;

        // Track for cleanup
        self.allocations.append(self.allocator, .{
            .ptr = buffer.ptr,
            .size = size,
        }) catch {
            self.allocator.free(buffer);
            return interface.MemoryError.OutOfMemory;
        };

        self.stats.allocation_count += 1;
        self.stats.current_memory_bytes += size;
        if (self.stats.current_memory_bytes > self.stats.peak_memory_bytes) {
            self.stats.peak_memory_bytes = self.stats.current_memory_bytes;
        }

        return buffer.ptr;
    }

    pub fn free(self: *Self, ptr: *anyopaque) void {
        const target: [*]u8 = @ptrCast(ptr);

        // Find and remove from tracking
        for (self.allocations.items, 0..) |alloc, i| {
            if (alloc.ptr == target) {
                self.stats.current_memory_bytes -= alloc.size;
                self.allocator.free(alloc.ptr[0..alloc.size]);
                _ = self.allocations.swapRemove(i);
                return;
            }
        }
    }

    pub fn copyToDevice(self: *Self, dst: *anyopaque, src: []const u8) interface.MemoryError!void {
        // Simulation: memcpy host-to-"device"
        const dest_ptr: [*]u8 = @ptrCast(dst);
        @memcpy(dest_ptr[0..src.len], src);
        self.stats.bytes_to_device += src.len;
    }

    pub fn copyFromDevice(self: *Self, dst: []u8, src: *anyopaque) interface.MemoryError!void {
        // Simulation: memcpy "device"-to-host
        const src_ptr: [*]const u8 = @ptrCast(src);
        @memcpy(dst, src_ptr[0..dst.len]);
        self.stats.bytes_from_device += dst.len;
    }

    pub fn copyToDeviceAsync(self: *Self, dst: *anyopaque, src: []const u8, stream: ?*anyopaque) interface.MemoryError!void {
        _ = stream;
        return self.copyToDevice(dst, src);
    }

    pub fn copyFromDeviceAsync(self: *Self, dst: []u8, src: *anyopaque, stream: ?*anyopaque) interface.MemoryError!void {
        _ = stream;
        return self.copyFromDevice(dst, src);
    }

    pub fn compileKernel(self: *Self, allocator: std.mem.Allocator, source: []const u8, name: []const u8) interface.KernelError!*anyopaque {
        _ = allocator;
        _ = source;

        // Classify kernel by name
        const kernel_name = name[0..@min(name.len, 128)];
        var kernel_class: tpu_types.KernelClass = .custom;

        if (std.mem.indexOf(u8, kernel_name, "flash_attention") != null) {
            kernel_class = .flash_attention;
        } else if (std.mem.indexOf(u8, kernel_name, "attention") != null or
            std.mem.indexOf(u8, kernel_name, "mha") != null)
        {
            kernel_class = .attention;
        } else if (std.mem.indexOf(u8, kernel_name, "matmul_fused") != null or
            std.mem.indexOf(u8, kernel_name, "fused_matmul") != null)
        {
            kernel_class = .matmul_fused;
        } else if (std.mem.indexOf(u8, kernel_name, "matmul") != null or
            std.mem.indexOf(u8, kernel_name, "gemm") != null)
        {
            kernel_class = .matmul;
        } else if (std.mem.indexOf(u8, kernel_name, "layernorm") != null) {
            kernel_class = .layernorm;
        } else if (std.mem.indexOf(u8, kernel_name, "rmsnorm") != null) {
            kernel_class = .rmsnorm;
        } else if (std.mem.indexOf(u8, kernel_name, "softmax") != null) {
            kernel_class = .softmax;
        } else if (std.mem.indexOf(u8, kernel_name, "silu") != null) {
            kernel_class = .silu;
        } else if (std.mem.indexOf(u8, kernel_name, "gelu") != null) {
            kernel_class = .gelu;
        } else if (std.mem.indexOf(u8, kernel_name, "rope") != null) {
            kernel_class = .rope;
        } else if (std.mem.indexOf(u8, kernel_name, "cosine") != null or
            std.mem.indexOf(u8, kernel_name, "l2_dist") != null or
            std.mem.indexOf(u8, kernel_name, "dot_product") != null)
        {
            kernel_class = .vector_distance;
        } else if (std.mem.indexOf(u8, kernel_name, "reduce") != null) {
            kernel_class = .reduce;
        }

        // Store compiled kernel
        const compiled = CompiledKernel{
            .name = self.allocator.dupe(u8, name) catch return interface.KernelError.CompileFailed,
            .kernel_class = kernel_class,
        };

        self.kernels.append(self.allocator, compiled) catch return interface.KernelError.CompileFailed;

        std.log.info("TPU: compiled kernel '{s}' (class: {s}, simulation)", .{
            name,
            @tagName(kernel_class),
        });

        // Return pointer to stored kernel entry
        return @ptrCast(&self.kernels.items[self.kernels.items.len - 1]);
    }

    pub fn launchKernel(self: *Self, kernel: *anyopaque, config: interface.LaunchConfig, args: []const *anyopaque) interface.KernelError!void {
        _ = config;
        _ = args;

        const kernel_ptr: *CompiledKernel = PointerCast.implCast(CompiledKernel, kernel);

        std.log.info("TPU: simulating kernel '{s}' execution (class: {s})", .{
            kernel_ptr.name,
            @tagName(kernel_ptr.kernel_class),
        });

        self.stats.kernel_invocations += 1;
    }

    pub fn destroyKernel(self: *Self, kernel: *anyopaque) void {
        const kernel_ptr: *CompiledKernel = PointerCast.implCast(CompiledKernel, kernel);

        for (self.kernels.items, 0..) |k, i| {
            if (std.mem.eql(u8, k.name, kernel_ptr.name)) {
                self.allocator.free(k.name);
                _ = self.kernels.swapRemove(i);
                return;
            }
        }
    }

    pub fn synchronize(self: *Self) interface.BackendError!void {
        _ = self;
        // Simulation: all ops are synchronous already
    }
};

/// Create a TPU backend VTable for the GPU interface registry.
pub fn createTpuVTable(allocator: std.mem.Allocator) interface.BackendError!interface.Backend {
    const impl = try TpuBackend.init(allocator);
    return interface.createBackend(TpuBackend, impl);
}

// ============================================================================
// Tests
// ============================================================================

test "TpuBackend init/deinit" {
    const allocator = std.testing.allocator;
    const impl = try TpuBackend.init(allocator);
    defer impl.deinit();

    try std.testing.expectEqual(@as(u32, 1), impl.getDeviceCount());
}

test "TpuBackend device caps" {
    const allocator = std.testing.allocator;
    const impl = try TpuBackend.init(allocator);
    defer impl.deinit();

    const caps = try impl.getDeviceCaps(0);
    try std.testing.expect(caps.name_len > 0);
    try std.testing.expect(caps.supports_bf16);
    try std.testing.expect(caps.supports_int8);
    try std.testing.expect(!caps.supports_fp16);
}

test "TpuBackend device caps out of range" {
    const allocator = std.testing.allocator;
    const impl = try TpuBackend.init(allocator);
    defer impl.deinit();

    try std.testing.expectError(interface.BackendError.DeviceNotFound, impl.getDeviceCaps(99));
}

test "TpuBackend allocate and free" {
    const allocator = std.testing.allocator;
    const impl = try TpuBackend.init(allocator);
    defer impl.deinit();

    const ptr = try impl.allocate(4096, .{});
    try std.testing.expectEqual(@as(u64, 1), impl.stats.allocation_count);
    try std.testing.expectEqual(@as(u64, 4096), impl.stats.current_memory_bytes);

    impl.free(ptr);
    try std.testing.expectEqual(@as(u64, 0), impl.stats.current_memory_bytes);
}

test "TpuBackend copy operations" {
    const allocator = std.testing.allocator;
    const impl = try TpuBackend.init(allocator);
    defer impl.deinit();

    const ptr = try impl.allocate(16, .{});
    defer impl.free(ptr);

    const src = [_]u8{ 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16 };
    try impl.copyToDevice(ptr, &src);
    try std.testing.expectEqual(@as(u64, 16), impl.stats.bytes_to_device);

    var dst: [16]u8 = undefined;
    try impl.copyFromDevice(&dst, ptr);
    try std.testing.expectEqualSlices(u8, &src, &dst);
    try std.testing.expectEqual(@as(u64, 16), impl.stats.bytes_from_device);
}

test "TpuBackend compile and destroy kernel" {
    const allocator = std.testing.allocator;
    const impl = try TpuBackend.init(allocator);
    defer impl.deinit();

    const kernel = try impl.compileKernel(allocator, "// matmul source", "matmul_bf16");
    try std.testing.expectEqual(@as(usize, 1), impl.kernels.items.len);

    impl.destroyKernel(kernel);
    try std.testing.expectEqual(@as(usize, 0), impl.kernels.items.len);
}

test "TpuBackend VTable creation" {
    const allocator = std.testing.allocator;
    const backend = try createTpuVTable(allocator);
    defer backend.deinit();

    try std.testing.expectEqual(@as(u32, 1), backend.getDeviceCount());
}

test "TpuBackend synchronize" {
    const allocator = std.testing.allocator;
    const impl = try TpuBackend.init(allocator);
    defer impl.deinit();

    try impl.synchronize();
}

test {
    std.testing.refAllDecls(@This());
}
