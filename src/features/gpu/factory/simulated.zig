const std = @import("std");
const interface = @import("../interface.zig");

/// Simulated/CPU backend implementing the full VTable interface.
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
        for (self.allocations.items) |alloc| {
            self.allocator.free(alloc.ptr[0..alloc.size]);
        }
        self.allocations.deinit(self.allocator);

        for (self.kernels.items) |kernel| {
            self.allocator.free(kernel.source);
            self.allocator.free(kernel.name);
        }
        self.kernels.deinit(self.allocator);

        self.allocator.destroy(self);
    }

    pub fn getDeviceCount(_: *SimulatedBackend) u32 {
        return 1;
    }

    pub fn getDeviceCaps(self: *SimulatedBackend, device_id: u32) interface.BackendError!interface.DeviceCaps {
        if (device_id != 0) return interface.BackendError.DeviceNotFound;

        var caps = interface.DeviceCaps{
            .total_memory = 8 * 1024 * 1024 * 1024,
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

        return @ptrCast(&self.kernels.items[self.kernels.items.len - 1]);
    }

    pub fn launchKernel(
        _: *SimulatedBackend,
        _: *anyopaque,
        _: interface.LaunchConfig,
        _: []const *anyopaque,
    ) interface.KernelError!void {
        return;
    }

    pub fn destroyKernel(_: *SimulatedBackend, _: *anyopaque) void {}

    pub fn synchronize(_: *SimulatedBackend) interface.BackendError!void {}
};

pub fn createSimulatedVTable(allocator: std.mem.Allocator) interface.BackendError!interface.Backend {
    const impl = SimulatedBackend.init(allocator) catch return interface.BackendError.OutOfMemory;
    return interface.createBackend(SimulatedBackend, impl);
}

test {
    std.testing.refAllDecls(@This());
}
