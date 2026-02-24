//! OpenGL ES VTable Backend Implementation
//!
//! Provides a complete VTable implementation for OpenGL ES, enabling GPU
//! kernel execution through the polymorphic backend interface.
//! Requires OpenGL ES 3.1+ for compute shader support.

const std = @import("std");
const build_options = @import("build_options");
const interface = @import("../interface.zig");
const opengles = @import("opengles.zig");
const gl_runtime = @import("gl/runtime.zig");

/// OpenGL ES VTable backend wrapper.
pub const OpenGLESBackend = struct {
    allocator: std.mem.Allocator,
    initialized: bool,

    allocations: std.ArrayListUnmanaged(Allocation),
    kernels: std.ArrayListUnmanaged(CompiledKernel),

    device_name: [256]u8 = undefined,
    device_name_len: usize = 0,

    const Allocation = struct {
        ptr: *anyopaque,
        size: usize,
    };

    const CompiledKernel = struct {
        handle: *anyopaque,
        name: []const u8,
    };

    const Self = @This();

    pub fn init(allocator: std.mem.Allocator) interface.BackendError!*Self {
        if (comptime !build_options.gpu_opengles) {
            return interface.BackendError.NotAvailable;
        }

        opengles.init() catch |err| {
            return switch (err) {
                opengles.OpenGlesError.LibraryNotFound => interface.BackendError.DriverNotFound,
                opengles.OpenGlesError.VersionNotSupported => interface.BackendError.NotAvailable,
                opengles.OpenGlesError.FunctionLoadFailed => interface.BackendError.InitFailed,
                else => interface.BackendError.InitFailed,
            };
        };

        const self = allocator.create(Self) catch {
            opengles.deinit();
            return interface.BackendError.OutOfMemory;
        };

        self.* = .{
            .allocator = allocator,
            .initialized = true,
            .allocations = .empty,
            .kernels = .empty,
        };

        const version = opengles.getVersion();
        const name = std.fmt.bufPrint(&self.device_name, "OpenGL ES {}.{} Compute Device", .{
            version.major,
            version.minor,
        }) catch "OpenGL ES Compute Device";
        self.device_name_len = name.len;

        opengles.setBufferAllocator(allocator);
        return self;
    }

    pub fn deinit(self: *Self) void {
        for (self.allocations.items) |alloc| {
            gl_runtime.freeDeviceMemory(.opengles, alloc.ptr);
        }
        self.allocations.deinit(self.allocator);

        for (self.kernels.items) |kernel| {
            gl_runtime.destroyKernel(.opengles, self.allocator, kernel.handle);
            self.allocator.free(kernel.name);
        }
        self.kernels.deinit(self.allocator);

        if (self.initialized) {
            opengles.deinit();
        }

        self.allocator.destroy(self);
    }

    pub fn getDeviceCount(_: *Self) u32 {
        if (opengles.isAvailable()) return 1;
        return 0;
    }

    pub fn getDeviceCaps(self: *Self, device_id: u32) interface.BackendError!interface.DeviceCaps {
        if (device_id != 0) return interface.BackendError.DeviceNotFound;

        var caps = interface.DeviceCaps{};
        @memcpy(caps.name[0..self.device_name_len], self.device_name[0..self.device_name_len]);
        caps.name_len = self.device_name_len;

        const version = opengles.getVersion();
        caps.compute_capability_major = @intCast(version.major);
        caps.compute_capability_minor = @intCast(version.minor);

        // Conservative OpenGL ES compute defaults.
        caps.max_threads_per_block = 512;
        caps.max_shared_memory = 16384;
        caps.warp_size = 32;
        caps.supports_fp16 = false;
        caps.supports_fp64 = false;
        caps.supports_int8 = true;
        caps.unified_memory = false;

        return caps;
    }

    pub fn allocate(self: *Self, size: usize, flags: interface.MemoryFlags) interface.MemoryError!*anyopaque {
        _ = flags;
        const ptr = gl_runtime.allocateDeviceMemory(.opengles, size) catch {
            return interface.MemoryError.OutOfMemory;
        };
        self.allocations.append(self.allocator, .{
            .ptr = ptr,
            .size = size,
        }) catch {
            gl_runtime.freeDeviceMemory(.opengles, ptr);
            return interface.MemoryError.OutOfMemory;
        };
        return ptr;
    }

    pub fn free(self: *Self, ptr: *anyopaque) void {
        for (self.allocations.items, 0..) |alloc, i| {
            if (alloc.ptr == ptr) {
                gl_runtime.freeDeviceMemory(.opengles, ptr);
                _ = self.allocations.swapRemove(i);
                return;
            }
        }
    }

    pub fn copyToDevice(_: *Self, dst: *anyopaque, src: []const u8) interface.MemoryError!void {
        gl_runtime.memcpyHostToDevice(.opengles, dst, src.ptr, src.len) catch {
            return interface.MemoryError.TransferFailed;
        };
    }

    pub fn copyFromDevice(_: *Self, dst: []u8, src: *anyopaque) interface.MemoryError!void {
        gl_runtime.memcpyDeviceToHost(.opengles, dst.ptr, src, dst.len) catch {
            return interface.MemoryError.TransferFailed;
        };
    }

    pub fn copyToDeviceAsync(self: *Self, dst: *anyopaque, src: []const u8, stream: ?*anyopaque) interface.MemoryError!void {
        _ = stream;
        return self.copyToDevice(dst, src);
    }

    pub fn copyFromDeviceAsync(self: *Self, dst: []u8, src: *anyopaque, stream: ?*anyopaque) interface.MemoryError!void {
        _ = stream;
        return self.copyFromDevice(dst, src);
    }

    pub fn compileKernel(
        self: *Self,
        allocator: std.mem.Allocator,
        source: []const u8,
        kernel_name: []const u8,
    ) interface.KernelError!*anyopaque {
        const kernel_types = @import("../kernel_types.zig");
        const backend_mod = @import("../backend.zig");

        const kernel_source = kernel_types.KernelSource{
            .source = source,
            .name = kernel_name,
            .entry_point = "main",
            .backend = backend_mod.Backend.opengles,
        };

        const handle = gl_runtime.compileKernel(.opengles, allocator, kernel_source) catch {
            return interface.KernelError.CompileFailed;
        };

        const name_copy = self.allocator.dupe(u8, kernel_name) catch {
            gl_runtime.destroyKernel(.opengles, allocator, handle);
            return interface.KernelError.CompileFailed;
        };

        self.kernels.append(self.allocator, .{
            .handle = handle,
            .name = name_copy,
        }) catch {
            self.allocator.free(name_copy);
            gl_runtime.destroyKernel(.opengles, allocator, handle);
            return interface.KernelError.CompileFailed;
        };

        return handle;
    }

    pub fn launchKernel(
        _: *Self,
        kernel: *anyopaque,
        config: interface.LaunchConfig,
        args: []const *anyopaque,
    ) interface.KernelError!void {
        const kernel_types = @import("../kernel_types.zig");

        if (config.block_x == 0 or config.block_y == 0 or config.block_z == 0) {
            return interface.KernelError.InvalidConfig;
        }
        if (config.grid_x == 0 or config.grid_y == 0 or config.grid_z == 0) {
            return interface.KernelError.InvalidConfig;
        }

        const kernel_config = kernel_types.KernelConfig{
            .grid_dim = .{ config.grid_x, config.grid_y, config.grid_z },
            .block_dim = .{ config.block_x, config.block_y, config.block_z },
            .shared_memory_bytes = config.shared_memory,
        };

        var opt_args: [32]?*const anyopaque = .{null} ** 32;
        const arg_count = @min(args.len, 32);
        for (0..arg_count) |i| {
            opt_args[i] = args[i];
        }

        gl_runtime.launchKernel(
            .opengles,
            std.heap.page_allocator,
            kernel,
            kernel_config,
            opt_args[0..arg_count],
        ) catch {
            return interface.KernelError.LaunchFailed;
        };
    }

    pub fn destroyKernel(self: *Self, kernel: *anyopaque) void {
        for (self.kernels.items, 0..) |k, i| {
            if (k.handle == kernel) {
                gl_runtime.destroyKernel(.opengles, self.allocator, kernel);
                self.allocator.free(k.name);
                _ = self.kernels.swapRemove(i);
                return;
            }
        }
    }

    pub fn synchronize(_: *Self) interface.BackendError!void {
        if (!opengles.isAvailable()) {
            return interface.BackendError.NotAvailable;
        }
        gl_runtime.synchronize(.opengles);
    }
};

pub fn createOpenGLESVTable(allocator: std.mem.Allocator) interface.BackendError!interface.Backend {
    const impl = try OpenGLESBackend.init(allocator);
    return interface.createBackend(OpenGLESBackend, impl);
}

test "OpenGLESBackend initialization" {
    const allocator = std.testing.allocator;

    const result = OpenGLESBackend.init(allocator);
    if (result) |backend| {
        defer backend.deinit();
        try std.testing.expect(backend.initialized);
    } else |err| {
        try std.testing.expect(err == error.NotAvailable or
            err == error.InitFailed or
            err == error.DriverNotFound);
    }
}

test "createOpenGLESVTable" {
    const allocator = std.testing.allocator;

    const result = createOpenGLESVTable(allocator);
    if (result) |backend| {
        defer backend.deinit();
        const count = backend.getDeviceCount();
        try std.testing.expect(count >= 0);
    } else |err| {
        try std.testing.expect(err == error.NotAvailable or
            err == error.InitFailed or
            err == error.DriverNotFound);
    }
}

test {
    std.testing.refAllDecls(@This());
}
