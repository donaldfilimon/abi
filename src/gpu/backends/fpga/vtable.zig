//! FPGA VTable Backend Implementation
//!
//! Implements the Backend interface for FPGA accelerators.
//! Targeted for AMD Alveo and Intel Agilex platforms as per research docs.

const std = @import("std");
const interface = @import("../../interface.zig");
const kernels = @import("kernels.zig");
const fpga_mod = @import("mod.zig");
const loader = @import("loader.zig");

pub const FpgaBackend = struct {
    allocator: std.mem.Allocator,

    const Self = @This();

    pub fn init(allocator: std.mem.Allocator) interface.BackendError!*Self {
        const self = allocator.create(Self) catch return interface.BackendError.OutOfMemory;
        self.* = .{
            .allocator = allocator,
        };

        // Initialize FPGA loader (does detection)
        loader.init() catch {
            // Even if loader fails, we can run in simulation mode
            std.log.warn("FPGA backend: Loader initialization failed, running in simulation mode", .{});
        };

        // Initialize FPGA module
        fpga_mod.init() catch return interface.BackendError.InitFailed;

        return self;
    }

    pub fn deinit(self: *Self) void {
        fpga_mod.deinit();
        loader.deinit();
        self.allocator.destroy(self);
    }

    pub fn getDeviceCount(self: *Self) u32 {
        _ = self;
        return loader.detectFpgaDevices();
    }

    pub fn getDeviceCaps(self: *Self, device_id: u32) interface.BackendError!interface.DeviceCaps {
        _ = self;

        // Get device info from loader
        const device_info = loader.getDeviceInfo(device_id) catch {
            return interface.BackendError.DeviceNotFound;
        };

        var caps = interface.DeviceCaps{};

        // Copy device name
        const name = device_info.getName();
        const copy_len = @min(name.len, caps.name.len);
        @memcpy(caps.name[0..copy_len], name[0..copy_len]);
        caps.name_len = copy_len;

        // Set memory size (use DDR or HBM, whichever is larger)
        caps.total_memory = if (device_info.hbm_size_bytes > 0)
            device_info.hbm_size_bytes
        else
            device_info.ddr_size_bytes;

        // FPGA-specific capabilities
        caps.max_threads_per_block = 1; // Task parallelism model
        caps.max_shared_memory = 32 * 1024 * 1024; // Typical FPGA URAM/BRAM
        caps.warp_size = 1; // No warps on FPGA
        caps.supports_fp16 = true; // Most FPGAs support FP16
        caps.supports_fp64 = false; // Limited FP64 support
        caps.supports_int8 = true; // Native quantized support
        caps.unified_memory = false; // Separate host/device memory
        caps.compute_capability_major = @intCast(device_info.num_compute_units);
        caps.compute_capability_minor = @intCast(device_info.clock_frequency_mhz / 100);
        caps.async_engine_count = 1; // Single async engine typical

        return caps;
    }

    pub fn allocate(self: *Self, size: usize, flags: interface.MemoryFlags) interface.MemoryError!*anyopaque {
        _ = self;
        _ = size;
        _ = flags;
        return interface.MemoryError.NotImplemented;
    }

    pub fn free(self: *Self, ptr: *anyopaque) void {
        _ = self;
        _ = ptr;
    }

    pub fn copyToDevice(self: *Self, dst: *anyopaque, src: []const u8) interface.MemoryError!void {
        _ = self;
        _ = dst;
        _ = src;
        return interface.MemoryError.NotImplemented;
    }

    pub fn copyFromDevice(self: *Self, dst: []u8, src: *anyopaque) interface.MemoryError!void {
        _ = self;
        _ = dst;
        _ = src;
        return interface.MemoryError.NotImplemented;
    }

    pub fn copyToDeviceAsync(self: *Self, dst: *anyopaque, src: []const u8, stream: ?*anyopaque) interface.MemoryError!void {
        _ = self;
        _ = dst;
        _ = src;
        _ = stream;
        return interface.MemoryError.NotImplemented;
    }

    pub fn copyFromDeviceAsync(self: *Self, dst: []u8, src: *anyopaque, stream: ?*anyopaque) interface.MemoryError!void {
        _ = self;
        _ = dst;
        _ = src;
        _ = stream;
        return interface.MemoryError.NotImplemented;
    }

    pub fn compileKernel(self: *Self, allocator: std.mem.Allocator, source: []const u8, name: []const u8) interface.KernelError!*anyopaque {
        _ = self;
        _ = allocator;
        _ = source;
        _ = name;
        return interface.KernelError.NotImplemented;
    }

    pub fn launchKernel(self: *Self, kernel: *anyopaque, config: interface.LaunchConfig, args: []const *anyopaque) interface.KernelError!void {
        _ = self;
        _ = kernel;
        _ = config;
        _ = args;
        return interface.KernelError.NotImplemented;
    }

    pub fn destroyKernel(self: *Self, kernel: *anyopaque) void {
        _ = self;
        _ = kernel;
    }

    pub fn synchronize(self: *Self) interface.BackendError!void {
        _ = self;
        return {};
    }
};

pub fn createFpgaVTable(allocator: std.mem.Allocator) interface.BackendError!interface.Backend {
    const impl = try FpgaBackend.init(allocator);
    return interface.createBackend(FpgaBackend, impl);
}
