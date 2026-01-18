//! FPGA Backend VTable Implementation
//!
//! Implements the GPU backend interface for FPGA devices,
//! enabling seamless integration with the unified GPU API.

const std = @import("std");
const build_options = @import("build_options");
const interface = @import("../../interface.zig");
const loader = @import("loader.zig");
const memory = @import("memory.zig");
const kernels = @import("kernels.zig");

pub const FpgaError = error{
    InitFailed,
    NotAvailable,
    DeviceNotFound,
    OutOfMemory,
    KernelNotFound,
    KernelLaunchFailed,
    InvalidConfig,
    BitstreamNotLoaded,
    Timeout,
    FpgaDisabled,
};

/// FPGA platform selection
pub const FpgaPlatform = enum {
    auto, // Auto-detect
    xilinx, // AMD/Xilinx via XRT
    intel, // Intel via oneAPI
};

/// Configuration for FPGA backend
pub const FpgaConfig = struct {
    /// Preferred platform (auto-detect if null)
    platform: FpgaPlatform = .auto,
    /// Device index to use
    device_index: u32 = 0,
    /// Path to pre-compiled bitstream (optional)
    bitstream_path: ?[]const u8 = null,
    /// Enable profiling
    enable_profiling: bool = false,
    /// Default memory bank
    default_memory_bank: memory.MemoryBank = .auto,
};

/// Kernel handle for FPGA
pub const FpgaKernelHandle = struct {
    allocator: std.mem.Allocator,
    name: []const u8,
    kernel_type: kernels.FpgaKernelType,
    compiled: bool = false,
};

/// FPGA Backend implementation
pub const FpgaBackend = struct {
    allocator: std.mem.Allocator,
    config: FpgaConfig,
    device_info: loader.DeviceInfo,
    mem_manager: memory.FpgaMemory,
    bitstream: ?loader.BitstreamHandle = null,
    is_initialized: bool = false,

    // Statistics
    kernel_launches: u64 = 0,
    total_transfer_bytes: u64 = 0,

    pub fn init(allocator: std.mem.Allocator, config: FpgaConfig) !*FpgaBackend {
        if (comptime !build_options.gpu_fpga) {
            return error.FpgaDisabled;
        }

        // Initialize loader if needed
        loader.init() catch |err| {
            std.log.err("FPGA: Failed to initialize loader: {t}", .{err});
            return error.InitFailed;
        };

        // Get device info
        const device_info = loader.getDeviceInfo(config.device_index) catch {
            return error.DeviceNotFound;
        };

        // Initialize memory manager
        var mem_manager = memory.FpgaMemory.init(allocator, config.device_index) catch {
            return error.OutOfMemory;
        };

        const self = try allocator.create(FpgaBackend);
        errdefer allocator.destroy(self);

        self.* = FpgaBackend{
            .allocator = allocator,
            .config = config,
            .device_info = device_info,
            .mem_manager = mem_manager,
        };

        // Load bitstream if provided
        if (config.bitstream_path) |path| {
            self.bitstream = loader.loadBitstream(allocator, config.device_index, path) catch |err| {
                std.log.warn("FPGA: Failed to load bitstream: {t}", .{err});
                // Continue without bitstream - kernels can be loaded later
            };
        }

        self.is_initialized = true;

        std.log.info("FPGA backend initialized: device={d} platform={s}", .{
            config.device_index,
            device_info.platform.name(),
        });

        return self;
    }

    pub fn deinit(self: *FpgaBackend) void {
        if (self.bitstream) |*bs| {
            loader.unloadBitstream(bs);
        }
        self.mem_manager.deinit();
        self.is_initialized = false;
        self.allocator.destroy(self);
    }

    // =========================================================================
    // GPU Interface Implementation
    // =========================================================================

    pub fn getDeviceCount(self: *FpgaBackend) u32 {
        _ = self;
        return loader.detectFpgaDevices();
    }

    pub fn getDeviceCaps(self: *FpgaBackend, device_id: u32) interface.BackendError!interface.DeviceCaps {
        const info = loader.getDeviceInfo(device_id) catch {
            return error.DeviceNotFound;
        };

        var caps = interface.DeviceCaps{
            .total_memory = info.ddr_size_bytes + info.hbm_size_bytes,
            .compute_capability_major = 1,
            .compute_capability_minor = 0,
            .max_threads_per_block = 256, // FPGA work-item concept differs
            .max_shared_memory = 4 * 1024 * 1024, // PLRAM
            .warp_size = 1, // FPGA doesn't have warps
            .supports_fp16 = true,
            .supports_fp64 = true,
            .supports_int8 = true,
            .unified_memory = false,
            .async_engine_count = info.num_compute_units,
        };

        const name = info.getName();
        @memcpy(caps.name[0..name.len], name);
        caps.name_len = name.len;

        _ = self;
        return caps;
    }

    pub fn allocate(self: *FpgaBackend, size: usize, flags: interface.MemoryFlags) interface.MemoryError!*anyopaque {
        const fpga_flags = memory.MemoryFlags{
            .device = flags.device,
            .host_visible = flags.host_visible,
            .host_coherent = flags.host_coherent,
            .cached = flags.cached,
        };

        var buffer = self.mem_manager.allocate(size, self.config.default_memory_bank, fpga_flags) catch {
            return error.AllocationFailed;
        };

        // Store buffer info for later retrieval
        const buffer_ptr = self.allocator.create(memory.FpgaBuffer) catch {
            return error.OutOfMemory;
        };
        buffer_ptr.* = buffer;

        return @ptrCast(buffer_ptr);
    }

    pub fn free(self: *FpgaBackend, ptr: *anyopaque) void {
        const buffer_ptr: *memory.FpgaBuffer = @ptrCast(@alignCast(ptr));
        self.mem_manager.free(buffer_ptr);
        self.allocator.destroy(buffer_ptr);
    }

    pub fn copyToDevice(self: *FpgaBackend, dst: *anyopaque, src: []const u8) interface.MemoryError!void {
        const buffer_ptr: *memory.FpgaBuffer = @ptrCast(@alignCast(dst));
        self.mem_manager.copyToDevice(buffer_ptr, src) catch {
            return error.TransferFailed;
        };
        self.total_transfer_bytes += src.len;
    }

    pub fn copyFromDevice(self: *FpgaBackend, dst: []u8, src: *anyopaque) interface.MemoryError!void {
        const buffer_ptr: *memory.FpgaBuffer = @ptrCast(@alignCast(src));
        self.mem_manager.copyFromDevice(buffer_ptr, dst) catch {
            return error.TransferFailed;
        };
        self.total_transfer_bytes += dst.len;
    }

    pub fn compileKernel(
        self: *FpgaBackend,
        allocator: std.mem.Allocator,
        source: []const u8,
        kernel_name: []const u8,
    ) interface.KernelError!*anyopaque {
        _ = source; // FPGA kernels are pre-compiled in bitstream

        // Check if kernel is available in loaded bitstream
        if (self.bitstream) |*bs| {
            if (!loader.hasKernel(bs, kernel_name)) {
                std.log.warn("FPGA: Kernel '{s}' not found in bitstream", .{kernel_name});
                return error.KernelNotFound;
            }
        }

        // Create kernel handle
        const handle = allocator.create(FpgaKernelHandle) catch {
            return error.CompileFailed;
        };
        errdefer allocator.destroy(handle);

        const name_copy = allocator.dupe(u8, kernel_name) catch {
            return error.CompileFailed;
        };

        handle.* = FpgaKernelHandle{
            .allocator = allocator,
            .name = name_copy,
            .kernel_type = kernels.kernelTypeFromName(kernel_name),
            .compiled = true,
        };

        return @ptrCast(handle);
    }

    pub fn launchKernel(
        self: *FpgaBackend,
        kernel: *anyopaque,
        config: interface.LaunchConfig,
        args: []const *anyopaque,
    ) interface.KernelError!void {
        const handle: *FpgaKernelHandle = @ptrCast(@alignCast(kernel));

        if (!handle.compiled) {
            return error.KernelNotFound;
        }

        // In a real implementation:
        // - Xilinx: xrt::run object with kernel arguments
        // - Intel: clSetKernelArg + clEnqueueNDRangeKernel

        // Execute simulated kernel
        kernels.executeKernel(handle.kernel_type, config, args) catch |err| {
            std.log.err("FPGA: Kernel launch failed: {t}", .{err});
            return error.LaunchFailed;
        };

        self.kernel_launches += 1;
    }

    pub fn destroyKernel(self: *FpgaBackend, kernel: *anyopaque) void {
        _ = self;
        const handle: *FpgaKernelHandle = @ptrCast(@alignCast(kernel));
        handle.allocator.free(handle.name);
        handle.allocator.destroy(handle);
    }

    pub fn synchronize(self: *FpgaBackend) interface.BackendError!void {
        // In a real implementation:
        // - Xilinx: xrt::run::wait()
        // - Intel: clFinish(queue)

        _ = self;
        // For simulation, nothing to do
    }

    // =========================================================================
    // FPGA-Specific Operations
    // =========================================================================

    /// Load a new bitstream
    pub fn loadBitstream(self: *FpgaBackend, path: []const u8) !void {
        if (self.bitstream) |*bs| {
            loader.unloadBitstream(bs);
        }

        self.bitstream = try loader.loadBitstream(
            self.allocator,
            self.config.device_index,
            path,
        );
    }

    /// Get memory statistics
    pub fn getMemoryStats(self: *const FpgaBackend) memory.MemoryStats {
        return self.mem_manager.getStats();
    }

    /// Get execution statistics
    pub fn getStats(self: *const FpgaBackend) FpgaStats {
        return FpgaStats{
            .kernel_launches = self.kernel_launches,
            .total_transfer_bytes = self.total_transfer_bytes,
            .memory_stats = self.mem_manager.getStats(),
        };
    }
};

pub const FpgaStats = struct {
    kernel_launches: u64,
    total_transfer_bytes: u64,
    memory_stats: memory.MemoryStats,
};

/// Create the VTable for GPU interface integration
pub fn createVTable() interface.Backend.VTable {
    return interface.Backend.VTable{
        .deinit = struct {
            fn f(ptr: *anyopaque) void {
                const self: *FpgaBackend = @ptrCast(@alignCast(ptr));
                self.deinit();
            }
        }.f,
        .getDeviceCount = struct {
            fn f(ptr: *anyopaque) u32 {
                const self: *FpgaBackend = @ptrCast(@alignCast(ptr));
                return self.getDeviceCount();
            }
        }.f,
        .getDeviceCaps = struct {
            fn f(ptr: *anyopaque, device_id: u32) interface.BackendError!interface.DeviceCaps {
                const self: *FpgaBackend = @ptrCast(@alignCast(ptr));
                return self.getDeviceCaps(device_id);
            }
        }.f,
        .allocate = struct {
            fn f(ptr: *anyopaque, size: usize, flags: interface.MemoryFlags) interface.MemoryError!*anyopaque {
                const self: *FpgaBackend = @ptrCast(@alignCast(ptr));
                return self.allocate(size, flags);
            }
        }.f,
        .free = struct {
            fn f(ptr: *anyopaque, mem: *anyopaque) void {
                const self: *FpgaBackend = @ptrCast(@alignCast(ptr));
                self.free(mem);
            }
        }.f,
        .copyToDevice = struct {
            fn f(ptr: *anyopaque, dst: *anyopaque, src: []const u8) interface.MemoryError!void {
                const self: *FpgaBackend = @ptrCast(@alignCast(ptr));
                return self.copyToDevice(dst, src);
            }
        }.f,
        .copyFromDevice = struct {
            fn f(ptr: *anyopaque, dst: []u8, src: *anyopaque) interface.MemoryError!void {
                const self: *FpgaBackend = @ptrCast(@alignCast(ptr));
                return self.copyFromDevice(dst, src);
            }
        }.f,
        .compileKernel = struct {
            fn f(ptr: *anyopaque, allocator: std.mem.Allocator, source: []const u8, kernel_name: []const u8) interface.KernelError!*anyopaque {
                const self: *FpgaBackend = @ptrCast(@alignCast(ptr));
                return self.compileKernel(allocator, source, kernel_name);
            }
        }.f,
        .launchKernel = struct {
            fn f(ptr: *anyopaque, kernel: *anyopaque, config: interface.LaunchConfig, args: []const *anyopaque) interface.KernelError!void {
                const self: *FpgaBackend = @ptrCast(@alignCast(ptr));
                return self.launchKernel(kernel, config, args);
            }
        }.f,
        .destroyKernel = struct {
            fn f(ptr: *anyopaque, kernel: *anyopaque) void {
                const self: *FpgaBackend = @ptrCast(@alignCast(ptr));
                self.destroyKernel(kernel);
            }
        }.f,
        .synchronize = struct {
            fn f(ptr: *anyopaque) interface.BackendError!void {
                const self: *FpgaBackend = @ptrCast(@alignCast(ptr));
                return self.synchronize();
            }
        }.f,
    };
}

test "fpga vtable creation" {
    const vtable = createVTable();
    try std.testing.expect(vtable.deinit != undefined);
    try std.testing.expect(vtable.allocate != undefined);
}
