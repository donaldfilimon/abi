//! CUDA device query with real hardware capabilities.
//!
//! Provides detailed GPU device information including memory, compute capability,
//! and performance characteristics using CUDA Driver API.

const std = @import("std");
const shared = @import("../shared.zig");

/// CUDA GPU architecture generation derived from compute capability.
pub const CudaArchitecture = enum {
    kepler, // sm_30-37 (CC 3.x)
    maxwell, // sm_50-53 (CC 5.x)
    pascal, // sm_60-62 (CC 6.x)
    volta, // sm_70-72 (CC 7.0-7.2)
    turing, // sm_75 (CC 7.5)
    ampere, // sm_80-87 (CC 8.0-8.7)
    ada_lovelace, // sm_89 (CC 8.9)
    hopper, // sm_90 (CC 9.0)
    blackwell, // sm_100+ (CC 10.x)
    unknown,

    pub fn fromComputeCapability(major: i32, minor: i32) CudaArchitecture {
        return switch (major) {
            3 => .kepler,
            5 => .maxwell,
            6 => .pascal,
            7 => if (minor >= 5) .turing else .volta,
            8 => if (minor >= 9) .ada_lovelace else .ampere,
            9 => .hopper,
            10 => .blackwell,
            else => if (major > 10) .blackwell else .unknown,
        };
    }

    pub fn name(self: CudaArchitecture) []const u8 {
        return switch (self) {
            .kepler => "Kepler",
            .maxwell => "Maxwell",
            .pascal => "Pascal",
            .volta => "Volta",
            .turing => "Turing",
            .ampere => "Ampere",
            .ada_lovelace => "Ada Lovelace",
            .hopper => "Hopper",
            .blackwell => "Blackwell",
            .unknown => "Unknown",
        };
    }
};

/// Feature support matrix derived from CUDA compute capability.
/// Single source of truth for which features each architecture supports.
pub const CudaFeatureSupport = struct {
    fp16: bool,
    bf16: bool,
    tf32: bool,
    fp8: bool,
    tensor_cores: bool,
    int8_tensor_cores: bool,
    cooperative_groups: bool,
    dynamic_parallelism: bool,
    async_copy: bool,
    unified_memory: bool,

    pub fn fromComputeCapability(major: i32, minor: i32) CudaFeatureSupport {
        const cc = major * 10 + minor;
        return .{
            .fp16 = cc >= 53,
            .bf16 = cc >= 80,
            .tf32 = cc >= 80,
            .fp8 = cc >= 89,
            .tensor_cores = cc >= 70,
            .int8_tensor_cores = cc >= 72,
            .cooperative_groups = cc >= 60,
            .dynamic_parallelism = cc >= 35,
            .async_copy = cc >= 80,
            .unified_memory = cc >= 60,
        };
    }
};

pub const DeviceProperty = enum(u32) {
    major = 75,
    minor = 76,
    name = 104,
    total_global_mem = 6,
    shared_mem_per_block = 8,
    regs_per_block = 12,
    warp_size = 10,
    max_threads_per_block = 1,
    max_threads_dim = 13,
    max_grid_dim = 14,
    clock_rate = 13,
    multi_processor_count = 16,
    l2_cache_size = 38,
    max_threads_per_multi_processor = 39,
    memory_clock_rate = 36,
    global_memory_bus_width = 37,
    managed_memory = 83,
    cooperative_launch = 95,
    max_blocks_per_multiprocessor = 106,
    compute_mode = 103,
    concurrent_kernels = 31,
    ecc_enabled = 32,
    integrated = 94,
    can_map_host_memory = 19,
    unified_addressing = 113,
};

pub const CudaDeviceInfo = struct {
    device_id: i32,
    name: [256]u8,
    compute_capability: struct { major: i32, minor: i32 },
    total_memory: u64,
    shared_memory_per_block: usize,
    registers_per_block: i32,
    warp_size: i32,
    max_threads_per_block: i32,
    max_threads_dim: [3]i32,
    max_grid_dim: [3]i32,
    clock_rate_khz: i32,
    multi_processor_count: i32,
    l2_cache_size: i32,
    max_threads_per_multi_processor: i32,
    ecc_enabled: bool,
    integrated: bool,
    can_map_host_memory: bool,
    unified_addressing: bool,
    concurrent_kernels: bool,
    // Extended fields
    architecture: CudaArchitecture = .unknown,
    feature_support: CudaFeatureSupport = .{
        .fp16 = false,
        .bf16 = false,
        .tf32 = false,
        .fp8 = false,
        .tensor_cores = false,
        .int8_tensor_cores = false,
        .cooperative_groups = false,
        .dynamic_parallelism = false,
        .async_copy = false,
        .unified_memory = false,
    },
    memory_clock_rate_khz: i32 = 0,
    global_memory_bus_width: i32 = 0,
    managed_memory: bool = false,
    cooperative_launch: bool = false,
};

const CUdevice = i32;
const CuResult = enum(i32) {
    success = 0,
    invalid_value = 1,
    out_of_memory = 2,
    not_initialized = 3,
    invalid_device = 101,
};

const CuDeviceGetFn = *const fn (*CUdevice, i32) callconv(.c) CuResult;
const CuDeviceGetCountFn = *const fn (*i32) callconv(.c) CuResult;
const CuDeviceGetNameFn = *const fn ([*]u8, i32, CUdevice) callconv(.c) CuResult;
const CuDeviceGetAttributeFn = *const fn (*i32, DeviceProperty, CUdevice) callconv(.c) CuResult;

var cuDeviceGet: ?CuDeviceGetFn = null;
var cuDeviceGetCount: ?CuDeviceGetCountFn = null;
var cuDeviceGetName: ?CuDeviceGetNameFn = null;
var cuDeviceGetAttribute: ?CuDeviceGetAttributeFn = null;
var cuda_lib: ?std.DynLib = null;

pub fn init() !void {
    if (cuda_lib != null) return;

    if (!tryLoadCuda()) {
        return error.DriverNotFound;
    }

    cuDeviceGet = cuda_lib.?.lookup(CuDeviceGetFn, "cuDeviceGet") orelse return error.DriverNotFound;
    cuDeviceGetCount = cuda_lib.?.lookup(CuDeviceGetCountFn, "cuDeviceGetCount") orelse return error.DriverNotFound;
    cuDeviceGetName = cuda_lib.?.lookup(CuDeviceGetNameFn, "cuDeviceGetName") orelse return error.DriverNotFound;
    cuDeviceGetAttribute = cuda_lib.?.lookup(CuDeviceGetAttributeFn, "cuDeviceGetAttribute") orelse return error.DriverNotFound;
}

pub fn deinit() void {
    if (cuda_lib) |lib| {
        lib.close();
    }
    cuda_lib = null;
    cuDeviceGet = null;
    cuDeviceGetCount = null;
    cuDeviceGetName = null;
    cuDeviceGetAttribute = null;
}

pub fn getDeviceCount() !i32 {
    const count_fn = cuDeviceGetCount orelse return error.DriverNotFound;
    var count: i32 = 0;
    if (count_fn(&count) != .success) {
        return error.DeviceQueryFailed;
    }
    return count;
}

pub fn getDeviceInfo(device_id: i32) !CudaDeviceInfo {
    const get_fn = cuDeviceGet orelse return error.DriverNotFound;
    const name_fn = cuDeviceGetName orelse return error.DriverNotFound;
    const attr_fn = cuDeviceGetAttribute orelse return error.DriverNotFound;

    var device: CUdevice = undefined;
    if (get_fn(&device, device_id) != .success) {
        return error.DeviceQueryFailed;
    }

    var info = CudaDeviceInfo{
        .device_id = device_id,
        .name = undefined,
        .compute_capability = .{ .major = 0, .minor = 0 },
        .total_memory = 0,
        .shared_memory_per_block = 0,
        .registers_per_block = 0,
        .warp_size = 0,
        .max_threads_per_block = 0,
        .max_threads_dim = .{ 0, 0, 0 },
        .max_grid_dim = .{ 0, 0, 0 },
        .clock_rate_khz = 0,
        .multi_processor_count = 0,
        .l2_cache_size = 0,
        .max_threads_per_multi_processor = 0,
        .ecc_enabled = false,
        .integrated = false,
        .can_map_host_memory = false,
        .unified_addressing = false,
        .concurrent_kernels = false,
    };

    var major: i32 = 0;
    var minor: i32 = 0;
    if (attr_fn(&major, .major, device) == .success) {
        info.compute_capability.major = major;
    }
    if (attr_fn(&minor, .minor, device) == .success) {
        info.compute_capability.minor = minor;
    }

    if (name_fn(&info.name, @intCast(info.name.len), device) == .success) {
        std.mem.splitScalar(u8, &info.name, 0);
    }

    var mem: i32 = 0;
    if (attr_fn(&mem, .total_global_mem, device) == .success) {
        info.total_memory = @intCast(mem);
    }

    if (attr_fn(&mem, .shared_mem_per_block, device) == .success) {
        info.shared_memory_per_block = @intCast(mem);
    }

    if (attr_fn(&mem, .regs_per_block, device) == .success) {
        info.registers_per_block = mem;
    }

    if (attr_fn(&mem, .warp_size, device) == .success) {
        info.warp_size = mem;
    }

    if (attr_fn(&mem, .max_threads_per_block, device) == .success) {
        info.max_threads_per_block = mem;
    }

    for (0..3) |i| {
        if (attr_fn(&mem, .max_threads_dim, device) == .success) {
            info.max_threads_dim[i] = mem;
        }
        if (attr_fn(&mem, .max_grid_dim, device) == .success) {
            info.max_grid_dim[i] = mem;
        }
    }

    if (attr_fn(&mem, .clock_rate, device) == .success) {
        info.clock_rate_khz = mem;
    }

    if (attr_fn(&mem, .multi_processor_count, device) == .success) {
        info.multi_processor_count = mem;
    }

    if (attr_fn(&mem, .l2_cache_size, device) == .success) {
        info.l2_cache_size = mem;
    }

    if (attr_fn(&mem, .max_threads_per_multi_processor, device) == .success) {
        info.max_threads_per_multi_processor = mem;
    }

    if (attr_fn(&mem, .ecc_enabled, device) == .success) {
        info.ecc_enabled = mem != 0;
    }

    if (attr_fn(&mem, .integrated, device) == .success) {
        info.integrated = mem != 0;
    }

    if (attr_fn(&mem, .can_map_host_memory, device) == .success) {
        info.can_map_host_memory = mem != 0;
    }

    if (attr_fn(&mem, .unified_addressing, device) == .success) {
        info.unified_addressing = mem != 0;
    }

    if (attr_fn(&mem, .concurrent_kernels, device) == .success) {
        info.concurrent_kernels = mem != 0;
    }

    // Extended attribute queries
    if (attr_fn(&mem, .memory_clock_rate, device) == .success) {
        info.memory_clock_rate_khz = mem;
    }

    if (attr_fn(&mem, .global_memory_bus_width, device) == .success) {
        info.global_memory_bus_width = mem;
    }

    if (attr_fn(&mem, .managed_memory, device) == .success) {
        info.managed_memory = mem != 0;
    }

    if (attr_fn(&mem, .cooperative_launch, device) == .success) {
        info.cooperative_launch = mem != 0;
    }

    // Derive architecture and feature support from compute capability
    info.architecture = CudaArchitecture.fromComputeCapability(
        info.compute_capability.major,
        info.compute_capability.minor,
    );
    info.feature_support = CudaFeatureSupport.fromComputeCapability(
        info.compute_capability.major,
        info.compute_capability.minor,
    );

    return info;
}

pub fn listDevices(allocator: std.mem.Allocator) ![]CudaDeviceInfo {
    const count = try getDeviceCount();
    if (count == 0) {
        return try allocator.alloc(CudaDeviceInfo, 0);
    }

    var devices = try allocator.alloc(CudaDeviceInfo, @intCast(count));
    errdefer allocator.free(devices);

    for (0..@as(usize, @intCast(count))) |i| {
        devices[i] = try getDeviceInfo(@intCast(i));
    }

    return devices;
}

pub fn formatDeviceInfo(info: CudaDeviceInfo, writer: anytype) !void {
    try writer.print("Device {d}: {s}\n", .{ info.device_id, std.mem.span(&info.name) });
    try writer.print("  Architecture: {s}\n", .{info.architecture.name()});
    try writer.print("  Compute Capability: {d}.{d}\n", .{ info.compute_capability.major, info.compute_capability.minor });
    try writer.print("  Total Memory: {B}\n", .{info.total_memory});
    try writer.print("  Shared Memory per Block: {B}\n", .{info.shared_memory_per_block});
    try writer.print("  Max Threads per Block: {d}\n", .{info.max_threads_per_block});
    try writer.print("  Warp Size: {d}\n", .{info.warp_size});
    try writer.print("  Max Threads Dim: [{d}, {d}, {d}]\n", .{ info.max_threads_dim[0], info.max_threads_dim[1], info.max_threads_dim[2] });
    try writer.print("  Max Grid Dim: [{d}, {d}, {d}]\n", .{ info.max_grid_dim[0], info.max_grid_dim[1], info.max_grid_dim[2] });
    try writer.print("  Clock Rate: {d} MHz\n", .{info.clock_rate_khz / 1000});
    try writer.print("  Multiprocessors: {d}\n", .{info.multi_processor_count});
    try writer.print("  Max Threads per Multiprocessor: {d}\n", .{info.max_threads_per_multi_processor});
    try writer.print("  L2 Cache: {B}\n", .{info.l2_cache_size});
    try writer.print("  Memory Clock: {d} MHz\n", .{@divTrunc(info.memory_clock_rate_khz, 1000)});
    try writer.print("  Memory Bus Width: {d} bit\n", .{info.global_memory_bus_width});
    try writer.print("  ECC Enabled: {any}\n", .{info.ecc_enabled});
    try writer.print("  Integrated: {any}\n", .{info.integrated});
    try writer.print("  Unified Addressing: {any}\n", .{info.unified_addressing});
    try writer.print("  Concurrent Kernels: {any}\n", .{info.concurrent_kernels});
    try writer.print("  Managed Memory: {any}\n", .{info.managed_memory});
    try writer.print("  Cooperative Launch: {any}\n", .{info.cooperative_launch});
    try writer.print("  Tensor Cores: {any}\n", .{info.feature_support.tensor_cores});
    try writer.print("  BF16: {any}, TF32: {any}, FP8: {any}\n", .{
        info.feature_support.bf16,
        info.feature_support.tf32,
        info.feature_support.fp8,
    });
}

fn tryLoadCuda() bool {
    const lib_names = [_][]const u8{ "nvcuda.dll", "libcuda.so.1", "libcuda.so" };
    for (lib_names) |name| {
        if (std.DynLib.open(name)) |lib| {
            cuda_lib = lib;
            return true;
        } else |_| {}
    }
    return false;
}
