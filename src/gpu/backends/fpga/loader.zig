//! FPGA Bitstream Loader
//!
//! Handles loading and managing FPGA bitstreams for AMD/Xilinx (XRT)
//! and Intel (oneAPI/OpenCL) platforms.

const std = @import("std");
const build_options = @import("build_options");

pub const LoaderError = error{
    DeviceNotFound,
    BitstreamLoadFailed,
    InvalidBitstream,
    PlatformNotSupported,
    DriverNotFound,
    DeviceBusy,
    InsufficientResources,
    InitializationFailed,
};

/// FPGA platform type
pub const Platform = enum {
    xilinx, // AMD/Xilinx via XRT
    intel, // Intel via oneAPI/OpenCL
    unknown,

    pub fn name(self: Platform) []const u8 {
        return switch (self) {
            .xilinx => "AMD/Xilinx (XRT)",
            .intel => "Intel (oneAPI)",
            .unknown => "Unknown",
        };
    }
};

/// Device information for an FPGA
pub const DeviceInfo = struct {
    index: u32,
    platform: Platform,
    name: [256]u8 = undefined,
    name_len: usize = 0,
    bdf: [16]u8 = undefined, // PCI Bus:Device.Function
    bdf_len: usize = 0,
    ddr_size_bytes: u64 = 0,
    hbm_size_bytes: u64 = 0,
    num_compute_units: u32 = 0,
    clock_frequency_mhz: u32 = 0,
    is_available: bool = false,

    pub fn getName(self: *const DeviceInfo) []const u8 {
        return self.name[0..self.name_len];
    }

    pub fn getBdf(self: *const DeviceInfo) []const u8 {
        return self.bdf[0..self.bdf_len];
    }
};

/// Handle to a loaded bitstream
pub const BitstreamHandle = struct {
    allocator: std.mem.Allocator,
    device_index: u32,
    platform: Platform,
    uuid: [16]u8 = undefined,
    kernel_names: std.ArrayListUnmanaged([]const u8) = .{},
    is_loaded: bool = false,

    pub fn deinit(self: *BitstreamHandle) void {
        for (self.kernel_names.items) |name| {
            self.allocator.free(name);
        }
        self.kernel_names.deinit(self.allocator);
    }

    pub fn getKernelNames(self: *const BitstreamHandle) []const []const u8 {
        return self.kernel_names.items;
    }
};

/// Global state for FPGA loader
var initialized: bool = false;
var detected_devices: u32 = 0;
var device_cache: [8]DeviceInfo = undefined;

/// Initialize the FPGA loader subsystem
pub fn init() LoaderError!void {
    if (initialized) return;

    if (comptime !build_options.gpu_fpga) {
        return error.PlatformNotSupported;
    }

    // Detect available FPGA devices
    detected_devices = detectFpgaDevicesInternal();

    if (detected_devices == 0) {
        std.log.info("FPGA loader: No FPGA devices detected", .{});
    } else {
        std.log.info("FPGA loader: Detected {d} FPGA device(s)", .{detected_devices});
    }

    initialized = true;
}

/// Deinitialize the FPGA loader subsystem
pub fn deinit() void {
    if (!initialized) return;
    detected_devices = 0;
    initialized = false;
}

/// Detect available FPGA devices
pub fn detectFpgaDevices() u32 {
    if (comptime !build_options.gpu_fpga) return 0;
    if (!initialized) {
        return detectFpgaDevicesInternal();
    }
    return detected_devices;
}

fn detectFpgaDevicesInternal() u32 {
    var count: u32 = 0;

    // Try Xilinx XRT detection
    count += detectXilinxDevices();

    // Try Intel oneAPI detection
    count += detectIntelDevices();

    return count;
}

fn detectXilinxDevices() u32 {
    // In a real implementation, this would:
    // 1. Check for XRT library (libxrt_coreutil.so)
    // 2. Call xrt::system::enumerate_devices()
    // 3. Query device properties

    // For now, check environment variable for development/testing
    if (std.posix.getenv("ABI_FPGA_XILINX_DEVICE")) |_| {
        // Populate device cache with simulated device
        device_cache[0] = .{
            .index = 0,
            .platform = .xilinx,
            .ddr_size_bytes = 64 * 1024 * 1024 * 1024, // 64 GB
            .num_compute_units = 4,
            .clock_frequency_mhz = 300,
            .is_available = true,
        };
        const name = "Alveo U250";
        @memcpy(device_cache[0].name[0..name.len], name);
        device_cache[0].name_len = name.len;
        return 1;
    }

    return 0;
}

fn detectIntelDevices() u32 {
    // In a real implementation, this would:
    // 1. Check for Intel FPGA SDK/oneAPI
    // 2. Use OpenCL platform enumeration
    // 3. Filter for FPGA devices

    // For now, check environment variable for development/testing
    if (std.posix.getenv("ABI_FPGA_INTEL_DEVICE")) |_| {
        const idx: usize = if (detected_devices < 8) detected_devices else 7;
        device_cache[idx] = .{
            .index = @intCast(idx),
            .platform = .intel,
            .ddr_size_bytes = 32 * 1024 * 1024 * 1024, // 32 GB
            .num_compute_units = 2,
            .clock_frequency_mhz = 400,
            .is_available = true,
        };
        const name = "Agilex 7";
        @memcpy(device_cache[idx].name[0..name.len], name);
        device_cache[idx].name_len = name.len;
        return 1;
    }

    return 0;
}

/// Get information about a specific FPGA device
pub fn getDeviceInfo(device_index: u32) LoaderError!DeviceInfo {
    if (device_index >= detected_devices) {
        return error.DeviceNotFound;
    }
    return device_cache[device_index];
}

/// Load a bitstream onto an FPGA device
pub fn loadBitstream(
    allocator: std.mem.Allocator,
    device_index: u32,
    bitstream_path: []const u8,
) LoaderError!BitstreamHandle {
    if (device_index >= detected_devices) {
        return error.DeviceNotFound;
    }

    const device = device_cache[device_index];

    // Validate bitstream file exists (Zig 0.16 I/O API)
    var io_backend = std.Io.Threaded.init(allocator, .{
        .environ = std.process.Environ.empty,
    });
    defer io_backend.deinit();
    const io = io_backend.io();

    std.Io.Dir.cwd().access(io, bitstream_path, .{}) catch {
        std.log.err("FPGA loader: Bitstream file not found: {s}", .{bitstream_path});
        return error.InvalidBitstream;
    };

    // In a real implementation:
    // - Xilinx: xrt::device::load_xclbin()
    // - Intel: clCreateProgramWithBinary()

    var handle = BitstreamHandle{
        .allocator = allocator,
        .device_index = device_index,
        .platform = device.platform,
        .is_loaded = true,
    };

    // Extract kernel names from bitstream metadata
    // For now, register common kernel names
    const common_kernels = [_][]const u8{
        "vector_distance",
        "quantized_matmul",
        "softmax",
        "kmeans_assign",
        "hnsw_search",
    };

    for (common_kernels) |kernel_name| {
        const name_copy = try allocator.dupe(u8, kernel_name);
        try handle.kernel_names.append(allocator, name_copy);
    }

    std.log.info("FPGA loader: Loaded bitstream on device {d} ({s})", .{
        device_index,
        device.getName(),
    });

    return handle;
}

/// Load a bitstream from memory
pub fn loadBitstreamFromMemory(
    allocator: std.mem.Allocator,
    device_index: u32,
    bitstream_data: []const u8,
) LoaderError!BitstreamHandle {
    if (device_index >= detected_devices) {
        return error.DeviceNotFound;
    }

    if (bitstream_data.len < 64) {
        return error.InvalidBitstream;
    }

    const device = device_cache[device_index];

    // In a real implementation, parse bitstream header and load
    var handle = BitstreamHandle{
        .allocator = allocator,
        .device_index = device_index,
        .platform = device.platform,
        .is_loaded = true,
    };

    std.log.info("FPGA loader: Loaded bitstream from memory on device {d}", .{device_index});

    return handle;
}

/// Unload a bitstream from an FPGA device
pub fn unloadBitstream(handle: *BitstreamHandle) void {
    if (!handle.is_loaded) return;

    // In a real implementation:
    // - Release device resources
    // - Clear programmed configuration

    handle.is_loaded = false;
    handle.deinit();

    std.log.info("FPGA loader: Unloaded bitstream from device {d}", .{handle.device_index});
}

/// Check if a specific kernel is available in the loaded bitstream
pub fn hasKernel(handle: *const BitstreamHandle, kernel_name: []const u8) bool {
    for (handle.kernel_names.items) |name| {
        if (std.mem.eql(u8, name, kernel_name)) {
            return true;
        }
    }
    return false;
}

test "loader initialization" {
    if (comptime !build_options.gpu_fpga) return;

    try init();
    defer deinit();

    const count = detectFpgaDevices();
    _ = count;
}
