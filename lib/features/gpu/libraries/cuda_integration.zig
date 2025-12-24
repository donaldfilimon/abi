//! CUDA Integration for GPU-Accelerated Computing
//!
//! This module provides integration with CUDA runtime API for
//! GPU-accelerated computations and parallel processing.

const std = @import("std");
const gpu = @import("../mod.zig");

/// CUDA-specific error set
pub const CUDAError = error{
    /// CUDA runtime not initialized
    CUDANotInitialized,
    /// CUDA initialization failed
    CUDAInitFailed,
    /// No CUDA devices available
    NoCUDADevices,
    /// Failed to get CUDA device
    CUDADeviceGetFailed,
    /// Failed to create CUDA context
    CUDAContextCreateFailed,
    /// Failed to create CUDA stream
    CUDAStreamCreateFailed,
    /// CUDA memory allocation failed
    CUDAMemoryAllocationFailed,
    /// CUDA memory copy failed
    CUDAMemoryCopyFailed,
    /// CUDA kernel launch failed
    CUDAKernelLaunchFailed,
    /// CUDA synchronization failed
    CUDASyncFailed,
    /// Failed to get CUDA device count
    CUDADeviceCountFailed,
    /// CUDA kernel compilation failed
    CUDAKernelCompilationFailed,
    /// CUDA kernel function not found
    CUDAKernelFunctionNotFound,
    /// Failed to get CUDA device name
    CUDADeviceNameFailed,
    /// Failed to query CUDA memory
    CUDAMemoryQueryFailed,
    /// Failed to get CUDA device attribute
    CUDADeviceAttributeFailed,
};

// CUDA Runtime API declarations
extern "c" fn cuInit(flags: u32) c_int;
extern "c" fn cuDeviceGet(device: *c_int, ordinal: c_int) c_int;
extern "c" fn cuDeviceGetCount(count: *c_int) c_int;
extern "c" fn cuDeviceGetName(name: [*]u8, len: c_int, dev: c_int) c_int;
extern "c" fn cuDeviceTotalMem_v2(bytes: *usize, dev: c_int) c_int;
extern "c" fn cuDeviceGetAttribute(pi: *c_int, attrib: CuDeviceAttr, dev: c_int) c_int;
extern "c" fn cuCtxCreate_v2(pctx: *?*anyopaque, flags: u32, dev: c_int) c_int;
extern "c" fn cuCtxDestroy_v2(ctx: ?*anyopaque) c_int;
extern "c" fn cuStreamCreate(phStream: *?*anyopaque, Flags: u32) c_int;
extern "c" fn cuStreamDestroy_v2(hStream: ?*anyopaque) c_int;
extern "c" fn cuStreamSynchronize(hStream: ?*anyopaque) c_int;
extern "c" fn cuMemAlloc_v2(dptr: *?*anyopaque, bytesize: usize) c_int;
extern "c" fn cuMemFree_v2(dptr: ?*anyopaque) c_int;
extern "c" fn cuMemcpyHtoD_v2(dstDevice: ?*anyopaque, srcHost: ?*const anyopaque, ByteCount: usize) c_int;
extern "c" fn cuMemcpyDtoH_v2(dstHost: ?*anyopaque, srcDevice: ?*anyopaque, ByteCount: usize) c_int;
extern "c" fn cuMemcpyDtoD_v2(dstDevice: ?*anyopaque, srcDevice: ?*anyopaque, ByteCount: usize) c_int;
extern "c" fn cuLaunchKernel(
    f: ?*anyopaque,
    gridDimX: c_uint,
    gridDimY: c_uint,
    gridDimZ: c_uint,
    blockDimX: c_uint,
    blockDimY: c_uint,
    blockDimZ: c_uint,
    sharedMemBytes: c_uint,
    hStream: ?*anyopaque,
    kernelParams: ?[*]?*anyopaque,
    extra: ?[*]?*anyopaque,
) c_int;
extern "c" fn cuModuleLoadData(module: *?*anyopaque, image: ?*const anyopaque) c_int;
extern "c" fn cuModuleGetFunction(hfunc: *?*anyopaque, hmod: ?*anyopaque, name: [*]const u8) c_int;
extern "c" fn cuModuleUnload(hmod: ?*anyopaque) c_int;

// CUDA types and constants
pub const CUresult = c_int;
pub const CUdevice = c_int;
pub const CUcontext = ?*anyopaque;
pub const CUstream = ?*anyopaque;
pub const CUdeviceptr = ?*anyopaque;
pub const CUmodule = ?*anyopaque;
pub const CUfunction = ?*anyopaque;

pub const CUDA_SUCCESS: CUresult = 0;
pub const CU_DEVICE_ATTRIBUTE_MAX_THREADS_PER_BLOCK = 1;
pub const CU_DEVICE_ATTRIBUTE_MAX_BLOCK_DIM_X = 2;
pub const CU_DEVICE_ATTRIBUTE_MAX_BLOCK_DIM_Y = 3;
pub const CU_DEVICE_ATTRIBUTE_MAX_BLOCK_DIM_Z = 4;
pub const CU_DEVICE_ATTRIBUTE_MAX_GRID_DIM_X = 5;
pub const CU_DEVICE_ATTRIBUTE_MAX_GRID_DIM_Y = 6;
pub const CU_DEVICE_ATTRIBUTE_MAX_GRID_DIM_Z = 7;
pub const CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_BLOCK = 8;
pub const CU_DEVICE_ATTRIBUTE_TOTAL_CONSTANT_MEMORY = 9;
pub const CU_DEVICE_ATTRIBUTE_WARP_SIZE = 10;
pub const CU_DEVICE_ATTRIBUTE_MAX_PITCH = 11;
pub const CU_DEVICE_ATTRIBUTE_MAX_REGISTERS_PER_BLOCK = 12;
pub const CU_DEVICE_ATTRIBUTE_CLOCK_RATE = 13;
pub const CU_DEVICE_ATTRIBUTE_TEXTURE_ALIGNMENT = 14;
pub const CU_DEVICE_ATTRIBUTE_GPU_OVERLAP = 15;
pub const CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT = 16;
pub const CU_DEVICE_ATTRIBUTE_KERNEL_EXEC_TIMEOUT = 17;
pub const CU_DEVICE_ATTRIBUTE_INTEGRATED = 18;
pub const CU_DEVICE_ATTRIBUTE_CAN_MAP_HOST_MEMORY = 19;
pub const CU_DEVICE_ATTRIBUTE_COMPUTE_MODE = 20;
pub const CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE1D_WIDTH = 21;
pub const CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_WIDTH = 22;
pub const CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_HEIGHT = 23;
pub const CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE3D_WIDTH = 24;
pub const CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE3D_HEIGHT = 25;
pub const CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE3D_DEPTH = 26;
pub const CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_LAYERED_WIDTH = 27;
pub const CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_LAYERED_HEIGHT = 28;
pub const CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_LAYERED_LAYERS = 29;
pub const CU_DEVICE_ATTRIBUTE_SURFACE_ALIGNMENT = 30;
pub const CU_DEVICE_ATTRIBUTE_CONCURRENT_KERNELS = 31;
pub const CU_DEVICE_ATTRIBUTE_ECC_ENABLED = 32;
pub const CU_DEVICE_ATTRIBUTE_PCI_BUS_ID = 33;
pub const CU_DEVICE_ATTRIBUTE_PCI_DEVICE_ID = 34;
pub const CU_DEVICE_ATTRIBUTE_TCC_DRIVER = 35;
pub const CU_DEVICE_ATTRIBUTE_MEMORY_CLOCK_RATE = 36;
pub const CU_DEVICE_ATTRIBUTE_GLOBAL_MEMORY_BUS_WIDTH = 37;
pub const CU_DEVICE_ATTRIBUTE_L2_CACHE_SIZE = 38;
pub const CU_DEVICE_ATTRIBUTE_MAX_THREADS_PER_MULTIPROCESSOR = 39;
pub const CU_DEVICE_ATTRIBUTE_ASYNC_ENGINE_COUNT = 40;
pub const CU_DEVICE_ATTRIBUTE_UNIFIED_ADDRESSING = 41;
pub const CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE1D_LAYERED_WIDTH = 42;
pub const CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE1D_LAYERED_LAYERS = 43;
pub const CU_DEVICE_ATTRIBUTE_CAN_TEX2D_GATHER = 44;
pub const CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_GATHER_WIDTH = 45;
pub const CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_GATHER_HEIGHT = 46;
pub const CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE3D_WIDTH_ALTERNATE = 47;
pub const CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE3D_HEIGHT_ALTERNATE = 48;
pub const CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE3D_DEPTH_ALTERNATE = 49;
pub const CU_DEVICE_ATTRIBUTE_PCI_DOMAIN_ID = 50;
pub const CU_DEVICE_ATTRIBUTE_TEXTURE_PITCH_ALIGNMENT = 51;
pub const CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURECUBEMAP_WIDTH = 52;
pub const CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURECUBEMAP_LAYERED_WIDTH = 53;
pub const CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURECUBEMAP_LAYERED_LAYERS = 54;
pub const CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACE1D_WIDTH = 55;
pub const CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACE2D_WIDTH = 56;
pub const CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACE2D_HEIGHT = 57;
pub const CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACE3D_WIDTH = 58;
pub const CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACE3D_HEIGHT = 59;
pub const CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACE3D_DEPTH = 60;
pub const CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACE1D_LAYERED_WIDTH = 61;
pub const CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACE1D_LAYERED_LAYERS = 62;
pub const CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACE2D_LAYERED_WIDTH = 63;
pub const CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACE2D_LAYERED_HEIGHT = 64;
pub const CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACE2D_LAYERED_LAYERS = 65;
pub const CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACECUBEMAP_WIDTH = 66;
pub const CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACECUBEMAP_LAYERED_WIDTH = 67;
pub const CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACECUBEMAP_LAYERED_LAYERS = 68;
pub const CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE1D_LINEAR_WIDTH = 69;
pub const CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_LINEAR_WIDTH = 70;
pub const CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_LINEAR_HEIGHT = 71;
pub const CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_LINEAR_PITCH = 72;
pub const CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_MIPMAPPED_WIDTH = 73;
pub const CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_MIPMAPPED_HEIGHT = 74;
pub const CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR = 75;
pub const CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR = 76;
pub const CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE1D_MIPMAPPED_WIDTH = 77;
pub const CU_DEVICE_ATTRIBUTE_STREAM_PRIORITIES_SUPPORTED = 78;
pub const CU_DEVICE_ATTRIBUTE_GLOBAL_L1_CACHE_SUPPORTED = 79;
pub const CU_DEVICE_ATTRIBUTE_LOCAL_L1_CACHE_SUPPORTED = 80;
pub const CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_MULTIPROCESSOR = 81;
pub const CU_DEVICE_ATTRIBUTE_MAX_REGISTERS_PER_MULTIPROCESSOR = 82;
pub const CU_DEVICE_ATTRIBUTE_MANAGED_MEMORY = 83;
pub const CU_DEVICE_ATTRIBUTE_MULTI_GPU_BOARD = 84;
pub const CU_DEVICE_ATTRIBUTE_MULTI_GPU_BOARD_GROUP_ID = 85;
pub const CU_DEVICE_ATTRIBUTE_HOST_NATIVE_ATOMIC_SUPPORTED = 86;
pub const CU_DEVICE_ATTRIBUTE_SINGLE_TO_DOUBLE_PRECISION_PERF_RATIO = 87;
pub const CU_DEVICE_ATTRIBUTE_PAGEABLE_MEMORY_ACCESS = 88;
pub const CU_DEVICE_ATTRIBUTE_CONCURRENT_MANAGED_ACCESS = 89;
pub const CU_DEVICE_ATTRIBUTE_COMPUTE_PREEMPTION_SUPPORTED = 90;
pub const CU_DEVICE_ATTRIBUTE_CAN_USE_HOST_POINTER_FOR_REGISTERED_MEM = 91;
pub const CU_DEVICE_ATTRIBUTE_CAN_USE_STREAM_MEM_OPS = 92;
pub const CU_DEVICE_ATTRIBUTE_CAN_USE_64_BIT_STREAM_MEM_OPS = 93;
pub const CU_DEVICE_ATTRIBUTE_CAN_USE_STREAM_WAIT_VALUE_NOR = 94;
pub const CU_DEVICE_ATTRIBUTE_COOPERATIVE_LAUNCH = 95;
pub const CU_DEVICE_ATTRIBUTE_COOPERATIVE_MULTI_DEVICE_LAUNCH = 96;
pub const CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_BLOCK_OPTIN = 97;
pub const CU_DEVICE_ATTRIBUTE_PAGEABLE_MEMORY_ACCESS_USES_HOST_PAGE_TABLES = 98;
pub const CU_DEVICE_ATTRIBUTE_REFRESH_RATE = 99;

pub const CuDeviceAttr = c_int;

/// CUDA device capabilities
pub const CUDACapabilities = struct {
    compute_capability_major: u32,
    compute_capability_minor: u32,
    device_name: []const u8,
    total_memory: u64,
    shared_memory_per_block: u32,
    max_threads_per_block: u32,
    max_block_dimensions: [3]u32,
    max_grid_dimensions: [3]u32,
    max_threads_per_multiprocessor: u32,
    multiprocessor_count: u32,
    memory_clock_rate: u32,
    memory_bus_width: u32,
    l2_cache_size: u32,
    max_texture_1d: u32,
    max_texture_2d: [2]u32,
    max_texture_3d: [3]u32,
    features: CUDAFeatures,

    pub const CUDAFeatures = packed struct {
        unified_addressing: bool = false,
        managed_memory: bool = false,
        concurrent_kernels: bool = false,
        cooperative_launch: bool = false,
        cooperative_multi_device_launch: bool = false,
        shared_memory_atomics: bool = false,
        global_memory_atomics: bool = false,
        surface_bindings: bool = false,
        texture_bindings: bool = false,
        double_precision: bool = false,
        half_precision: bool = false,
        int64_atomics: bool = false,
        unified_memory: bool = false,
        _padding: u19 = 0,
    };
};

/// CUDA renderer implementation
pub const CUDARenderer = struct {
    allocator: std.mem.Allocator,
    context: ?*anyopaque = null, // CUcontext
    device: ?*anyopaque = null, // CUdevice
    stream: ?*anyopaque = null, // CUstream
    capabilities: ?CUDACapabilities = null,
    is_initialized: bool = false,

    const Self = @This();

    pub fn init(allocator: std.mem.Allocator) !Self {
        return Self{
            .allocator = allocator,
        };
    }

    pub fn deinit(self: *Self) void {
        if (self.capabilities) |caps| {
            self.allocator.free(caps.device_name);
            self.capabilities = null;
        }

        if (self.stream) |stream| {
            _ = cuStreamDestroy_v2(stream);
            self.stream = null;
        }

        if (self.context) |ctx| {
            _ = cuCtxDestroy_v2(ctx);
            self.context = null;
        }

        self.is_initialized = false;
    }

    /// Initialize CUDA context and device
    pub fn initialize(self: *Self) !void {
        // Initialize CUDA
        if (cuInit(0) != CUDA_SUCCESS) {
            return error.CUDAInitFailed;
        }

        // Get device count
        var device_count: c_int = 0;
        if (cuDeviceGetCount(&device_count) != CUDA_SUCCESS or device_count == 0) {
            return error.NoCUDADevices;
        }

        // Get first device
        var device: c_int = 0;
        if (cuDeviceGet(&device, 0) != CUDA_SUCCESS) {
            return error.CUDADeviceGetFailed;
        }

        // Create context
        if (cuCtxCreate_v2(&self.context, 0, device) != CUDA_SUCCESS) {
            return error.CUDAContextCreateFailed;
        }

        // Create stream
        if (cuStreamCreate(&self.stream, 0) != CUDA_SUCCESS) {
            _ = cuCtxDestroy_v2(self.context);
            self.context = null;
            return error.CUDAStreamCreateFailed;
        }

        // Get device capabilities
        self.capabilities = try self.getCapabilities();
        self.is_initialized = true;

        std.log.info("🔧 CUDA renderer initialized successfully", .{});
    }

    /// Get device capabilities
    pub fn getCapabilities(self: *Self) !CUDACapabilities {
        if (!self.is_initialized) {
            return error.CUDANotInitialized;
        }

        var device: c_int = 0;
        if (cuDeviceGet(&device, 0) != CUDA_SUCCESS) {
            return error.CUDADeviceGetFailed;
        }

        var name_buf: [256]u8 = undefined;
        if (cuDeviceGetName(&name_buf, name_buf.len, device) != CUDA_SUCCESS) {
            return error.CUDADeviceNameFailed;
        }
        const name_len = std.mem.indexOfScalar(u8, &name_buf, 0) orelse name_buf.len;
        const device_name = try self.allocator.dupe(u8, name_buf[0..name_len]);

        var total_memory: usize = 0;
        if (cuDeviceTotalMem_v2(&total_memory, device) != CUDA_SUCCESS) {
            self.allocator.free(device_name);
            return error.CUDAMemoryQueryFailed;
        }

        // Query device attributes
        var compute_capability_major: c_int = 0;
        var compute_capability_minor: c_int = 0;
        var shared_memory_per_block: c_int = 0;
        var max_threads_per_block: c_int = 0;
        var max_block_dim_x: c_int = 0;
        var max_block_dim_y: c_int = 0;
        var max_block_dim_z: c_int = 0;
        var max_grid_dim_x: c_int = 0;
        var max_grid_dim_y: c_int = 0;
        var max_grid_dim_z: c_int = 0;
        var max_threads_per_multiprocessor: c_int = 0;
        var multiprocessor_count: c_int = 0;
        var memory_clock_rate: c_int = 0;
        var memory_bus_width: c_int = 0;
        var l2_cache_size: c_int = 0;
        var max_texture_1d: c_int = 0;
        var max_texture_2d_width: c_int = 0;
        var max_texture_2d_height: c_int = 0;
        var max_texture_3d_width: c_int = 0;
        var max_texture_3d_height: c_int = 0;
        var max_texture_3d_depth: c_int = 0;

        var unified_addressing: c_int = 0;
        var managed_memory: c_int = 0;
        var concurrent_kernels: c_int = 0;
        var cooperative_launch: c_int = 0;
        var cooperative_multi_device_launch: c_int = 0;
        var shared_memory_atomics: c_int = 0;
        var global_memory_atomics: c_int = 0;
        var surface_bindings: c_int = 0;
        var texture_bindings: c_int = 0;
        var double_precision: c_int = 0;
        var half_precision: c_int = 0;
        var int64_atomics: c_int = 0;
        var unified_memory: c_int = 0;

        // Query all attributes
        const queries = [_]struct { attr: CuDeviceAttr, ptr: *c_int }{
            .{ .attr = CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR, .ptr = &compute_capability_major },
            .{ .attr = CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR, .ptr = &compute_capability_minor },
            .{ .attr = CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_BLOCK, .ptr = &shared_memory_per_block },
            .{ .attr = CU_DEVICE_ATTRIBUTE_MAX_THREADS_PER_BLOCK, .ptr = &max_threads_per_block },
            .{ .attr = CU_DEVICE_ATTRIBUTE_MAX_BLOCK_DIM_X, .ptr = &max_block_dim_x },
            .{ .attr = CU_DEVICE_ATTRIBUTE_MAX_BLOCK_DIM_Y, .ptr = &max_block_dim_y },
            .{ .attr = CU_DEVICE_ATTRIBUTE_MAX_BLOCK_DIM_Z, .ptr = &max_block_dim_z },
            .{ .attr = CU_DEVICE_ATTRIBUTE_MAX_GRID_DIM_X, .ptr = &max_grid_dim_x },
            .{ .attr = CU_DEVICE_ATTRIBUTE_MAX_GRID_DIM_Y, .ptr = &max_grid_dim_y },
            .{ .attr = CU_DEVICE_ATTRIBUTE_MAX_GRID_DIM_Z, .ptr = &max_grid_dim_z },
            .{ .attr = CU_DEVICE_ATTRIBUTE_MAX_THREADS_PER_MULTIPROCESSOR, .ptr = &max_threads_per_multiprocessor },
            .{ .attr = CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT, .ptr = &multiprocessor_count },
            .{ .attr = CU_DEVICE_ATTRIBUTE_MEMORY_CLOCK_RATE, .ptr = &memory_clock_rate },
            .{ .attr = CU_DEVICE_ATTRIBUTE_GLOBAL_MEMORY_BUS_WIDTH, .ptr = &memory_bus_width },
            .{ .attr = CU_DEVICE_ATTRIBUTE_L2_CACHE_SIZE, .ptr = &l2_cache_size },
            .{ .attr = CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE1D_WIDTH, .ptr = &max_texture_1d },
            .{ .attr = CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_WIDTH, .ptr = &max_texture_2d_width },
            .{ .attr = CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_HEIGHT, .ptr = &max_texture_2d_height },
            .{ .attr = CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE3D_WIDTH, .ptr = &max_texture_3d_width },
            .{ .attr = CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE3D_HEIGHT, .ptr = &max_texture_3d_height },
            .{ .attr = CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE3D_DEPTH, .ptr = &max_texture_3d_depth },
            .{ .attr = CU_DEVICE_ATTRIBUTE_UNIFIED_ADDRESSING, .ptr = &unified_addressing },
            .{ .attr = CU_DEVICE_ATTRIBUTE_MANAGED_MEMORY, .ptr = &managed_memory },
            .{ .attr = CU_DEVICE_ATTRIBUTE_CONCURRENT_KERNELS, .ptr = &concurrent_kernels },
            .{ .attr = CU_DEVICE_ATTRIBUTE_COOPERATIVE_LAUNCH, .ptr = &cooperative_launch },
            .{ .attr = CU_DEVICE_ATTRIBUTE_COOPERATIVE_MULTI_DEVICE_LAUNCH, .ptr = &cooperative_multi_device_launch },
            .{ .attr = CU_DEVICE_ATTRIBUTE_HOST_NATIVE_ATOMIC_SUPPORTED, .ptr = &shared_memory_atomics },
            .{ .attr = CU_DEVICE_ATTRIBUTE_GLOBAL_L1_CACHE_SUPPORTED, .ptr = &global_memory_atomics },
            .{ .attr = CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACE1D_WIDTH, .ptr = &surface_bindings },
            .{ .attr = CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE1D_WIDTH, .ptr = &texture_bindings },
            .{ .attr = CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_WIDTH, .ptr = &double_precision },
            .{ .attr = CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_HEIGHT, .ptr = &half_precision },
            .{ .attr = CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE3D_WIDTH, .ptr = &int64_atomics },
            .{ .attr = CU_DEVICE_ATTRIBUTE_MANAGED_MEMORY, .ptr = &unified_memory },
        };

        for (queries) |query| {
            if (cuDeviceGetAttribute(query.ptr, query.attr, device) != CUDA_SUCCESS) {
                self.allocator.free(device_name);
                return error.CUDADeviceAttributeFailed;
            }
        }

        return CUDACapabilities{
            .compute_capability_major = @intCast(compute_capability_major),
            .compute_capability_minor = @intCast(compute_capability_minor),
            .device_name = device_name,
            .total_memory = total_memory,
            .shared_memory_per_block = @intCast(shared_memory_per_block),
            .max_threads_per_block = @intCast(max_threads_per_block),
            .max_block_dimensions = .{
                @intCast(max_block_dim_x),
                @intCast(max_block_dim_y),
                @intCast(max_block_dim_z),
            },
            .max_grid_dimensions = .{
                @intCast(max_grid_dim_x),
                @intCast(max_grid_dim_y),
                @intCast(max_grid_dim_z),
            },
            .max_threads_per_multiprocessor = @intCast(max_threads_per_multiprocessor),
            .multiprocessor_count = @intCast(multiprocessor_count),
            .memory_clock_rate = @intCast(memory_clock_rate),
            .memory_bus_width = @intCast(memory_bus_width),
            .l2_cache_size = @intCast(l2_cache_size),
            .max_texture_1d = @intCast(max_texture_1d),
            .max_texture_2d = .{
                @intCast(max_texture_2d_width),
                @intCast(max_texture_2d_height),
            },
            .max_texture_3d = .{
                @intCast(max_texture_3d_width),
                @intCast(max_texture_3d_height),
                @intCast(max_texture_3d_depth),
            },
            .features = .{
                .unified_addressing = unified_addressing != 0,
                .managed_memory = managed_memory != 0,
                .concurrent_kernels = concurrent_kernels != 0,
                .cooperative_launch = cooperative_launch != 0,
                .cooperative_multi_device_launch = cooperative_multi_device_launch != 0,
                .shared_memory_atomics = shared_memory_atomics != 0,
                .global_memory_atomics = global_memory_atomics != 0,
                .surface_bindings = surface_bindings != 0,
                .texture_bindings = texture_bindings != 0,
                .double_precision = double_precision != 0,
                .half_precision = half_precision != 0,
                .int64_atomics = int64_atomics != 0,
                .unified_memory = unified_memory != 0,
            },
        };
    }

    /// Launch CUDA kernel
    pub fn launchKernel(self: *Self, kernel: *anyopaque, grid_dim: [3]u32, block_dim: [3]u32, shared_mem_bytes: u32, args: []const *anyopaque) !void {
        if (!self.is_initialized) {
            return error.CUDANotInitialized;
        }

        const result = cuLaunchKernel(
            kernel,
            grid_dim[0],
            grid_dim[1],
            grid_dim[2],
            block_dim[0],
            block_dim[1],
            block_dim[2],
            shared_mem_bytes,
            self.stream,
            @constCast(args.ptr),
            null,
        );

        if (result != CUDA_SUCCESS) {
            return error.CUDAKernelLaunchFailed;
        }
    }

    /// Allocate device memory
    pub fn allocateDeviceMemory(self: *Self, size: u64) !*anyopaque {
        if (!self.is_initialized) {
            return error.CUDANotInitialized;
        }

        var ptr: ?*anyopaque = null;
        if (cuMemAlloc_v2(&ptr, size) != CUDA_SUCCESS or ptr == null) {
            return error.CUDAMemoryAllocationFailed;
        }

        return ptr.?;
    }

    /// Free device memory
    pub fn freeDeviceMemory(self: *Self, memory: *anyopaque) void {
        _ = self;
        _ = cuMemFree_v2(memory);
    }

    /// Copy memory between host and device
    pub fn copyMemory(self: *Self, dst: *anyopaque, src: *anyopaque, size: u64, kind: MemoryCopyKind) !void {
        if (!self.is_initialized) {
            return error.CUDANotInitialized;
        }

        const result = switch (kind) {
            .host_to_device => cuMemcpyHtoD_v2(dst, src, size),
            .device_to_host => cuMemcpyDtoH_v2(dst, src, size),
            .device_to_device => cuMemcpyDtoD_v2(dst, src, size),
            .host_to_host => blk: {
                const dst_bytes = @as([*]u8, @ptrCast(dst))[0..size];
                const src_bytes = @as([*]const u8, @ptrCast(src))[0..size];
                @memcpy(dst_bytes, src_bytes);
                break :blk CUDA_SUCCESS;
            },
        };

        if (result != CUDA_SUCCESS) {
            return error.CUDAMemoryCopyFailed;
        }
    }

    pub const MemoryCopyKind = enum {
        host_to_device,
        device_to_host,
        device_to_device,
        host_to_host,
    };

    /// Synchronize CUDA stream
    pub fn synchronize(self: *Self) !void {
        if (!self.is_initialized) {
            return error.CUDANotInitialized;
        }

        if (cuStreamSynchronize(self.stream) != CUDA_SUCCESS) {
            return error.CUDASyncFailed;
        }
    }
};

/// CUDA utility functions
pub const CUDAUtils = struct {
    /// Check if CUDA is available
    pub fn isCUDAAvailable() bool {
        // Try to initialize CUDA to check availability
        const result = cuInit(0);
        return result == CUDA_SUCCESS;
    }

    /// Get number of CUDA devices
    pub fn getDeviceCount() !u32 {
        var count: c_int = 0;
        if (cuDeviceGetCount(&count) != CUDA_SUCCESS) {
            return error.CUDADeviceCountFailed;
        }
        return @intCast(count);
    }

    /// Compile CUDA kernel from source (PTX)
    pub fn compileKernel(source: []const u8, kernel_name: []const u8) !*anyopaque {
        var module: ?*anyopaque = null;
        if (cuModuleLoadData(&module, source.ptr) != CUDA_SUCCESS or module == null) {
            return error.CUDAKernelCompilationFailed;
        }

        var function: ?*anyopaque = null;
        if (cuModuleGetFunction(&function, module, kernel_name.ptr) != CUDA_SUCCESS or function == null) {
            _ = cuModuleUnload(module);
            return error.CUDAKernelFunctionNotFound;
        }

        return function.?;
    }

    /// Create CUDA stream
    pub fn createStream() !*anyopaque {
        var stream: ?*anyopaque = null;
        if (cuStreamCreate(&stream, 0) != CUDA_SUCCESS or stream == null) {
            return error.CUDAStreamCreateFailed;
        }
        return stream.?;
    }

    /// Destroy CUDA stream
    pub fn destroyStream(stream: *anyopaque) void {
        _ = cuStreamDestroy_v2(stream);
    }
};
