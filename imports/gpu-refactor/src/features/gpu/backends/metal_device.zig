//! Metal device initialization, capabilities, selection, and enumeration.
//!
//! Handles loading the Objective-C runtime and Metal framework,
//! initializing selectors, creating the Metal device and command queue,
//! querying device properties, and enumerating available devices.

const std = @import("std");
const builtin = @import("builtin");
const s = @import("metal_state.zig");
const metal_types = @import("metal_types.zig");
const gpu_family = @import("metal/gpu_family.zig");
const capabilities = @import("metal/capabilities.zig");

const Device = @import("../device.zig").Device;
const DeviceType = @import("../device.zig").DeviceType;

// Re-export public types
pub const DeviceInfo = metal_types.DeviceInfo;

/// Initialize the Metal backend: load libraries, create device and command queue.
pub fn init() !void {
    if (s.metal_initialized.load(.acquire)) return;

    if (builtin.target.os.tag != .macos) {
        return s.MetalError.InitializationFailed;
    }

    // Load Objective-C runtime first
    if (!tryLoadObjcRuntime()) {
        std.log.err("Failed to load Objective-C runtime", .{});
        return s.MetalError.ObjcRuntimeUnavailable;
    }
    errdefer {
        if (s.objc_lib) |*lib| lib.close();
        s.objc_lib = null;
    }

    if (!tryLoadMetal()) {
        std.log.err("Failed to load Metal framework", .{});
        return s.MetalError.InitializationFailed;
    }
    errdefer {
        if (s.metal_lib) |*lib| lib.close();
        s.metal_lib = null;
    }

    if (!loadMetalFunctions()) {
        std.log.err("Failed to load Metal functions", .{});
        return s.MetalError.InitializationFailed;
    }

    // Initialize selectors
    try initializeSelectors();

    // Create Metal device using the C-callable function
    const create_device_fn = s.mtlCreateSystemDefaultDevice orelse return s.MetalError.DeviceNotFound;
    const device = create_device_fn();
    if (device == null) {
        std.log.err("MTLCreateSystemDefaultDevice returned null", .{});
        return s.MetalError.DeviceNotFound;
    }
    errdefer {
        if (s.objc_msgSend_void) |release_fn| {
            if (device != null) release_fn(device.?, s.sel_release);
        }
    }

    // Query device properties
    queryDeviceProperties(device);
    if (!s.cached_metal_level.atLeast(capabilities.required_runtime_level)) {
        std.log.err(
            "Metal backend requires {s} capability, detected level {s}",
            .{ capabilities.required_runtime_level.name(), s.cached_metal_level.name() },
        );
        return s.MetalError.UnsupportedFeature;
    }

    // Create command queue using Objective-C message dispatch
    const msg_send = s.objc_msgSend orelse return s.MetalError.ObjcRuntimeUnavailable;
    const command_queue = msg_send(device, s.sel_newCommandQueue);
    if (command_queue == null) {
        std.log.err("Failed to create Metal command queue", .{});
        return s.MetalError.CommandQueueCreationFailed;
    }
    errdefer {
        if (s.objc_msgSend_void) |release_fn| {
            if (command_queue != null) release_fn(command_queue.?, s.sel_release);
        }
    }

    s.metal_device = device;
    s.metal_command_queue = command_queue;
    s.metal_initialized.store(true, .release);

    std.log.debug("Metal backend initialized: {s}, {d:.2} GB VRAM", .{
        s.device_name_buf[0..s.device_name_len],
        @as(f64, @floatFromInt(s.device_total_memory)) / (1024 * 1024 * 1024),
    });
}

/// Deinitialize the Metal backend and release all resources.
pub fn deinit() void {
    // Release Metal objects using Objective-C runtime
    if (s.objc_msgSend_void) |release_fn| {
        // Release pending command buffers
        for (s.pending_command_buffers.items) |cmd_buffer| {
            if (cmd_buffer != null) {
                release_fn(cmd_buffer, s.sel_release);
            }
        }

        // Release cached pipeline states
        var pipe_it = s.pipeline_cache.iterator();
        while (pipe_it.next()) |entry| {
            if (entry.value_ptr.* != null) {
                release_fn(entry.value_ptr.*, s.sel_release);
            }
        }

        if (s.metal_command_queue != null) {
            release_fn(s.metal_command_queue, s.sel_release);
        }
        // Device is typically not released - managed by system
    }

    // Deinit pending command buffers ArrayList
    if (s.pending_buffers_allocator) |alloc| {
        s.pending_command_buffers.deinit(alloc);
    }
    s.pending_command_buffers = .empty;
    s.pending_buffers_allocator = null;

    // Deinit pipeline cache HashMap
    if (s.pipeline_cache_allocator) |alloc| {
        s.pipeline_cache.deinit(alloc);
    }
    s.pipeline_cache = .empty;
    s.pipeline_cache_allocator = null;

    s.metal_device = null;
    s.metal_command_queue = null;

    if (s.metal_lib != null) {
        var lib = s.metal_lib.?;
        lib.close();
    }
    if (s.objc_lib != null) {
        var lib = s.objc_lib.?;
        lib.close();
    }
    s.metal_lib = null;
    s.objc_lib = null;
    s.metal_initialized.store(false, .release);
    s.selectors_initialized.store(false, .release);
    s.cached_feature_set = null;
    s.cached_metal_level = .none;

    std.log.debug("Metal backend deinitialized", .{});
}

/// Query device properties from the MTLDevice object.
fn queryDeviceProperties(device: s.ID) void {
    const msg_send = s.objc_msgSend orelse return;
    const msg_send_u64 = s.objc_msgSend_u64 orelse return;

    // Query device name: [device name] returns NSString
    const name_nsstring = msg_send(device, s.sel_name);
    if (name_nsstring != null) {
        // Get UTF8 string from NSString
        const utf8_fn: *const fn (s.ID, s.SEL) callconv(.c) ?[*:0]const u8 = @ptrCast(s.objc_msgSend);
        const utf8_ptr = utf8_fn(name_nsstring, s.sel_UTF8String);
        if (utf8_ptr) |ptr| {
            const name_slice = std.mem.span(ptr);
            const copy_len = @min(name_slice.len, s.device_name_buf.len);
            @memcpy(s.device_name_buf[0..copy_len], name_slice[0..copy_len]);
            s.device_name_len = copy_len;
        }
    }

    // Query recommended max working set size (available VRAM)
    s.device_total_memory = msg_send_u64(device, s.sel_recommendedMaxWorkingSetSize);

    // Query max buffer length
    s.device_max_buffer_length = msg_send_u64(device, s.sel_maxBufferLength);

    // Query max threads per threadgroup using MTLSize
    // For now, use default values based on Apple Silicon capabilities
    s.device_max_threads_per_group = 1024; // Common default for Apple Silicon

    // Detect GPU family and build feature set
    if (s.sel_registerName) |sel_fn| {
        const family_sel = sel_fn("supportsFamily:");
        const family_fn: *const fn (
            s.ID,
            s.SEL,
            u32,
        ) callconv(.c) bool = @ptrCast(s.objc_msgSend);
        const family = gpu_family.detectGpuFamily(
            @ptrCast(device),
            @ptrCast(family_sel),
            family_fn,
        );
        s.cached_feature_set = gpu_family.buildFeatureSet(family);
        s.cached_metal_level = capabilities.levelFromFamily(family);
    }
}

/// Set the allocator to use for buffer metadata allocations.
pub fn setBufferAllocator(allocator: std.mem.Allocator) void {
    s.buffer_allocator = allocator;
}

/// Set the allocator to use for tracking pending command buffers.
pub fn setPendingBuffersAllocator(allocator: std.mem.Allocator) void {
    s.pending_buffers_allocator = allocator;
}

/// Check if Metal backend is available on this system.
pub fn isAvailable() bool {
    if (builtin.target.os.tag != .macos) {
        return false;
    }
    // Quick check if Metal framework exists
    if (std.DynLib.open("/System/Library/Frameworks/Metal.framework/Metal")) |lib_val| {
        var lib = lib_val;
        lib.close();
        return true;
    } else |_| {
        return false;
    }
}

/// Query properties from a Metal device object.
fn queryDeviceInfo(mtl_device: s.ID, device_id: u32) !Device {
    const msg_send = s.objc_msgSend orelse return s.MetalError.DeviceQueryFailed;
    const msg_send_u64 = s.objc_msgSend_u64 orelse return s.MetalError.DeviceQueryFailed;
    const msg_send_bool = s.objc_msgSend_bool orelse return s.MetalError.DeviceQueryFailed;

    // Query device name
    var name_slice: []const u8 = "Unknown Metal GPU";
    const name_nsstring = msg_send(mtl_device, s.sel_name);
    if (name_nsstring != null) {
        const utf8_fn: *const fn (s.ID, s.SEL) callconv(.c) ?[*:0]const u8 = @ptrCast(s.objc_msgSend);
        const utf8_ptr = utf8_fn(name_nsstring, s.sel_UTF8String);
        if (utf8_ptr) |ptr| {
            name_slice = std.mem.span(ptr);
        }
    }

    // Query memory
    const total_memory = msg_send_u64(mtl_device, s.sel_recommendedMaxWorkingSetSize);

    // Query device properties
    const is_low_power = msg_send_bool(mtl_device, s.sel_isLowPower);
    const has_unified_memory = msg_send_bool(mtl_device, s.sel_hasUnifiedMemory);

    // Determine device type based on properties
    const device_type: DeviceType = if (has_unified_memory)
        .integrated // Apple Silicon or integrated GPU
    else if (is_low_power)
        .integrated
    else
        .discrete;

    return Device{
        .id = device_id,
        .backend = .metal,
        .name = name_slice,
        .device_type = device_type,
        .vendor = .unknown,
        .total_memory = if (total_memory > 0) total_memory else null,
        .available_memory = null,
        .is_emulated = false,
        .capability = .{
            .supports_fp16 = true,
            .supports_int8 = true,
            .supports_async_transfers = true,
            .unified_memory = has_unified_memory,
            .max_threads_per_block = s.device_max_threads_per_group,
            .max_shared_memory_bytes = 32 * 1024,
        },
        .compute_units = null,
        .clock_mhz = null,
        .pci_bus_id = null,
        .driver_version = null,
    };
}

/// Enumerate all Metal devices available on this Mac.
pub fn enumerateDevices(allocator: std.mem.Allocator) ![]Device {
    if (!isAvailable()) {
        return &[_]Device{};
    }

    var devices = std.ArrayListUnmanaged(Device).empty;
    errdefer devices.deinit(allocator);

    // Initialize Metal if not already done
    if (!s.metal_initialized.load(.acquire)) {
        init() catch {
            return &[_]Device{};
        };
    }

    // Try to enumerate all devices using MTLCopyAllDevices (if available)
    if (s.mtlCopyAllDevices) |copy_all_fn| {
        const device_array = copy_all_fn();
        if (device_array != null) {
            _ = s.objc_msgSend orelse return &[_]Device{};
            const msg_send_u64 = s.objc_msgSend_u64 orelse return &[_]Device{};

            // Get count: [array count]
            const count = msg_send_u64(device_array, s.sel_count);

            // Enumerate each device
            var i: u32 = 0;
            while (i < count) : (i += 1) {
                const get_obj_fn: *const fn (s.ID, s.SEL, usize) callconv(.c) s.ID = @ptrCast(s.objc_msgSend);
                const mtl_device = get_obj_fn(device_array, s.sel_objectAtIndex, i);
                if (mtl_device != null) {
                    const dev_info = queryDeviceInfo(mtl_device, i) catch continue;
                    try devices.append(allocator, dev_info);
                }
            }

            // Release the array
            if (s.objc_msgSend_void) |release_fn| {
                release_fn(device_array, s.sel_release);
            }

            if (devices.items.len > 0) {
                return devices.toOwnedSlice(allocator);
            }
        }
    }

    // Fallback: use the default device if MTLCopyAllDevices is not available
    if (s.metal_device != null) {
        const dev_info = queryDeviceInfo(s.metal_device, 0) catch {
            // Fallback to basic device info
            try devices.append(allocator, .{
                .id = 0,
                .backend = .metal,
                .name = if (s.device_name_len > 0)
                    s.device_name_buf[0..s.device_name_len]
                else
                    "Metal GPU",
                .device_type = if (builtin.target.cpu.arch == .aarch64) .integrated else .discrete,
                .vendor = .unknown,
                .total_memory = if (s.device_total_memory > 0) s.device_total_memory else null,
                .available_memory = null,
                .is_emulated = false,
                .capability = .{
                    .supports_fp16 = true,
                    .supports_int8 = true,
                    .supports_async_transfers = true,
                    .unified_memory = builtin.target.cpu.arch == .aarch64,
                },
                .compute_units = null,
                .clock_mhz = null,
                .pci_bus_id = null,
                .driver_version = null,
            });
            return devices.toOwnedSlice(allocator);
        };
        try devices.append(allocator, dev_info);
    }

    return devices.toOwnedSlice(allocator);
}

/// Get detailed information about the current default Metal device.
pub fn getDeviceInfo() ?DeviceInfo {
    if (!s.metal_initialized.load(.acquire) or s.metal_device == null) return null;

    var info = DeviceInfo{
        .name = if (s.device_name_len > 0) s.device_name_buf[0..s.device_name_len] else "Unknown",
        .total_memory = s.device_total_memory,
        .max_buffer_length = s.device_max_buffer_length,
        .max_threads_per_threadgroup = s.device_max_threads_per_group,
        .has_unified_memory = builtin.target.cpu.arch == .aarch64,
    };

    if (s.cached_feature_set) |fs| {
        info.gpu_family = @intFromEnum(fs.gpu_family);
        info.supports_mesh_shaders = fs.supports_mesh_shaders;
        info.supports_ray_tracing = fs.supports_ray_tracing;
        info.supports_mps = fs.supports_mps;
        info.supports_neural_engine = fs.has_neural_engine;
        info.metal_level = @intFromEnum(s.cached_metal_level);
    }

    return info;
}

/// Get the detected GPU feature set (populated during init).
pub fn getFeatureSet() ?s.MetalFeatureSet {
    return s.cached_feature_set;
}

pub fn getMetalLevel() s.MetalLevel {
    return s.cached_metal_level;
}

pub fn supportsMetal4() bool {
    return s.cached_metal_level.atLeast(.metal4);
}

// ============================================================================
// Library loading and selector initialization (internal)
// ============================================================================

fn tryLoadObjcRuntime() bool {
    const objc_paths = [_][]const u8{
        "/usr/lib/libobjc.dylib",
        "/usr/lib/libobjc.A.dylib",
    };

    for (objc_paths) |path| {
        if (std.DynLib.open(path)) |lib_val| {
            s.objc_lib = lib_val;
            var lib = lib_val;

            s.objc_msgSend = lib.lookup(s.ObjcMsgSendFn, "objc_msgSend");
            s.objc_msgSend_int = @ptrCast(@alignCast(lib.lookup(*anyopaque, "objc_msgSend")));
            s.objc_msgSend_ptr = @ptrCast(@alignCast(lib.lookup(*anyopaque, "objc_msgSend")));
            s.objc_msgSend_ptr2 = @ptrCast(@alignCast(lib.lookup(*anyopaque, "objc_msgSend")));
            s.objc_msgSend_ptr3 = @ptrCast(@alignCast(lib.lookup(*anyopaque, "objc_msgSend")));
            s.objc_msgSend_void = @ptrCast(@alignCast(lib.lookup(*anyopaque, "objc_msgSend")));
            s.objc_msgSend_void_ptr = @ptrCast(@alignCast(lib.lookup(*anyopaque, "objc_msgSend")));
            s.objc_msgSend_void_ptr_int_int = @ptrCast(@alignCast(lib.lookup(*anyopaque, "objc_msgSend")));
            s.objc_msgSend_u64 = @ptrCast(@alignCast(lib.lookup(*anyopaque, "objc_msgSend")));
            s.objc_msgSend_u32 = @ptrCast(@alignCast(lib.lookup(*anyopaque, "objc_msgSend")));
            s.objc_msgSend_bool = @ptrCast(@alignCast(lib.lookup(*anyopaque, "objc_msgSend")));
            s.objc_msgSend_mtlsize2 = @ptrCast(@alignCast(lib.lookup(*anyopaque, "objc_msgSend")));
            s.objc_msgSend_nsstring = @ptrCast(@alignCast(lib.lookup(*anyopaque, "objc_msgSend")));
            s.sel_registerName = lib.lookup(s.SelRegisterNameFn, "sel_registerName");
            s.objc_getClass = lib.lookup(s.ObjcGetClassFn, "objc_getClass");

            if (s.objc_msgSend != null and s.sel_registerName != null and s.objc_getClass != null) {
                return true;
            }
        } else |_| {}
    }

    return false;
}

fn tryLoadMetal() bool {
    if (builtin.target.os.tag != .macos) {
        return false;
    }

    const framework_paths = [_][]const u8{
        "/System/Library/Frameworks/Metal.framework/Metal",
        "/System/Library/Frameworks/Metal.framework/Versions/A/Metal",
    };

    for (framework_paths) |path| {
        if (std.DynLib.open(path)) |lib| {
            s.metal_lib = lib;
            return true;
        } else |_| {}
    }

    if (std.DynLib.open("Metal.framework/Metal")) |lib| {
        s.metal_lib = lib;
        return true;
    } else |_| {}

    return false;
}

fn loadMetalFunctions() bool {
    if (s.metal_lib == null) return false;
    var lib = s.metal_lib.?;

    s.mtlCreateSystemDefaultDevice = lib.lookup(
        s.MtlCreateSystemDefaultDeviceFn,
        "MTLCreateSystemDefaultDevice",
    ) orelse return false;

    s.mtlCopyAllDevices = lib.lookup(
        s.MtlCopyAllDevicesFn,
        "MTLCopyAllDevices",
    );

    return true;
}

fn initializeSelectors() s.MetalError!void {
    if (s.selectors_initialized.load(.acquire)) return;

    const sel_fn = s.sel_registerName orelse return s.MetalError.SelectorNotFound;

    // Command queue and buffer selectors
    s.sel_newCommandQueue = sel_fn("newCommandQueue");
    s.sel_commandBuffer = sel_fn("commandBuffer");
    s.sel_computeCommandEncoder = sel_fn("computeCommandEncoder");
    s.sel_commit = sel_fn("commit");
    s.sel_waitUntilCompleted = sel_fn("waitUntilCompleted");
    s.sel_endEncoding = sel_fn("endEncoding");

    // Library and kernel compilation selectors
    s.sel_newLibraryWithSource = sel_fn("newLibraryWithSource:options:error:");
    s.sel_newFunctionWithName = sel_fn("newFunctionWithName:");
    s.sel_newComputePipelineStateWithFunction = sel_fn("newComputePipelineStateWithFunction:error:");

    // Buffer and memory selectors
    s.sel_newBufferWithLength = sel_fn("newBufferWithLength:options:");
    s.sel_contents = sel_fn("contents");
    s.sel_length = sel_fn("length");

    // Compute encoder selectors
    s.sel_setComputePipelineState = sel_fn("setComputePipelineState:");
    s.sel_setBuffer = sel_fn("setBuffer:offset:atIndex:");
    s.sel_dispatchThreads = sel_fn("dispatchThreads:threadsPerThreadgroup:");
    s.sel_dispatchThreadgroups = sel_fn("dispatchThreadgroups:threadsPerThreadgroup:");

    // Memory management selectors
    s.sel_release = sel_fn("release");
    s.sel_retain = sel_fn("retain");

    // Device property selectors
    s.sel_name = sel_fn("name");
    s.sel_recommendedMaxWorkingSetSize = sel_fn("recommendedMaxWorkingSetSize");
    s.sel_maxThreadsPerThreadgroup = sel_fn("maxThreadsPerThreadgroup");
    s.sel_maxBufferLength = sel_fn("maxBufferLength");
    s.sel_supportsFamily = sel_fn("supportsFamily:");
    s.sel_registryID = sel_fn("registryID");
    s.sel_isLowPower = sel_fn("isLowPower");
    s.sel_isHeadless = sel_fn("isHeadless");
    s.sel_hasUnifiedMemory = sel_fn("hasUnifiedMemory");

    // NSString selectors
    s.sel_stringWithUTF8String = sel_fn("stringWithUTF8String:");
    s.sel_UTF8String = sel_fn("UTF8String");

    // NSArray selectors
    s.sel_count = sel_fn("count");
    s.sel_objectAtIndex = sel_fn("objectAtIndex:");

    // Pipeline state selectors
    s.sel_maxTotalThreadsPerThreadgroup = sel_fn("maxTotalThreadsPerThreadgroup");
    s.sel_threadExecutionWidth = sel_fn("threadExecutionWidth");

    s.selectors_initialized.store(true, .release);
}
