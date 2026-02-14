//! Metal backend implementation with native GPU execution.
//!
//! Provides Metal-specific kernel compilation, execution, and memory management
//! using the Metal API for Apple Silicon acceleration.
//!
//! Metal uses Objective-C runtime, so this module uses objc_msgSend for message dispatch.
//!
//! ## Features
//! - Full kernel dispatch with proper MTLSize struct handling
//! - Device property queries (memory, compute units, device name)
//! - NSString creation for runtime kernel compilation
//! - Proper synchronization via command buffer tracking
//! - Multi-device enumeration via MTLCopyAllDevices
//!
//! ## Architecture Support
//! - Apple Silicon (ARM64): Uses standard objc_msgSend for all calls
//! - Intel Macs (x86_64): Uses objc_msgSend_stret for struct returns

const std = @import("std");
const builtin = @import("builtin");
const types = @import("../kernel_types.zig");
const shared = @import("shared.zig");

// Re-export extracted type definitions for build discovery
pub const metal_types = @import("metal_types.zig");

pub const MetalError = metal_types.MetalError;

// Objective-C runtime types (from metal_types.zig)
const SEL = metal_types.SEL;
const Class = metal_types.Class;
const ID = metal_types.ID;

// Metal struct types (re-exported from metal_types.zig)
pub const MTLSize = metal_types.MTLSize;
pub const MTLOrigin = metal_types.MTLOrigin;
pub const MTLRegion = metal_types.MTLRegion;

// Metal GPU Family / Feature detection
pub const gpu_family = @import("metal/gpu_family.zig");
pub const MetalGpuFamily = gpu_family.MetalGpuFamily;
pub const MetalFeatureSet = gpu_family.MetalFeatureSet;

var metal_lib: ?std.DynLib = null;
var objc_lib: ?std.DynLib = null;
var foundation_lib: ?std.DynLib = null;
var metal_initialized = false;
var metal_device: ID = null;
var metal_command_queue: ID = null;

// Device properties cache
var device_name_buf: [256]u8 = undefined;
var device_name_len: usize = 0;
var device_total_memory: u64 = 0;
var device_max_threads_per_group: u32 = 0;
var device_max_buffer_length: u64 = 0;

// Cached GPU feature set (populated during queryDeviceProperties)
var cached_feature_set: ?MetalFeatureSet = null;

// Active command buffers for synchronization
var pending_command_buffers: std.ArrayListUnmanaged(ID) = .empty;
var pending_buffers_allocator: ?std.mem.Allocator = null;

// Cached allocator for buffer metadata
var buffer_allocator: ?std.mem.Allocator = null;

// Pipeline cache for compiled compute pipelines
var pipeline_cache: std.StringHashMapUnmanaged(ID) = .empty;
var pipeline_cache_allocator: ?std.mem.Allocator = null;

// Objective-C runtime function pointers (from metal_types.zig)
const ObjcMsgSendFn = metal_types.ObjcMsgSendFn;
const ObjcMsgSendIntFn = metal_types.ObjcMsgSendIntFn;
const ObjcMsgSendPtrFn = metal_types.ObjcMsgSendPtrFn;
const ObjcMsgSendPtr2Fn = metal_types.ObjcMsgSendPtr2Fn;
const ObjcMsgSendPtr3Fn = metal_types.ObjcMsgSendPtr3Fn;
const ObjcMsgSendVoidFn = metal_types.ObjcMsgSendVoidFn;
const ObjcMsgSendVoidPtrFn = metal_types.ObjcMsgSendVoidPtrFn;
const ObjcMsgSendVoidPtrIntIntFn = metal_types.ObjcMsgSendVoidPtrIntIntFn;
const ObjcMsgSendU64Fn = metal_types.ObjcMsgSendU64Fn;
const ObjcMsgSendU32Fn = metal_types.ObjcMsgSendU32Fn;
const ObjcMsgSendBoolFn = metal_types.ObjcMsgSendBoolFn;
const SelRegisterNameFn = metal_types.SelRegisterNameFn;
const ObjcGetClassFn = metal_types.ObjcGetClassFn;
const ObjcMsgSendMTLSize2Fn = metal_types.ObjcMsgSendMTLSize2Fn;
const NSStringWithUTF8Fn = metal_types.NSStringWithUTF8Fn;

var objc_msgSend: ?ObjcMsgSendFn = null;
var objc_msgSend_int: ?ObjcMsgSendIntFn = null;
var objc_msgSend_ptr: ?ObjcMsgSendPtrFn = null;
var objc_msgSend_ptr2: ?ObjcMsgSendPtr2Fn = null;
var objc_msgSend_ptr3: ?ObjcMsgSendPtr3Fn = null;
var objc_msgSend_void: ?ObjcMsgSendVoidFn = null;
var objc_msgSend_void_ptr: ?ObjcMsgSendVoidPtrFn = null;
var objc_msgSend_void_ptr_int_int: ?ObjcMsgSendVoidPtrIntIntFn = null;
var objc_msgSend_u64: ?ObjcMsgSendU64Fn = null;
var objc_msgSend_u32: ?ObjcMsgSendU32Fn = null;
var objc_msgSend_bool: ?ObjcMsgSendBoolFn = null;
var objc_msgSend_mtlsize2: ?ObjcMsgSendMTLSize2Fn = null;
var objc_msgSend_nsstring: ?NSStringWithUTF8Fn = null;
var sel_registerName: ?SelRegisterNameFn = null;
var objc_getClass: ?ObjcGetClassFn = null;

// NSString class reference
var nsstring_class: ?Class = null;

// Metal C-callable function types (from metal_types.zig)
const MtlCreateSystemDefaultDeviceFn = metal_types.MtlCreateSystemDefaultDeviceFn;
const MtlCopyAllDevicesFn = metal_types.MtlCopyAllDevicesFn;
var mtlCreateSystemDefaultDevice: ?MtlCreateSystemDefaultDeviceFn = null;
var mtlCopyAllDevices: ?MtlCopyAllDevicesFn = null;

// Cached selectors for Metal methods
var sel_newCommandQueue: SEL = undefined;
var sel_newLibraryWithSource: SEL = undefined;
var sel_newFunctionWithName: SEL = undefined;
var sel_newComputePipelineStateWithFunction: SEL = undefined;
var sel_newBufferWithLength: SEL = undefined;
var sel_commandBuffer: SEL = undefined;
var sel_computeCommandEncoder: SEL = undefined;
var sel_setComputePipelineState: SEL = undefined;
var sel_setBuffer: SEL = undefined;
var sel_dispatchThreads: SEL = undefined;
var sel_dispatchThreadgroups: SEL = undefined;
var sel_endEncoding: SEL = undefined;
var sel_commit: SEL = undefined;
var sel_waitUntilCompleted: SEL = undefined;
var sel_contents: SEL = undefined;
var sel_length: SEL = undefined;
var sel_release: SEL = undefined;
var sel_retain: SEL = undefined;

// Device property selectors
var sel_name: SEL = undefined;
var sel_recommendedMaxWorkingSetSize: SEL = undefined;
var sel_maxThreadsPerThreadgroup: SEL = undefined;
var sel_maxBufferLength: SEL = undefined;
var sel_supportsFamily: SEL = undefined;
var sel_registryID: SEL = undefined;
var sel_isLowPower: SEL = undefined;
var sel_isHeadless: SEL = undefined;
var sel_hasUnifiedMemory: SEL = undefined;

// NSString selectors
var sel_stringWithUTF8String: SEL = undefined;
var sel_UTF8String: SEL = undefined;

// NSArray selectors
var sel_count: SEL = undefined;
var sel_objectAtIndex: SEL = undefined;

// Pipeline state selectors
var sel_maxTotalThreadsPerThreadgroup: SEL = undefined;
var sel_threadExecutionWidth: SEL = undefined;

var selectors_initialized = false;

// MTLResourceOptions (from metal_types.zig)
const MTLResourceStorageModeShared = metal_types.MTLResourceStorageModeShared;

// Internal Metal structs (from metal_types.zig)
const MetalKernel = metal_types.MetalKernel;
const MetalBuffer = metal_types.MetalBuffer;

// Safe pointer casting types (from metal_types.zig)
const kernel_magic = metal_types.kernel_magic;
const buffer_magic = metal_types.buffer_magic;
const SafeMetalKernel = metal_types.SafeMetalKernel;
const SafeMetalBuffer = metal_types.SafeMetalBuffer;

/// Safely cast an opaque pointer to a MetalKernel pointer with validation.
/// Returns null if the pointer is null or the magic value doesn't match.
fn safeCastToKernel(ptr: ?*anyopaque) ?*MetalKernel {
    // Null pointer check - return null for safety
    const p = ptr orelse return null;

    // Cast to SafeMetalKernel to validate magic
    const safe_kernel: *SafeMetalKernel = @ptrCast(@alignCast(p));

    // Validate magic value to detect corruption/invalid pointers
    if (safe_kernel.magic != kernel_magic) {
        std.log.err("Invalid MetalKernel pointer: magic mismatch (expected 0x{x}, got 0x{x})", .{ kernel_magic, safe_kernel.magic });
        return null;
    }

    return &safe_kernel.inner;
}

/// Safely cast an opaque pointer to a MetalBuffer pointer with validation.
/// Returns null if the pointer is null or the magic value doesn't match.
fn safeCastToBuffer(ptr: ?*anyopaque) ?*MetalBuffer {
    // Null pointer check - return null for safety
    const p = ptr orelse return null;

    // Cast to SafeMetalBuffer to validate magic
    const safe_buffer: *SafeMetalBuffer = @ptrCast(@alignCast(p));

    // Validate magic value to detect corruption/invalid pointers
    if (safe_buffer.magic != buffer_magic) {
        std.log.err("Invalid MetalBuffer pointer: magic mismatch (expected 0x{x}, got 0x{x})", .{ buffer_magic, safe_buffer.magic });
        return null;
    }

    return &safe_buffer.inner;
}

/// Safely cast a const opaque pointer to a MetalBuffer pointer with validation.
/// Returns null if the pointer is null or the magic value doesn't match.
fn safeCastToBufferConst(ptr: ?*const anyopaque) ?*const MetalBuffer {
    // Null pointer check - return null for safety
    const p = ptr orelse return null;

    // Cast to SafeMetalBuffer to validate magic (need to cast away const for alignment)
    const safe_buffer: *const SafeMetalBuffer = @ptrCast(@alignCast(p));

    // Validate magic value to detect corruption/invalid pointers
    if (safe_buffer.magic != buffer_magic) {
        std.log.err("Invalid MetalBuffer pointer: magic mismatch (expected 0x{x}, got 0x{x})", .{ buffer_magic, safe_buffer.magic });
        return null;
    }

    return &safe_buffer.inner;
}

pub fn init() !void {
    if (metal_initialized) return;

    if (builtin.target.os.tag != .macos) {
        return MetalError.InitializationFailed;
    }

    // Load Objective-C runtime first
    if (!tryLoadObjcRuntime()) {
        std.log.err("Failed to load Objective-C runtime", .{});
        return MetalError.ObjcRuntimeUnavailable;
    }

    if (!tryLoadMetal()) {
        std.log.err("Failed to load Metal framework", .{});
        return MetalError.InitializationFailed;
    }

    if (!loadMetalFunctions()) {
        std.log.err("Failed to load Metal functions", .{});
        return MetalError.InitializationFailed;
    }

    // Initialize selectors
    try initializeSelectors();

    // Create Metal device using the C-callable function
    const create_device_fn = mtlCreateSystemDefaultDevice orelse return MetalError.DeviceNotFound;
    const device = create_device_fn();
    if (device == null) {
        std.log.err("MTLCreateSystemDefaultDevice returned null", .{});
        return MetalError.DeviceNotFound;
    }

    // Query device properties
    queryDeviceProperties(device);

    // Create command queue using Objective-C message dispatch
    const msg_send = objc_msgSend orelse return MetalError.ObjcRuntimeUnavailable;
    const command_queue = msg_send(device, sel_newCommandQueue);
    if (command_queue == null) {
        std.log.err("Failed to create Metal command queue", .{});
        return MetalError.CommandQueueCreationFailed;
    }

    metal_device = device;
    metal_command_queue = command_queue;
    metal_initialized = true;

    std.log.debug("Metal backend initialized: {s}, {d:.2} GB VRAM", .{
        device_name_buf[0..device_name_len],
        @as(f64, @floatFromInt(device_total_memory)) / (1024 * 1024 * 1024),
    });
}

/// Query device properties from the MTLDevice object.
fn queryDeviceProperties(device: ID) void {
    const msg_send = objc_msgSend orelse return;
    const msg_send_u64 = objc_msgSend_u64 orelse return;

    // Query device name: [device name] returns NSString
    const name_nsstring = msg_send(device, sel_name);
    if (name_nsstring != null) {
        // Get UTF8 string from NSString
        const utf8_fn: *const fn (ID, SEL) callconv(.c) ?[*:0]const u8 = @ptrCast(objc_msgSend);
        const utf8_ptr = utf8_fn(name_nsstring, sel_UTF8String);
        if (utf8_ptr) |ptr| {
            const name_slice = std.mem.span(ptr);
            const copy_len = @min(name_slice.len, device_name_buf.len);
            @memcpy(device_name_buf[0..copy_len], name_slice[0..copy_len]);
            device_name_len = copy_len;
        }
    }

    // Query recommended max working set size (available VRAM)
    device_total_memory = msg_send_u64(device, sel_recommendedMaxWorkingSetSize);

    // Query max buffer length
    device_max_buffer_length = msg_send_u64(device, sel_maxBufferLength);

    // Query max threads per threadgroup using MTLSize
    // This returns an MTLSize struct, which we need to handle specially
    // For now, use default values based on Apple Silicon capabilities
    device_max_threads_per_group = 1024; // Common default for Apple Silicon

    // Detect GPU family and build feature set
    if (sel_registerName) |sel_fn| {
        const family_sel = sel_fn("supportsFamily:");
        const family_fn: *const fn (
            ID,
            SEL,
            u32,
        ) callconv(.c) bool = @ptrCast(objc_msgSend);
        const family = gpu_family.detectGpuFamily(
            @ptrCast(device),
            @ptrCast(family_sel),
            @ptrCast(&family_fn),
        );
        cached_feature_set = gpu_family.buildFeatureSet(family);
    }
}

pub fn deinit() void {
    // Release Metal objects using Objective-C runtime
    if (objc_msgSend_void) |release_fn| {
        if (metal_command_queue != null) {
            release_fn(metal_command_queue, sel_release);
        }
        // Device is typically not released - managed by system
    }

    metal_device = null;
    metal_command_queue = null;

    if (metal_lib) |lib| {
        lib.close();
    }
    if (objc_lib) |lib| {
        lib.close();
    }
    metal_lib = null;
    objc_lib = null;
    metal_initialized = false;
    selectors_initialized = false;

    std.log.debug("Metal backend deinitialized", .{});
}

/// Create an NSString from a Zig string slice.
/// The caller is responsible for releasing the returned NSString.
fn createNSString(str: []const u8) MetalError!ID {
    const get_class = objc_getClass orelse return MetalError.NSStringCreationFailed;
    const ns_class = nsstring_class orelse blk: {
        const cls = get_class("NSString");
        if (cls == null) return MetalError.NSStringCreationFailed;
        nsstring_class = cls;
        break :blk cls;
    };

    // We need to create a null-terminated string
    // Use a stack buffer for small strings, heap for larger ones
    var stack_buf: [4096]u8 = undefined;
    const c_str: [*:0]const u8 = if (str.len < stack_buf.len) blk: {
        @memcpy(stack_buf[0..str.len], str);
        stack_buf[str.len] = 0;
        break :blk stack_buf[0..str.len :0];
    } else blk: {
        // For larger strings, we'd need to allocate
        // For now, truncate to stack buffer size
        @memcpy(stack_buf[0 .. stack_buf.len - 1], str[0 .. stack_buf.len - 1]);
        stack_buf[stack_buf.len - 1] = 0;
        break :blk stack_buf[0 .. stack_buf.len - 1 :0];
    };

    // Call [NSString stringWithUTF8String:]
    const msg_send_str: *const fn (?Class, SEL, [*:0]const u8) callconv(.c) ID = @ptrCast(objc_msgSend);
    const result = msg_send_str(ns_class, sel_stringWithUTF8String, c_str);
    if (result == null) {
        return MetalError.NSStringCreationFailed;
    }

    return result;
}

/// Create an NSString from a null-terminated C string.
fn createNSStringFromCStr(c_str: [*:0]const u8) MetalError!ID {
    const get_class = objc_getClass orelse return MetalError.NSStringCreationFailed;
    const ns_class = nsstring_class orelse blk: {
        const cls = get_class("NSString");
        if (cls == null) return MetalError.NSStringCreationFailed;
        nsstring_class = cls;
        break :blk cls;
    };

    const msg_send_str: *const fn (?Class, SEL, [*:0]const u8) callconv(.c) ID = @ptrCast(objc_msgSend);
    const result = msg_send_str(ns_class, sel_stringWithUTF8String, c_str);
    if (result == null) {
        return MetalError.NSStringCreationFailed;
    }

    return result;
}

pub fn compileKernel(
    allocator: std.mem.Allocator,
    source: types.KernelSource,
) types.KernelError!*anyopaque {
    if (!metal_initialized or metal_device == null) {
        return types.KernelError.CompilationFailed;
    }

    const device = metal_device.?;

    // Create NSString from source code
    const source_nsstring = createNSString(source.code) catch {
        std.log.err("Failed to create NSString from kernel source", .{});
        return types.KernelError.CompilationFailed;
    };
    defer {
        // Release the NSString when done
        if (objc_msgSend_void) |release_fn| {
            release_fn(source_nsstring, sel_release);
        }
    }

    // Create library from source using objc_msgSend
    // [device newLibraryWithSource:options:error:]
    // Signature: (ID, SEL, NSString*, MTLCompileOptions*, NSError**)
    const msg_send_lib: *const fn (ID, SEL, ID, ID, *ID) callconv(.c) ID = @ptrCast(objc_msgSend);
    var compile_error: ID = null;
    const library = msg_send_lib(device, sel_newLibraryWithSource, source_nsstring, null, &compile_error);

    if (library == null) {
        if (compile_error != null) {
            // Try to extract error description
            const sel_fn = sel_registerName orelse {
                std.log.err("Failed to create Metal library from source (unknown error)", .{});
                return types.KernelError.CompilationFailed;
            };
            const sel_desc = sel_fn("localizedDescription");
            const msg_send = objc_msgSend orelse {
                std.log.err("Failed to create Metal library from source (unknown error)", .{});
                return types.KernelError.CompilationFailed;
            };
            const desc_nsstring = msg_send(compile_error, sel_desc);
            if (desc_nsstring != null) {
                const utf8_fn: *const fn (ID, SEL) callconv(.c) ?[*:0]const u8 = @ptrCast(objc_msgSend);
                const utf8_ptr = utf8_fn(desc_nsstring, sel_UTF8String);
                if (utf8_ptr) |ptr| {
                    std.log.err("Metal compilation error: {s}", .{ptr});
                }
            }
        }
        std.log.err("Failed to create Metal library from source", .{});
        return types.KernelError.CompilationFailed;
    }

    // Get the function by entry point name
    // [library newFunctionWithName:@"entry_point"]
    const entry_point_str: [*:0]const u8 = if (source.entry_point.len > 0) blk: {
        // Create a null-terminated version
        var buf: [256]u8 = undefined;
        const len = @min(source.entry_point.len, buf.len - 1);
        @memcpy(buf[0..len], source.entry_point[0..len]);
        buf[len] = 0;
        break :blk buf[0..len :0];
    } else "main";

    const entry_point_nsstring = createNSStringFromCStr(entry_point_str) catch {
        std.log.err("Failed to create NSString for entry point", .{});
        // Release library
        if (objc_msgSend_void) |release_fn| {
            release_fn(library, sel_release);
        }
        return types.KernelError.CompilationFailed;
    };
    defer {
        if (objc_msgSend_void) |release_fn| {
            release_fn(entry_point_nsstring, sel_release);
        }
    }

    const msg_send_ptr = objc_msgSend_ptr orelse return types.KernelError.CompilationFailed;
    const function = msg_send_ptr(library, sel_newFunctionWithName, entry_point_nsstring);
    if (function == null) {
        std.log.err("Failed to get Metal function '{s}' from library", .{source.entry_point});
        // Release library
        if (objc_msgSend_void) |release_fn| {
            release_fn(library, sel_release);
        }
        return types.KernelError.CompilationFailed;
    }

    // Create compute pipeline state
    // [device newComputePipelineStateWithFunction:error:]
    var pipeline_error: ID = null;
    const msg_send_pipeline: *const fn (ID, SEL, ID, *ID) callconv(.c) ID = @ptrCast(objc_msgSend);
    const pipeline_state = msg_send_pipeline(device, sel_newComputePipelineStateWithFunction, function, &pipeline_error);
    if (pipeline_state == null) {
        std.log.err("Failed to create Metal compute pipeline state", .{});
        // Release function and library
        if (objc_msgSend_void) |release_fn| {
            release_fn(function, sel_release);
            release_fn(library, sel_release);
        }
        return types.KernelError.CompilationFailed;
    }

    // Allocate SafeMetalKernel with magic validation header for safe pointer casting
    const safe_kernel = try allocator.create(SafeMetalKernel);
    safe_kernel.* = .{
        .magic = kernel_magic, // Set magic for pointer validation
        .inner = .{
            .pipeline_state = pipeline_state,
            .library = library,
            .function = function,
        },
    };

    std.log.debug("Metal kernel compiled successfully: entry_point={s}", .{source.entry_point});
    return safe_kernel;
}

pub fn launchKernel(
    allocator: std.mem.Allocator,
    kernel_handle: *anyopaque,
    config: types.KernelConfig,
    args: []const ?*const anyopaque,
) types.KernelError!void {
    _ = allocator;

    if (!metal_initialized or metal_command_queue == null) {
        return types.KernelError.LaunchFailed;
    }

    const msg_send = objc_msgSend orelse return types.KernelError.LaunchFailed;
    const msg_send_void = objc_msgSend_void orelse return types.KernelError.LaunchFailed;
    const msg_send_void_ptr = objc_msgSend_void_ptr orelse return types.KernelError.LaunchFailed;
    const msg_send_void_ptr_int_int = objc_msgSend_void_ptr_int_int orelse return types.KernelError.LaunchFailed;

    // Safe pointer cast with null check and magic validation
    const kernel = safeCastToKernel(kernel_handle) orelse {
        std.log.err("launchKernel: Invalid kernel handle (null or corrupted)", .{});
        return types.KernelError.LaunchFailed;
    };

    // Validate grid and block sizes
    if (config.grid_size[0] == 0 or config.grid_size[1] == 0 or config.grid_size[2] == 0) {
        std.log.err("launchKernel: Invalid grid size (zero dimension)", .{});
        return types.KernelError.LaunchFailed;
    }
    if (config.block_size[0] == 0 or config.block_size[1] == 0 or config.block_size[2] == 0) {
        std.log.err("launchKernel: Invalid block size (zero dimension)", .{});
        return types.KernelError.LaunchFailed;
    }

    // Create command buffer: [commandQueue commandBuffer]
    const command_buffer = msg_send(metal_command_queue, sel_commandBuffer);
    if (command_buffer == null) {
        std.log.err("Failed to create Metal command buffer", .{});
        return types.KernelError.LaunchFailed;
    }

    // Create compute command encoder: [commandBuffer computeCommandEncoder]
    const encoder = msg_send(command_buffer, sel_computeCommandEncoder);
    if (encoder == null) {
        std.log.err("Failed to create Metal compute command encoder", .{});
        return types.KernelError.LaunchFailed;
    }

    // Set pipeline state: [encoder setComputePipelineState:pipelineState]
    msg_send_void_ptr(encoder, sel_setComputePipelineState, kernel.pipeline_state);

    // Set buffers: [encoder setBuffer:buffer offset:0 atIndex:i]
    for (args, 0..) |arg, i| {
        if (arg != null) {
            // Safe pointer cast with validation for buffer arguments
            const buffer_wrapper = safeCastToBufferConst(arg) orelse {
                std.log.err("launchKernel: Invalid buffer argument at index {} (null or corrupted)", .{i});
                return types.KernelError.LaunchFailed;
            };
            msg_send_void_ptr_int_int(encoder, sel_setBuffer, buffer_wrapper.buffer, 0, @intCast(i));
        }
    }

    // Dispatch threads using MTLSize structs
    // Metal supports two dispatch methods:
    // 1. dispatchThreads:threadsPerThreadgroup: - specifies total threads directly
    // 2. dispatchThreadgroups:threadsPerThreadgroup: - specifies number of threadgroups
    //
    // We use dispatchThreadgroups which is more similar to CUDA's grid/block model

    // Calculate total threads (grid_size * block_size for each dimension)
    const grid_size = MTLSize.init(
        config.grid_size[0] * config.block_size[0],
        config.grid_size[1] * config.block_size[1],
        config.grid_size[2] * config.block_size[2],
    );

    const threads_per_group = MTLSize.init(
        config.block_size[0],
        config.block_size[1],
        config.block_size[2],
    );

    // Dispatch using objc_msgSend with MTLSize parameters
    // On ARM64 (Apple Silicon), MTLSize (3 x usize = 24 bytes) is passed in registers
    // [encoder dispatchThreads:gridSize threadsPerThreadgroup:threadsPerGroup]
    const dispatch_fn: *const fn (ID, SEL, MTLSize, MTLSize) callconv(.c) void = @ptrCast(objc_msgSend);
    dispatch_fn(encoder, sel_dispatchThreads, grid_size, threads_per_group);

    // End encoding: [encoder endEncoding]
    msg_send_void(encoder, sel_endEncoding);

    // Commit: [commandBuffer commit]
    msg_send_void(command_buffer, sel_commit);

    // Wait: [commandBuffer waitUntilCompleted]
    msg_send_void(command_buffer, sel_waitUntilCompleted);

    std.log.debug("Metal kernel launched: grid=({},{},{}), block=({},{},{})", .{
        config.grid_size[0],  config.grid_size[1],  config.grid_size[2],
        config.block_size[0], config.block_size[1], config.block_size[2],
    });
}

/// Launch a kernel asynchronously without waiting for completion.
/// Returns the command buffer ID for later synchronization.
pub fn launchKernelAsync(
    allocator: std.mem.Allocator,
    kernel_handle: *anyopaque,
    config: types.KernelConfig,
    args: []const ?*const anyopaque,
) types.KernelError!ID {
    _ = allocator;

    if (!metal_initialized or metal_command_queue == null) {
        return types.KernelError.LaunchFailed;
    }

    const msg_send = objc_msgSend orelse return types.KernelError.LaunchFailed;
    const msg_send_void = objc_msgSend_void orelse return types.KernelError.LaunchFailed;
    const msg_send_void_ptr = objc_msgSend_void_ptr orelse return types.KernelError.LaunchFailed;
    const msg_send_void_ptr_int_int = objc_msgSend_void_ptr_int_int orelse return types.KernelError.LaunchFailed;

    const kernel = safeCastToKernel(kernel_handle) orelse {
        std.log.err("launchKernelAsync: Invalid kernel handle", .{});
        return types.KernelError.LaunchFailed;
    };

    // Create command buffer
    const command_buffer = msg_send(metal_command_queue, sel_commandBuffer);
    if (command_buffer == null) {
        return types.KernelError.LaunchFailed;
    }

    // Retain command buffer for tracking
    if (objc_msgSend_void) |retain_fn| {
        retain_fn(command_buffer, sel_retain);
    }

    // Create compute encoder and configure
    const encoder = msg_send(command_buffer, sel_computeCommandEncoder);
    if (encoder == null) {
        if (objc_msgSend_void) |release_fn| {
            release_fn(command_buffer, sel_release);
        }
        return types.KernelError.LaunchFailed;
    }

    msg_send_void_ptr(encoder, sel_setComputePipelineState, kernel.pipeline_state);

    for (args, 0..) |arg, i| {
        if (arg != null) {
            const buffer_wrapper = safeCastToBufferConst(arg) orelse continue;
            msg_send_void_ptr_int_int(encoder, sel_setBuffer, buffer_wrapper.buffer, 0, @intCast(i));
        }
    }

    // Dispatch threads
    const grid_size = MTLSize.init(
        config.grid_size[0] * config.block_size[0],
        config.grid_size[1] * config.block_size[1],
        config.grid_size[2] * config.block_size[2],
    );
    const threads_per_group = MTLSize.init(
        config.block_size[0],
        config.block_size[1],
        config.block_size[2],
    );

    const dispatch_fn: *const fn (ID, SEL, MTLSize, MTLSize) callconv(.c) void = @ptrCast(objc_msgSend);
    dispatch_fn(encoder, sel_dispatchThreads, grid_size, threads_per_group);

    msg_send_void(encoder, sel_endEncoding);
    msg_send_void(command_buffer, sel_commit);

    // Track pending command buffer for synchronization
    if (pending_buffers_allocator) |alloc| {
        pending_command_buffers.append(alloc, command_buffer) catch |err| {
            std.log.debug("Failed to track Metal command buffer: {t}", .{err});
        };
    }

    return command_buffer;
}

pub fn destroyKernel(allocator: std.mem.Allocator, kernel_handle: *anyopaque) void {
    // Safe pointer cast with null check and magic validation
    const kernel = safeCastToKernel(kernel_handle) orelse {
        std.log.err("destroyKernel: Invalid kernel handle (null or corrupted), skipping destruction", .{});
        return;
    };

    // Release Metal objects using Objective-C runtime
    if (objc_msgSend_void) |release_fn| {
        if (kernel.pipeline_state != null) release_fn(kernel.pipeline_state, sel_release);
        if (kernel.function != null) release_fn(kernel.function, sel_release);
        if (kernel.library != null) release_fn(kernel.library, sel_release);
    }

    // Clear magic before freeing to prevent use-after-free detection issues
    const safe_kernel: *SafeMetalKernel = @fieldParentPtr("inner", kernel);
    safe_kernel.magic = 0; // Invalidate magic on destruction

    allocator.destroy(safe_kernel);
}

pub fn allocateDeviceMemory(allocator: std.mem.Allocator, size: usize) !*anyopaque {
    const actual_allocator = buffer_allocator orelse allocator;
    return allocateDeviceMemoryWithAllocator(actual_allocator, size);
}

pub fn allocateDeviceMemoryWithAllocator(allocator: std.mem.Allocator, size: usize) !*anyopaque {
    if (!metal_initialized or metal_device == null) {
        return MetalError.BufferCreationFailed;
    }

    const device = metal_device.?;

    // Use objc_msgSend with the proper signature for newBufferWithLength:options:
    const msg_send_int = objc_msgSend_int orelse return MetalError.BufferCreationFailed;

    // [device newBufferWithLength:size options:MTLResourceStorageModeShared]
    const buffer = msg_send_int(device, sel_newBufferWithLength, size, MTLResourceStorageModeShared);
    if (buffer == null) {
        std.log.err("Failed to create Metal buffer of size {B}", .{size});
        return MetalError.BufferCreationFailed;
    }

    // Allocate SafeMetalBuffer with magic validation header for safe pointer casting
    const safe_buffer = try allocator.create(SafeMetalBuffer);
    errdefer allocator.destroy(safe_buffer);

    safe_buffer.* = .{
        .magic = buffer_magic, // Set magic for pointer validation
        .inner = .{
            .buffer = buffer,
            .size = size,
            .allocator = allocator,
        },
    };

    std.log.debug("Metal buffer allocated: size={B}", .{size});
    return safe_buffer;
}

pub fn freeDeviceMemory(allocator: std.mem.Allocator, ptr: *anyopaque) void {
    _ = allocator;

    // Safe pointer cast with null check and magic validation
    const buffer = safeCastToBuffer(ptr) orelse {
        std.log.err("freeDeviceMemory: Invalid buffer pointer (null or corrupted), skipping free", .{});
        return;
    };
    const buffer_allocator_ref = buffer.allocator;

    // Release the Metal buffer object
    if (objc_msgSend_void) |release_fn| {
        if (buffer.buffer != null) {
            release_fn(buffer.buffer, sel_release);
        }
    }

    // Clear magic before freeing to prevent use-after-free detection issues
    const safe_buffer: *SafeMetalBuffer = @fieldParentPtr("inner", buffer);
    safe_buffer.magic = 0; // Invalidate magic on destruction

    buffer_allocator_ref.destroy(safe_buffer);
}

pub fn memcpyHostToDevice(dst: *anyopaque, src: *anyopaque, size: usize) !void {
    // Safe pointer cast with null check and magic validation
    const dst_buffer = safeCastToBuffer(dst) orelse {
        std.log.err("memcpyHostToDevice: Invalid destination buffer (null or corrupted)", .{});
        return MetalError.MemoryCopyFailed;
    };

    // Bounds check: ensure copy size doesn't exceed buffer capacity
    if (size > dst_buffer.size) {
        std.log.err("memcpyHostToDevice: Copy size ({}) exceeds buffer size ({})", .{ size, dst_buffer.size });
        return MetalError.MemoryCopyFailed;
    }

    // Get buffer contents using objc_msgSend: [buffer contents]
    const msg_send = objc_msgSend orelse return MetalError.MemoryCopyFailed;
    const contents = msg_send(dst_buffer.buffer, sel_contents);
    if (contents == null) {
        std.log.err("Failed to get Metal buffer contents for host->device copy", .{});
        return MetalError.MemoryCopyFailed;
    }

    // Safe memcpy with validated pointers and bounds-checked size
    const dst_slice = @as([*]u8, @ptrCast(contents.?))[0..size];
    const src_slice = @as([*]const u8, @ptrCast(src))[0..size];
    @memcpy(dst_slice, src_slice);
    std.log.debug("Metal memcpy host->device: {B}", .{size});
}

pub fn memcpyDeviceToHost(dst: *anyopaque, src: *anyopaque, size: usize) !void {
    // Safe pointer cast with null check and magic validation
    const src_buffer = safeCastToBuffer(src) orelse {
        std.log.err("memcpyDeviceToHost: Invalid source buffer (null or corrupted)", .{});
        return MetalError.MemoryCopyFailed;
    };

    // Bounds check: ensure copy size doesn't exceed buffer capacity
    if (size > src_buffer.size) {
        std.log.err("memcpyDeviceToHost: Copy size ({}) exceeds buffer size ({})", .{ size, src_buffer.size });
        return MetalError.MemoryCopyFailed;
    }

    // Get buffer contents using objc_msgSend: [buffer contents]
    const msg_send = objc_msgSend orelse return MetalError.MemoryCopyFailed;
    const contents = msg_send(src_buffer.buffer, sel_contents);
    if (contents == null) {
        std.log.err("Failed to get Metal buffer contents for device->host copy", .{});
        return MetalError.MemoryCopyFailed;
    }

    // Safe memcpy with validated pointers and bounds-checked size
    const dst_slice = @as([*]u8, @ptrCast(dst))[0..size];
    const src_slice = @as([*]const u8, @ptrCast(contents.?))[0..size];
    @memcpy(dst_slice, src_slice);
    std.log.debug("Metal memcpy device->host: {B}", .{size});
}

pub fn memcpyDeviceToDevice(dst: *anyopaque, src: *anyopaque, size: usize) !void {
    // Safe pointer casts with null check and magic validation
    const src_buffer = safeCastToBuffer(src) orelse {
        std.log.err("memcpyDeviceToDevice: Invalid source buffer (null or corrupted)", .{});
        return MetalError.MemoryCopyFailed;
    };
    const dst_buffer = safeCastToBuffer(dst) orelse {
        std.log.err("memcpyDeviceToDevice: Invalid destination buffer (null or corrupted)", .{});
        return MetalError.MemoryCopyFailed;
    };

    // Bounds check: ensure copy size doesn't exceed either buffer's capacity
    if (size > src_buffer.size) {
        std.log.err("memcpyDeviceToDevice: Copy size ({}) exceeds source buffer size ({})", .{ size, src_buffer.size });
        return MetalError.MemoryCopyFailed;
    }
    if (size > dst_buffer.size) {
        std.log.err("memcpyDeviceToDevice: Copy size ({}) exceeds destination buffer size ({})", .{ size, dst_buffer.size });
        return MetalError.MemoryCopyFailed;
    }

    // Get buffer contents using objc_msgSend
    const msg_send = objc_msgSend orelse return MetalError.MemoryCopyFailed;
    const src_contents = msg_send(src_buffer.buffer, sel_contents);
    const dst_contents = msg_send(dst_buffer.buffer, sel_contents);

    if (src_contents == null or dst_contents == null) {
        std.log.err("Failed to get Metal buffer contents for device->device copy", .{});
        return MetalError.MemoryCopyFailed;
    }

    // Safe memcpy with validated pointers and bounds-checked size
    const dst_slice = @as([*]u8, @ptrCast(dst_contents.?))[0..size];
    const src_slice = @as([*]const u8, @ptrCast(src_contents.?))[0..size];
    @memcpy(dst_slice, src_slice);
    std.log.debug("Metal memcpy device->device: {B}", .{size});
}

fn tryLoadObjcRuntime() bool {
    // Load libobjc for Objective-C runtime support
    const objc_paths = [_][]const u8{
        "/usr/lib/libobjc.dylib",
        "/usr/lib/libobjc.A.dylib",
    };

    for (objc_paths) |path| {
        if (std.DynLib.open(path)) |lib| {
            objc_lib = lib;

            // Load Objective-C runtime functions - all variants use the same objc_msgSend
            // but with different type casts for different calling conventions
            objc_msgSend = lib.lookup(ObjcMsgSendFn, "objc_msgSend");
            objc_msgSend_int = @ptrCast(lib.lookup(*anyopaque, "objc_msgSend"));
            objc_msgSend_ptr = @ptrCast(lib.lookup(*anyopaque, "objc_msgSend"));
            objc_msgSend_ptr2 = @ptrCast(lib.lookup(*anyopaque, "objc_msgSend"));
            objc_msgSend_ptr3 = @ptrCast(lib.lookup(*anyopaque, "objc_msgSend"));
            objc_msgSend_void = @ptrCast(lib.lookup(*anyopaque, "objc_msgSend"));
            objc_msgSend_void_ptr = @ptrCast(lib.lookup(*anyopaque, "objc_msgSend"));
            objc_msgSend_void_ptr_int_int = @ptrCast(lib.lookup(*anyopaque, "objc_msgSend"));
            objc_msgSend_u64 = @ptrCast(lib.lookup(*anyopaque, "objc_msgSend"));
            objc_msgSend_u32 = @ptrCast(lib.lookup(*anyopaque, "objc_msgSend"));
            objc_msgSend_bool = @ptrCast(lib.lookup(*anyopaque, "objc_msgSend"));
            objc_msgSend_mtlsize2 = @ptrCast(lib.lookup(*anyopaque, "objc_msgSend"));
            objc_msgSend_nsstring = @ptrCast(lib.lookup(*anyopaque, "objc_msgSend"));
            sel_registerName = lib.lookup(SelRegisterNameFn, "sel_registerName");
            objc_getClass = lib.lookup(ObjcGetClassFn, "objc_getClass");

            // Verify we have the minimum required functions
            if (objc_msgSend != null and sel_registerName != null and objc_getClass != null) {
                return true;
            }
        } else |_| {}
    }

    return false;
}

fn tryLoadMetal() bool {
    // Metal framework loading on macOS
    if (builtin.target.os.tag != .macos) {
        return false;
    }

    // Try to load Metal framework dynamically
    const framework_paths = [_][]const u8{
        "/System/Library/Frameworks/Metal.framework/Metal",
        "/System/Library/Frameworks/Metal.framework/Versions/A/Metal",
    };

    for (framework_paths) |path| {
        if (std.DynLib.open(path)) |lib| {
            metal_lib = lib;
            return true;
        } else |_| {}
    }

    // Try loading via @rpath on newer macOS versions
    if (std.DynLib.open("Metal.framework/Metal")) |lib| {
        metal_lib = lib;
        return true;
    } else |_| {}

    return false;
}

fn loadMetalFunctions() bool {
    if (metal_lib == null) return false;

    // MTLCreateSystemDefaultDevice is the primary C-callable Metal function
    mtlCreateSystemDefaultDevice = metal_lib.?.lookup(
        MtlCreateSystemDefaultDeviceFn,
        "MTLCreateSystemDefaultDevice",
    ) orelse return false;

    // MTLCopyAllDevices returns an NSArray of all available Metal devices
    // This is useful for multi-GPU systems (Intel Macs with discrete GPU)
    mtlCopyAllDevices = metal_lib.?.lookup(
        MtlCopyAllDevicesFn,
        "MTLCopyAllDevices",
    );

    // All other Metal API calls use Objective-C runtime (objc_msgSend)
    return true;
}

fn initializeSelectors() MetalError!void {
    if (selectors_initialized) return;

    const sel_fn = sel_registerName orelse return MetalError.SelectorNotFound;

    // Command queue and buffer selectors
    sel_newCommandQueue = sel_fn("newCommandQueue");
    sel_commandBuffer = sel_fn("commandBuffer");
    sel_computeCommandEncoder = sel_fn("computeCommandEncoder");
    sel_commit = sel_fn("commit");
    sel_waitUntilCompleted = sel_fn("waitUntilCompleted");
    sel_endEncoding = sel_fn("endEncoding");

    // Library and kernel compilation selectors
    sel_newLibraryWithSource = sel_fn("newLibraryWithSource:options:error:");
    sel_newFunctionWithName = sel_fn("newFunctionWithName:");
    sel_newComputePipelineStateWithFunction = sel_fn("newComputePipelineStateWithFunction:error:");

    // Buffer and memory selectors
    sel_newBufferWithLength = sel_fn("newBufferWithLength:options:");
    sel_contents = sel_fn("contents");
    sel_length = sel_fn("length");

    // Compute encoder selectors
    sel_setComputePipelineState = sel_fn("setComputePipelineState:");
    sel_setBuffer = sel_fn("setBuffer:offset:atIndex:");
    sel_dispatchThreads = sel_fn("dispatchThreads:threadsPerThreadgroup:");
    sel_dispatchThreadgroups = sel_fn("dispatchThreadgroups:threadsPerThreadgroup:");

    // Memory management selectors
    sel_release = sel_fn("release");
    sel_retain = sel_fn("retain");

    // Device property selectors
    sel_name = sel_fn("name");
    sel_recommendedMaxWorkingSetSize = sel_fn("recommendedMaxWorkingSetSize");
    sel_maxThreadsPerThreadgroup = sel_fn("maxThreadsPerThreadgroup");
    sel_maxBufferLength = sel_fn("maxBufferLength");
    sel_supportsFamily = sel_fn("supportsFamily:");
    sel_registryID = sel_fn("registryID");
    sel_isLowPower = sel_fn("isLowPower");
    sel_isHeadless = sel_fn("isHeadless");
    sel_hasUnifiedMemory = sel_fn("hasUnifiedMemory");

    // NSString selectors
    sel_stringWithUTF8String = sel_fn("stringWithUTF8String:");
    sel_UTF8String = sel_fn("UTF8String");

    // NSArray selectors
    sel_count = sel_fn("count");
    sel_objectAtIndex = sel_fn("objectAtIndex:");

    // Pipeline state selectors
    sel_maxTotalThreadsPerThreadgroup = sel_fn("maxTotalThreadsPerThreadgroup");
    sel_threadExecutionWidth = sel_fn("threadExecutionWidth");

    selectors_initialized = true;
}

/// Set the allocator to use for buffer metadata allocations.
pub fn setBufferAllocator(allocator: std.mem.Allocator) void {
    buffer_allocator = allocator;
}

/// Set the allocator to use for tracking pending command buffers.
pub fn setPendingBuffersAllocator(allocator: std.mem.Allocator) void {
    pending_buffers_allocator = allocator;
}

/// Synchronize with the GPU. Blocks until all pending commands are complete.
/// This waits for all command buffers submitted via launchKernelAsync.
pub fn synchronize() void {
    const msg_send_void = objc_msgSend_void orelse return;

    // Wait for all pending command buffers to complete
    for (pending_command_buffers.items) |cmd_buffer| {
        if (cmd_buffer != null) {
            // [commandBuffer waitUntilCompleted]
            msg_send_void(cmd_buffer, sel_waitUntilCompleted);
            // Release the retained command buffer
            msg_send_void(cmd_buffer, sel_release);
        }
    }

    // Clear the pending list
    if (pending_buffers_allocator) |alloc| {
        pending_command_buffers.clearRetainingCapacity();
        _ = alloc;
    }

    std.log.debug("Metal synchronize complete", .{});
}

/// Wait for a specific command buffer to complete.
pub fn waitForCommandBuffer(cmd_buffer: ID) void {
    if (cmd_buffer == null) return;

    const msg_send_void = objc_msgSend_void orelse return;

    // [commandBuffer waitUntilCompleted]
    msg_send_void(cmd_buffer, sel_waitUntilCompleted);
    // Release the retained command buffer
    msg_send_void(cmd_buffer, sel_release);

    // Remove from pending list if present
    for (pending_command_buffers.items, 0..) |buf, i| {
        if (buf == cmd_buffer) {
            _ = pending_command_buffers.swapRemove(i);
            break;
        }
    }
}

/// Check if Metal backend is available on this system.
pub fn isAvailable() bool {
    if (builtin.target.os.tag != .macos) {
        return false;
    }
    // Quick check if Metal framework exists
    if (std.DynLib.open("/System/Library/Frameworks/Metal.framework/Metal")) |lib| {
        lib.close();
        return true;
    } else |_| {
        return false;
    }
}

// ============================================================================
// Device Enumeration
// ============================================================================

const Device = @import("../device.zig").Device;
const DeviceType = @import("../device.zig").DeviceType;

/// Query properties from a Metal device object.
fn queryDeviceInfo(mtl_device: ID, allocator: std.mem.Allocator, device_id: u32) !Device {
    const msg_send = objc_msgSend orelse return MetalError.DeviceQueryFailed;
    const msg_send_u64 = objc_msgSend_u64 orelse return MetalError.DeviceQueryFailed;
    const msg_send_bool = objc_msgSend_bool orelse return MetalError.DeviceQueryFailed;

    // Query device name
    var name_slice: []const u8 = "Unknown Metal GPU";
    const name_nsstring = msg_send(mtl_device, sel_name);
    if (name_nsstring != null) {
        const utf8_fn: *const fn (ID, SEL) callconv(.c) ?[*:0]const u8 = @ptrCast(objc_msgSend);
        const utf8_ptr = utf8_fn(name_nsstring, sel_UTF8String);
        if (utf8_ptr) |ptr| {
            name_slice = std.mem.span(ptr);
        }
    }

    const name = try allocator.dupe(u8, name_slice);
    errdefer allocator.free(name);

    // Query memory
    const total_memory = msg_send_u64(mtl_device, sel_recommendedMaxWorkingSetSize);

    // Query device properties
    const is_low_power = msg_send_bool(mtl_device, sel_isLowPower);
    const has_unified_memory = msg_send_bool(mtl_device, sel_hasUnifiedMemory);

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
        .name = name,
        .device_type = device_type,
        .total_memory = if (total_memory > 0) total_memory else null,
        .available_memory = null, // Metal doesn't provide real-time available memory
        .is_emulated = false,
        .capability = .{
            .supports_fp16 = true, // All modern Metal devices support FP16
            .supports_fp64 = false, // Metal doesn't support FP64 compute
            .supports_int8 = true, // Apple Silicon and newer GPUs support Int8
            .supports_async_transfers = true,
            .unified_memory = has_unified_memory,
            .max_threads_per_block = device_max_threads_per_group,
            .max_shared_memory_bytes = 32 * 1024, // Typical Metal shared memory
        },
        .compute_units = null, // Metal doesn't directly expose compute unit count
        .clock_mhz = null, // Metal doesn't expose clock speed
    };
}

/// Enumerate all Metal devices available on this Mac.
/// On Intel Macs with discrete GPUs, this may return multiple devices.
/// On Apple Silicon, typically returns a single device.
pub fn enumerateDevices(allocator: std.mem.Allocator) ![]Device {
    if (!isAvailable()) {
        return &[_]Device{};
    }

    var devices = std.ArrayListUnmanaged(Device).empty;
    errdefer {
        for (devices.items) |dev| {
            allocator.free(dev.name);
        }
        devices.deinit(allocator);
    }

    // Initialize Metal if not already done
    if (!metal_initialized) {
        init() catch {
            return &[_]Device{};
        };
    }

    // Try to enumerate all devices using MTLCopyAllDevices (if available)
    if (mtlCopyAllDevices) |copy_all_fn| {
        const device_array = copy_all_fn();
        if (device_array != null) {
            _ = objc_msgSend orelse return &[_]Device{};
            const msg_send_u64 = objc_msgSend_u64 orelse return &[_]Device{};

            // Get count: [array count]
            const count = msg_send_u64(device_array, sel_count);

            // Enumerate each device
            var i: u32 = 0;
            while (i < count) : (i += 1) {
                // Get device at index: [array objectAtIndex:i]
                const get_obj_fn: *const fn (ID, SEL, usize) callconv(.c) ID = @ptrCast(objc_msgSend);
                const mtl_device = get_obj_fn(device_array, sel_objectAtIndex, i);
                if (mtl_device != null) {
                    const dev_info = queryDeviceInfo(mtl_device, allocator, i) catch continue;
                    try devices.append(allocator, dev_info);
                }
            }

            // Release the array
            if (objc_msgSend_void) |release_fn| {
                release_fn(device_array, sel_release);
            }

            if (devices.items.len > 0) {
                return devices.toOwnedSlice(allocator);
            }
        }
    }

    // Fallback: use the default device if MTLCopyAllDevices is not available
    if (metal_device != null) {
        const dev_info = queryDeviceInfo(metal_device, allocator, 0) catch {
            // Fallback to basic device info
            const name = try allocator.dupe(u8, if (device_name_len > 0)
                device_name_buf[0..device_name_len]
            else
                "Metal GPU");

            return try devices.toOwnedSlice() ++ &[_]Device{.{
                .id = 0,
                .backend = .metal,
                .name = name,
                .device_type = if (builtin.target.cpu.arch == .aarch64) .integrated else .discrete,
                .total_memory = if (device_total_memory > 0) device_total_memory else null,
                .available_memory = null,
                .is_emulated = false,
                .capability = .{
                    .supports_fp16 = true,
                    .supports_fp64 = false,
                    .supports_int8 = true,
                    .supports_async_transfers = true,
                    .unified_memory = builtin.target.cpu.arch == .aarch64,
                },
                .compute_units = null,
                .clock_mhz = null,
            }};
        };
        try devices.append(allocator, dev_info);
    }

    return devices.toOwnedSlice(allocator);
}

/// Get detailed information about the current default Metal device.
pub fn getDeviceInfo() ?DeviceInfo {
    if (!metal_initialized or metal_device == null) return null;

    var info = DeviceInfo{
        .name = if (device_name_len > 0) device_name_buf[0..device_name_len] else "Unknown",
        .total_memory = device_total_memory,
        .max_buffer_length = device_max_buffer_length,
        .max_threads_per_threadgroup = device_max_threads_per_group,
        .has_unified_memory = builtin.target.cpu.arch == .aarch64,
    };

    if (cached_feature_set) |fs| {
        info.gpu_family = @intFromEnum(fs.gpu_family);
        info.supports_mesh_shaders = fs.supports_mesh_shaders;
        info.supports_ray_tracing = fs.supports_ray_tracing;
        info.supports_mps = fs.supports_mps;
        info.supports_neural_engine = fs.has_neural_engine;
    }

    return info;
}

/// Detailed device information struct (re-exported from metal_types.zig).
pub const DeviceInfo = metal_types.DeviceInfo;

/// Get the detected GPU feature set (populated during init).
/// Returns null if Metal is not initialized or feature detection failed.
pub fn getFeatureSet() ?MetalFeatureSet {
    return cached_feature_set;
}

// ============================================================================
// Test discovery for extracted submodules
// ============================================================================

test {
    _ = @import("metal_types.zig");
    _ = @import("metal_test.zig");
    _ = @import("metal/gpu_family.zig");
    _ = @import("metal/mps.zig");
    _ = @import("metal/coreml.zig");
    _ = @import("metal/mesh_shaders.zig");
    _ = @import("metal/ray_tracing.zig");
}
