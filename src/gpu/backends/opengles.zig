//! OpenGL ES backend implementation with compute shader support.
//!
//! Provides OpenGL ES-specific kernel compilation and execution using compute shaders
//! for mobile and embedded platforms. Requires OpenGL ES 3.1+ for compute shader support.

const std = @import("std");
const types = @import("../kernel_types.zig");
const shared = @import("shared.zig");
const fallback = @import("fallback.zig");

pub const OpenGlesError = error{
    InitializationFailed,
    ContextCreationFailed,
    ShaderCompilationFailed,
    ProgramLinkingFailed,
    BufferCreationFailed,
    DispatchFailed,
    VersionNotSupported,
    FunctionLoadFailed,
    LibraryNotFound,
    HintNotSupported,
};

/// Buffer usage hints for OpenGL ES memory allocation
/// Maps to GL buffer usage patterns for optimal memory placement
pub const BufferUsageHint = enum {
    /// GPU-only buffer, written once, used many times (GL_STATIC_DRAW)
    static_draw,
    /// GPU-only buffer, read-heavy workloads (GL_STATIC_READ)
    static_read,
    /// Host-visible buffer for frequent CPU access (GL_DYNAMIC_DRAW)
    dynamic_storage,
    /// Host-visible + coherent for CPU/GPU shared access (GL_DYNAMIC_READ)
    dynamic_coherent,
};

var opengles_lib: ?std.DynLib = null;
var opengles_initialized = false;
var init_mutex = std.Thread.Mutex{};

// OpenGL ES function pointers (subset of OpenGL)
const GlesGetStringFn = *const fn (u32) callconv(.c) ?[*:0]const u8;
const GlesCreateShaderFn = *const fn (u32) callconv(.c) u32;
const GlesShaderSourceFn = *const fn (u32, i32, ?[*]const [*:0]const u8, ?[*]const i32) callconv(.c) void;
const GlesCompileShaderFn = *const fn (u32) callconv(.c) void;
const GlesGetShaderivFn = *const fn (u32, u32, *i32) callconv(.c) void;
const GlesGetShaderInfoLogFn = *const fn (u32, i32, *i32, [*]u8) callconv(.c) void;
const GlesCreateProgramFn = *const fn () callconv(.c) u32;
const GlesAttachShaderFn = *const fn (u32, u32) callconv(.c) void;
const GlesLinkProgramFn = *const fn (u32) callconv(.c) void;
const GlesGetProgramivFn = *const fn (u32, u32, *i32) callconv(.c) void;
const GlesGetProgramInfoLogFn = *const fn (u32, i32, *i32, [*]u8) callconv(.c) void;
const GlesUseProgramFn = *const fn (u32) callconv(.c) void;
const GlesGenBuffersFn = *const fn (i32, [*]u32) callconv(.c) void;
const GlesBindBufferFn = *const fn (u32, u32) callconv(.c) void;
const GlesBufferDataFn = *const fn (u32, isize, ?*anyopaque, u32) callconv(.c) void;
const GlesBindBufferBaseFn = *const fn (u32, u32, u32) callconv(.c) void;
const GlesDispatchComputeFn = *const fn (u32, u32, u32) callconv(.c) void;
const GlesMemoryBarrierFn = *const fn (u32) callconv(.c) void;
const GlesGetBufferSubDataFn = *const fn (u32, isize, isize, *anyopaque) callconv(.c) void;
const GlesDeleteBuffersFn = *const fn (i32, [*]const u32) callconv(.c) void;
const GlesDeleteProgramFn = *const fn (u32) callconv(.c) void;
const GlesDeleteShaderFn = *const fn (u32) callconv(.c) void;
const GlesMapBufferRangeFn = *const fn (u32, isize, isize, u32) callconv(.c) ?*anyopaque;
const GlesUnmapBufferFn = *const fn (u32) callconv(.c) u8;
const GlesFlushMappedBufferRangeFn = *const fn (u32, isize, isize) callconv(.c) void;
const GlesFinishFn = *const fn () callconv(.c) void;
const GlesFlushFn = *const fn () callconv(.c) void;
const GlesGetErrorFn = *const fn () callconv(.c) u32;
const GlesGetIntegervFn = *const fn (u32, *i32) callconv(.c) void;
const GlesCopyBufferSubDataFn = *const fn (u32, u32, isize, isize, isize) callconv(.c) void;

var glesGetString: ?GlesGetStringFn = null;
var glesCreateShader: ?GlesCreateShaderFn = null;
var glesShaderSource: ?GlesShaderSourceFn = null;
var glesCompileShader: ?GlesCompileShaderFn = null;
var glesGetShaderiv: ?GlesGetShaderivFn = null;
var glesGetShaderInfoLog: ?GlesGetShaderInfoLogFn = null;
var glesCreateProgram: ?GlesCreateProgramFn = null;
var glesAttachShader: ?GlesAttachShaderFn = null;
var glesLinkProgram: ?GlesLinkProgramFn = null;
var glesGetProgramiv: ?GlesGetProgramivFn = null;
var glesGetProgramInfoLog: ?GlesGetProgramInfoLogFn = null;
var glesUseProgram: ?GlesUseProgramFn = null;
var glesGenBuffers: ?GlesGenBuffersFn = null;
var glesBindBuffer: ?GlesBindBufferFn = null;
var glesBufferData: ?GlesBufferDataFn = null;
var glesBindBufferBase: ?GlesBindBufferBaseFn = null;
var glesDispatchCompute: ?GlesDispatchComputeFn = null;
var glesMemoryBarrier: ?GlesMemoryBarrierFn = null;
var glesGetBufferSubData: ?GlesGetBufferSubDataFn = null;
var glesDeleteBuffers: ?GlesDeleteBuffersFn = null;
var glesDeleteProgram: ?GlesDeleteProgramFn = null;
var glesDeleteShader: ?GlesDeleteShaderFn = null;
var glesMapBufferRange: ?GlesMapBufferRangeFn = null;
var glesUnmapBuffer: ?GlesUnmapBufferFn = null;
var glesFlushMappedBufferRange: ?GlesFlushMappedBufferRangeFn = null;
var glesFinish: ?GlesFinishFn = null;
var glesFlush: ?GlesFlushFn = null;
var glesGetError: ?GlesGetErrorFn = null;
var glesGetIntegerv: ?GlesGetIntegervFn = null;
var glesCopyBufferSubData: ?GlesCopyBufferSubDataFn = null;

// Cached allocator for buffer metadata
var buffer_allocator: ?std.mem.Allocator = null;

// Cached OpenGL ES version info
var gles_major_version: i32 = 0;
var gles_minor_version: i32 = 0;

const OpenGlesKernel = struct {
    program: u32,
    shader: u32,
};

const OpenGlesBuffer = struct {
    buffer_id: u32,
    size: usize,
    allocator: std.mem.Allocator,
};

const GL_COMPUTE_SHADER = 0x91B9;
const GL_SHADER_STORAGE_BUFFER = 0x90D2;
const GL_SHADER_STORAGE_BARRIER_BIT = 0x00002000;
const GL_STATIC_DRAW = 0x88E4;
const GL_STATIC_READ = 0x88E5;
const GL_DYNAMIC_DRAW = 0x88E8;
const GL_DYNAMIC_READ = 0x88E9;
const GL_COMPILE_STATUS = 0x8B81;
const GL_LINK_STATUS = 0x8B82;
const GL_MAP_READ_BIT = 0x0001;
const GL_MAP_WRITE_BIT = 0x0002;
const GL_MAP_INVALIDATE_BUFFER_BIT = 0x0008;
const GL_MAP_FLUSH_EXPLICIT_BIT = 0x0010;
const GL_COPY_READ_BUFFER = 0x8F36;
const GL_COPY_WRITE_BUFFER = 0x8F37;
const GL_MAJOR_VERSION = 0x821B;
const GL_MINOR_VERSION = 0x821C;
const GL_NO_ERROR = 0;
const GL_INVALID_ENUM = 0x0500;
const GL_INVALID_VALUE = 0x0501;
const GL_INVALID_OPERATION = 0x0502;
const GL_OUT_OF_MEMORY = 0x0505;

pub fn init() OpenGlesError!void {
    init_mutex.lock();
    defer init_mutex.unlock();

    if (opengles_initialized) return;

    if (!tryLoadOpenGles()) {
        return OpenGlesError.LibraryNotFound;
    }
    errdefer if (opengles_lib) |lib| lib.close();

    if (!loadOpenGlesFunctions()) {
        return OpenGlesError.FunctionLoadFailed;
    }

    // Check if compute shaders are supported (OpenGL ES 3.1+)
    const version_string = glesGetString orelse return OpenGlesError.InitializationFailed;
    const version = version_string(0x1F02); // GL_VERSION
    if (version == null) {
        return OpenGlesError.VersionNotSupported;
    }

    // Parse OpenGL ES version using glGetIntegerv for accurate major/minor version
    const get_integer_fn = glesGetIntegerv orelse {
        // Fallback: parse version string
        if (parseVersionString(std.mem.span(version))) |parsed| {
            gles_major_version = parsed.major;
            gles_minor_version = parsed.minor;
        } else {
            return OpenGlesError.VersionNotSupported;
        }
        opengles_initialized = true;
        std.log.debug("OpenGL ES backend initialized (parsed version): {}.{}", .{ gles_major_version, gles_minor_version });
        return;
    };

    get_integer_fn(GL_MAJOR_VERSION, &gles_major_version);
    get_integer_fn(GL_MINOR_VERSION, &gles_minor_version);

    // Verify compute shader support (OpenGL ES 3.1+)
    if (gles_major_version < 3 or (gles_major_version == 3 and gles_minor_version < 1)) {
        std.log.err("OpenGL ES {}.{} does not support compute shaders (requires 3.1+)", .{
            gles_major_version,
            gles_minor_version,
        });
        return OpenGlesError.VersionNotSupported;
    }

    opengles_initialized = true;
    std.log.debug("OpenGL ES backend initialized successfully: version {}.{}", .{ gles_major_version, gles_minor_version });
}

pub fn deinit() void {
    init_mutex.lock();
    defer init_mutex.unlock();

    if (!opengles_initialized) return;

    if (opengles_lib) |lib| {
        lib.close();
    }
    opengles_lib = null;
    opengles_initialized = false;

    std.log.debug("OpenGL ES backend deinitialized", .{});
}

pub fn compileKernel(
    allocator: std.mem.Allocator,
    source: types.KernelSource,
) (types.KernelError || OpenGlesError)!*anyopaque {
    if (!opengles_initialized) {
        return OpenGlesError.InitializationFailed;
    }

    // Create compute shader
    const create_shader_fn = glesCreateShader orelse return OpenGlesError.ShaderCompilationFailed;
    const shader = create_shader_fn(GL_COMPUTE_SHADER);
    if (shader == 0) {
        return OpenGlesError.ShaderCompilationFailed;
    }
    errdefer if (glesDeleteShader) |delete_fn| delete_fn(shader);

    // Set shader source
    const source_ptr = &[_][*:0]const u8{source.source.ptr};
    const set_source_fn = glesShaderSource orelse return OpenGlesError.ShaderCompilationFailed;
    set_source_fn(shader, 1, source_ptr.ptr, null);

    // Compile shader
    const compile_fn = glesCompileShader orelse return OpenGlesError.ShaderCompilationFailed;
    compile_fn(shader);

    // Check compilation status
    const get_shader_iv_fn = glesGetShaderiv orelse return OpenGlesError.ShaderCompilationFailed;
    var compile_status: i32 = 0;
    get_shader_iv_fn(shader, GL_COMPILE_STATUS, &compile_status);

    if (compile_status == 0) {
        // Get error log
        var log_length: i32 = 0;
        get_shader_iv_fn(shader, 0x8B84, &log_length); // GL_INFO_LOG_LENGTH
        if (log_length > 0) {
            const log = try allocator.alloc(u8, @intCast(log_length));
            defer allocator.free(log);
            const get_log_fn = glesGetShaderInfoLog orelse return OpenGlesError.ShaderCompilationFailed;
            get_log_fn(shader, log_length, null, log.ptr);
            std.log.err("OpenGL ES shader compilation failed: {s}", .{log});
        }
        return OpenGlesError.ShaderCompilationFailed;
    }

    // Create program
    const create_program_fn = glesCreateProgram orelse return OpenGlesError.ProgramLinkingFailed;
    const program = create_program_fn();
    if (program == 0) {
        return OpenGlesError.ProgramLinkingFailed;
    }
    errdefer if (glesDeleteProgram) |delete_fn| delete_fn(program);

    // Attach shader
    const attach_fn = glesAttachShader orelse return OpenGlesError.ProgramLinkingFailed;
    attach_fn(program, shader);

    // Link program
    const link_fn = glesLinkProgram orelse return OpenGlesError.ProgramLinkingFailed;
    link_fn(program);

    // Check link status
    const get_program_iv_fn = glesGetProgramiv orelse return OpenGlesError.ProgramLinkingFailed;
    var link_status: i32 = 0;
    get_program_iv_fn(program, GL_LINK_STATUS, &link_status);

    if (link_status == 0) {
        // Get error log
        var log_length: i32 = 0;
        get_program_iv_fn(program, 0x8B84, &log_length); // GL_INFO_LOG_LENGTH
        if (log_length > 0) {
            const log = try allocator.alloc(u8, @intCast(log_length));
            defer allocator.free(log);
            const get_log_fn = glesGetProgramInfoLog orelse return OpenGlesError.ProgramLinkingFailed;
            get_log_fn(program, log_length, null, log.ptr);
            std.log.err("OpenGL ES program linking failed: {s}", .{log});
        }
        return OpenGlesError.ProgramLinkingFailed;
    }

    const kernel = try allocator.create(OpenGlesKernel);
    kernel.* = .{
        .program = program,
        .shader = shader,
    };

    return kernel;
}

pub fn launchKernel(
    allocator: std.mem.Allocator,
    kernel_handle: *anyopaque,
    config: types.KernelConfig,
    args: []const ?*const anyopaque,
) (types.KernelError || OpenGlesError)!void {
    _ = allocator;

    if (!opengles_initialized) {
        return OpenGlesError.InitializationFailed;
    }

    const kernel: *OpenGlesKernel = @ptrCast(@alignCast(kernel_handle));

    // Use program
    const use_program_fn = glesUseProgram orelse return OpenGlesError.DispatchFailed;
    use_program_fn(kernel.program);

    // Bind buffers
    const bind_buffer_base_fn = glesBindBufferBase orelse return OpenGlesError.DispatchFailed;
    for (args, 0..) |arg, i| {
        if (arg != null) {
            const buffer: *OpenGlesBuffer = @ptrCast(@alignCast(arg.?));
            bind_buffer_base_fn(GL_SHADER_STORAGE_BUFFER, @intCast(i), buffer.buffer_id);
        }
    }

    // Dispatch compute
    const dispatch_fn = glesDispatchCompute orelse return OpenGlesError.DispatchFailed;
    dispatch_fn(config.grid_dim[0], config.grid_dim[1], config.grid_dim[2]);

    // Memory barrier
    const barrier_fn = glesMemoryBarrier orelse return OpenGlesError.DispatchFailed;
    barrier_fn(GL_SHADER_STORAGE_BARRIER_BIT);

    std.log.debug("OpenGL ES kernel dispatched: {}x{}x{}", .{
        config.grid_dim[0],
        config.grid_dim[1],
        config.grid_dim[2],
    });
}

pub fn destroyKernel(allocator: std.mem.Allocator, kernel_handle: *anyopaque) void {
    if (!opengles_initialized) {
        return;
    }

    const kernel: *OpenGlesKernel = @ptrCast(@alignCast(kernel_handle));

    const delete_program_fn = glesDeleteProgram orelse return;
    delete_program_fn(kernel.program);

    const delete_shader_fn = glesDeleteShader orelse return;
    delete_shader_fn(kernel.shader);

    allocator.destroy(kernel);
}

pub fn allocateDeviceMemory(size: usize) OpenGlesError!*anyopaque {
    // Use cached allocator or fallback to page_allocator
    const allocator = buffer_allocator orelse std.heap.page_allocator;
    return allocateDeviceMemoryWithAllocator(allocator, size);
}

/// Allocate device memory with a specific usage hint for optimal placement
pub fn allocateDeviceMemoryWithHint(size: usize, hint: BufferUsageHint) OpenGlesError!*anyopaque {
    const allocator = buffer_allocator orelse std.heap.page_allocator;

    if (!opengles_initialized) {
        return OpenGlesError.InitializationFailed;
    }

    const gen_buffers_fn = glesGenBuffers orelse return OpenGlesError.BufferCreationFailed;
    var buffer_id: u32 = 0;
    gen_buffers_fn(1, &buffer_id);
    if (buffer_id == 0) {
        checkAndLogGlesError("glGenBuffers");
        return OpenGlesError.BufferCreationFailed;
    }

    const bind_buffer_fn = glesBindBuffer orelse return OpenGlesError.BufferCreationFailed;
    bind_buffer_fn(GL_SHADER_STORAGE_BUFFER, buffer_id);

    // Map BufferUsageHint to OpenGL ES usage constants
    const gl_usage: u32 = switch (hint) {
        .static_draw => GL_STATIC_DRAW,
        .static_read => GL_STATIC_READ,
        .dynamic_storage => GL_DYNAMIC_DRAW,
        .dynamic_coherent => GL_DYNAMIC_READ,
    };

    const buffer_data_fn = glesBufferData orelse return OpenGlesError.BufferCreationFailed;
    buffer_data_fn(GL_SHADER_STORAGE_BUFFER, @intCast(size), null, gl_usage);

    // Check for GL errors after buffer creation
    if (checkAndLogGlesError("glBufferData")) {
        const delete_buffers_fn = glesDeleteBuffers orelse return OpenGlesError.BufferCreationFailed;
        delete_buffers_fn(1, &buffer_id);
        return OpenGlesError.BufferCreationFailed;
    }

    const opengles_buffer = allocator.create(OpenGlesBuffer) catch {
        const delete_buffers_fn = glesDeleteBuffers orelse return OpenGlesError.BufferCreationFailed;
        delete_buffers_fn(1, &buffer_id);
        return OpenGlesError.BufferCreationFailed;
    };
    errdefer allocator.destroy(opengles_buffer);

    opengles_buffer.* = .{
        .buffer_id = buffer_id,
        .size = size,
        .allocator = allocator,
    };

    std.log.debug("OpenGL ES buffer allocated with hint {t}: ID={}, size={B}", .{ hint, buffer_id, size });
    return opengles_buffer;
}

pub fn allocateDeviceMemoryWithAllocator(allocator: std.mem.Allocator, size: usize) OpenGlesError!*anyopaque {
    if (!opengles_initialized) {
        return OpenGlesError.InitializationFailed;
    }

    const gen_buffers_fn = glesGenBuffers orelse return OpenGlesError.BufferCreationFailed;
    var buffer_id: u32 = 0;
    gen_buffers_fn(1, &buffer_id);
    if (buffer_id == 0) {
        checkAndLogGlesError("glGenBuffers");
        return OpenGlesError.BufferCreationFailed;
    }

    const bind_buffer_fn = glesBindBuffer orelse return OpenGlesError.BufferCreationFailed;
    bind_buffer_fn(GL_SHADER_STORAGE_BUFFER, buffer_id);

    const buffer_data_fn = glesBufferData orelse return OpenGlesError.BufferCreationFailed;
    buffer_data_fn(GL_SHADER_STORAGE_BUFFER, @intCast(size), null, GL_DYNAMIC_READ);

    // Check for GL errors after buffer creation
    if (checkAndLogGlesError("glBufferData")) {
        const delete_buffers_fn = glesDeleteBuffers orelse return OpenGlesError.BufferCreationFailed;
        delete_buffers_fn(1, &buffer_id);
        return OpenGlesError.BufferCreationFailed;
    }

    const opengles_buffer = allocator.create(OpenGlesBuffer) catch {
        const delete_buffers_fn = glesDeleteBuffers orelse return OpenGlesError.BufferCreationFailed;
        delete_buffers_fn(1, &buffer_id);
        return OpenGlesError.BufferCreationFailed;
    };
    errdefer allocator.destroy(opengles_buffer);

    opengles_buffer.* = .{
        .buffer_id = buffer_id,
        .size = size,
        .allocator = allocator,
    };

    std.log.debug("OpenGL ES buffer allocated: ID={}, size={B}", .{ buffer_id, size });
    return opengles_buffer;
}

pub fn freeDeviceMemory(ptr: *anyopaque) void {
    if (!opengles_initialized) {
        return;
    }

    const buffer: *OpenGlesBuffer = @ptrCast(@alignCast(ptr));
    const allocator = buffer.allocator;

    const delete_buffers_fn = glesDeleteBuffers orelse return;
    delete_buffers_fn(1, &buffer.buffer_id);
    _ = checkAndLogGlesError("glDeleteBuffers");

    allocator.destroy(buffer);
}

pub fn memcpyDeviceToDevice(dst: *anyopaque, src: *anyopaque, size: usize) OpenGlesError!void {
    if (!opengles_initialized) {
        return OpenGlesError.InitializationFailed;
    }

    const src_buffer: *OpenGlesBuffer = @ptrCast(@alignCast(src));
    const dst_buffer: *OpenGlesBuffer = @ptrCast(@alignCast(dst));

    if (size > src_buffer.size) {
        std.log.err("OpenGL ES memcpy size ({B}) exceeds source buffer size ({B})", .{ size, src_buffer.size });
        return OpenGlesError.BufferCreationFailed;
    }
    if (size > dst_buffer.size) {
        std.log.err("OpenGL ES memcpy size ({B}) exceeds destination buffer size ({B})", .{ size, dst_buffer.size });
        return OpenGlesError.BufferCreationFailed;
    }

    const bind_buffer_fn = glesBindBuffer orelse return OpenGlesError.BufferCreationFailed;
    const copy_buffer_fn = glesCopyBufferSubData orelse {
        // Fallback: copy through host memory if glCopyBufferSubData not available
        return memcpyDeviceToDeviceFallback(dst_buffer, src_buffer, size);
    };

    bind_buffer_fn(GL_COPY_READ_BUFFER, src_buffer.buffer_id);
    bind_buffer_fn(GL_COPY_WRITE_BUFFER, dst_buffer.buffer_id);
    copy_buffer_fn(GL_COPY_READ_BUFFER, GL_COPY_WRITE_BUFFER, 0, 0, @intCast(size));

    if (checkAndLogGlesError("glCopyBufferSubData")) {
        return OpenGlesError.BufferCreationFailed;
    }

    std.log.debug("OpenGL ES memcpy device->device: {B}", .{size});
}

fn memcpyDeviceToDeviceFallback(dst: *OpenGlesBuffer, src: *OpenGlesBuffer, size: usize) OpenGlesError!void {
    // Fallback implementation using host memory as intermediate via glMapBufferRange
    const bind_buffer_fn = glesBindBuffer orelse return OpenGlesError.BufferCreationFailed;
    const map_buffer_range_fn = glesMapBufferRange orelse return OpenGlesError.BufferCreationFailed;
    const unmap_buffer_fn = glesUnmapBuffer orelse return OpenGlesError.BufferCreationFailed;

    var temp_buffer: [4096]u8 = undefined;
    var offset: usize = 0;

    while (offset < size) {
        const chunk_size = @min(temp_buffer.len, size - offset);

        // Read from source using glMapBufferRange
        bind_buffer_fn(GL_SHADER_STORAGE_BUFFER, src.buffer_id);
        const src_ptr = map_buffer_range_fn(GL_SHADER_STORAGE_BUFFER, @intCast(offset), @intCast(chunk_size), GL_MAP_READ_BIT);
        if (src_ptr == null) {
            return OpenGlesError.BufferCreationFailed;
        }

        const src_slice: [*]const u8 = @ptrCast(src_ptr.?);
        @memcpy(temp_buffer[0..chunk_size], src_slice[0..chunk_size]);
        _ = unmap_buffer_fn(GL_SHADER_STORAGE_BUFFER);

        // Write to destination using glMapBufferRange
        bind_buffer_fn(GL_SHADER_STORAGE_BUFFER, dst.buffer_id);
        const dst_ptr = map_buffer_range_fn(GL_SHADER_STORAGE_BUFFER, @intCast(offset), @intCast(chunk_size), GL_MAP_WRITE_BIT);
        if (dst_ptr == null) {
            return OpenGlesError.BufferCreationFailed;
        }

        const dst_slice: [*]u8 = @ptrCast(dst_ptr.?);
        @memcpy(dst_slice[0..chunk_size], temp_buffer[0..chunk_size]);
        _ = unmap_buffer_fn(GL_SHADER_STORAGE_BUFFER);

        offset += chunk_size;
    }

    std.log.debug("OpenGL ES memcpy device->device (fallback): {B}", .{size});
}

pub fn memcpyHostToDevice(dst: *anyopaque, src: *anyopaque, size: usize) OpenGlesError!void {
    if (!opengles_initialized) {
        return OpenGlesError.InitializationFailed;
    }

    const dst_buffer: *OpenGlesBuffer = @ptrCast(@alignCast(dst));
    if (size > dst_buffer.size) {
        std.log.err("OpenGL ES memcpy size ({B}) exceeds buffer size ({B})", .{ size, dst_buffer.size });
        return OpenGlesError.BufferCreationFailed;
    }

    const bind_buffer_fn = glesBindBuffer orelse return OpenGlesError.BufferCreationFailed;
    bind_buffer_fn(GL_SHADER_STORAGE_BUFFER, dst_buffer.buffer_id);

    const buffer_data_fn = glesBufferData orelse return OpenGlesError.BufferCreationFailed;
    buffer_data_fn(GL_SHADER_STORAGE_BUFFER, @intCast(size), src, GL_STATIC_DRAW);

    std.log.debug("OpenGL ES memcpy host->device: {B}", .{size});
}

pub fn memcpyDeviceToHost(dst: *anyopaque, src: *anyopaque, size: usize) OpenGlesError!void {
    if (!opengles_initialized) {
        return OpenGlesError.InitializationFailed;
    }

    const src_buffer: *OpenGlesBuffer = @ptrCast(@alignCast(src));
    if (size > src_buffer.size) {
        std.log.err("OpenGL ES memcpy size ({B}) exceeds buffer size ({B})", .{ size, src_buffer.size });
        return OpenGlesError.BufferCreationFailed;
    }

    const bind_buffer_fn = glesBindBuffer orelse return OpenGlesError.BufferCreationFailed;
    bind_buffer_fn(GL_SHADER_STORAGE_BUFFER, src_buffer.buffer_id);

    // OpenGL ES doesn't have glGetBufferSubData - use glMapBufferRange instead
    const map_buffer_range_fn = glesMapBufferRange orelse {
        // Try glGetBufferSubData as fallback (may exist on some implementations)
        if (glesGetBufferSubData) |get_fn| {
            get_fn(GL_SHADER_STORAGE_BUFFER, 0, @intCast(size), dst);
            std.log.debug("OpenGL ES memcpy device->host (glGetBufferSubData): {B}", .{size});
            return;
        }
        return OpenGlesError.BufferCreationFailed;
    };
    const unmap_buffer_fn = glesUnmapBuffer orelse return OpenGlesError.BufferCreationFailed;

    const mapped_ptr = map_buffer_range_fn(GL_SHADER_STORAGE_BUFFER, 0, @intCast(size), GL_MAP_READ_BIT);
    if (mapped_ptr == null) {
        checkAndLogGlesError("glMapBufferRange");
        return OpenGlesError.BufferCreationFailed;
    }

    // Copy data from mapped buffer to destination
    const src_slice: [*]const u8 = @ptrCast(mapped_ptr.?);
    const dst_slice: [*]u8 = @ptrCast(dst);
    @memcpy(dst_slice[0..size], src_slice[0..size]);

    // Unmap buffer
    _ = unmap_buffer_fn(GL_SHADER_STORAGE_BUFFER);

    std.log.debug("OpenGL ES memcpy device->host (glMapBufferRange): {B}", .{size});
}

fn tryLoadOpenGles() bool {
    const lib_names = [_][]const u8{
        "libGLESv2.dll",
        "libEGL.dll",
        "libGLESv2.so.2",
        "libGLESv2.so",
        "/System/Library/Frameworks/OpenGLES.framework/OpenGLES",
    };
    for (lib_names) |name| {
        if (std.DynLib.open(name)) |lib| {
            opengles_lib = lib;
            return true;
        } else |_| {}
    }
    return false;
}

fn loadOpenGlesFunctions() bool {
    if (opengles_lib == null) return false;

    // Required functions
    glesGetString = opengles_lib.?.lookup(GlesGetStringFn, "glGetString") orelse return false;
    glesCreateShader = opengles_lib.?.lookup(GlesCreateShaderFn, "glCreateShader") orelse return false;
    glesShaderSource = opengles_lib.?.lookup(GlesShaderSourceFn, "glShaderSource") orelse return false;
    glesCompileShader = opengles_lib.?.lookup(GlesCompileShaderFn, "glCompileShader") orelse return false;
    glesGetShaderiv = opengles_lib.?.lookup(GlesGetShaderivFn, "glGetShaderiv") orelse return false;
    glesGetShaderInfoLog = opengles_lib.?.lookup(GlesGetShaderInfoLogFn, "glGetShaderInfoLog") orelse return false;
    glesCreateProgram = opengles_lib.?.lookup(GlesCreateProgramFn, "glCreateProgram") orelse return false;
    glesAttachShader = opengles_lib.?.lookup(GlesAttachShaderFn, "glAttachShader") orelse return false;
    glesLinkProgram = opengles_lib.?.lookup(GlesLinkProgramFn, "glLinkProgram") orelse return false;
    glesGetProgramiv = opengles_lib.?.lookup(GlesGetProgramivFn, "glGetProgramiv") orelse return false;
    glesGetProgramInfoLog = opengles_lib.?.lookup(GlesGetProgramInfoLogFn, "glGetProgramInfoLog") orelse return false;
    glesUseProgram = opengles_lib.?.lookup(GlesUseProgramFn, "glUseProgram") orelse return false;
    glesGenBuffers = opengles_lib.?.lookup(GlesGenBuffersFn, "glGenBuffers") orelse return false;
    glesBindBuffer = opengles_lib.?.lookup(GlesBindBufferFn, "glBindBuffer") orelse return false;
    glesBufferData = opengles_lib.?.lookup(GlesBufferDataFn, "glBufferData") orelse return false;
    glesBindBufferBase = opengles_lib.?.lookup(GlesBindBufferBaseFn, "glBindBufferBase") orelse return false;
    glesDispatchCompute = opengles_lib.?.lookup(GlesDispatchComputeFn, "glDispatchCompute") orelse return false;
    glesMemoryBarrier = opengles_lib.?.lookup(GlesMemoryBarrierFn, "glMemoryBarrier") orelse return false;
    glesDeleteBuffers = opengles_lib.?.lookup(GlesDeleteBuffersFn, "glDeleteBuffers") orelse return false;
    glesDeleteProgram = opengles_lib.?.lookup(GlesDeleteProgramFn, "glDeleteProgram") orelse return false;
    glesDeleteShader = opengles_lib.?.lookup(GlesDeleteShaderFn, "glDeleteShader") orelse return false;

    // Optional functions (some may not be available on all implementations)
    // glGetBufferSubData is NOT part of OpenGL ES spec but may exist as extension
    glesGetBufferSubData = opengles_lib.?.lookup(GlesGetBufferSubDataFn, "glGetBufferSubData");
    glesMapBufferRange = opengles_lib.?.lookup(GlesMapBufferRangeFn, "glMapBufferRange");
    glesUnmapBuffer = opengles_lib.?.lookup(GlesUnmapBufferFn, "glUnmapBuffer");
    glesFlushMappedBufferRange = opengles_lib.?.lookup(GlesFlushMappedBufferRangeFn, "glFlushMappedBufferRange");
    glesFinish = opengles_lib.?.lookup(GlesFinishFn, "glFinish");
    glesFlush = opengles_lib.?.lookup(GlesFlushFn, "glFlush");
    glesGetError = opengles_lib.?.lookup(GlesGetErrorFn, "glGetError");
    glesGetIntegerv = opengles_lib.?.lookup(GlesGetIntegervFn, "glGetIntegerv");
    glesCopyBufferSubData = opengles_lib.?.lookup(GlesCopyBufferSubDataFn, "glCopyBufferSubData");

    return true;
}

/// Parse OpenGL ES version string (e.g., "OpenGL ES 3.2 V@435.0")
fn parseVersionString(version: []const u8) ?struct { major: i32, minor: i32 } {
    if (version.len == 0) return null;

    // Skip "OpenGL ES " prefix if present
    var start: usize = 0;
    if (std.mem.startsWith(u8, version, "OpenGL ES ")) {
        start = 10;
    }

    // Find major version (first digit sequence after prefix)
    while (start < version.len and !std.ascii.isDigit(version[start])) {
        start += 1;
    }
    if (start >= version.len) return null;

    var major_end = start;
    while (major_end < version.len and std.ascii.isDigit(version[major_end])) {
        major_end += 1;
    }

    const major = std.fmt.parseInt(i32, version[start..major_end], 10) catch return null;

    // Find minor version (after the dot)
    if (major_end >= version.len or version[major_end] != '.') return null;
    const minor_start = major_end + 1;
    if (minor_start >= version.len) return null;

    var minor_end = minor_start;
    while (minor_end < version.len and std.ascii.isDigit(version[minor_end])) {
        minor_end += 1;
    }

    const minor = std.fmt.parseInt(i32, version[minor_start..minor_end], 10) catch return null;

    return .{ .major = major, .minor = minor };
}

/// Check for GL errors and log them. Returns true if an error occurred.
fn checkAndLogGlesError(operation: []const u8) bool {
    const get_error_fn = glesGetError orelse return false;
    const err = get_error_fn();

    if (err == GL_NO_ERROR) return false;

    const error_name: []const u8 = switch (err) {
        GL_INVALID_ENUM => "GL_INVALID_ENUM",
        GL_INVALID_VALUE => "GL_INVALID_VALUE",
        GL_INVALID_OPERATION => "GL_INVALID_OPERATION",
        GL_OUT_OF_MEMORY => "GL_OUT_OF_MEMORY",
        else => "Unknown error",
    };

    std.log.err("OpenGL ES error in {s}: {s} (0x{X:0>4})", .{ operation, error_name, err });
    return true;
}

/// Synchronize with the GPU. Blocks until all previous commands are complete.
pub fn synchronize() void {
    if (glesFinish) |finish_fn| {
        finish_fn();
    }
}

/// Flush pending commands to the GPU without waiting for completion.
pub fn flush() void {
    if (glesFlush) |flush_fn| {
        flush_fn();
    }
}

/// Set the allocator to use for buffer metadata allocations.
pub fn setBufferAllocator(allocator: std.mem.Allocator) void {
    buffer_allocator = allocator;
}

/// Get OpenGL ES version information.
pub fn getVersion() struct { major: i32, minor: i32 } {
    return .{ .major = gles_major_version, .minor = gles_minor_version };
}
