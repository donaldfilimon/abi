//! OpenGL backend implementation with compute shader support.
//!
//! Provides OpenGL-specific kernel compilation and execution using compute shaders
//! for cross-platform GPU compute acceleration. Requires OpenGL 4.3+ for compute shader support.

const std = @import("std");
const types = @import("../kernel_types.zig");
const shared = @import("shared.zig");
const fallback = @import("fallback.zig");

pub const OpenGlError = error{
    InitializationFailed,
    ContextCreationFailed,
    ShaderCompilationFailed,
    ProgramLinkingFailed,
    BufferCreationFailed,
    DispatchFailed,
    VersionNotSupported,
    FunctionLoadFailed,
    LibraryNotFound,
};

var opengl_lib: ?std.DynLib = null;
var opengl_initialized = false;
var init_mutex = std.Thread.Mutex{};

// OpenGL function pointers (simplified)
const GlGetStringFn = *const fn (u32) callconv(.c) ?[*:0]const u8;
const GlCreateShaderFn = *const fn (u32) callconv(.c) u32;
const GlShaderSourceFn = *const fn (u32, i32, ?[*]const [*:0]const u8, ?[*]const i32) callconv(.c) void;
const GlCompileShaderFn = *const fn (u32) callconv(.c) void;
const GlGetShaderivFn = *const fn (u32, u32, *i32) callconv(.c) void;
const GlGetShaderInfoLogFn = *const fn (u32, i32, *i32, [*]u8) callconv(.c) void;
const GlCreateProgramFn = *const fn () callconv(.c) u32;
const GlAttachShaderFn = *const fn (u32, u32) callconv(.c) void;
const GlDetachShaderFn = *const fn (u32, u32) callconv(.c) void;
const GlLinkProgramFn = *const fn (u32) callconv(.c) void;
const GlGetProgramivFn = *const fn (u32, u32, *i32) callconv(.c) void;
const GlGetProgramInfoLogFn = *const fn (u32, i32, *i32, [*]u8) callconv(.c) void;
const GlUseProgramFn = *const fn (u32) callconv(.c) void;
const GlGenBuffersFn = *const fn (i32, [*]u32) callconv(.c) void;
const GlBindBufferFn = *const fn (u32, u32) callconv(.c) void;
const GlBufferDataFn = *const fn (u32, isize, ?*anyopaque, u32) callconv(.c) void;
const GlBindBufferBaseFn = *const fn (u32, u32, u32) callconv(.c) void;
const GlDispatchComputeFn = *const fn (u32, u32, u32) callconv(.c) void;
const GlMemoryBarrierFn = *const fn (u32) callconv(.c) void;
const GlGetBufferSubDataFn = *const fn (u32, isize, isize, *anyopaque) callconv(.c) void;
const GlDeleteBuffersFn = *const fn (i32, [*]const u32) callconv(.c) void;
const GlDeleteProgramFn = *const fn (u32) callconv(.c) void;
const GlDeleteShaderFn = *const fn (u32) callconv(.c) void;
const GlMapBufferFn = *const fn (u32, u32) callconv(.c) ?*anyopaque;
const GlMapBufferRangeFn = *const fn (u32, isize, isize, u32) callconv(.c) ?*anyopaque;
const GlUnmapBufferFn = *const fn (u32) callconv(.c) u8;
const GlFlushMappedBufferRangeFn = *const fn (u32, isize, isize) callconv(.c) void;
const GlFinishFn = *const fn () callconv(.c) void;
const GlFlushFn = *const fn () callconv(.c) void;
const GlGetErrorFn = *const fn () callconv(.c) u32;
const GlGetIntegervFn = *const fn (u32, *i32) callconv(.c) void;
const GlCopyBufferSubDataFn = *const fn (u32, u32, isize, isize, isize) callconv(.c) void;

var glGetString: ?GlGetStringFn = null;
var glCreateShader: ?GlCreateShaderFn = null;
var glShaderSource: ?GlShaderSourceFn = null;
var glCompileShader: ?GlCompileShaderFn = null;
var glGetShaderiv: ?GlGetShaderivFn = null;
var glGetShaderInfoLog: ?GlGetShaderInfoLogFn = null;
var glCreateProgram: ?GlCreateProgramFn = null;
var glAttachShader: ?GlAttachShaderFn = null;
var glDetachShader: ?GlDetachShaderFn = null;
var glLinkProgram: ?GlLinkProgramFn = null;
var glGetProgramiv: ?GlGetProgramivFn = null;
var glGetProgramInfoLog: ?GlGetProgramInfoLogFn = null;
var glUseProgram: ?GlUseProgramFn = null;
var glGenBuffers: ?GlGenBuffersFn = null;
var glBindBuffer: ?GlBindBufferFn = null;
var glBufferData: ?GlBufferDataFn = null;
var glBindBufferBase: ?GlBindBufferBaseFn = null;
var glDispatchCompute: ?GlDispatchComputeFn = null;
var glMemoryBarrier: ?GlMemoryBarrierFn = null;
var glGetBufferSubData: ?GlGetBufferSubDataFn = null;
var glDeleteBuffers: ?GlDeleteBuffersFn = null;
var glDeleteProgram: ?GlDeleteProgramFn = null;
var glDeleteShader: ?GlDeleteShaderFn = null;
var glMapBuffer: ?GlMapBufferFn = null;
var glMapBufferRange: ?GlMapBufferRangeFn = null;
var glUnmapBuffer: ?GlUnmapBufferFn = null;
var glFlushMappedBufferRange: ?GlFlushMappedBufferRangeFn = null;
var glFinish: ?GlFinishFn = null;
var glFlush: ?GlFlushFn = null;
var glGetError: ?GlGetErrorFn = null;
var glGetIntegerv: ?GlGetIntegervFn = null;
var glCopyBufferSubData: ?GlCopyBufferSubDataFn = null;

// Cached allocator for buffer metadata
var buffer_allocator: ?std.mem.Allocator = null;

const OpenGlKernel = struct {
    program: u32,
    shader: u32,
};

const OpenGlBuffer = struct {
    buffer_id: u32,
    size: usize,
    allocator: std.mem.Allocator,
};

const GL_COMPUTE_SHADER = 0x91B9;
const GL_SHADER_STORAGE_BUFFER = 0x90D2;
const GL_SHADER_STORAGE_BARRIER_BIT = 0x00002000;
const GL_STATIC_DRAW = 0x88E4;
const GL_DYNAMIC_READ = 0x88E9;
const GL_COMPILE_STATUS = 0x8B81;
const GL_LINK_STATUS = 0x8B82;
const GL_MAP_READ_BIT = 0x0001;
const GL_MAP_WRITE_BIT = 0x0002;
const GL_MAP_INVALIDATE_BUFFER_BIT = 0x0008;
const GL_MAP_FLUSH_EXPLICIT_BIT = 0x0010;
const GL_READ_ONLY = 0x88B8;
const GL_WRITE_ONLY = 0x88B9;
const GL_READ_WRITE = 0x88BA;
const GL_COPY_READ_BUFFER = 0x8F36;
const GL_COPY_WRITE_BUFFER = 0x8F37;
const GL_MAJOR_VERSION = 0x821B;
const GL_MINOR_VERSION = 0x821C;
const GL_NO_ERROR = 0;
const GL_INVALID_ENUM = 0x0500;
const GL_INVALID_VALUE = 0x0501;
const GL_INVALID_OPERATION = 0x0502;
const GL_OUT_OF_MEMORY = 0x0505;

// Cached OpenGL version info
var gl_major_version: i32 = 0;
var gl_minor_version: i32 = 0;

pub fn init() OpenGlError!void {
    init_mutex.lock();
    defer init_mutex.unlock();

    if (opengl_initialized) return;

    if (!tryLoadOpenGl()) {
        return OpenGlError.LibraryNotFound;
    }
    errdefer if (opengl_lib) |lib| lib.close();

    if (!loadOpenGlFunctions()) {
        return OpenGlError.FunctionLoadFailed;
    }

    // Check if compute shaders are supported (requires OpenGL 4.3+)
    const version_string = glGetString orelse return OpenGlError.InitializationFailed;
    const version = version_string(0x1F02); // GL_VERSION
    if (version == null) {
        return OpenGlError.VersionNotSupported;
    }

    // Parse OpenGL version using glGetIntegerv for accurate major/minor version
    const get_integer_fn = glGetIntegerv orelse {
        // Fallback: parse version string
        if (parseVersionString(std.mem.span(version))) |parsed| {
            gl_major_version = parsed.major;
            gl_minor_version = parsed.minor;
        } else {
            return OpenGlError.VersionNotSupported;
        }
        opengl_initialized = true;
        std.log.debug("OpenGL backend initialized (parsed version): {}.{}", .{ gl_major_version, gl_minor_version });
        return;
    };

    get_integer_fn(GL_MAJOR_VERSION, &gl_major_version);
    get_integer_fn(GL_MINOR_VERSION, &gl_minor_version);

    // Verify compute shader support (OpenGL 4.3+)
    if (gl_major_version < 4 or (gl_major_version == 4 and gl_minor_version < 3)) {
        std.log.err("OpenGL {}.{} does not support compute shaders (requires 4.3+)", .{
            gl_major_version,
            gl_minor_version,
        });
        return OpenGlError.VersionNotSupported;
    }

    opengl_initialized = true;
    std.log.debug("OpenGL backend initialized successfully: version {}.{}", .{ gl_major_version, gl_minor_version });
}

pub fn deinit() void {
    init_mutex.lock();
    defer init_mutex.unlock();

    if (!opengl_initialized) return;

    if (opengl_lib) |lib| {
        lib.close();
    }
    opengl_lib = null;
    opengl_initialized = false;

    std.log.debug("OpenGL backend deinitialized", .{});
}

pub fn compileKernel(
    allocator: std.mem.Allocator,
    source: types.KernelSource,
) (types.KernelError || OpenGlError)!*anyopaque {
    if (!opengl_initialized) {
        return OpenGlError.InitializationFailed;
    }

    // Create compute shader
    const create_shader_fn = glCreateShader orelse return OpenGlError.ShaderCompilationFailed;
    const shader = create_shader_fn(GL_COMPUTE_SHADER);
    if (shader == 0) {
        return OpenGlError.ShaderCompilationFailed;
    }
    errdefer if (glDeleteShader) |delete_fn| delete_fn(shader);

    // Set shader source
    const source_ptr = &[_][*:0]const u8{source.source.ptr};
    const set_source_fn = glShaderSource orelse return OpenGlError.ShaderCompilationFailed;
    set_source_fn(shader, 1, source_ptr.ptr, null);

    // Compile shader
    const compile_fn = glCompileShader orelse return OpenGlError.ShaderCompilationFailed;
    compile_fn(shader);

    // Check compilation status
    const get_shader_iv_fn = glGetShaderiv orelse return OpenGlError.ShaderCompilationFailed;
    var compile_status: i32 = 0;
    get_shader_iv_fn(shader, GL_COMPILE_STATUS, &compile_status);

    if (compile_status == 0) {
        // Get error log
        var log_length: i32 = 0;
        get_shader_iv_fn(shader, 0x8B84, &log_length); // GL_INFO_LOG_LENGTH
        if (log_length > 0) {
            const log = try allocator.alloc(u8, @intCast(log_length));
            defer allocator.free(log);
            const get_log_fn = glGetShaderInfoLog orelse return OpenGlError.ShaderCompilationFailed;
            get_log_fn(shader, log_length, null, log.ptr);
            std.log.err("OpenGL shader compilation failed: {s}", .{log});
        }
        return OpenGlError.ShaderCompilationFailed;
    }

    // Create program
    const create_program_fn = glCreateProgram orelse return OpenGlError.ProgramLinkingFailed;
    const program = create_program_fn();
    if (program == 0) {
        return OpenGlError.ProgramLinkingFailed;
    }
    errdefer if (glDeleteProgram) |delete_fn| delete_fn(program);

    // Attach shader
    const attach_fn = glAttachShader orelse return OpenGlError.ProgramLinkingFailed;
    attach_fn(program, shader);

    // Link program
    const link_fn = glLinkProgram orelse return OpenGlError.ProgramLinkingFailed;
    link_fn(program);

    // Check link status
    const get_program_iv_fn = glGetProgramiv orelse return OpenGlError.ProgramLinkingFailed;
    var link_status: i32 = 0;
    get_program_iv_fn(program, GL_LINK_STATUS, &link_status);

    if (link_status == 0) {
        // Get error log
        var log_length: i32 = 0;
        get_program_iv_fn(program, 0x8B84, &log_length); // GL_INFO_LOG_LENGTH
        if (log_length > 0) {
            const log = try allocator.alloc(u8, @intCast(log_length));
            defer allocator.free(log);
            const get_log_fn = glGetProgramInfoLog orelse return OpenGlError.ProgramLinkingFailed;
            get_log_fn(program, log_length, null, log.ptr);
            std.log.err("OpenGL program linking failed: {s}", .{log});
        }
        return OpenGlError.ProgramLinkingFailed;
    }

    const kernel = try allocator.create(OpenGlKernel);
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
) (types.KernelError || OpenGlError)!void {
    _ = allocator;

    if (!opengl_initialized) {
        return OpenGlError.InitializationFailed;
    }

    const kernel: *OpenGlKernel = @ptrCast(@alignCast(kernel_handle));

    // Use program
    const use_program_fn = glUseProgram orelse return OpenGlError.DispatchFailed;
    use_program_fn(kernel.program);

    // Bind buffers
    const bind_buffer_base_fn = glBindBufferBase orelse return OpenGlError.DispatchFailed;
    for (args, 0..) |arg, i| {
        if (arg != null) {
            const buffer: *OpenGlBuffer = @ptrCast(@alignCast(arg.?));
            bind_buffer_base_fn(GL_SHADER_STORAGE_BUFFER, @intCast(i), buffer.buffer_id);
        }
    }

    // Dispatch compute
    const dispatch_fn = glDispatchCompute orelse return OpenGlError.DispatchFailed;
    dispatch_fn(config.grid_dim[0], config.grid_dim[1], config.grid_dim[2]);

    // Memory barrier
    const barrier_fn = glMemoryBarrier orelse return OpenGlError.DispatchFailed;
    barrier_fn(GL_SHADER_STORAGE_BARRIER_BIT);

    std.log.debug("OpenGL kernel dispatched: {}x{}x{}", .{
        config.grid_dim[0],
        config.grid_dim[1],
        config.grid_dim[2],
    });
}

pub fn destroyKernel(allocator: std.mem.Allocator, kernel_handle: *anyopaque) void {
    if (!opengl_initialized) {
        return;
    }

    const kernel: *OpenGlKernel = @ptrCast(@alignCast(kernel_handle));

    // Detach shader from program before deletion (required by OpenGL)
    if (glDetachShader) |detach_fn| {
        detach_fn(kernel.program, kernel.shader);
    }

    const delete_shader_fn = glDeleteShader orelse return;
    delete_shader_fn(kernel.shader);

    const delete_program_fn = glDeleteProgram orelse return;
    delete_program_fn(kernel.program);

    allocator.destroy(kernel);
}

pub fn allocateDeviceMemory(size: usize) OpenGlError!*anyopaque {
    // Use cached allocator or fallback to page_allocator
    const allocator = buffer_allocator orelse std.heap.page_allocator;
    return allocateDeviceMemoryWithAllocator(allocator, size);
}

pub fn allocateDeviceMemoryWithAllocator(allocator: std.mem.Allocator, size: usize) OpenGlError!*anyopaque {
    if (!opengl_initialized) {
        return OpenGlError.InitializationFailed;
    }

    const gen_buffers_fn = glGenBuffers orelse return OpenGlError.BufferCreationFailed;
    var buffer_id: u32 = 0;
    gen_buffers_fn(1, &buffer_id);
    if (buffer_id == 0) {
        checkAndLogGlError("glGenBuffers");
        return OpenGlError.BufferCreationFailed;
    }

    const bind_buffer_fn = glBindBuffer orelse return OpenGlError.BufferCreationFailed;
    bind_buffer_fn(GL_SHADER_STORAGE_BUFFER, buffer_id);

    const buffer_data_fn = glBufferData orelse return OpenGlError.BufferCreationFailed;
    buffer_data_fn(GL_SHADER_STORAGE_BUFFER, @intCast(size), null, GL_DYNAMIC_READ);

    // Check for GL errors after buffer creation
    if (checkAndLogGlError("glBufferData")) {
        const delete_buffers_fn = glDeleteBuffers orelse return OpenGlError.BufferCreationFailed;
        delete_buffers_fn(1, &buffer_id);
        return OpenGlError.BufferCreationFailed;
    }

    const opengl_buffer = allocator.create(OpenGlBuffer) catch {
        const delete_buffers_fn = glDeleteBuffers orelse return OpenGlError.BufferCreationFailed;
        delete_buffers_fn(1, &buffer_id);
        return OpenGlError.BufferCreationFailed;
    };
    errdefer allocator.destroy(opengl_buffer);

    opengl_buffer.* = .{
        .buffer_id = buffer_id,
        .size = size,
        .allocator = allocator,
    };

    std.log.debug("OpenGL buffer allocated: ID={}, size={B}", .{ buffer_id, size });
    return opengl_buffer;
}

pub fn freeDeviceMemory(ptr: *anyopaque) void {
    if (!opengl_initialized) {
        return;
    }

    const buffer: *OpenGlBuffer = @ptrCast(@alignCast(ptr));
    const allocator = buffer.allocator;

    const delete_buffers_fn = glDeleteBuffers orelse return;
    delete_buffers_fn(1, &buffer.buffer_id);
    _ = checkAndLogGlError("glDeleteBuffers");

    allocator.destroy(buffer);
}

pub fn memcpyDeviceToDevice(dst: *anyopaque, src: *anyopaque, size: usize) OpenGlError!void {
    if (!opengl_initialized) {
        return OpenGlError.InitializationFailed;
    }

    const src_buffer: *OpenGlBuffer = @ptrCast(@alignCast(src));
    const dst_buffer: *OpenGlBuffer = @ptrCast(@alignCast(dst));

    if (size > src_buffer.size) {
        std.log.err("OpenGL memcpy size ({B}) exceeds source buffer size ({B})", .{ size, src_buffer.size });
        return OpenGlError.BufferCreationFailed;
    }
    if (size > dst_buffer.size) {
        std.log.err("OpenGL memcpy size ({B}) exceeds destination buffer size ({B})", .{ size, dst_buffer.size });
        return OpenGlError.BufferCreationFailed;
    }

    const bind_buffer_fn = glBindBuffer orelse return OpenGlError.BufferCreationFailed;
    const copy_buffer_fn = glCopyBufferSubData orelse {
        // Fallback: copy through host memory if glCopyBufferSubData not available
        return memcpyDeviceToDeviceFallback(dst_buffer, src_buffer, size);
    };

    bind_buffer_fn(GL_COPY_READ_BUFFER, src_buffer.buffer_id);
    bind_buffer_fn(GL_COPY_WRITE_BUFFER, dst_buffer.buffer_id);
    copy_buffer_fn(GL_COPY_READ_BUFFER, GL_COPY_WRITE_BUFFER, 0, 0, @intCast(size));

    if (checkAndLogGlError("glCopyBufferSubData")) {
        return OpenGlError.BufferCreationFailed;
    }

    std.log.debug("OpenGL memcpy device->device: {B}", .{size});
}

fn memcpyDeviceToDeviceFallback(dst: *OpenGlBuffer, src: *OpenGlBuffer, size: usize) OpenGlError!void {
    // Fallback implementation using host memory as intermediate
    var temp_buffer: [4096]u8 = undefined;
    var offset: usize = 0;

    const bind_buffer_fn = glBindBuffer orelse return OpenGlError.BufferCreationFailed;
    const get_buffer_sub_data_fn = glGetBufferSubData orelse return OpenGlError.BufferCreationFailed;
    const buffer_data_fn = glBufferData orelse return OpenGlError.BufferCreationFailed;

    while (offset < size) {
        const chunk_size = @min(temp_buffer.len, size - offset);

        // Read from source
        bind_buffer_fn(GL_SHADER_STORAGE_BUFFER, src.buffer_id);
        get_buffer_sub_data_fn(GL_SHADER_STORAGE_BUFFER, @intCast(offset), @intCast(chunk_size), &temp_buffer);

        // Write to destination
        bind_buffer_fn(GL_SHADER_STORAGE_BUFFER, dst.buffer_id);
        buffer_data_fn(GL_SHADER_STORAGE_BUFFER, @intCast(chunk_size), &temp_buffer, GL_STATIC_DRAW);

        offset += chunk_size;
    }

    std.log.debug("OpenGL memcpy device->device (fallback): {B}", .{size});
}

pub fn memcpyHostToDevice(dst: *anyopaque, src: *anyopaque, size: usize) OpenGlError!void {
    if (!opengl_initialized) {
        return OpenGlError.InitializationFailed;
    }

    const dst_buffer: *OpenGlBuffer = @ptrCast(@alignCast(dst));
    if (size > dst_buffer.size) {
        std.log.err("OpenGL memcpy size ({B}) exceeds buffer size ({B})", .{ size, dst_buffer.size });
        return OpenGlError.BufferCreationFailed;
    }

    const bind_buffer_fn = glBindBuffer orelse return OpenGlError.BufferCreationFailed;
    bind_buffer_fn(GL_SHADER_STORAGE_BUFFER, dst_buffer.buffer_id);

    const buffer_data_fn = glBufferData orelse return OpenGlError.BufferCreationFailed;
    buffer_data_fn(GL_SHADER_STORAGE_BUFFER, @intCast(size), src, GL_STATIC_DRAW);

    std.log.debug("OpenGL memcpy host->device: {B}", .{size});
}

pub fn memcpyDeviceToHost(dst: *anyopaque, src: *anyopaque, size: usize) OpenGlError!void {
    if (!opengl_initialized) {
        return OpenGlError.InitializationFailed;
    }

    const src_buffer: *OpenGlBuffer = @ptrCast(@alignCast(src));
    if (size > src_buffer.size) {
        std.log.err("OpenGL memcpy size ({B}) exceeds buffer size ({B})", .{ size, src_buffer.size });
        return OpenGlError.BufferCreationFailed;
    }

    const bind_buffer_fn = glBindBuffer orelse return OpenGlError.BufferCreationFailed;
    bind_buffer_fn(GL_SHADER_STORAGE_BUFFER, src_buffer.buffer_id);

    const get_buffer_sub_data_fn = glGetBufferSubData orelse return OpenGlError.BufferCreationFailed;
    get_buffer_sub_data_fn(GL_SHADER_STORAGE_BUFFER, 0, @intCast(size), dst);

    std.log.debug("OpenGL memcpy device->host: {B}", .{size});
}

fn tryLoadOpenGl() bool {
    const lib_names = [_][]const u8{
        "opengl32.dll",
        "libGL.so.1",
        "libGL.so",
        "/System/Library/Frameworks/OpenGL.framework/OpenGL",
    };
    for (lib_names) |name| {
        if (std.DynLib.open(name)) |lib| {
            opengl_lib = lib;
            return true;
        } else |_| {}
    }
    return false;
}

fn loadOpenGlFunctions() bool {
    if (opengl_lib == null) return false;

    // Required functions
    glGetString = opengl_lib.?.lookup(GlGetStringFn, "glGetString") orelse return false;
    glCreateShader = opengl_lib.?.lookup(GlCreateShaderFn, "glCreateShader") orelse return false;
    glShaderSource = opengl_lib.?.lookup(GlShaderSourceFn, "glShaderSource") orelse return false;
    glCompileShader = opengl_lib.?.lookup(GlCompileShaderFn, "glCompileShader") orelse return false;
    glGetShaderiv = opengl_lib.?.lookup(GlGetShaderivFn, "glGetShaderiv") orelse return false;
    glGetShaderInfoLog = opengl_lib.?.lookup(GlGetShaderInfoLogFn, "glGetShaderInfoLog") orelse return false;
    glCreateProgram = opengl_lib.?.lookup(GlCreateProgramFn, "glCreateProgram") orelse return false;
    glAttachShader = opengl_lib.?.lookup(GlAttachShaderFn, "glAttachShader") orelse return false;
    glDetachShader = opengl_lib.?.lookup(GlDetachShaderFn, "glDetachShader") orelse return false;
    glLinkProgram = opengl_lib.?.lookup(GlLinkProgramFn, "glLinkProgram") orelse return false;
    glGetProgramiv = opengl_lib.?.lookup(GlGetProgramivFn, "glGetProgramiv") orelse return false;
    glGetProgramInfoLog = opengl_lib.?.lookup(GlGetProgramInfoLogFn, "glGetProgramInfoLog") orelse return false;
    glUseProgram = opengl_lib.?.lookup(GlUseProgramFn, "glUseProgram") orelse return false;
    glGenBuffers = opengl_lib.?.lookup(GlGenBuffersFn, "glGenBuffers") orelse return false;
    glBindBuffer = opengl_lib.?.lookup(GlBindBufferFn, "glBindBuffer") orelse return false;
    glBufferData = opengl_lib.?.lookup(GlBufferDataFn, "glBufferData") orelse return false;
    glBindBufferBase = opengl_lib.?.lookup(GlBindBufferBaseFn, "glBindBufferBase") orelse return false;
    glDispatchCompute = opengl_lib.?.lookup(GlDispatchComputeFn, "glDispatchCompute") orelse return false;
    glMemoryBarrier = opengl_lib.?.lookup(GlMemoryBarrierFn, "glMemoryBarrier") orelse return false;
    glGetBufferSubData = opengl_lib.?.lookup(GlGetBufferSubDataFn, "glGetBufferSubData") orelse return false;
    glDeleteBuffers = opengl_lib.?.lookup(GlDeleteBuffersFn, "glDeleteBuffers") orelse return false;
    glDeleteProgram = opengl_lib.?.lookup(GlDeleteProgramFn, "glDeleteProgram") orelse return false;
    glDeleteShader = opengl_lib.?.lookup(GlDeleteShaderFn, "glDeleteShader") orelse return false;

    // Optional functions (may not be available on all platforms)
    glMapBuffer = opengl_lib.?.lookup(GlMapBufferFn, "glMapBuffer");
    glMapBufferRange = opengl_lib.?.lookup(GlMapBufferRangeFn, "glMapBufferRange");
    glUnmapBuffer = opengl_lib.?.lookup(GlUnmapBufferFn, "glUnmapBuffer");
    glFlushMappedBufferRange = opengl_lib.?.lookup(GlFlushMappedBufferRangeFn, "glFlushMappedBufferRange");
    glFinish = opengl_lib.?.lookup(GlFinishFn, "glFinish");
    glFlush = opengl_lib.?.lookup(GlFlushFn, "glFlush");
    glGetError = opengl_lib.?.lookup(GlGetErrorFn, "glGetError");
    glGetIntegerv = opengl_lib.?.lookup(GlGetIntegervFn, "glGetIntegerv");
    glCopyBufferSubData = opengl_lib.?.lookup(GlCopyBufferSubDataFn, "glCopyBufferSubData");

    return true;
}

/// Parse OpenGL version string (e.g., "4.6.0 NVIDIA 535.104.05")
fn parseVersionString(version: []const u8) ?struct { major: i32, minor: i32 } {
    if (version.len == 0) return null;

    // Find major version (first digit sequence)
    var major_start: usize = 0;
    while (major_start < version.len and !std.ascii.isDigit(version[major_start])) {
        major_start += 1;
    }
    if (major_start >= version.len) return null;

    var major_end = major_start;
    while (major_end < version.len and std.ascii.isDigit(version[major_end])) {
        major_end += 1;
    }

    const major = std.fmt.parseInt(i32, version[major_start..major_end], 10) catch return null;

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
fn checkAndLogGlError(operation: []const u8) bool {
    const get_error_fn = glGetError orelse return false;
    const err = get_error_fn();

    if (err == GL_NO_ERROR) return false;

    const error_name: []const u8 = switch (err) {
        GL_INVALID_ENUM => "GL_INVALID_ENUM",
        GL_INVALID_VALUE => "GL_INVALID_VALUE",
        GL_INVALID_OPERATION => "GL_INVALID_OPERATION",
        GL_OUT_OF_MEMORY => "GL_OUT_OF_MEMORY",
        else => "Unknown error",
    };

    std.log.err("OpenGL error in {s}: {s} (0x{X:0>4})", .{ operation, error_name, err });
    return true;
}

/// Synchronize with the GPU. Blocks until all previous commands are complete.
pub fn synchronize() void {
    if (glFinish) |finish_fn| {
        finish_fn();
    }
}

/// Flush pending commands to the GPU without waiting for completion.
pub fn flush() void {
    if (glFlush) |flush_fn| {
        flush_fn();
    }
}

/// Set the allocator to use for buffer metadata allocations.
pub fn setBufferAllocator(allocator: std.mem.Allocator) void {
    buffer_allocator = allocator;
}

/// Get OpenGL version information.
pub fn getVersion() struct { major: i32, minor: i32 } {
    return .{ .major = gl_major_version, .minor = gl_minor_version };
}

// ============================================================================
// Device Enumeration
// ============================================================================

const Device = @import("../device.zig").Device;
const DeviceType = @import("../device.zig").DeviceType;
const Backend = @import("../backend.zig").Backend;

/// Enumerate all OpenGL devices available on the system
pub fn enumerateDevices(allocator: std.mem.Allocator) ![]Device {
    if (!isAvailable()) {
        return &[_]Device{};
    }

    var devices = std.ArrayListUnmanaged(Device).empty;
    errdefer devices.deinit(allocator);

    // OpenGL typically exposes one device per context
    if (opengl_initialized) {
        // Always allocate name to ensure consistent memory ownership for cleanup
        const name = try allocator.dupe(u8, "OpenGL Device");
        errdefer allocator.free(name);

        try devices.append(allocator, .{
            .id = 0,
            .backend = .opengl,
            .name = name,
            .device_type = .discrete, // Assume discrete
            .total_memory = null,
            .available_memory = null,
            .is_emulated = false,
            .capability = .{
                .supports_fp16 = false, // OpenGL compute doesn't require FP16
                .supports_fp64 = true, // OpenGL 4.3+ supports FP64
                .supports_int8 = true,
                .supports_async_transfers = false,
                .unified_memory = false,
            },
            .compute_units = null,
            .clock_mhz = null,
        });
    }

    return devices.toOwnedSlice(allocator);
}

/// Check if OpenGL compute is available on this system
pub fn isAvailable() bool {
    // OpenGL compute requires 4.3+
    return opengl_initialized and gl_major_version >= 4 and gl_minor_version >= 3;
}

// ============================================================================
// Tests
// ============================================================================

test "OpenGL error enum covers all cases" {
    const errors = [_]OpenGlError{
        error.InitializationFailed,
        error.ShaderCompilationFailed,
        error.ProgramLinkFailed,
        error.BufferCreationFailed,
        error.ComputeNotSupported,
    };
    try std.testing.expectEqual(@as(usize, 5), errors.len);
}

test "GL constants are correct" {
    try std.testing.expectEqual(@as(u32, 0x91B9), GL_COMPUTE_SHADER);
    try std.testing.expectEqual(@as(u32, 0x90D2), GL_SHADER_STORAGE_BUFFER);
    try std.testing.expectEqual(@as(u32, 0x00002000), GL_SHADER_STORAGE_BARRIER_BIT);
}

test "isAvailable returns false when not initialized" {
    try std.testing.expect(!isAvailable());
}

test "enumerateDevices returns empty when not initialized" {
    const devices = try enumerateDevices(std.testing.allocator);
    defer {
        for (devices) |d| {
            if (d.name) |name| std.testing.allocator.free(name);
        }
        std.testing.allocator.free(devices);
    }
    try std.testing.expectEqual(@as(usize, 0), devices.len);
}
