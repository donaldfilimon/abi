//! OpenGL backend implementation with compute shader support.
//!
//! Provides OpenGL-specific kernel compilation and execution using compute shaders
//! for cross-platform GPU compute acceleration.

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
};

var opengl_lib: ?std.DynLib = null;
var opengl_initialized = false;

// OpenGL function pointers (simplified)
const GlGetStringFn = *const fn (u32) callconv(.c) ?[*:0]const u8;
const GlCreateShaderFn = *const fn (u32) callconv(.c) u32;
const GlShaderSourceFn = *const fn (u32, i32, ?[*]const [*:0]const u8, ?[*]const i32) callconv(.c) void;
const GlCompileShaderFn = *const fn (u32) callconv(.c) void;
const GlGetShaderivFn = *const fn (u32, u32, *i32) callconv(.c) void;
const GlGetShaderInfoLogFn = *const fn (u32, i32, *i32, [*]u8) callconv(.c) void;
const GlCreateProgramFn = *const fn () callconv(.c) u32;
const GlAttachShaderFn = *const fn (u32, u32) callconv(.c) void;
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

var glGetString: ?GlGetStringFn = null;
var glCreateShader: ?GlCreateShaderFn = null;
var glShaderSource: ?GlShaderSourceFn = null;
var glCompileShader: ?GlCompileShaderFn = null;
var glGetShaderiv: ?GlGetShaderivFn = null;
var glGetShaderInfoLog: ?GlGetShaderInfoLogFn = null;
var glCreateProgram: ?GlCreateProgramFn = null;
var glAttachShader: ?GlAttachShaderFn = null;
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

const OpenGlKernel = struct {
    program: u32,
    shader: u32,
};

const OpenGlBuffer = struct {
    buffer_id: u32,
    size: usize,
};

const GL_COMPUTE_SHADER = 0x91B9;
const GL_SHADER_STORAGE_BUFFER = 0x90D2;
const GL_SHADER_STORAGE_BARRIER_BIT = 0x00002000;
const GL_STATIC_DRAW = 0x88E4;
const GL_DYNAMIC_READ = 0x88E9;
const GL_COMPILE_STATUS = 0x8B81;
const GL_LINK_STATUS = 0x8B82;

pub fn init() !void {
    if (opengl_initialized) return;

    if (!tryLoadOpenGl()) {
        return OpenGlError.InitializationFailed;
    }

    if (!loadOpenGlFunctions()) {
        return OpenGlError.InitializationFailed;
    }

    // Check if compute shaders are supported
    const version_string = glGetString orelse return OpenGlError.InitializationFailed;
    const version = version_string(0x1F02); // GL_VERSION
    if (version == null) {
        return OpenGlError.InitializationFailed;
    }

    // Parse version to check if >= 4.3
    // Simplified check - in practice, would parse the version string
    opengl_initialized = true;
}

pub fn deinit() void {
    if (opengl_lib) |lib| {
        lib.close();
    }
    opengl_lib = null;
    opengl_initialized = false;
}

pub fn compileKernel(
    allocator: std.mem.Allocator,
    source: types.KernelSource,
) types.KernelError!*anyopaque {
    if (!opengl_initialized) {
        return types.KernelError.CompilationFailed;
    }

    // Create compute shader
    const create_shader_fn = glCreateShader orelse return types.KernelError.CompilationFailed;
    const shader = create_shader_fn(GL_COMPUTE_SHADER);

    // Set shader source
    const source_ptr = &[_][*:0]const u8{source.source.ptr};
    const set_source_fn = glShaderSource orelse return types.KernelError.CompilationFailed;
    set_source_fn(shader, 1, source_ptr.ptr, null);

    // Compile shader
    const compile_fn = glCompileShader orelse return types.KernelError.CompilationFailed;
    compile_fn(shader);

    // Check compilation status
    const get_shader_iv_fn = glGetShaderiv orelse return types.KernelError.CompilationFailed;
    var compile_status: i32 = 0;
    get_shader_iv_fn(shader, GL_COMPILE_STATUS, &compile_status);

    if (compile_status == 0) {
        // Get error log
        var log_length: i32 = 0;
        get_shader_iv_fn(shader, 0x8B84, &log_length); // GL_INFO_LOG_LENGTH
        if (log_length > 0) {
            var log = try allocator.alloc(u8, @intCast(log_length));
            defer allocator.free(log);
            const get_log_fn = glGetShaderInfoLog orelse return types.KernelError.CompilationFailed;
            get_log_fn(shader, log_length, null, log.ptr);
            std.log.err("OpenGL Shader Compilation Failed: {s}", .{log});
        }
        return types.KernelError.CompilationFailed;
    }

    // Create program
    const create_program_fn = glCreateProgram orelse return types.KernelError.CompilationFailed;
    const program = create_program_fn();

    // Attach shader
    const attach_fn = glAttachShader orelse return types.KernelError.CompilationFailed;
    attach_fn(program, shader);

    // Link program
    const link_fn = glLinkProgram orelse return types.KernelError.CompilationFailed;
    link_fn(program);

    // Check link status
    const get_program_iv_fn = glGetProgramiv orelse return types.KernelError.CompilationFailed;
    var link_status: i32 = 0;
    get_program_iv_fn(program, GL_LINK_STATUS, &link_status);

    if (link_status == 0) {
        // Get error log
        var log_length: i32 = 0;
        get_program_iv_fn(program, 0x8B84, &log_length); // GL_INFO_LOG_LENGTH
        if (log_length > 0) {
            var log = try allocator.alloc(u8, @intCast(log_length));
            defer allocator.free(log);
            const get_log_fn = glGetProgramInfoLog orelse return types.KernelError.CompilationFailed;
            get_log_fn(program, log_length, null, log.ptr);
            std.log.err("OpenGL Program Linking Failed: {s}", .{log});
        }
        return types.KernelError.CompilationFailed;
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
) types.KernelError!void {
    _ = allocator;

    if (!opengl_initialized) {
        return types.KernelError.LaunchFailed;
    }

    const kernel: *OpenGlKernel = @ptrCast(@alignCast(kernel_handle));

    // Use program
    const use_program_fn = glUseProgram orelse return types.KernelError.LaunchFailed;
    use_program_fn(kernel.program);

    // Bind buffers
    const bind_buffer_base_fn = glBindBufferBase orelse return types.KernelError.LaunchFailed;
    for (args, 0..) |arg, i| {
        if (arg != null) {
            const buffer: *OpenGlBuffer = @ptrCast(@alignCast(arg.?));
            bind_buffer_base_fn(GL_SHADER_STORAGE_BUFFER, @intCast(i), buffer.buffer_id);
        }
    }

    // Dispatch compute
    const dispatch_fn = glDispatchCompute orelse return types.KernelError.LaunchFailed;
    dispatch_fn(config.grid_dim[0], config.grid_dim[1], config.grid_dim[2]);

    // Memory barrier
    const barrier_fn = glMemoryBarrier orelse return types.KernelError.LaunchFailed;
    barrier_fn(GL_SHADER_STORAGE_BARRIER_BIT);
}

pub fn destroyKernel(allocator: std.mem.Allocator, kernel_handle: *anyopaque) void {
    if (!opengl_initialized) {
        return;
    }

    const kernel: *OpenGlKernel = @ptrCast(@alignCast(kernel_handle));

    const delete_program_fn = glDeleteProgram orelse return;
    delete_program_fn(kernel.program);

    const delete_shader_fn = glDeleteShader orelse return;
    delete_shader_fn(kernel.shader);

    allocator.destroy(kernel);
}

pub fn allocateDeviceMemory(size: usize) !*anyopaque {
    if (!opengl_initialized) {
        return OpenGlError.BufferCreationFailed;
    }

    const gen_buffers_fn = glGenBuffers orelse return OpenGlError.BufferCreationFailed;
    var buffer_id: u32 = 0;
    gen_buffers_fn(1, &buffer_id);

    const bind_buffer_fn = glBindBuffer orelse return OpenGlError.BufferCreationFailed;
    bind_buffer_fn(GL_SHADER_STORAGE_BUFFER, buffer_id);

    const buffer_data_fn = glBufferData orelse return OpenGlError.BufferCreationFailed;
    buffer_data_fn(GL_SHADER_STORAGE_BUFFER, @intCast(size), null, GL_DYNAMIC_READ);

    const opengl_buffer = try std.heap.page_allocator.create(OpenGlBuffer);
    opengl_buffer.* = .{
        .buffer_id = buffer_id,
        .size = size,
    };

    return opengl_buffer;
}

pub fn freeDeviceMemory(ptr: *anyopaque) void {
    if (!opengl_initialized) {
        return;
    }

    const buffer: *OpenGlBuffer = @ptrCast(@alignCast(ptr));

    const delete_buffers_fn = glDeleteBuffers orelse return;
    delete_buffers_fn(1, &buffer.buffer_id);

    std.heap.page_allocator.destroy(buffer);
}

pub fn memcpyHostToDevice(dst: *anyopaque, src: *anyopaque, size: usize) !void {
    if (!opengl_initialized) {
        return OpenGlError.BufferCreationFailed;
    }

    const dst_buffer: *OpenGlBuffer = @ptrCast(@alignCast(dst));

    const bind_buffer_fn = glBindBuffer orelse return OpenGlError.BufferCreationFailed;
    bind_buffer_fn(GL_SHADER_STORAGE_BUFFER, dst_buffer.buffer_id);

    const buffer_data_fn = glBufferData orelse return OpenGlError.BufferCreationFailed;
    buffer_data_fn(GL_SHADER_STORAGE_BUFFER, @intCast(size), src, GL_STATIC_DRAW);
}

pub fn memcpyDeviceToHost(dst: *anyopaque, src: *anyopaque, size: usize) !void {
    if (!opengl_initialized) {
        return OpenGlError.BufferCreationFailed;
    }

    const src_buffer: *OpenGlBuffer = @ptrCast(@alignCast(src));

    const bind_buffer_fn = glBindBuffer orelse return OpenGlError.BufferCreationFailed;
    bind_buffer_fn(GL_SHADER_STORAGE_BUFFER, src_buffer.buffer_id);

    const get_buffer_sub_data_fn = glGetBufferSubData orelse return OpenGlError.BufferCreationFailed;
    get_buffer_sub_data_fn(GL_SHADER_STORAGE_BUFFER, 0, @intCast(size), dst);
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

    glGetString = opengl_lib.?.lookup(GlGetStringFn, "glGetString") orelse return false;
    glCreateShader = opengl_lib.?.lookup(GlCreateShaderFn, "glCreateShader") orelse return false;
    glShaderSource = opengl_lib.?.lookup(GlShaderSourceFn, "glShaderSource") orelse return false;
    glCompileShader = opengl_lib.?.lookup(GlCompileShaderFn, "glCompileShader") orelse return false;
    glGetShaderiv = opengl_lib.?.lookup(GlGetShaderivFn, "glGetShaderiv") orelse return false;
    glGetShaderInfoLog = opengl_lib.?.lookup(GlGetShaderInfoLogFn, "glGetShaderInfoLog") orelse return false;
    glCreateProgram = opengl_lib.?.lookup(GlCreateProgramFn, "glCreateProgram") orelse return false;
    glAttachShader = opengl_lib.?.lookup(GlAttachShaderFn, "glAttachShader") orelse return false;
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

    return true;
}
