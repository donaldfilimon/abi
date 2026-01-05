//! OpenGL ES backend implementation with compute shader support.
//!
//! Provides OpenGL ES-specific kernel compilation and execution using compute shaders
//! for mobile and embedded platforms.

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
};

var opengles_lib: ?std.DynLib = null;
var opengles_initialized = false;

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

const OpenGlesKernel = struct {
    program: u32,
    shader: u32,
};

const OpenGlesBuffer = struct {
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
    if (opengles_initialized) return;

    if (!tryLoadOpenGles()) {
        return OpenGlesError.InitializationFailed;
    }

    if (!loadOpenGlesFunctions()) {
        return OpenGlesError.InitializationFailed;
    }

    // Check if compute shaders are supported (OpenGL ES 3.1+)
    const version_string = glesGetString orelse return OpenGlesError.InitializationFailed;
    const version = version_string(0x1F02); // GL_VERSION
    if (version == null) {
        return OpenGlesError.InitializationFailed;
    }

    opengles_initialized = true;
}

pub fn deinit() void {
    if (opengles_lib) |lib| {
        lib.close();
    }
    opengles_lib = null;
    opengles_initialized = false;
}

pub fn compileKernel(
    allocator: std.mem.Allocator,
    source: types.KernelSource,
) types.KernelError!*anyopaque {
    if (!opengles_initialized) {
        return types.KernelError.CompilationFailed;
    }

    // Create compute shader
    const create_shader_fn = glesCreateShader orelse return types.KernelError.CompilationFailed;
    const shader = create_shader_fn(GL_COMPUTE_SHADER);

    // Set shader source
    const source_ptr = &[_][*:0]const u8{source.source.ptr};
    const set_source_fn = glesShaderSource orelse return types.KernelError.CompilationFailed;
    set_source_fn(shader, 1, source_ptr.ptr, null);

    // Compile shader
    const compile_fn = glesCompileShader orelse return types.KernelError.CompilationFailed;
    compile_fn(shader);

    // Check compilation status
    const get_shader_iv_fn = glesGetShaderiv orelse return types.KernelError.CompilationFailed;
    var compile_status: i32 = 0;
    get_shader_iv_fn(shader, GL_COMPILE_STATUS, &compile_status);

    if (compile_status == 0) {
        // Get error log
        var log_length: i32 = 0;
        get_shader_iv_fn(shader, 0x8B84, &log_length); // GL_INFO_LOG_LENGTH
        if (log_length > 0) {
            var log = try allocator.alloc(u8, @intCast(log_length));
            defer allocator.free(log);
            const get_log_fn = glesGetShaderInfoLog orelse return types.KernelError.CompilationFailed;
            get_log_fn(shader, log_length, null, log.ptr);
            std.log.err("OpenGL ES Shader Compilation Failed: {s}", .{log});
        }
        return types.KernelError.CompilationFailed;
    }

    // Create program
    const create_program_fn = glesCreateProgram orelse return types.KernelError.CompilationFailed;
    const program = create_program_fn();

    // Attach shader
    const attach_fn = glesAttachShader orelse return types.KernelError.CompilationFailed;
    attach_fn(program, shader);

    // Link program
    const link_fn = glesLinkProgram orelse return types.KernelError.CompilationFailed;
    link_fn(program);

    // Check link status
    const get_program_iv_fn = glesGetProgramiv orelse return types.KernelError.CompilationFailed;
    var link_status: i32 = 0;
    get_program_iv_fn(program, GL_LINK_STATUS, &link_status);

    if (link_status == 0) {
        // Get error log
        var log_length: i32 = 0;
        get_program_iv_fn(program, 0x8B84, &log_length); // GL_INFO_LOG_LENGTH
        if (log_length > 0) {
            var log = try allocator.alloc(u8, @intCast(log_length));
            defer allocator.free(log);
            const get_log_fn = glesGetProgramInfoLog orelse return types.KernelError.CompilationFailed;
            get_log_fn(program, log_length, null, log.ptr);
            std.log.err("OpenGL ES Program Linking Failed: {s}", .{log});
        }
        return types.KernelError.CompilationFailed;
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
) types.KernelError!void {
    _ = allocator;

    if (!opengles_initialized) {
        return types.KernelError.LaunchFailed;
    }

    const kernel: *OpenGlesKernel = @ptrCast(@alignCast(kernel_handle));

    // Use program
    const use_program_fn = glesUseProgram orelse return types.KernelError.LaunchFailed;
    use_program_fn(kernel.program);

    // Bind buffers
    const bind_buffer_base_fn = glesBindBufferBase orelse return types.KernelError.LaunchFailed;
    for (args, 0..) |arg, i| {
        if (arg != null) {
            const buffer: *OpenGlesBuffer = @ptrCast(@alignCast(arg.?));
            bind_buffer_base_fn(GL_SHADER_STORAGE_BUFFER, @intCast(i), buffer.buffer_id);
        }
    }

    // Dispatch compute
    const dispatch_fn = glesDispatchCompute orelse return types.KernelError.LaunchFailed;
    dispatch_fn(config.grid_dim[0], config.grid_dim[1], config.grid_dim[2]);

    // Memory barrier
    const barrier_fn = glesMemoryBarrier orelse return types.KernelError.LaunchFailed;
    barrier_fn(GL_SHADER_STORAGE_BARRIER_BIT);
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

pub fn allocateDeviceMemory(size: usize) !*anyopaque {
    if (!opengles_initialized) {
        return OpenGlesError.BufferCreationFailed;
    }

    const gen_buffers_fn = glesGenBuffers orelse return OpenGlesError.BufferCreationFailed;
    var buffer_id: u32 = 0;
    gen_buffers_fn(1, &buffer_id);

    const bind_buffer_fn = glesBindBuffer orelse return OpenGlesError.BufferCreationFailed;
    bind_buffer_fn(GL_SHADER_STORAGE_BUFFER, buffer_id);

    const buffer_data_fn = glesBufferData orelse return OpenGlesError.BufferCreationFailed;
    buffer_data_fn(GL_SHADER_STORAGE_BUFFER, @intCast(size), null, GL_DYNAMIC_READ);

    const opengles_buffer = try std.heap.page_allocator.create(OpenGlesBuffer);
    opengles_buffer.* = .{
        .buffer_id = buffer_id,
        .size = size,
    };

    return opengles_buffer;
}

pub fn freeDeviceMemory(ptr: *anyopaque) void {
    if (!opengles_initialized) {
        return;
    }

    const buffer: *OpenGlesBuffer = @ptrCast(@alignCast(ptr));

    const delete_buffers_fn = glesDeleteBuffers orelse return;
    delete_buffers_fn(1, &buffer.buffer_id);

    std.heap.page_allocator.destroy(buffer);
}

pub fn memcpyHostToDevice(dst: *anyopaque, src: *anyopaque, size: usize) !void {
    if (!opengles_initialized) {
        return OpenGlesError.BufferCreationFailed;
    }

    const dst_buffer: *OpenGlesBuffer = @ptrCast(@alignCast(dst));

    const bind_buffer_fn = glesBindBuffer orelse return OpenGlesError.BufferCreationFailed;
    bind_buffer_fn(GL_SHADER_STORAGE_BUFFER, dst_buffer.buffer_id);

    const buffer_data_fn = glesBufferData orelse return OpenGlesError.BufferCreationFailed;
    buffer_data_fn(GL_SHADER_STORAGE_BUFFER, @intCast(size), src, GL_STATIC_DRAW);
}

pub fn memcpyDeviceToHost(dst: *anyopaque, src: *anyopaque, size: usize) !void {
    if (!opengles_initialized) {
        return OpenGlesError.BufferCreationFailed;
    }

    const src_buffer: *OpenGlesBuffer = @ptrCast(@alignCast(src));

    const bind_buffer_fn = glesBindBuffer orelse return OpenGlesError.BufferCreationFailed;
    bind_buffer_fn(GL_SHADER_STORAGE_BUFFER, src_buffer.buffer_id);

    const get_buffer_sub_data_fn = glesGetBufferSubData orelse return OpenGlesError.BufferCreationFailed;
    get_buffer_sub_data_fn(GL_SHADER_STORAGE_BUFFER, 0, @intCast(size), dst);
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
    glesGetBufferSubData = opengles_lib.?.lookup(GlesGetBufferSubDataFn, "glGetBufferSubData") orelse return false;
    glesDeleteBuffers = opengles_lib.?.lookup(GlesDeleteBuffersFn, "glDeleteBuffers") orelse return false;
    glesDeleteProgram = opengles_lib.?.lookup(GlesDeleteProgramFn, "glDeleteProgram") orelse return false;
    glesDeleteShader = opengles_lib.?.lookup(GlesDeleteShaderFn, "glDeleteShader") orelse return false;

    return true;
}
