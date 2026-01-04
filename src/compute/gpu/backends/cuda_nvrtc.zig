//! NVRTC (NVIDIA Runtime Compiler) for PTX compilation.
//!
//! Provides runtime compilation of CUDA C++ source to PTX with caching.

const std = @import("std");

pub const NvrtcError = error{
    CompilationFailed,
    InvalidProgram,
    InvalidInput,
    OutOfMemory,
    ProgramNotFound,
    InternalError,
};

pub const CompileOptions = struct {
    max_registers: i32 = 0,
    min_blocks_per_multiprocessor: i32 = 0,
    optimization_level: u32 = 3,
    generate_debug_info: bool = false,
    generate_line_info: bool = false,
    cache_dir: ?[]const u8 = null,
};

pub const CompileResult = struct {
    ptx: []u8,
    log: []u8,
    size_log: usize,
};

const NvrtcResult = enum(i32) {
    success = 0,
    out_of_memory = 2,
    program_creation_failed = 3,
    invalid_input = 4,
    invalid_program = 5,
    compilation_failed = 6,
    internal_error = 7,
    error_unknown = 999,
};

const NvrtcProgram = *anyopaque;

const NvrtcCreateProgramFn = *const fn (
    *NvrtcProgram,
    [*:0]const u8,
    [*:0]const u8,
    i32,
    [*]const [*:0]const u8,
    [*]const [*:0]const u8,
) callconv(.c) NvrtcResult;
const NvrtcDestroyProgramFn = *const fn (*NvrtcProgram) callconv(.c) NvrtcResult;
const NvrtcCompileProgramFn = *const fn (NvrtcProgram, i32, [*]const [*:0]const u8) callconv(.c) NvrtcResult;
const NvrtcGetPTXSizeFn = *const fn (NvrtcProgram, *usize) callconv(.c) NvrtcResult;
const NvrtcGetPTXFn = *const fn (NvrtcProgram, [*]u8) callconv(.c) NvrtcResult;
const NvrtcGetProgramLogSizeFn = *const fn (NvrtcProgram, *usize) callconv(.c) NvrtcResult;
const NvrtcGetProgramLogFn = *const fn (NvrtcProgram, [*]u8) callconv(.c) NvrtcResult;
const NvrtcAddNameExpressionFn = *const fn (NvrtcProgram, [*:0]const u8) callconv(.c) NvrtcResult;

var nvrtcCreateProgram: ?NvrtcCreateProgramFn = null;
var nvrtcDestroyProgram: ?NvrtcDestroyProgramFn = null;
var nvrtcCompileProgram: ?NvrtcCompileProgramFn = null;
var nvrtcGetPTXSize: ?NvrtcGetPTXSizeFn = null;
var nvrtcGetPTX: ?NvrtcGetPTXFn = null;
var nvrtcGetProgramLogSize: ?NvrtcGetProgramLogSizeFn = null;
var nvrtcGetProgramLog: ?NvrtcGetProgramLogFn = null;
var nvrtcAddNameExpression: ?NvrtcAddNameExpressionFn = null;
var nvrtc_lib: ?std.DynLib = null;
var nvrtc_initialized = false;

pub fn init() !void {
    if (nvrtc_initialized) return;

    if (!tryLoadNvrtc()) {
        return NvrtcError.CompilationFailed;
    }

    if (!loadNvrtcFunctions()) {
        return NvrtcError.CompilationFailed;
    }

    nvrtc_initialized = true;
}

pub fn deinit() void {
    if (nvrtc_lib) |lib| {
        lib.close();
    }
    nvrtc_lib = null;
    nvrtc_initialized = false;
}

pub fn compileToPTX(
    allocator: std.mem.Allocator,
    source: []const u8,
    name: []const u8,
    options: CompileOptions,
) !CompileResult {
    const create_fn = nvrtcCreateProgram orelse return NvrtcError.CompilationFailed;

    var program: NvrtcProgram = undefined;
    const source_ptr = allocator.dupeZ(u8, source) catch return NvrtcError.OutOfMemory;
    defer allocator.free(source_ptr);

    const name_ptr = allocator.dupeZ(u8, name) catch return NvrtcError.OutOfMemory;
    defer allocator.free(name_ptr);

    if (create_fn(&program, source_ptr, name_ptr, 0, null, null) != .success) {
        return NvrtcError.CompilationFailed;
    }

    const add_expr_fn = nvrtcAddNameExpression orelse {
        const destroy_fn = nvrtcDestroyProgram orelse return NvrtcError.CompilationFailed;
        destroy_fn(&program);
        return NvrtcError.CompilationFailed;
    };

    if (add_expr_fn(program, name_ptr) != .success) {
        const destroy_fn = nvrtcDestroyProgram orelse return NvrtcError.CompilationFailed;
        destroy_fn(&program);
        return NvrtcError.CompilationFailed;
    }

    const compile_opts = try buildCompileOptions(allocator, options);
    defer {
        for (compile_opts) |opt| {
            allocator.free(opt);
        }
        allocator.free(compile_opts);
    }

    const compile_fn = nvrtcCompileProgram orelse {
        const destroy_fn = nvrtcDestroyProgram orelse return NvrtcError.CompilationFailed;
        destroy_fn(&program);
        return NvrtcError.CompilationFailed;
    };

    if (compile_fn(program, @intCast(compile_opts.len), compile_opts.ptr) != .success) {
        var log_size: usize = 0;
        const log_size_fn = nvrtcGetProgramLogSize orelse {
            const destroy_fn = nvrtcDestroyProgram orelse return NvrtcError.CompilationFailed;
            destroy_fn(&program);
            return NvrtcError.CompilationFailed;
        };

        if (log_size_fn(program, &log_size) == .success and log_size > 0) {
            const log = try allocator.alloc(u8, log_size);
            defer allocator.free(log);
            const log_fn = nvrtcGetProgramLog orelse {
                const destroy_fn = nvrtcDestroyProgram orelse return NvrtcError.CompilationFailed;
                destroy_fn(&program);
                return NvrtcError.CompilationFailed;
            };
            _ = log_fn(program, log.ptr);
            std.log.err("NVRTC Compilation Failed: {s}\n", .{log});
        }

        const destroy_fn = nvrtcDestroyProgram orelse return NvrtcError.CompilationFailed;
        destroy_fn(&program);
        return NvrtcError.CompilationFailed;
    }

    var ptx_size: usize = 0;
    const size_fn = nvrtcGetPTXSize orelse {
        const destroy_fn = nvrtcDestroyProgram orelse return NvrtcError.CompilationFailed;
        destroy_fn(&program);
        return NvrtcError.CompilationFailed;
    };

    if (size_fn(program, &ptx_size) != .success or ptx_size == 0) {
        const destroy_fn = nvrtcDestroyProgram orelse return NvrtcError.CompilationFailed;
        destroy_fn(&program);
        return NvrtcError.CompilationFailed;
    }

    const ptx = try allocator.alloc(u8, ptx_size);

    const get_fn = nvrtcGetPTX orelse {
        const destroy_fn = nvrtcDestroyProgram orelse return NvrtcError.CompilationFailed;
        destroy_fn(&program);
        return NvrtcError.CompilationFailed;
    };

    if (get_fn(program, ptx.ptr) != .success) {
        allocator.free(ptx);
        const destroy_fn = nvrtcDestroyProgram orelse return NvrtcError.CompilationFailed;
        destroy_fn(&program);
        return NvrtcError.CompilationFailed;
    }

    var log_size: usize = 0;
    var log: []u8 = &.{};
    const log_size_fn = nvrtcGetProgramLogSize;
    const log_fn = nvrtcGetProgramLog;

    if (log_size_fn) |ls_fn| {
        if (ls_fn(program, &log_size) == .success and log_size > 0) {
            log = try allocator.alloc(u8, log_size);
            if (log_fn) |lf| {
                _ = lf(program, log.ptr);
            }
        }
    }

    const destroy_fn = nvrtcDestroyProgram orelse {
        allocator.free(ptx);
        allocator.free(log);
        return NvrtcError.CompilationFailed;
    };

    destroy_fn(&program);

    return .{
        .ptx = ptx,
        .log = log,
        .size_log = log_size,
    };
}

fn buildCompileOptions(allocator: std.mem.Allocator, options: CompileOptions) ![][]const u8 {
    var opts = std.ArrayList([]const u8).init(allocator);

    if (options.max_registers != 0) {
        const opt = try std.fmt.allocPrint(allocator, "-maxrregcount={d}", .{options.max_registers});
        try opts.append(opt);
    }

    if (options.min_blocks_per_multiprocessor != 0) {
        const opt = try std.fmt.allocPrint(allocator, "-minblockspermp={d}", .{options.min_blocks_per_multiprocessor});
        try opts.append(opt);
    }

    const arch = try std.fmt.allocPrint(allocator, "-arch=native", .{});
    try opts.append(arch);

    const opt_level = switch (options.optimization_level) {
        0 => "-O0",
        1 => "-O1",
        2 => "-O2",
        3 => "-O3",
        else => "-O3",
    };
    const opt_str = try allocator.dupe(u8, opt_level);
    try opts.append(opt_str);

    if (options.generate_debug_info) {
        const opt = try allocator.dupe(u8, "-lineinfo");
        try opts.append(opt);
    }

    if (options.generate_line_info) {
        const opt = try allocator.dupe(u8, "-device-debug");
        try opts.append(opt);
    }

    try opts.append(try allocator.dupe(u8, "-rdc=true"));
    try opts.append(try allocator.dupe(u8, "-default-device"));

    return try opts.toOwnedSlice();
}

fn tryLoadNvrtc() bool {
    const lib_names = [_][]const u8{ "nvrtc64.dll", "libnvrtc.so.11", "libnvrtc.so" };
    for (lib_names) |name| {
        if (std.DynLib.open(name)) |lib| {
            nvrtc_lib = lib;
            return true;
        } else |_| {}
    }
    return false;
}

fn loadNvrtcFunctions() bool {
    if (nvrtc_lib == null) return false;

    nvrtcCreateProgram = nvrtc_lib.?.lookup(NvrtcCreateProgramFn, "nvrtcCreateProgram") orelse return false;
    nvrtcDestroyProgram = nvrtc_lib.?.lookup(NvrtcDestroyProgramFn, "nvrtcDestroyProgram") orelse return false;
    nvrtcCompileProgram = nvrtc_lib.?.lookup(NvrtcCompileProgramFn, "nvrtcCompileProgram") orelse return false;
    nvrtcGetPTXSize = nvrtc_lib.?.lookup(NvrtcGetPTXSizeFn, "nvrtcGetPTXSize") orelse return false;
    nvrtcGetPTX = nvrtc_lib.?.lookup(NvrtcGetPTXFn, "nvrtcGetPTX") orelse return false;
    nvrtcGetProgramLogSize = nvrtc_lib.?.lookup(NvrtcGetProgramLogSizeFn, "nvrtcGetProgramLogSize") orelse return false;
    nvrtcGetProgramLog = nvrtc_lib.?.lookup(NvrtcGetProgramLogFn, "nvrtcGetProgramLog") orelse return false;
    nvrtcAddNameExpression = nvrtc_lib.?.lookup(NvrtcAddNameExpressionFn, "nvrtcAddNameExpression") orelse return false;

    return true;
}
