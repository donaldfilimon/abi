//! Zig to SPIR-V Compiler Bridge
//!
//! Utilities to compile Zig kernel source code directly to SPIR-V bytecode
//! using the Zig compiler's native SPIR-V backend (`-target spirv64-vulkan`).
//! Also provides helpers to generate kernel source skeletons.

const std = @import("std");
const builtin = @import("builtin");
const Allocator = std.mem.Allocator;
const Io = std.Io;

// ---------------------------------------------------------------------------
// Public types
// ---------------------------------------------------------------------------

/// Optimisation level forwarded to the Zig compiler.
pub const OptLevel = enum {
    debug,
    release_safe,
    release_small,
    release_fast,

    fn toFlag(self: OptLevel) []const u8 {
        return switch (self) {
            .debug => "Debug",
            .release_safe => "ReleaseSafe",
            .release_small => "ReleaseSmall",
            .release_fast => "ReleaseFast",
        };
    }
};

/// Options that control how `compileZigToSpirv` invokes the compiler.
pub const SpvCompileOptions = struct {
    /// Override the path to the Zig executable. When `null`, the path is
    /// resolved automatically via `resolveZigPath`.
    zig_exe_path: ?[]const u8 = null,
    /// Optimisation level.
    optimize: OptLevel = .release_small,
    /// Target triple passed to `-target`.
    target: []const u8 = "spirv64-vulkan",
};

/// Owns the compiled SPIR-V bytecode returned by `compileZigToSpirv`.
pub const SpvResult = struct {
    bytecode: []u8,
    allocator: Allocator,

    pub fn deinit(self: *SpvResult) void {
        self.allocator.free(self.bytecode);
        self.* = undefined;
    }
};

/// Buffer parameter descriptor used by `generateKernelSource`.
pub const KernelParam = struct {
    name: []const u8,
    /// Zig type name for the element, e.g. `"f32"`, `"u32"`.
    elem_type: []const u8,
    /// `true` ⇒ `*const addrspace(.global)`, `false` ⇒ `*addrspace(.global)`.
    read_only: bool = false,
};

// ---------------------------------------------------------------------------
// Zig path resolution
// ---------------------------------------------------------------------------

/// Resolve the path to a usable Zig compiler.
///
/// Search order:
///  1. `ABI_ZIG_PATH` environment variable.
///  2. `~/.zvm/master/zig` (ZVM-managed compiler).
///  3. `"zig"` (assumed to be on `PATH`).
pub fn resolveZigPath(allocator: Allocator) ![]const u8 {
    // 1. Environment override
    if (std.c.getenv("ABI_ZIG_PATH")) |ptr| {
        const val = std.mem.sliceTo(ptr, 0);
        if (val.len > 0)
            return allocator.dupe(u8, val);
    }

    // 2. ZVM default location
    if (std.c.getenv("HOME")) |home_ptr| {
        const home = std.mem.sliceTo(home_ptr, 0);
        const zvm_path = try std.fmt.allocPrint(allocator, "{s}/.zvm/master/zig", .{home});
        // Check existence via C access().
        const path_z = std.posix.toPosixPath(zvm_path) catch {
            allocator.free(zvm_path);
            return allocator.dupe(u8, "zig");
        };
        if (std.c.access(&path_z, std.posix.F_OK) == 0) {
            return zvm_path;
        }
        allocator.free(zvm_path);
        return allocator.dupe(u8, "zig");
    }

    // 3. Fallback
    return allocator.dupe(u8, "zig");
}

// ---------------------------------------------------------------------------
// Compilation
// ---------------------------------------------------------------------------

pub const CompileError = error{
    CompilerNotFound,
    CompilationFailed,
    OutputMissing,
    OutOfMemory,
    SpawnFailed,
    ReadFailed,
};

/// Compile Zig source code to SPIR-V bytecode.
///
/// The function:
///  1. Writes `source` to a temporary `.zig` file.
///  2. Spawns `zig build-obj … -target spirv64-vulkan -femit-bin=<out>.spv`.
///  3. Reads back the `.spv` binary.
///  4. Cleans up temporary files.
pub fn compileZigToSpirv(
    allocator: Allocator,
    source: []const u8,
    options: SpvCompileOptions,
) CompileError!SpvResult {
    // Resolve Zig compiler path.
    const zig_path = if (options.zig_exe_path) |p|
        allocator.dupe(u8, p) catch return error.OutOfMemory
    else
        resolveZigPath(allocator) catch return error.CompilerNotFound;
    defer allocator.free(zig_path);

    // Determine temp directory.
    const tmp_base = getTmpDir();

    // Build unique-ish file names using arc4random.
    var rand_bytes: [8]u8 = undefined;
    std.c.arc4random_buf(&rand_bytes, rand_bytes.len);
    const hex = std.fmt.bytesToHex(rand_bytes, .lower);
    const src_name = std.fmt.allocPrint(allocator, "{s}/abi_spirv_{s}.zig", .{ tmp_base, hex }) catch
        return error.OutOfMemory;
    defer allocator.free(src_name);
    const spv_name = std.fmt.allocPrint(allocator, "{s}/abi_spirv_{s}.spv", .{ tmp_base, hex }) catch
        return error.OutOfMemory;
    defer allocator.free(spv_name);

    // Write source to temp file.
    const emit_flag = std.fmt.allocPrint(allocator, "-femit-bin={s}", .{spv_name}) catch
        return error.OutOfMemory;
    defer allocator.free(emit_flag);

    // Create Io backend for file and process operations.
    var io_backend = Io.Threaded.init(allocator, .{ .environ = std.process.Environ.empty });
    defer io_backend.deinit();
    const io = io_backend.io();

    // Write source to temp file.
    const cwd = Io.Dir.cwd();
    cwd.writeFile(io, .{ .sub_path = src_name, .data = source }) catch
        return error.CompilationFailed;
    defer cwd.deleteFile(io, src_name) catch {};
    defer cwd.deleteFile(io, spv_name) catch {};

    // Spawn the compiler.
    const argv = [_][]const u8{
        zig_path,
        "build-obj",
        src_name,
        "-target",
        options.target,
        "-O",
        options.optimize.toFlag(),
        emit_flag,
    };

    var child = std.process.spawn(io, .{
        .argv = &argv,
        .stdout = .pipe,
        .stderr = .pipe,
    }) catch return error.SpawnFailed;

    // Drain stderr/stdout so the child doesn't block.
    var stderr_buf: [4096]u8 = undefined;
    var stderr_reader = child.stderr.?.readerStreaming(io, &stderr_buf);
    const stderr_data = stderr_reader.interface.allocRemaining(
        allocator,
        .limited(64 * 1024),
    ) catch null;
    defer if (stderr_data) |d| allocator.free(d);

    var stdout_buf: [4096]u8 = undefined;
    var stdout_reader = child.stdout.?.readerStreaming(io, &stdout_buf);
    const stdout_data = stdout_reader.interface.allocRemaining(
        allocator,
        .limited(64 * 1024),
    ) catch null;
    defer if (stdout_data) |d| allocator.free(d);

    const term = child.wait(io) catch return error.CompilationFailed;
    const exit_ok = switch (term) {
        .exited => |code| code == 0,
        else => false,
    };
    if (!exit_ok) return error.CompilationFailed;

    // Read the generated .spv binary.
    const bytecode = cwd.readFileAlloc(io, spv_name, allocator, .limited(16 * 1024 * 1024)) catch
        return error.OutputMissing;

    return .{ .bytecode = bytecode, .allocator = allocator };
}

// ---------------------------------------------------------------------------
// Source generation
// ---------------------------------------------------------------------------

/// Generate a complete kernel source file from structured parameters.
///
/// The generated function uses `callconv(.spirv_kernel)` and
/// `addrspace(.global)` pointers.
pub fn generateKernelSource(
    allocator: Allocator,
    name: []const u8,
    params: []const KernelParam,
    body: []const u8,
) ![]u8 {
    var buf: std.ArrayList(u8) = .empty;
    defer buf.deinit(allocator);

    try buf.appendSlice(allocator, "const std = @import(\"std\");\n\n");

    const fn_header = try std.fmt.allocPrint(allocator, "pub export fn {s}(\n", .{name});
    defer allocator.free(fn_header);
    try buf.appendSlice(allocator, fn_header);

    try buf.appendSlice(allocator, "    global_id: u32,\n");
    for (params) |p| {
        const const_qual: []const u8 = if (p.read_only) "const " else "";
        const param_line = try std.fmt.allocPrint(
            allocator,
            "    {s}: [*]{s}addrspace(.global) {s},\n",
            .{ p.name, const_qual, p.elem_type },
        );
        defer allocator.free(param_line);
        try buf.appendSlice(allocator, param_line);
    }
    try buf.appendSlice(allocator, ") callconv(.spirv_kernel) void {\n");

    // Indent each line of body
    var body_iter = std.mem.splitScalar(u8, body, '\n');
    while (body_iter.next()) |line| {
        if (line.len > 0) {
            const indented = try std.fmt.allocPrint(allocator, "    {s}\n", .{line});
            defer allocator.free(indented);
            try buf.appendSlice(allocator, indented);
        } else {
            try buf.appendSlice(allocator, "\n");
        }
    }
    try buf.appendSlice(allocator, "}\n");

    return buf.toOwnedSlice(allocator);
}

/// Generate a simple skeleton kernel (backward-compatible API).
pub fn generateSkeleton(allocator: Allocator, name: []const u8) ![]u8 {
    return std.fmt.allocPrint(allocator,
        \\const std = @import("std");
        \\
        \\pub export fn {s}(
        \\    global_id: u32,
        \\    data: [*]addrspace(.global) f32,
        \\) callconv(.spirv_kernel) void {{
        \\    const idx = global_id;
        \\    data[idx] *= 2.0;
        \\}}
    , .{name});
}

// ---------------------------------------------------------------------------
// Internal helpers
// ---------------------------------------------------------------------------

fn getTmpDir() []const u8 {
    if (std.c.getenv("TMPDIR")) |ptr| {
        const val = std.mem.sliceTo(ptr, 0);
        if (val.len > 0) return val;
    }
    return "/tmp";
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

test "resolveZigPath finds a Zig compiler" {
    const allocator = std.testing.allocator;
    const path = try resolveZigPath(allocator);
    defer allocator.free(path);
    try std.testing.expect(path.len > 0);
}

test "generateSkeleton produces valid output" {
    const allocator = std.testing.allocator;
    const src = try generateSkeleton(allocator, "my_kernel");
    defer allocator.free(src);

    // Must contain the function name and spirv_kernel callconv
    try std.testing.expect(std.mem.indexOf(u8, src, "my_kernel") != null);
    try std.testing.expect(std.mem.indexOf(u8, src, "spirv_kernel") != null);
    try std.testing.expect(std.mem.indexOf(u8, src, "addrspace(.global)") != null);
}

test "generateKernelSource produces correct Zig source" {
    const allocator = std.testing.allocator;
    const params = [_]KernelParam{
        .{ .name = "input", .elem_type = "f32", .read_only = true },
        .{ .name = "output", .elem_type = "f32" },
    };
    const src = try generateKernelSource(
        allocator,
        "scale",
        &params,
        "const idx = global_id;\noutput[idx] = input[idx] * 2.0;",
    );
    defer allocator.free(src);

    try std.testing.expect(std.mem.indexOf(u8, src, "pub export fn scale(") != null);
    try std.testing.expect(std.mem.indexOf(u8, src, "callconv(.spirv_kernel)") != null);
    try std.testing.expect(std.mem.indexOf(u8, src, "[*]const addrspace(.global) f32") != null);
    try std.testing.expect(std.mem.indexOf(u8, src, "[*]addrspace(.global) f32") != null);
    try std.testing.expect(std.mem.indexOf(u8, src, "output[idx] = input[idx] * 2.0;") != null);
}

test "compileZigToSpirv skips when SPIR-V target unavailable" {
    // The SPIR-V target may not be available in all environments.
    // We attempt a tiny compile; if the compiler itself cannot be found or
    // the target is unsupported we skip the test gracefully.
    const allocator = std.testing.allocator;

    const source = try generateSkeleton(allocator, "test_kern");
    defer allocator.free(source);

    var result = compileZigToSpirv(allocator, source, .{}) catch |err| {
        switch (err) {
            error.CompilerNotFound,
            error.SpawnFailed,
            error.CompilationFailed,
            => return error.SkipZigTest,
            else => return err,
        }
    };
    defer result.deinit();

    // SPIR-V magic number: 0x07230203
    try std.testing.expect(result.bytecode.len >= 4);
    const magic = std.mem.readInt(u32, result.bytecode[0..4], .little);
    try std.testing.expect(magic == 0x07230203);
}

test {
    std.testing.refAllDecls(@This());
}
