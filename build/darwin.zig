//! Darwin 25+ (macOS 26+) build abstraction.
//!
//! Zig 0.16-dev's internal Mach-O linker cannot resolve system symbols on
//! macOS 26+.  This module encapsulates the compile-as-object → relink-with-
//! Apple-ld workaround so that `build.zig` does not need per-site forks.
const std = @import("std");
const builtin = @import("builtin");
const link = @import("link.zig");

// ── Context ──────────────────────────────────────────────────────────────

/// Pre-computed Darwin degraded-mode state, threaded through the build.
pub const DarwinCtx = struct {
    is_blocked: bool,
    compiler_rt: ?[]const u8,
};

/// Detect whether the current host is Darwin 25+ with a stock (non-host-built)
/// Zig, and pre-compute the compiler_rt path for relink operations.
pub fn initDarwinCtx(b: *std.Build) DarwinCtx {
    const blocked = useDarwinDegradedMode(b);
    return .{
        .is_blocked = blocked,
        .compiler_rt = if (blocked) link.findCompilerRt(b) else null,
    };
}

// ── Artifact Helpers ─────────────────────────────────────────────────────

/// Result of `addExeOrObject`: a Compile artifact that is either an
/// executable (normal) or an object file (Darwin degraded — needs relink).
pub const DarwinArtifact = struct {
    compile: *std.Build.Step.Compile,
    is_object: bool,
};

/// Create an executable (normal path) or a compiled object (Darwin degraded
/// path) from the given module.  On Darwin, sets `use_llvm = true` so the
/// LLVM backend produces a .o file that Apple's ld can link.
pub fn addExeOrObject(
    b: *std.Build,
    name: []const u8,
    root_module: *std.Build.Module,
    ctx: DarwinCtx,
) DarwinArtifact {
    if (ctx.is_blocked) {
        const obj = b.addObject(.{ .name = name, .root_module = root_module });
        obj.use_llvm = true;
        return .{ .compile = obj, .is_object = true };
    }
    return .{
        .compile = b.addExecutable(.{ .name = name, .root_module = root_module }),
        .is_object = false,
    };
}

/// Produce a Run step from a `DarwinArtifact`.  On Darwin degraded, this
/// relinks via Apple's `/usr/bin/ld`; otherwise it simply runs the executable.
pub fn addRunStep(
    b: *std.Build,
    artifact: DarwinArtifact,
    output_name: []const u8,
    ctx: DarwinCtx,
) *std.Build.Step.Run {
    if (artifact.is_object) {
        return link.darwinRelink(b, artifact.compile, output_name, ctx.compiler_rt);
    }
    return b.addRunArtifact(artifact.compile);
}

/// Set `use_llvm = true` on an artifact when Darwin is blocked.
/// Use this for test artifacts and libraries that don't need the full
/// addExeOrObject / addRunStep dance.
pub fn enableLlvm(artifact: *std.Build.Step.Compile, ctx: DarwinCtx) void {
    if (ctx.is_blocked) artifact.use_llvm = true;
}

/// Create a test run step that handles Darwin degraded mode.
/// On Darwin, tests compile but cannot link/run, so we depend on the
/// compile step only.  On other platforms, we run the test artifact.
pub fn addTestRunStep(
    b: *std.Build,
    tests: *std.Build.Step.Compile,
    ctx: DarwinCtx,
) *std.Build.Step {
    if (ctx.is_blocked) {
        return &tests.step;
    }
    const run = b.addRunArtifact(tests);
    run.skip_foreign_checks = true;
    return &run.step;
}

// ── Host Script Step ─────────────────────────────────────────────────────

/// Build and run a Zig-based host script (tools/scripts/*.zig).
/// On Darwin degraded, compiles to .o and relinks via Apple ld.
pub fn addHostScriptStep(
    b: *std.Build,
    name: []const u8,
    source: []const u8,
    target: std.Build.ResolvedTarget,
    optimize: std.builtin.OptimizeMode,
    args: []const []const u8,
    deps: []const struct { name: []const u8, module: *std.Build.Module },
    ctx: DarwinCtx,
) *std.Build.Step {
    const mod = b.createModule(.{
        .root_source_file = b.path(source),
        .target = target,
        .optimize = optimize,
        .link_libc = true,
    });
    for (deps) |dep| mod.addImport(dep.name, dep.module);

    if (ctx.is_blocked) {
        const obj = b.addObject(.{ .name = name, .root_module = mod });
        obj.use_llvm = true;
        const run = link.darwinRelink(b, obj, b.fmt("{s}_linked", .{name}), ctx.compiler_rt);
        for (args) |arg| run.addArg(arg);
        return &run.step;
    }

    const exe = b.addExecutable(.{ .name = name, .root_module = mod });
    const run = b.addRunArtifact(exe);
    for (args) |arg| run.addArg(arg);
    return &run.step;
}

// ── Target Resolution ────────────────────────────────────────────────────

const host_is_affected_darwin = builtin.os.tag == .macos and builtin.os.version_range.semver.min.major >= 26;

/// Resolve the native build target, clamping macOS version when needed.
///
/// Zig 0.16-dev's linker does not support macOS 26+ (Tahoe).  When the
/// build host reports a version the toolchain cannot handle, we clamp the
/// deployment target to macOS 15.0 so the linker can resolve
/// libSystem.B.dylib and friends from the installed SDK.  An explicit
/// `-Dtarget=` from the user is never overridden.
pub fn resolveNativeTarget(b: *std.Build) std.Build.ResolvedTarget {
    var query = b.standardTargetOptionsQueryOnly(.{});

    if (query.os_tag == null and builtin.os.tag == .macos) {
        const native_ver = builtin.os.version_range.semver;
        if (native_ver.min.major >= 26) {
            const clamped: std.Target.Query.OsVersion = .{
                .semver = .{ .major = 15, .minor = 0, .patch = 0 },
            };
            if (query.os_version_min == null) query.os_version_min = clamped;
            if (query.os_version_max == null) query.os_version_max = clamped;
        }
    }

    return b.resolveTargetQuery(query);
}

fn useDarwinDegradedMode(b: *std.Build) bool {
    if (!host_is_affected_darwin) return false;
    return !usesKnownGoodHostZig(b);
}

fn usesKnownGoodHostZig(b: *std.Build) bool {
    const expected_version = readZigVersion();
    const zig_exe = b.graph.zig_exe;

    if (b.graph.environ_map.get("ABI_HOST_ZIG")) |explicit_host_zig| {
        if (std.mem.eql(u8, zig_exe, explicit_host_zig)) return true;
    }

    const cache_root = b.graph.environ_map.get("ABI_HOST_ZIG_CACHE_DIR") orelse blk: {
        const home = b.graph.environ_map.get("HOME") orelse return false;
        break :blk b.fmt("{s}/.cache/abi-host-zig", .{home});
    };
    const canonical_host_zig = b.fmt("{s}/{s}/bin/zig", .{ cache_root, expected_version });
    return std.mem.eql(u8, zig_exe, canonical_host_zig);
}

/// Read the pinned Zig version from `.zigversion` at comptime.
/// Returns the version string with trailing whitespace stripped.
fn readZigVersion() []const u8 {
    // Comptime: embed file and strip trailing whitespace in one step.
    return comptime blk: {
        const raw: []const u8 = @embedFile("../.zigversion");
        if (raw.len == 0) @compileError(".zigversion is empty — expected a Zig version string");
        var end: usize = raw.len;
        while (end > 0) {
            if (raw[end - 1] == '\n' or raw[end - 1] == '\r' or
                raw[end - 1] == ' ' or raw[end - 1] == '\t')
            {
                end -= 1;
            } else break;
        }
        if (end == 0) @compileError(".zigversion contains only whitespace");
        break :blk raw[0..end];
    };
}
