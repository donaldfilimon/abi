//! Build-system compatibility layer for Zig version detection and
//! graceful degradation on older 0.16-dev toolchains.
//!
//! Centralizes all `b.graph.*` access behind comptime-gated helpers so
//! that `build.zig` and its sub-modules compile cleanly even when the
//! `graph` field does not exist (older 0.16-dev builds).
//!
//! Pattern: each helper uses `if (comptime has_graph)` to gate the new
//! API; the dead branch is never analyzed by the compiler, so references
//! to `std.Io`, `b.graph.io`, etc. are safe in the `true` branch even
//! when building with a toolchain that lacks them.

const std = @import("std");

// ── Version detection ─────────────────────────────────────────────────

/// Comptime flag: `true` when `std.Build.Graph` has the full IO-based API
/// (`io`, `environ_map`, `zig_exe` — introduced around dev.2000).
/// Older 0.16-dev builds have a `Graph` type but without these fields.
/// Dead branches behind this flag are never analyzed, so references to
/// `std.Io`, `b.graph.io`, etc. in the `true` branch are safe when
/// building with an older toolchain.
pub const has_graph = @hasField(std.Build, "graph") and @hasField(std.Build.Graph, "environ_map");

/// Pinned Zig version from `.zigversion`, whitespace-stripped at comptime.
/// Shared by `darwin.zig` (replaces its `readZigVersion()`) and the
/// version guard in `build.zig`.
pub const pinned_version: []const u8 = blk: {
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

// ── Environment access ────────────────────────────────────────────────

/// Retrieve an environment variable.  Uses `b.graph.environ_map` when
/// available; returns `null` on older toolchains.
///
/// Accepts `[:0]const u8` so callers can pass string literals directly.
/// The sentinel is dropped when forwarding to `environ_map.get()`.
pub fn getEnv(b: *std.Build, key: [:0]const u8) ?[]const u8 {
    if (comptime has_graph) {
        return b.graph.environ_map.get(key);
    }
    return null;
}

/// Return the Zig executable path from the build graph, or `null` on
/// older toolchains where `b.graph.zig_exe` is unavailable.
pub fn getZigExe(b: *std.Build) ?[]const u8 {
    if (comptime has_graph) {
        return b.graph.zig_exe;
    }
    return null;
}

// ── Filesystem probes ─────────────────────────────────────────────────

/// Check whether a relative path exists within the build root.
/// Returns `false` on older toolchains (conservative default).
pub fn pathExists(b: *std.Build, path: []const u8) bool {
    if (comptime has_graph) {
        b.build_root.handle.access(b.graph.io, path, .{}) catch return false;
        return true;
    }
    return false;
}

/// Check whether an absolute directory path exists.
/// Uses `std.Io.Dir` when available; returns `false` otherwise.
pub fn dirExists(b: *std.Build, path: []const u8) bool {
    if (comptime has_graph) {
        var dir = std.Io.Dir.openDirAbsolute(b.graph.io, path, .{}) catch return false;
        dir.close(b.graph.io);
        return true;
    }
    return false;
}

/// Run a command and return `true` if it exits with status 0.
/// Returns `false` on older toolchains where `std.process.spawn`
/// requires `std.Io`.
pub fn commandSucceeds(b: *std.Build, argv: []const []const u8) bool {
    if (comptime has_graph) {
        var child = std.process.spawn(b.graph.io, .{
            .argv = argv,
            .stdin = .ignore,
            .stdout = .ignore,
            .stderr = .ignore,
        }) catch return false;

        const term = child.wait(b.graph.io) catch return false;
        return switch (term) {
            .exited => |code| code == 0,
            else => false,
        };
    }
    return false;
}

// ── Version guard ─────────────────────────────────────────────────────

/// Print a clear diagnostic when the active Zig toolchain is too old.
/// On toolchains without `b.graph`, emits a warning directing the user
/// to bootstrap or download the pinned version.  Does NOT stop the
/// build — the caller should register format-only steps and return.
pub fn checkVersion(_: *std.Build) void {
    if (comptime !has_graph) {
        std.log.warn(
            \\
            \\  ╔══════════════════════════════════════════════════════════╗
            \\  ║  Zig too old — full build system unavailable.           ║
            \\  ║  Required: {s}
            \\  ║  Your Zig lacks the b.graph API (added ~dev.2000).     ║
            \\  ║                                                        ║
            \\  ║  Available steps: lint, fix  (format checks only)      ║
            \\  ║                                                        ║
            \\  ║  To unlock all build steps:                            ║
            \\  ║    1. Download pinned Zig from ziglang.org/builds      ║
            \\  ║    2. Run: ./tools/scripts/bootstrap_host_zig.sh       ║
            \\  ║    3. Run: ./build.sh (auto-resolves correct Zig)      ║
            \\  ╚══════════════════════════════════════════════════════════╝
            \\
        , .{pinned_version});
    }
}
