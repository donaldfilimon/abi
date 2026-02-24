//! Shared helpers for GPU backend dynamic loading and target checks.
const std = @import("std");
const builtin = @import("builtin");

const dynlib_supported = switch (builtin.target.os.tag) {
    .linux,
    .driverkit,
    .ios,
    .maccatalyst,
    .macos,
    .tvos,
    .visionos,
    .watchos,
    .freebsd,
    .netbsd,
    .openbsd,
    .dragonfly,
    .illumos,
    => true,
    else => false,
};

pub const dynlibSupported = dynlib_supported;

pub fn canUseDynLib() bool {
    if (!dynlib_supported) return false;
    return switch (builtin.target.cpu.arch) {
        .wasm32, .wasm64 => false,
        else => true,
    };
}

pub fn isWebTarget() bool {
    return switch (builtin.target.cpu.arch) {
        .wasm32, .wasm64 => true,
        else => false,
    };
}

pub fn tryLoadAny(libs: []const []const u8) bool {
    if (comptime !dynlib_supported) return false;
    if (!canUseDynLib()) return false;
    if (openFirst(libs)) |lib| {
        var mut_lib = lib;
        mut_lib.close();
        return true;
    }
    return false;
}

pub fn openFirst(libs: []const []const u8) ?std.DynLib {
    if (!canUseDynLib()) return null;
    if (dynlib_supported) {
        for (libs) |name| {
            const lib = std.DynLib.open(name) catch continue;
            return lib;
        }
    }
    return null;
}

test {
    std.testing.refAllDecls(@This());
}
