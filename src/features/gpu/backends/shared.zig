//! Shared helpers for GPU backend dynamic loading and target checks.
const std = @import("std");
const builtin = @import("builtin");

pub fn canUseDynLib() bool {
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
    var lib = openFirst(libs) orelse return false;
    lib.close();
    return true;
}

pub fn openFirst(libs: []const []const u8) ?std.DynLib {
    for (libs) |name| {
        if (std.DynLib.open(name)) |lib| {
            return lib;
        } else |_| {}
    }
    return null;
}
