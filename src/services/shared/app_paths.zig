//! Cross-platform ABI application path resolution.
//!
//! Resolves the primary per-OS config root and builds paths under it.

const std = @import("std");
const builtin = @import("builtin");

pub const PathError = error{NoHomeDirectory} || std.mem.Allocator.Error;

pub const EnvValues = struct {
    appdata: ?[]const u8 = null,
    localappdata: ?[]const u8 = null,
    userprofile: ?[]const u8 = null,
    home: ?[]const u8 = null,
    xdg_config_home: ?[]const u8 = null,
};

pub fn resolvePrimaryRoot(allocator: std.mem.Allocator) PathError![]u8 {
    return resolvePrimaryRootFor(allocator, builtin.os.tag, readRuntimeEnv());
}

pub fn resolvePath(allocator: std.mem.Allocator, relative: []const u8) PathError![]u8 {
    return resolvePathFor(allocator, builtin.os.tag, readRuntimeEnv(), relative);
}

pub fn resolvePrimaryRootFor(
    allocator: std.mem.Allocator,
    os_tag: std.Target.Os.Tag,
    env: EnvValues,
) PathError![]u8 {
    switch (os_tag) {
        .windows => {
            if (env.appdata) |base| {
                return std.fs.path.join(allocator, &.{ base, "abi" });
            }
            if (env.localappdata) |base| {
                return std.fs.path.join(allocator, &.{ base, "abi" });
            }
            const userprofile = env.userprofile orelse return error.NoHomeDirectory;
            return std.fs.path.join(allocator, &.{ userprofile, ".abi" });
        },
        .macos => {
            const home = env.home orelse return error.NoHomeDirectory;
            return std.fs.path.join(allocator, &.{ home, "Library", "Application Support", "abi" });
        },
        else => {
            if (env.xdg_config_home) |xdg| {
                return std.fs.path.join(allocator, &.{ xdg, "abi" });
            }
            const home = env.home orelse return error.NoHomeDirectory;
            return std.fs.path.join(allocator, &.{ home, ".config", "abi" });
        },
    }
}

pub fn resolvePathFor(
    allocator: std.mem.Allocator,
    os_tag: std.Target.Os.Tag,
    env: EnvValues,
    relative: []const u8,
) PathError![]u8 {
    const primary_root = try resolvePrimaryRootFor(allocator, os_tag, env);
    defer allocator.free(primary_root);
    return std.fs.path.join(allocator, &.{ primary_root, relative });
}

test "resolvePathFor tolerates missing legacy HOME on linux when XDG config exists" {
    const allocator = std.testing.allocator;
    const env: EnvValues = .{
        .xdg_config_home = "/home/tester/.xdg",
    };
    const path = try resolvePathFor(allocator, .linux, env, "config.json");
    defer allocator.free(path);

    const expected = try std.fs.path.join(allocator, &.{ "/home/tester/.xdg", "abi", "config.json" });
    defer allocator.free(expected);

    try std.testing.expectEqualStrings(expected, path);
}

fn readRuntimeEnv() EnvValues {
    return .{
        .appdata = getenvSlice("APPDATA"),
        .localappdata = getenvSlice("LOCALAPPDATA"),
        .userprofile = getenvSlice("USERPROFILE"),
        .home = getenvSlice("HOME"),
        .xdg_config_home = getenvSlice("XDG_CONFIG_HOME"),
    };
}

fn getenvSlice(name: [:0]const u8) ?[]const u8 {
    if (comptime builtin.target.os.tag == .freestanding or
        builtin.target.cpu.arch == .wasm32 or
        builtin.target.cpu.arch == .wasm64)
    {
        return null;
    }

    const value_ptr = std.c.getenv(name.ptr);
    if (value_ptr) |ptr| {
        return std.mem.span(ptr);
    }
    return null;
}

test "resolvePrimaryRootFor windows prefers APPDATA" {
    const allocator = std.testing.allocator;
    const env: EnvValues = .{
        .appdata = "/win/appdata",
        .localappdata = "/win/local",
        .userprofile = "/win/user",
    };
    const path = try resolvePrimaryRootFor(allocator, .windows, env);
    defer allocator.free(path);
    const expected = try std.fs.path.join(allocator, &.{ "/win/appdata", "abi" });
    defer allocator.free(expected);
    try std.testing.expectEqualStrings(expected, path);
}

test "resolvePrimaryRootFor windows falls back to LOCALAPPDATA" {
    const allocator = std.testing.allocator;
    const env: EnvValues = .{
        .localappdata = "/win/local",
        .userprofile = "/win/user",
    };
    const path = try resolvePrimaryRootFor(allocator, .windows, env);
    defer allocator.free(path);
    const expected = try std.fs.path.join(allocator, &.{ "/win/local", "abi" });
    defer allocator.free(expected);
    try std.testing.expectEqualStrings(expected, path);
}

test "resolvePrimaryRootFor windows falls back to USERPROFILE dot abi" {
    const allocator = std.testing.allocator;
    const env: EnvValues = .{ .userprofile = "/win/user" };
    const path = try resolvePrimaryRootFor(allocator, .windows, env);
    defer allocator.free(path);
    const expected = try std.fs.path.join(allocator, &.{ "/win/user", ".abi" });
    defer allocator.free(expected);
    try std.testing.expectEqualStrings(expected, path);
}

test "resolvePrimaryRootFor windows requires USERPROFILE when app dirs missing" {
    const allocator = std.testing.allocator;
    const env: EnvValues = .{};
    try std.testing.expectError(error.NoHomeDirectory, resolvePrimaryRootFor(allocator, .windows, env));
}

test "resolvePrimaryRootFor macOS uses Application Support" {
    const allocator = std.testing.allocator;
    const env: EnvValues = .{ .home = "/Users/tester" };
    const path = try resolvePrimaryRootFor(allocator, .macos, env);
    defer allocator.free(path);
    const expected = try std.fs.path.join(allocator, &.{ "/Users/tester", "Library", "Application Support", "abi" });
    defer allocator.free(expected);
    try std.testing.expectEqualStrings(expected, path);
}

test "resolvePrimaryRootFor linux uses XDG_CONFIG_HOME when set" {
    const allocator = std.testing.allocator;
    const env: EnvValues = .{
        .home = "/home/tester",
        .xdg_config_home = "/home/tester/.xdg",
    };
    const path = try resolvePrimaryRootFor(allocator, .linux, env);
    defer allocator.free(path);
    const expected = try std.fs.path.join(allocator, &.{ "/home/tester/.xdg", "abi" });
    defer allocator.free(expected);
    try std.testing.expectEqualStrings(expected, path);
}

test "resolvePrimaryRootFor linux falls back to home .config" {
    const allocator = std.testing.allocator;
    const env: EnvValues = .{ .home = "/home/tester" };
    const path = try resolvePrimaryRootFor(allocator, .linux, env);
    defer allocator.free(path);
    const expected = try std.fs.path.join(allocator, &.{ "/home/tester", ".config", "abi" });
    defer allocator.free(expected);
    try std.testing.expectEqualStrings(expected, path);
}

test "resolvePrimaryRootFor linux requires HOME when XDG is unset" {
    const allocator = std.testing.allocator;
    const env: EnvValues = .{};
    try std.testing.expectError(error.NoHomeDirectory, resolvePrimaryRootFor(allocator, .linux, env));
}

test {
    std.testing.refAllDecls(@This());
}
