const std = @import("std");
const common = @import("../src/tools/cli/common.zig");
const router = @import("../src/tools/cli/router.zig");
const registry = @import("../src/tools/cli/registry.zig");
const gpu = @import("../src/tools/cli/gpu.zig");
const db = @import("../src/tools/cli/db.zig");
const config = @import("../src/tools/cli/config.zig");
const neural = @import("../src/tools/cli/neural.zig");
const simd = @import("../src/tools/cli/simd.zig");
const plugin = @import("../src/tools/cli/plugin.zig");
const server = @import("../src/tools/cli/server.zig");
const weather = @import("../src/tools/cli/weather.zig");
const llm = @import("../src/tools/cli/llm.zig");
const chat = @import("../src/tools/cli/chat.zig");

fn makeArgs(list: []const []const u8) ![][:0]u8 {
    var items = try std.testing.allocator.alloc([:0]u8, list.len);
    for (list, 0..) |item, idx| {
        items[idx] = try std.testing.allocator.dupeZ(u8, item);
    }
    return items;
}

fn freeArgs(args: [][:0]u8) void {
    for (args) |arg| {
        const len = std.mem.len(arg);
        std.testing.allocator.free(arg[0 .. len + 1]);
    }
    std.testing.allocator.free(args);
}

fn makeCtx() common.Context {
    return .{ .allocator = std.testing.allocator };
}

fn runCommand(command: anytype, list: []const []const u8) !void {
    var args = try makeArgs(list);
    defer freeArgs(args);
    var ctx = makeCtx();
    try command.run(&ctx, args);
}

test "router prints global help" {
    var args = try makeArgs(&.{ "abi", "--help" });
    defer freeArgs(args);
    var ctx = makeCtx();
    try router.run(&ctx, args);
}

test "registry finds commands" {
    const cmds = registry.all();
    try std.testing.expect(cmds.len > 0);
    try std.testing.expect(registry.find("gpu") != null);
    try std.testing.expect(registry.find("wdbx") != null);
}

test "gpu command displays help" {
    try runCommand(gpu, &.{ "abi", "gpu" });
}

test "db command displays help" {
    try runCommand(db, &.{ "abi", "db" });
}

test "config command handles help token" {
    try runCommand(config, &.{ "abi", "config", "--help" });
}

test "neural command displays usage" {
    try runCommand(neural, &.{ "abi", "neural" });
}

test "simd command displays usage" {
    try runCommand(simd, &.{ "abi", "simd" });
}

test "plugin command displays usage" {
    try runCommand(plugin, &.{ "abi", "plugin" });
}

test "server command displays usage" {
    try runCommand(server, &.{ "abi", "server" });
}

test "weather command displays usage" {
    try runCommand(weather, &.{ "abi", "weather" });
}

test "llm command displays usage" {
    try runCommand(llm, &.{ "abi", "llm" });
}

test "chat command displays usage" {
    try runCommand(chat, &.{ "abi", "chat" });
}
