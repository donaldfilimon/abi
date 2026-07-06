//! CLI test aggregator — gives the `src/cli` framework a real test artifact.
//!
//! The inline tests in `arg.zig`, `registry.zig`, and `dispatch.zig` are not
//! reachable from any other test target (the CLI ships as an executable, not a
//! module), so they would never run. This aggregator pulls them in via
//! `refAllDecls` and adds focused coverage that exercises the *actual* argument
//! specs that `dispatch` walks, proving the migrated `complete`/`train` parsing
//! matches the historical hand-written parser.

const std = @import("std");
const registry = @import("cli/registry.zig");
const arg = @import("cli/arg.zig");
const dispatch = @import("cli/dispatch.zig");

fn specFor(comptime name: []const u8) []const arg.Arg {
    return comptime blk: {
        for (registry.commands) |command| {
            if (std.mem.eql(u8, command.name, name)) break :blk command.args;
        }
        @compileError("no registry command named '" ++ name ++ "'");
    };
}

test "registry `complete` spec parses like the legacy parseCompleteArgs" {
    const spec = specFor("complete");

    // Order-independent flags + value + positional.
    var all = try arg.parse(std.testing.allocator, spec, &.{ "abi", "complete", "--model", "fable-5", "--live", "--learn", "hi" });
    defer all.deinit();
    try std.testing.expect(all.flag("live"));
    try std.testing.expect(all.flag("learn"));
    try std.testing.expect(!all.flag("confirm"));
    try std.testing.expectEqualStrings("fable-5", all.value("model").?);
    try std.testing.expectEqualStrings("hi", all.value("input").?);

    // Plain positional, flags default false.
    var plain = try arg.parse(std.testing.allocator, spec, &.{ "abi", "complete", "hello" });
    defer plain.deinit();
    try std.testing.expect(!plain.flag("learn"));
    try std.testing.expect(plain.value("model") == null);
    try std.testing.expectEqualStrings("hello", plain.value("input").?);

    // Missing input and a dangling `--model` are usage errors (→ exit 2).
    try std.testing.expectError(error.Usage, arg.parse(std.testing.allocator, spec, &.{ "abi", "complete", "--learn" }));
    try std.testing.expectError(error.Usage, arg.parse(std.testing.allocator, spec, &.{ "abi", "complete", "--model" }));
}

test "registry `train` spec requires exactly one positional" {
    const spec = specFor("train");

    var ok = try arg.parse(std.testing.allocator, spec, &.{ "abi", "train", "hello" });
    defer ok.deinit();
    try std.testing.expectEqualStrings("hello", ok.value("input").?);

    try std.testing.expectError(error.Usage, arg.parse(std.testing.allocator, spec, &.{ "abi", "train" }));
    try std.testing.expectError(error.Usage, arg.parse(std.testing.allocator, spec, &.{ "abi", "train", "a", "b" }));
}

test "argument-free commands reject stray tokens" {
    const spec = specFor("backends");
    var ok = try arg.parse(std.testing.allocator, spec, &.{ "abi", "backends" });
    defer ok.deinit();
    try std.testing.expectError(error.Usage, arg.parse(std.testing.allocator, spec, &.{ "abi", "backends", "extra" }));
}

test "registry `scheduler` spec requires status subcommand" {
    const spec = specFor("scheduler");

    var ok = try arg.parse(std.testing.allocator, spec, &.{ "abi", "scheduler", "status" });
    defer ok.deinit();
    try std.testing.expectEqualStrings("status", ok.value("status").?);

    try std.testing.expectError(error.Usage, arg.parse(std.testing.allocator, spec, &.{ "abi", "scheduler" }));
    try std.testing.expectError(error.Usage, arg.parse(std.testing.allocator, spec, &.{ "abi", "scheduler", "status", "extra" }));
}

test {
    std.testing.refAllDecls(@This());
    std.testing.refAllDecls(dispatch);
}
