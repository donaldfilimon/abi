//! Comptime command protocol.
//!
//! Each CLI command declares `pub const meta: command.Meta` and `pub fn run`.
//! The registry, help, completions, and dispatch are all auto-derived at
//! comptime from these declarations — eliminating the manual three-layer
//! bridge (catalog → descriptor → wiring) that previously existed.

const std = @import("std");
const types = @import("framework/types.zig");

pub const CommandDescriptor = types.CommandDescriptor;
pub const CommandHandler = types.CommandHandler;
pub const CommandForward = types.CommandForward;
pub const CommandIoMode = types.CommandIoMode;
pub const CommandKind = types.CommandKind;

/// Child command metadata for group commands (e.g. llm children, ui children).
pub const ChildMeta = struct {
    name: []const u8,
    description: []const u8,
    handler: CommandHandler,
};

/// Declarative command metadata. Every registered command module exports
/// `pub const meta: command.Meta` with these fields.
pub const Meta = struct {
    name: []const u8,
    description: []const u8,
    aliases: []const []const u8 = &.{},
    subcommands: []const []const u8 = &.{},
    io_mode: CommandIoMode = .basic,
    kind: CommandKind = .action,
    forward: ?CommandForward = null,
    children: []const ChildMeta = &.{},
};

/// Comptime validation: compile-error if a module is missing the required
/// command protocol exports (`meta` and `run`).
pub fn validate(comptime Module: type) void {
    if (!@hasDecl(Module, "meta")) {
        @compileError("Command module missing `pub const meta: command.Meta`");
    }
    if (!@hasDecl(Module, "run")) {
        @compileError("Command module missing `pub fn run`");
    }
}

/// Convert ChildMeta array to CommandDescriptor array at comptime.
/// Uses nested struct holder pattern for stable static addresses.
fn convertChildren(comptime children: []const ChildMeta) []const CommandDescriptor {
    if (children.len == 0) return &.{};
    const Holder = struct {
        const data: [children.len]CommandDescriptor = blk: {
            var result: [children.len]CommandDescriptor = undefined;
            for (children, 0..) |child, i| {
                result[i] = .{
                    .name = child.name,
                    .description = child.description,
                    .handler = child.handler,
                };
            }
            break :blk result;
        };
    };
    return &Holder.data;
}

/// Convert a command module's Meta + run function into a CommandDescriptor.
/// Bridges the declarative `Meta` to the runtime `CommandDescriptor` used
/// by the router, help, and completion systems.
pub fn toDescriptor(comptime Module: type) CommandDescriptor {
    comptime validate(Module);
    const m: Meta = Module.meta;
    return .{
        .name = m.name,
        .description = m.description,
        .aliases = m.aliases,
        .subcommands = m.subcommands,
        .children = convertChildren(m.children),
        .kind = m.kind,
        .handler = switch (m.io_mode) {
            .basic => .{ .basic = Module.run },
            .io => .{ .io = Module.run },
        },
        .forward = m.forward,
    };
}

// ── Tests ────────────────────────────────────────────────────────────

/// Minimal valid command module (basic handler, no extras).
const TestBasicModule = struct {
    pub const meta: Meta = .{
        .name = "test-basic",
        .description = "A basic test command",
    };
    pub fn run(_: std.mem.Allocator, _: []const [:0]const u8) !void {}
};

/// IO command module with aliases, subcommands, and forward.
const TestIoModule = struct {
    pub const meta: Meta = .{
        .name = "test-io",
        .description = "An IO test command",
        .aliases = &.{ "tio", "io-test" },
        .subcommands = &.{ "sub1", "sub2", "help" },
        .io_mode = .io,
        .kind = .group,
        .forward = .{ .target = "other-cmd", .prepend_args = &.{"launch"}, .warning = "deprecated" },
    };
    pub fn run(_: std.mem.Allocator, _: std.Io, _: []const [:0]const u8) !void {}
};

fn stubChild1(_: std.mem.Allocator, _: []const [:0]const u8) !void {}
fn stubChild2(_: std.mem.Allocator, _: []const [:0]const u8) !void {}

/// Group command with children.
const TestGroupModule = struct {
    pub const meta: Meta = .{
        .name = "test-group",
        .description = "A group command",
        .kind = .group,
        .children = &.{
            .{ .name = "child-a", .description = "First child", .handler = .{ .basic = stubChild1 } },
            .{ .name = "child-b", .description = "Second child", .handler = .{ .basic = stubChild2 } },
        },
    };
    pub fn run(_: std.mem.Allocator, _: []const [:0]const u8) !void {}
};

test "toDescriptor: basic module produces correct descriptor" {
    const d = comptime toDescriptor(TestBasicModule);
    try std.testing.expectEqualStrings("test-basic", d.name);
    try std.testing.expectEqualStrings("A basic test command", d.description);
    try std.testing.expectEqual(@as(usize, 0), d.aliases.len);
    try std.testing.expectEqual(@as(usize, 0), d.subcommands.len);
    try std.testing.expectEqual(@as(usize, 0), d.children.len);
    try std.testing.expectEqual(CommandKind.action, d.kind);
    try std.testing.expect(d.forward == null);
    // Handler should be .basic variant
    switch (d.handler) {
        .basic => {},
        .io => return error.TestUnexpectedResult,
    }
}

test "toDescriptor: io module preserves aliases, subcommands, forward" {
    const d = comptime toDescriptor(TestIoModule);
    try std.testing.expectEqualStrings("test-io", d.name);
    try std.testing.expectEqual(@as(usize, 2), d.aliases.len);
    try std.testing.expectEqualStrings("tio", d.aliases[0]);
    try std.testing.expectEqualStrings("io-test", d.aliases[1]);
    try std.testing.expectEqual(@as(usize, 3), d.subcommands.len);
    try std.testing.expectEqual(CommandKind.group, d.kind);
    // Handler should be .io variant
    switch (d.handler) {
        .io => {},
        .basic => return error.TestUnexpectedResult,
    }
    // Forward
    try std.testing.expect(d.forward != null);
    try std.testing.expectEqualStrings("other-cmd", d.forward.?.target);
    try std.testing.expectEqual(@as(usize, 1), d.forward.?.prepend_args.len);
    try std.testing.expectEqualStrings("launch", d.forward.?.prepend_args[0]);
    try std.testing.expectEqualStrings("deprecated", d.forward.?.warning.?);
}

test "toDescriptor: group module converts children correctly" {
    const d = comptime toDescriptor(TestGroupModule);
    try std.testing.expectEqual(@as(usize, 2), d.children.len);
    try std.testing.expectEqualStrings("child-a", d.children[0].name);
    try std.testing.expectEqualStrings("First child", d.children[0].description);
    try std.testing.expectEqualStrings("child-b", d.children[1].name);
    try std.testing.expectEqualStrings("Second child", d.children[1].description);
    // Verify child handlers are .basic variant
    switch (d.children[0].handler) {
        .basic => {},
        .io => return error.TestUnexpectedResult,
    }
}

test "convertChildren: empty input returns empty slice" {
    const empty = comptime convertChildren(&.{});
    try std.testing.expectEqual(@as(usize, 0), empty.len);
}

test "validate: accepts valid modules" {
    // These should not compile-error.
    comptime validate(TestBasicModule);
    comptime validate(TestIoModule);
    comptime validate(TestGroupModule);
}

test "Meta defaults: verify zero-value defaults" {
    const m: Meta = .{ .name = "x", .description = "y" };
    try std.testing.expectEqual(@as(usize, 0), m.aliases.len);
    try std.testing.expectEqual(@as(usize, 0), m.subcommands.len);
    try std.testing.expectEqual(@as(usize, 0), m.children.len);
    try std.testing.expectEqual(CommandIoMode.basic, m.io_mode);
    try std.testing.expectEqual(CommandKind.action, m.kind);
    try std.testing.expect(m.forward == null);
}

test {
    std.testing.refAllDecls(@This());
}
