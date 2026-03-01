//! Comptime command protocol.
//!
//! Each CLI command declares `pub const meta: command.Meta` and `pub fn run`.
//! The registry, help, completions, and dispatch are all auto-derived at
//! comptime from these declarations — eliminating the manual three-layer
//! bridge (catalog → descriptor → wiring) that previously existed.

const std = @import("std");
const context_mod = @import("framework/context.zig");
const types = @import("framework/types.zig");
const args_mod = @import("utils/args.zig");

pub const CommandDescriptor = types.CommandDescriptor;
pub const CommandHandler = types.CommandHandler;
pub const CommandForward = types.CommandForward;
pub const CommandKind = types.CommandKind;
pub const OptionInfo = types.OptionInfo;
pub const UiMeta = types.UiMeta;
pub const UiCategory = types.UiCategory;
pub const Visibility = types.Visibility;
pub const RiskLevel = types.RiskLevel;

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
    kind: CommandKind = .action,
    forward: ?CommandForward = null,
    children: []const ChildMeta = &.{},
    options: []const OptionInfo = &.{},
    ui: UiMeta = .{},
    visibility: Visibility = .public,
    risk: RiskLevel = .safe,
    source_id: ?[]const u8 = null,
    default_subcommand: ?[:0]const u8 = null,
    middleware_tags: []const []const u8 = &.{},
};

pub const AllocatorArgHandler = *const fn (allocator: std.mem.Allocator, args: []const [:0]const u8) anyerror!void;
pub const ArgParserHandler = *const fn (allocator: std.mem.Allocator, parser: *args_mod.ArgParser) anyerror!void;
pub const Middleware = *const fn (
    ctx: *const context_mod.CommandContext,
    args: []const [:0]const u8,
    next: CommandHandler,
) anyerror!void;

pub fn allocatorHandler(comptime handler: AllocatorArgHandler) CommandHandler {
    return struct {
        fn call(ctx: *const context_mod.CommandContext, args: []const [:0]const u8) anyerror!void {
            try handler(ctx.allocator, args);
        }
    }.call;
}

pub fn parserHandler(comptime handler: ArgParserHandler) CommandHandler {
    return struct {
        fn call(ctx: *const context_mod.CommandContext, args: []const [:0]const u8) anyerror!void {
            var parser = args_mod.ArgParser.init(ctx.allocator, args);
            try handler(ctx.allocator, &parser);
        }
    }.call;
}

pub fn withMiddleware(
    comptime next: CommandHandler,
    comptime middleware: Middleware,
) CommandHandler {
    return struct {
        fn call(ctx: *const context_mod.CommandContext, args: []const [:0]const u8) anyerror!void {
            try middleware(ctx, args, next);
        }
    }.call;
}

pub fn withMiddlewares(
    comptime handler: CommandHandler,
    comptime middlewares: []const Middleware,
) CommandHandler {
    if (middlewares.len == 0) return handler;
    var current = handler;
    inline for (middlewares) |mw| {
        current = withMiddleware(current, mw);
    }
    return current;
}

/// Comptime validation: compile-error if a module is missing the required
/// command protocol exports (`meta` and `run`).
pub fn validate(comptime Module: type) void {
    if (!@hasDecl(Module, "meta")) {
        @compileError("Command module missing `pub const meta: command.Meta`");
    }
    if (!@hasDecl(Module, "run")) {
        @compileError("Command module missing `pub fn run`");
    }

    const m: Meta = Module.meta;

    inline for (m.aliases, 0..) |alias, i| {
        if (alias.len == 0) {
            @compileError(std.fmt.comptimePrint(
                "Command '{s}' has an empty alias at index {d}",
                .{ m.name, i },
            ));
        }
        inline for (m.aliases[0..i]) |prev| {
            if (std.mem.eql(u8, prev, alias)) {
                @compileError(std.fmt.comptimePrint(
                    "Command '{s}' has duplicate alias '{s}'",
                    .{ m.name, alias },
                ));
            }
        }
        if (std.mem.eql(u8, alias, m.name)) {
            @compileError(std.fmt.comptimePrint(
                "Command '{s}' alias '{s}' duplicates the command name",
                .{ m.name, alias },
            ));
        }
    }

    if (m.ui.shortcut) |shortcut| {
        if (shortcut < 1 or shortcut > 9) {
            @compileError(std.fmt.comptimePrint(
                "Command '{s}' UI shortcut {d} is outside [1..9]",
                .{ m.name, shortcut },
            ));
        }
    }

    if (m.visibility == .hidden and m.ui.include_in_launcher) {
        @compileError(std.fmt.comptimePrint(
            "Command '{s}' is hidden but include_in_launcher=true",
            .{m.name},
        ));
    }

    if (m.ui.include_in_dashboard and !m.ui.include_in_launcher) {
        @compileError(std.fmt.comptimePrint(
            "Command '{s}' has include_in_dashboard=true but include_in_launcher=false",
            .{m.name},
        ));
    }

    if (m.default_subcommand) |default_sub| {
        var found = false;
        inline for (m.subcommands) |sub| {
            if (std.mem.eql(u8, sub, default_sub)) {
                found = true;
                break;
            }
        }
        if (!found) {
            inline for (m.children) |child| {
                if (std.mem.eql(u8, child.name, default_sub)) {
                    found = true;
                    break;
                }
            }
        }
        if (!found) {
            @compileError(std.fmt.comptimePrint(
                "Command '{s}' default_subcommand '{s}' not found in subcommands/children",
                .{ m.name, default_sub },
            ));
        }
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

/// Derive subcommand names from children metadata at comptime.
/// Only used when `meta.subcommands` is empty but `meta.children` is non-empty.
fn deriveSubcommandNames(comptime children: []const ChildMeta) []const []const u8 {
    if (children.len == 0) return &.{};
    const Holder = struct {
        const data: [children.len][]const u8 = blk: {
            var result: [children.len][]const u8 = undefined;
            for (children, 0..) |child, i| {
                result[i] = child.name;
            }
            break :blk result;
        };
    };
    return &Holder.data;
}

pub fn subcommandNames(comptime m: Meta) []const []const u8 {
    return if (m.subcommands.len == 0 and m.children.len > 0)
        deriveSubcommandNames(m.children)
    else
        m.subcommands;
}

pub fn suggestSubcommand(comptime m: Meta, raw: []const u8) ?[]const u8 {
    return args_mod.suggestCommand(raw, subcommandNames(m));
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
        .subcommands = if (m.subcommands.len == 0 and m.children.len > 0)
            deriveSubcommandNames(m.children)
        else
            m.subcommands,
        .children = convertChildren(m.children),
        .kind = m.kind,
        .handler = Module.run,
        .forward = m.forward,
        .options = m.options,
        .ui = m.ui,
        .visibility = m.visibility,
        .risk = m.risk,
        .source_id = if (m.source_id) |source_id| source_id else m.name,
        .default_subcommand = m.default_subcommand,
        .middleware_tags = m.middleware_tags,
    };
}

// ── Tests ────────────────────────────────────────────────────────────

/// Minimal valid command module (basic handler, no extras).
const TestBasicModule = struct {
    pub const meta: Meta = .{
        .name = "test-basic",
        .description = "A basic test command",
    };
    pub fn run(_: *const context_mod.CommandContext, _: []const [:0]const u8) !void {}
};

/// Group command module with aliases, subcommands, and forward.
const TestGroupForwardModule = struct {
    pub const meta: Meta = .{
        .name = "test-group-forward",
        .description = "A group-forward test command",
        .aliases = &.{ "tio", "io-test" },
        .subcommands = &.{ "sub1", "sub2", "help" },
        .kind = .group,
        .forward = .{ .target = "other-cmd", .prepend_args = &.{"launch"}, .warning = "deprecated" },
    };
    pub fn run(_: *const context_mod.CommandContext, _: []const [:0]const u8) !void {}
};

fn stubChild1(_: *const context_mod.CommandContext, _: []const [:0]const u8) !void {}
fn stubChild2(_: *const context_mod.CommandContext, _: []const [:0]const u8) !void {}

/// Group command with children.
const TestGroupModule = struct {
    pub const meta: Meta = .{
        .name = "test-group",
        .description = "A group command",
        .kind = .group,
        .children = &.{
            .{ .name = "child-a", .description = "First child", .handler = stubChild1 },
            .{ .name = "child-b", .description = "Second child", .handler = stubChild2 },
        },
    };
    pub fn run(_: *const context_mod.CommandContext, _: []const [:0]const u8) !void {}
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
    try std.testing.expectEqual(@as(CommandHandler, TestBasicModule.run), d.handler);
}

test "toDescriptor: group module preserves aliases, subcommands, forward" {
    const d = comptime toDescriptor(TestGroupForwardModule);
    try std.testing.expectEqualStrings("test-group-forward", d.name);
    try std.testing.expectEqual(@as(usize, 2), d.aliases.len);
    try std.testing.expectEqualStrings("tio", d.aliases[0]);
    try std.testing.expectEqualStrings("io-test", d.aliases[1]);
    try std.testing.expectEqual(@as(usize, 3), d.subcommands.len);
    try std.testing.expectEqual(CommandKind.group, d.kind);
    try std.testing.expectEqual(@as(CommandHandler, TestGroupForwardModule.run), d.handler);
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
    try std.testing.expectEqual(@as(CommandHandler, stubChild1), d.children[0].handler);
}

test "convertChildren: empty input returns empty slice" {
    const empty = comptime convertChildren(&.{});
    try std.testing.expectEqual(@as(usize, 0), empty.len);
}

test "validate: accepts valid modules" {
    // These should not compile-error.
    comptime validate(TestBasicModule);
    comptime validate(TestGroupForwardModule);
    comptime validate(TestGroupModule);
}

test "Meta defaults: verify zero-value defaults" {
    const m: Meta = .{ .name = "x", .description = "y" };
    try std.testing.expectEqual(@as(usize, 0), m.aliases.len);
    try std.testing.expectEqual(@as(usize, 0), m.subcommands.len);
    try std.testing.expectEqual(@as(usize, 0), m.children.len);
    try std.testing.expectEqual(CommandKind.action, m.kind);
    try std.testing.expect(m.forward == null);
}

test "toDescriptor: auto-derives subcommands from children when empty" {
    const d = comptime toDescriptor(TestGroupModule);
    // TestGroupModule has children but no explicit subcommands
    try std.testing.expectEqual(@as(usize, 2), d.subcommands.len);
    try std.testing.expectEqualStrings("child-a", d.subcommands[0]);
    try std.testing.expectEqualStrings("child-b", d.subcommands[1]);
}

test "toDescriptor: explicit subcommands preserved over children" {
    const d = comptime toDescriptor(TestGroupForwardModule);
    // TestGroupForwardModule has explicit subcommands — should NOT be overwritten
    try std.testing.expectEqual(@as(usize, 3), d.subcommands.len);
    try std.testing.expectEqualStrings("sub1", d.subcommands[0]);
}

test {
    std.testing.refAllDecls(@This());
}
