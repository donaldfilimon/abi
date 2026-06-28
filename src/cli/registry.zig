//! Declarative CLI command registry — the framework backbone for the `abi`
//! command surface.
//!
//! The frozen command *metadata* (name / usage / summary) is owned by the
//! `std`-only `usage.zig` module so the `cli_usage` build module (which is wired
//! without the `abi`/`build_options` imports) can keep projecting the help
//! surface from it. This module imports that metadata and augments each command
//! with its argument spec and handler, producing the single table that
//! `dispatch.zig` walks. Handler *invocation* lives here because this module is
//! only ever compiled inside the CLI executable graph (which has the handler
//! imports), never inside the standalone `cli_usage` module.

const std = @import("std");
const usage_mod = @import("usage.zig");
const handlers = @import("handlers/mod.zig");
const arg = @import("arg.zig");

pub const Arg = arg.Arg;
pub const Parsed = arg.Parsed;

/// Execution context handed to typed handlers.
pub const Ctx = struct {
    io: std.Io,
    allocator: std.mem.Allocator,
};

/// Legacy handler shape: receives the raw argv slice. Matches the historical
/// `CommandHandler` signature so existing handler functions wire in unchanged.
pub const RawHandler = *const fn (std.Io, std.mem.Allocator, []const []const u8) anyerror!u8;

/// Typed handler shape: receives the parsed argument set. Used by commands
/// migrated onto the generic argument parser (`arg.zig`).
pub const Handler = *const fn (Ctx, Parsed) anyerror!u8;

pub const Command = struct {
    name: []const u8,
    summary: []const u8,
    usage: []const u8,
    args: []const Arg = &.{},
    subcommands: []const Command = &.{},
    handler: ?Handler = null,
    raw_handler: ?RawHandler = null,
};

/// Look up the frozen metadata for `name` in the `usage` table at comptime so
/// the registry never re-declares the contract-tested name/usage/summary text.
fn meta(comptime name: []const u8) usage_mod.Command {
    inline for (usage_mod.commands) |command| {
        if (std.mem.eql(u8, command.name, name)) return command;
    }
    @compileError("registry: no usage metadata for command '" ++ name ++ "'");
}

/// Metadata-only command (e.g. `help`, which `dispatch` intercepts before the
/// table walk).
fn metaCmd(comptime name: []const u8) Command {
    const m = meta(name);
    return .{ .name = m.name, .summary = m.summary, .usage = m.usage };
}

/// Command wired to a raw `(io, allocator, argv)` handler.
fn rawCmd(comptime name: []const u8, handler: RawHandler) Command {
    const m = meta(name);
    return .{ .name = m.name, .summary = m.summary, .usage = m.usage, .raw_handler = handler };
}

/// Command wired to a typed handler driven by the generic argument parser.
fn typedCmd(comptime name: []const u8, args: []const Arg, handler: Handler) Command {
    const m = meta(name);
    return .{ .name = m.name, .summary = m.summary, .usage = m.usage, .args = args, .handler = handler };
}

// --- Typed argument specs ----------------------------------------------------

const complete_args = [_]Arg{
    .{ .name = "live", .kind = .flag, .help = "serve anthropic models over the live transport" },
    .{ .name = "confirm", .kind = .flag, .help = "confirm on-device FoundationModels execution" },
    .{ .name = "learn", .kind = .flag, .help = "run the SEA self-learning loop" },
    .{ .name = "model", .kind = .value, .help = "select a catalog model id (e.g. claude-fable-5)" },
    .{ .name = "input", .kind = .positional, .required = true, .help = "completion prompt" },
};

const train_args = [_]Arg{
    .{ .name = "input", .kind = .positional, .required = true, .help = "training input" },
};

// --- Typed handlers ----------------------------------------------------------
// Thin adapters that forward parsed arguments to the existing handler functions.
// Handler output text is preserved verbatim (never re-authored).

fn completeHandler(ctx: Ctx, parsed: Parsed) anyerror!u8 {
    return handlers.handleComplete(
        ctx.io,
        ctx.allocator,
        parsed.value("input").?,
        parsed.value("model"),
        parsed.flag("live"),
        parsed.flag("confirm"),
        parsed.flag("learn"),
    );
}

fn trainHandler(ctx: Ctx, parsed: Parsed) anyerror!u8 {
    return handlers.handleTrain(ctx.allocator, parsed.value("input").?);
}

fn backendsHandler(ctx: Ctx, parsed: Parsed) anyerror!u8 {
    _ = ctx;
    _ = parsed;
    return handlers.handleBackends();
}

fn dashboardHandler(ctx: Ctx, parsed: Parsed) anyerror!u8 {
    _ = parsed;
    return handlers.handleDashboard(ctx.allocator);
}

// --- Raw handler shims -------------------------------------------------------
// Commands still on the legacy `(io, allocator, argv)` contract. Thin adapters
// drop unused parameters and forward to the existing handler functions,
// reproducing the historical dispatch wrappers exactly.

fn handlePluginRaw(io: std.Io, alloc: std.mem.Allocator, args: []const []const u8) anyerror!u8 {
    _ = io;
    return handlers.handlePlugin(alloc, args);
}

fn handleTwilioRaw(io: std.Io, alloc: std.mem.Allocator, args: []const []const u8) anyerror!u8 {
    _ = io;
    return handlers.handleTwilio(alloc, args);
}

fn handleSchedulerRaw(io: std.Io, alloc: std.mem.Allocator, args: []const []const u8) anyerror!u8 {
    _ = io;
    return handlers.handleScheduler(alloc, args);
}

/// The frozen 13-command surface, in the same order as `usage.commands`.
/// `help` is metadata-only; `dispatch` intercepts it before the table walk.
pub const commands = [_]Command{
    metaCmd("help"),
    typedCmd("complete", &complete_args, completeHandler),
    typedCmd("train", &train_args, trainHandler),
    rawCmd("agent", handlers.handleAgent),
    typedCmd("backends", &.{}, backendsHandler),
    rawCmd("plugin", handlePluginRaw),
    rawCmd("auth", handlers.handleAuth),
    rawCmd("twilio", handleTwilioRaw),
    typedCmd("tui", &.{}, dashboardHandler),
    typedCmd("dashboard", &.{}, dashboardHandler),
    rawCmd("wdbx", handlers.handleWdbx),
    rawCmd("scheduler", handleSchedulerRaw),
    rawCmd("nn", handlers.handleNn),
};

test {
    std.testing.refAllDecls(@This());
}
