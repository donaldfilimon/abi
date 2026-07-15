//! Agent subcommand help text and shared arg-parsing helpers.
//!
//! Extracted from `agent.zig` so the main handler module focuses on dispatch
//! and the scheduler-backed execution paths. The help functions are pure
//! stdout renders; the arg parsers are pure extraction helpers used by the
//! spawn and browser subcommand handlers.

const std = @import("std");
const usage_mod = @import("../usage.zig");

pub fn agentPlanHelp() u8 {
    std.debug.print(
        \\usage: abi agent plan <input>
        \\
        \\Run a dry-run agent planning task through the scheduler and print memory tracker statistics.
        \\
    , .{});
    return 0;
}

pub fn agentTrainHelp() u8 {
    std.debug.print(
        \\usage: abi agent train <abbey|aviva|abi|all>
        \\
        \\Train one known local profile or all known profiles against the durable WDBX store.
        \\
    , .{});
    return 0;
}

pub fn agentTuiHelp() u8 {
    std.debug.print(
        \\usage: abi agent tui
        \\
        \\Launch the interactive ABI agent REPL. Non-tty input falls back to line mode.
        \\
    , .{});
    return 0;
}

pub fn agentOsHelp() u8 {
    std.debug.print(
        \\usage: abi agent os <dry-run|execute --confirm> <cmd> [args...]
        \\
        \\Audit an OS-control command request. execute requires --confirm and the policy allow-list.
        \\
    , .{});
    return 0;
}

pub fn agentMultiHelp() u8 {
    std.debug.print(
        \\usage: abi agent multi <input>
        \\
        \\Run Abbey, Aviva, and Abi concurrently via the scheduler and print aggregated output.
        \\
    , .{});
    return 0;
}

pub fn agentSpawnHelp() u8 {
    std.debug.print(
        \\usage: abi agent spawn [--background] [--workers <spec>] <input>
        \\
        \\Create custom smart-agent workers (name|instructions|hints;...). Submits via scheduler;
        \\--background prints task ids before running workers.
        \\
    , .{});
    return 0;
}

pub fn agentBrowserHelp() u8 {
    std.debug.print(
        \\usage: abi agent browser [--url <url>] [--execute --confirm] <task>
        \\
        \\Plan local browser automation (dry-run by default). Real navigation requires external MCP;
        \\ABI does not embed a headless browser.
        \\
    , .{});
    return 0;
}

/// Parsed result of the `spawn` subcommand's argv tail.
pub const SpawnArgs = struct {
    background: bool,
    workers_spec: ?[]const u8,
    input: []const u8,
};

/// Parse the `spawn` argv tail (everything after `agent spawn`) into structured
/// fields. Returns `null` (after printing nothing — the caller emits the usage
/// error) when the grammar is invalid.
pub fn parseSpawnArgs(allocator: std.mem.Allocator, cli_args: []const []const u8) !?SpawnArgs {
    var background = false;
    var workers_spec: ?[]const u8 = null;
    var input_parts = std.ArrayListUnmanaged([]const u8).empty;
    defer input_parts.deinit(allocator);

    var i: usize = 0;
    while (i < cli_args.len) : (i += 1) {
        const arg = cli_args[i];
        if (std.mem.eql(u8, arg, "--background")) {
            background = true;
        } else if (std.mem.eql(u8, arg, "--workers")) {
            i += 1;
            if (i >= cli_args.len) return null;
            workers_spec = cli_args[i];
        } else {
            try input_parts.append(allocator, arg);
        }
    }
    if (input_parts.items.len == 0) return null;

    var input_buf = std.ArrayListUnmanaged(u8).empty;
    defer input_buf.deinit(allocator);
    for (input_parts.items, 0..) |part, idx| {
        if (idx > 0) try input_buf.append(allocator, ' ');
        try input_buf.appendSlice(allocator, part);
    }
    const input = try input_buf.toOwnedSlice(allocator);
    return .{ .background = background, .workers_spec = workers_spec, .input = input };
}

/// Parsed result of the `browser` subcommand's argv tail.
pub const BrowserArgs = struct {
    url: ?[]const u8,
    execute_confirmed: bool,
    task: []const u8,
};

/// Parse the `browser` argv tail. Returns `null` when the grammar is invalid.
/// `--execute` without an adjacent `--confirm` is a hard usage error (returns
/// `error.Usage`) so the caller can distinguish "missing task" from
/// "unconfirmed execute".
pub fn parseBrowserArgs(allocator: std.mem.Allocator, cli_args: []const []const u8) !?BrowserArgs {
    var url: ?[]const u8 = null;
    var execute_confirmed = false;
    var task_parts = std.ArrayListUnmanaged([]const u8).empty;
    defer task_parts.deinit(allocator);

    var i: usize = 0;
    while (i < cli_args.len) : (i += 1) {
        const arg = cli_args[i];
        if (std.mem.eql(u8, arg, "--url")) {
            i += 1;
            if (i >= cli_args.len) return null;
            url = cli_args[i];
        } else if (std.mem.eql(u8, arg, "--execute")) {
            if (i + 1 < cli_args.len and std.mem.eql(u8, cli_args[i + 1], "--confirm")) {
                i += 1;
                execute_confirmed = true;
            } else {
                return error.Usage;
            }
        } else {
            try task_parts.append(allocator, arg);
        }
    }
    if (task_parts.items.len == 0) return null;

    var task_buf = std.ArrayListUnmanaged(u8).empty;
    defer task_buf.deinit(allocator);
    for (task_parts.items, 0..) |part, idx| {
        if (idx > 0) try task_buf.append(allocator, ' ');
        try task_buf.appendSlice(allocator, part);
    }
    const task = try task_buf.toOwnedSlice(allocator);
    return .{ .url = url, .execute_confirmed = execute_confirmed, .task = task };
}

test "parseSpawnArgs extracts flags and joins input" {
    const allocator = std.testing.allocator;
    const result = (try parseSpawnArgs(allocator, &.{ "--background", "hello", "world" })) orelse return error.ExpectedResult;
    defer allocator.free(result.input);
    try std.testing.expect(result.background);
    try std.testing.expect(result.workers_spec == null);
    try std.testing.expectEqualStrings("hello world", result.input);

    const workers_result = (try parseSpawnArgs(allocator, &.{ "--workers", "a|b|c;d|e|f", "task" })) orelse return error.ExpectedResult;
    defer allocator.free(workers_result.input);
    try std.testing.expect(!workers_result.background);
    try std.testing.expectEqualStrings("a|b|c;d|e|f", workers_result.workers_spec.?);

    try std.testing.expect((try parseSpawnArgs(allocator, &.{})) == null);
    try std.testing.expect((try parseSpawnArgs(allocator, &.{"--background"})) == null);
}

test "parseBrowserArgs extracts url and execute+confirm" {
    const allocator = std.testing.allocator;
    const result = (try parseBrowserArgs(allocator, &.{ "--url", "https://example.com", "open", "docs" })) orelse return error.ExpectedResult;
    defer allocator.free(result.task);
    try std.testing.expectEqualStrings("https://example.com", result.url.?);
    try std.testing.expect(!result.execute_confirmed);
    try std.testing.expectEqualStrings("open docs", result.task);

    const exec_result = (try parseBrowserArgs(allocator, &.{ "--execute", "--confirm", "do", "thing" })) orelse return error.ExpectedResult;
    defer allocator.free(exec_result.task);
    try std.testing.expect(exec_result.execute_confirmed);

    try std.testing.expectError(error.Usage, parseBrowserArgs(allocator, &.{ "--execute", "task" }));
    try std.testing.expect((try parseBrowserArgs(allocator, &.{})) == null);
}

test {
    std.testing.refAllDecls(@This());
}
