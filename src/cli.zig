//! Shared CLI helpers used by the `abi` binary and integration tests.

const std = @import("std");
const build_options = @import("build_options");
const feature_catalog = @import("core/feature_catalog.zig");
const acp = @import("protocols/acp/mod.zig");

const default_host = "127.0.0.1";
const default_port: u16 = 8080;
const command_usage_width: usize = 18;

pub const dashboard_command_summary = "Launch developer diagnostics shell";
pub const dashboard_command_detail =
    "Launch developer diagnostics shell (overview, features, runtime; requires -Dfeat-tui=true)";
pub const dashboard_fallback_note =
    "Dashboard note: non-interactive runs fall back to 'abi doctor'.";

pub const CommandDescriptor = struct {
    name: []const u8,
    description: []const u8,
};

pub const single_token_commands = [_]CommandDescriptor{
    .{ .name = "version", .description = "Print version and build info" },
    .{ .name = "doctor", .description = "Run diagnostics (features, platform, GPU)" },
    .{ .name = "features", .description = "List all features with status" },
    .{ .name = "platform", .description = "Show platform detection info" },
    .{ .name = "connectors", .description = "List available LLM connectors" },
    .{ .name = "info", .description = "Framework architecture summary" },
    .{ .name = "serve", .description = "Start the ACP HTTP server" },
    .{ .name = "dashboard", .description = dashboard_command_summary },
    .{ .name = "lsp", .description = "Start Language Server Protocol (LSP) server" },
};

pub const HelpSection = enum {
    diagnostics,
    ai_data,
    interactive,
};

pub const FeatureGate = enum {
    database,
    tui,
};

pub const RenderOptions = struct {
    stdout_is_tty: bool = true,
};

pub const DisplayCommand = struct {
    usage: []const u8,
    description: []const u8,
    section: HelpSection,
    feature_gate: ?FeatureGate = null,
};

pub const displayed_commands = [_]DisplayCommand{
    .{ .usage = "version", .description = "Print version and build info", .section = .diagnostics },
    .{ .usage = "doctor", .description = "Run diagnostics (features, platform, GPU)", .section = .diagnostics },
    .{ .usage = "features", .description = "List all features with enabled/disabled status", .section = .diagnostics },
    .{ .usage = "platform", .description = "Show platform detection (OS, arch, CPU)", .section = .diagnostics },
    .{ .usage = "connectors", .description = "List available LLM provider connectors", .section = .diagnostics },
    .{ .usage = "info", .description = "Show framework architecture summary", .section = .diagnostics },
    .{ .usage = "help", .description = "Show detailed help", .section = .diagnostics },
    .{ .usage = "chat <message...>", .description = "Route a message through the profile pipeline", .section = .ai_data },
    .{
        .usage = "db <cmd>",
        .description = "Vector database operations (add, query, stats, optimize, backup, restore, serve)",
        .section = .ai_data,
        .feature_gate = .database,
    },
    .{ .usage = "serve", .description = "Start the ACP HTTP server", .section = .ai_data },
    .{ .usage = "acp serve", .description = "Start the ACP HTTP server", .section = .ai_data },
    .{ .usage = "lsp", .description = "Start the Language Server Protocol (LSP) server", .section = .ai_data },
    .{
        .usage = "dashboard",
        .description = dashboard_command_detail,
        .section = .interactive,
        .feature_gate = .tui,
    },
};

pub const ChatPipelineReport = struct {
    input: []const u8,
    primary: []const u8,
    strategy: []const u8,
    reason: []const u8,
    confidence_pct: f32,
    abbey_pct: f32,
    aviva_pct: f32,
    abi_pct: f32,
    execution_summary: []const u8 = "",
};

pub fn joinChatMessage(allocator: std.mem.Allocator, message_args: []const [:0]const u8) ![]u8 {
    var full_message = std.ArrayListUnmanaged(u8).empty;
    errdefer full_message.deinit(allocator);

    for (message_args, 0..) |arg, i| {
        if (i > 0) try full_message.append(allocator, ' ');
        try full_message.appendSlice(allocator, arg);
    }

    return full_message.toOwnedSlice(allocator);
}

fn formatServeAddress(allocator: std.mem.Allocator, host: []const u8, port: u16) ![]u8 {
    const bracketed = host.len >= 2 and host[0] == '[' and host[host.len - 1] == ']';
    if (bracketed or std.mem.indexOfScalar(u8, host, ':') == null) {
        return std.fmt.allocPrint(allocator, "{s}:{d}", .{ host, port });
    }

    return std.fmt.allocPrint(allocator, "[{s}]:{d}", .{ host, port });
}

pub fn isServeInvocation(args: []const [:0]const u8) bool {
    if (args.len == 0) return false;
    if (std.mem.eql(u8, args[0], "serve")) return true;
    return args.len >= 2 and std.mem.eql(u8, args[0], "acp") and std.mem.eql(u8, args[1], "serve");
}

pub fn parseServeAddress(allocator: std.mem.Allocator, args: []const [:0]const u8) ![]u8 {
    var host: []const u8 = default_host;
    var port = default_port;
    var explicit_address: ?[]const u8 = null;

    var i: usize = 0;
    while (i < args.len) : (i += 1) {
        const arg = args[i];
        if (std.mem.eql(u8, arg, "--addr")) {
            if (i + 1 >= args.len) return error.InvalidServeArgs;
            explicit_address = args[i + 1];
            i += 1;
            continue;
        }
        if (std.mem.eql(u8, arg, "--host")) {
            if (i + 1 >= args.len) return error.InvalidServeArgs;
            host = args[i + 1];
            i += 1;
            continue;
        }
        if (std.mem.eql(u8, arg, "--port")) {
            if (i + 1 >= args.len) return error.InvalidServeArgs;
            port = std.fmt.parseInt(u16, args[i + 1], 10) catch return error.InvalidServePort;
            i += 1;
            continue;
        }
        if (std.mem.startsWith(u8, arg, "-")) return error.InvalidServeArgs;
        if (explicit_address == null) {
            explicit_address = arg;
        } else {
            return error.InvalidServeArgs;
        }
    }

    if (explicit_address) |address| {
        return allocator.dupe(u8, address);
    }

    return formatServeAddress(allocator, host, port);
}

fn initIoBackend(allocator: std.mem.Allocator) !std.Io.Threaded {
    const options: std.Io.Threaded.InitOptions = .{ .environ = std.process.Environ.empty };
    const InitResult = @TypeOf(std.Io.Threaded.init(allocator, options));
    if (@typeInfo(InitResult) == .error_union) {
        return try std.Io.Threaded.init(allocator, options);
    }
    return std.Io.Threaded.init(allocator, options);
}

fn isFeatureEnabled(gate: FeatureGate) bool {
    return switch (gate) {
        .database => build_options.feat_database,
        .tui => build_options.feat_tui,
    };
}

fn featureStateLabel(gate: FeatureGate) []const u8 {
    return if (isFeatureEnabled(gate)) "[enabled]" else "[disabled]";
}

fn writeUsageLine(writer: anytype, usage: []const u8, description: []const u8, feature_gate: ?FeatureGate) !void {
    var out = writer;
    try out.writeAll("  ");
    try out.writeAll(usage);

    const padding: usize = if (usage.len < command_usage_width)
        command_usage_width - usage.len
    else
        1;

    var i: usize = 0;
    while (i < padding) : (i += 1) try out.writeByte(' ');

    try out.writeAll(description);
    if (feature_gate) |gate| {
        try out.print(" {s}", .{featureStateLabel(gate)});
    }
    try out.writeByte('\n');
}

pub fn writeOptionalHeader(
    writer: anytype,
    title: []const u8,
    subtitle: ?[]const u8,
    options: RenderOptions,
) !void {
    if (!options.stdout_is_tty) return;

    var out = writer;
    try out.print("{s}\n", .{title});
    if (subtitle) |value| {
        try out.print("{s}\n", .{value});
        return;
    }

    for (title) |_| try out.writeByte('=');
    try out.writeAll("\n\n");
}

pub fn writeStatus(writer: anytype) !void {
    const enabled = comptime blk: {
        var count: u32 = 0;
        for (feature_catalog.all) |entry| {
            if (@field(build_options, entry.compile_flag_field)) count += 1;
        }
        break :blk count;
    };

    var out = writer;
    try out.print(
        \\ABI Framework v{s}
        \\Zig 0.16.0-dev | {d}/{d} features enabled
        \\
        \\Commands:
    , .{ build_options.package_version, enabled, feature_catalog.feature_count });
    try out.writeByte('\n');

    for (displayed_commands) |command| {
        try writeUsageLine(out, command.usage, command.description, command.feature_gate);
    }

    try out.print(
        \\
        \\{s}
        \\
        \\
        \\Run 'abi <command>' for details. 'abi help' for full reference.
        \\
    , .{dashboard_fallback_note});
}

fn writeHelpSection(writer: anytype, title: []const u8, section: HelpSection) !void {
    var out = writer;
    try out.print("{s}:\n", .{title});
    for (displayed_commands) |command| {
        if (command.section != section) continue;
        try writeUsageLine(out, command.usage, command.description, command.feature_gate);
    }
}

pub fn writeHelp(writer: anytype) !void {
    var out = writer;
    try out.writeAll(
        \\ABI - Multi-Profile AI Framework with WDBX
        \\
        \\Usage: abi <command> [args]
        \\
    );

    try writeHelpSection(out, "Diagnostics", .diagnostics);
    try out.writeByte('\n');
    try writeHelpSection(out, "AI & Data", .ai_data);
    try out.writeByte('\n');
    try writeHelpSection(out, "Interactive", .interactive);
    try out.print(
        \\
        \\{s}
        \\
        \\
        \\Build:
        \\  zig build cli      Build this CLI binary
        \\  zig build mcp      Build MCP stdio server
        \\  zig build lib      Build static library
        \\  zig build test     Run all tests
        \\  zig build check    Full gate (lint + test + parity)
        \\
    , .{dashboard_fallback_note});
}

pub fn writeServeHelp(writer: *std.Io.Writer) !void {
    const help_text =
        \\Usage: abi serve [options]
        \\
        \\Start the Agent Communication Protocol (ACP) server.
        \\
        \\Options:
        \\  --addr <host:port>   Listen on explicit address (e.g. 0.0.0.0:8080)
        \\  --host <hostname>    Listen host (default: 127.0.0.1)
        \\  --port <port>        Listen port (default: 8080)
        \\  --help, -h           Show this help message
        \\
    ;
    _ = try writer.writeVec(&.{help_text});
}

pub fn printServeHelp() void {
    var io_backend = initIoBackend(std.heap.page_allocator) catch return;
    defer io_backend.deinit();

    var stdout_file = std.Io.File.stdout();
    var write_buf: [1024]u8 = undefined;
    var writer = stdout_file.writer(io_backend.io(), &write_buf);
    writeServeHelp(&writer.interface) catch return;
    writer.flush() catch return;
}

pub fn wantsServeHelp(args: []const [:0]const u8) bool {
    for (args) |a| {
        if (std.mem.eql(u8, a, "help") or std.mem.eql(u8, a, "--help") or std.mem.eql(u8, a, "-h")) {
            return true;
        }
    }
    return false;
}

pub fn writeChatPipelineReport(writer: anytype, options: RenderOptions, report: ChatPipelineReport) !void {
    var out = writer;
    try writeOptionalHeader(out, "ABI Chat - Profile Pipeline", null, options);
    try out.print(
        \\Input: {s}
        \\
        \\Routing Decision:
        \\  Primary: {s}
        \\  Strategy: {s}
        \\  Confidence: {d:.0}%
        \\  Reason: {s}
        \\
        \\Weights:
        \\  Abbey: {d:.0}%
        \\  Aviva: {d:.0}%
        \\  Abi:   {d:.0}%
        \\
        \\Execution:
        \\{s}
    , .{
        report.input,
        report.primary,
        report.strategy,
        report.confidence_pct,
        report.reason,
        report.abbey_pct,
        report.aviva_pct,
        report.abi_pct,
        report.execution_summary,
    });
}

pub fn runServeWithWriter(allocator: std.mem.Allocator, args: []const [:0]const u8, writer: anytype) !void {
    if (wantsServeHelp(args)) {
        try writeServeHelp(writer);
        return;
    }

    const address = try parseServeAddress(allocator, args);
    defer allocator.free(address);

    var io_backend = try initIoBackend(allocator);
    defer io_backend.deinit();

    const url = try std.fmt.allocPrint(allocator, "http://{s}", .{address});
    defer allocator.free(url);

    const card = acp.AgentCard{
        .name = "abi",
        .description = "ABI Agent Communication Protocol server",
        .version = build_options.package_version,
        .url = url,
        .capabilities = .{},
    };

    try acp.serveHttp(allocator, io_backend.io(), address, card);
}

pub fn runServe(allocator: std.mem.Allocator, args: []const [:0]const u8) !void {
    var io_backend = try initIoBackend(allocator);
    defer io_backend.deinit();

    var stdout_file = std.Io.File.stdout();
    var write_buf: [1024]u8 = undefined;
    var writer = stdout_file.writer(io_backend.io(), &write_buf);
    try runServeWithWriter(allocator, args, &writer.interface);
    try writer.flush();
}

test "serve invocation recognises both aliases" {
    try std.testing.expect(isServeInvocation(&.{"serve"}));
    try std.testing.expect(isServeInvocation(&.{ "acp", "serve" }));
    try std.testing.expect(!isServeInvocation(&.{ "acp", "status" }));
}

test "serve address parsing honors addr and port flags" {
    const port_args = [_][:0]const u8{ "--port", "9090" };
    const port_address = try parseServeAddress(std.testing.allocator, &port_args);
    defer std.testing.allocator.free(port_address);
    try std.testing.expectEqualStrings("127.0.0.1:9090", port_address);

    const ipv6_args = [_][:0]const u8{ "--host", "::1", "--port", "9090" };
    const ipv6_address = try parseServeAddress(std.testing.allocator, &ipv6_args);
    defer std.testing.allocator.free(ipv6_address);
    try std.testing.expectEqualStrings("[::1]:9090", ipv6_address);

    const addr_args = [_][:0]const u8{ "--addr", "0.0.0.0:8080" };
    const explicit_address = try parseServeAddress(std.testing.allocator, &addr_args);
    defer std.testing.allocator.free(explicit_address);
    try std.testing.expectEqualStrings("0.0.0.0:8080", explicit_address);
}

test "displayed commands cover single-token commands" {
    for (single_token_commands) |single| {
        var found = false;
        for (displayed_commands) |displayed| {
            if (std.mem.eql(u8, displayed.usage, single.name)) {
                found = true;
                break;
            }
        }
        try std.testing.expect(found);
    }
}

test {
    std.testing.refAllDecls(@This());
}
