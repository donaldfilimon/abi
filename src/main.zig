//! ABI CLI — Command-line interface for the ABI framework.
//!
//! Provides user-facing commands for interacting with the multi-profile
//! AI system, WDBX database, full-text search, and framework diagnostics.
//!
//! Usage:
//!   abi                   Show status overview with enabled features
//!   abi version           Print version and build info
//!   abi doctor            Run diagnostics (features, platform, GPU)
//!   abi features          List all features with status
//!   abi platform          Show platform detection info
//!   abi connectors        List available LLM connectors
//!   abi plugins           List configured LLM provider plugins
//!   abi skills            List bundled ABI agent skills
//!   abi info              Show framework architecture summary
//!   abi chat <message...>  Route a message through the profile pipeline
//!   abi serve             Start the ACP HTTP server
//!   abi acp serve         Start the ACP HTTP server
//!   abi db <subcommand>   Vector database operations
//!   abi search <cmd>      Full-text search (create, index, query, delete, stats)
//!   abi dashboard         Launch developer diagnostics shell
//!   abi help              Show this help message

const std = @import("std");
const build_options = @import("build_options");

// Framework modules (relative imports within src/)
const root = @import("root.zig");
const cli = @import("cli.zig");
const os = @import("foundation/os.zig");
const feature_catalog = root.meta.features;
const llm_plugin_manifest = @import("features/ai/llm/providers/plugins/manifest.zig");

const CliError = error{
    InvalidArguments,
    CommandFailed,
};

var current_environ_map: ?*std.process.Environ.Map = null;

// ── Shared Constants ────────────────────────────────────────────────────

/// All GPU backend names and their build-time enabled state.
/// Used by both `printPlatform()` and `runDoctor()`.
const gpu_backends = .{
    .{ "metal", build_options.gpu_metal },
    .{ "cuda", build_options.gpu_cuda },
    .{ "vulkan", build_options.gpu_vulkan },
    .{ "webgpu", build_options.gpu_webgpu },
    .{ "opengl", build_options.gpu_opengl },
    .{ "opengles", build_options.gpu_opengles },
    .{ "webgl2", build_options.gpu_webgl2 },
    .{ "stdgpu", build_options.gpu_stdgpu },
    .{ "fpga", build_options.gpu_fpga },
    .{ "tpu", build_options.gpu_tpu },
};

// ── Helpers ─────────────────────────────────────────────────────────────

/// Write data to stdout using std.Io. Separates response output (stdout)
/// from diagnostic metadata (stderr via std.debug.print).
fn writeToStdout(data: []const u8) !void {
    var io_backend = initIoBackend(std.heap.page_allocator);
    defer io_backend.deinit();

    var stdout_file = std.Io.File.stdout();
    var write_buf: [1024]u8 = undefined;
    var writer = stdout_file.writer(io_backend.io(), &write_buf);
    try std.Io.Writer.writeAll(&writer.interface, data);
    try writer.flush();
}

fn printSharedCliText(comptime renderFn: anytype) !void {
    var writer: std.Io.Writer.Allocating = .init(std.heap.page_allocator);
    defer writer.deinit();
    renderFn(&writer.writer) catch return;
    const output = writer.toOwnedSlice() catch return;
    defer std.heap.page_allocator.free(output);

    var io_backend = initIoBackend(std.heap.page_allocator);
    defer io_backend.deinit();

    var stdout_file = std.Io.File.stdout();
    var write_buf: [1024]u8 = undefined;
    var stdout_writer = stdout_file.writer(io_backend.io(), &write_buf);
    try std.Io.Writer.writeAll(&stdout_writer.interface, output);
    try stdout_writer.flush();
}

fn initIoBackend(allocator: std.mem.Allocator) std.Io.Threaded {
    return std.Io.Threaded.init(allocator, .{});
}

fn countEnabledFeatures() struct { enabled: usize, total: usize } {
    const enabled = comptime blk: {
        var count: usize = 0;
        for (feature_catalog.all) |entry| {
            if (@field(build_options, entry.compile_flag_field)) count += 1;
        }
        break :blk count;
    };
    return .{ .enabled = enabled, .total = feature_catalog.all.len };
}

fn printHeader(title: []const u8, subtitle: ?[]const u8) void {
    if (!os.isatty()) return; // Strip non-diagnostic metadata when piped

    std.debug.print("{s}\n", .{title});
    if (subtitle) |sub| {
        std.debug.print("{s}\n", .{sub});
    } else {
        for (title) |_| std.debug.print("═", .{});
        std.debug.print("\n\n", .{});
    }
}

// ── Entry Point ─────────────────────────────────────────────────────────

pub fn main(init: std.process.Init) !void {
    const allocator = init.gpa;
    const arena = init.arena.allocator();
    const args = try init.minimal.args.toSlice(arena);
    current_environ_map = init.environ_map;

    const exit_code = dispatch(allocator, args[1..]) catch |err| blk: {
        std.debug.print("Error: {s}\n", .{@errorName(err)});
        break :blk 1;
    };
    if (exit_code != 0) std.process.exit(exit_code);
}

// ── Command Dispatch ────────────────────────────────────────────────────

pub fn dispatch(allocator: std.mem.Allocator, args: []const [:0]const u8) !u8 {
    var start: usize = 0;
    while (start < args.len and std.mem.startsWith(u8, args[start], "--")) : (start += 1) {
        const flag = args[start];
        if (std.mem.eql(u8, flag, "--help")) {
            printHelp();
            return 0;
        }
        if (std.mem.eql(u8, flag, "--version")) {
            printVersion();
            return 0;
        }
        if (std.mem.eql(u8, flag, "--quiet") or
            std.mem.eql(u8, flag, "--json") or
            std.mem.eql(u8, flag, "--no-color"))
        {
            continue;
        }
        std.debug.print("Unsupported global flag: {s}\n", .{flag});
        std.debug.print("Supported global flags: --help, --version, --quiet, --json, --no-color\n", .{});
        return 1;
    }

    const command_args = args[start..];

    if (command_args.len == 0) {
        printStatus();
        return 0;
    }

    if (cli.isServeInvocation(command_args)) {
        const serve_args = if (std.mem.eql(u8, command_args[0], "acp")) command_args[2..] else command_args[1..];
        try cli.runServe(allocator, serve_args);
        return 0;
    }

    const cmd = command_args[0];

    if (std.mem.eql(u8, cmd, "version")) {
        printVersion();
    } else if (std.mem.eql(u8, cmd, "doctor")) {
        try runDoctor(allocator);
    } else if (std.mem.eql(u8, cmd, "check-env")) {
        return try runCheckEnv();
    } else if (std.mem.eql(u8, cmd, "features")) {
        printFeatures();
    } else if (std.mem.eql(u8, cmd, "platform")) {
        printPlatform();
    } else if (std.mem.eql(u8, cmd, "connectors")) {
        printConnectors();
    } else if (std.mem.eql(u8, cmd, "plugins") or std.mem.eql(u8, cmd, "plugin")) {
        try runPlugins(allocator);
    } else if (std.mem.eql(u8, cmd, "skills") or std.mem.eql(u8, cmd, "skill")) {
        try runSkills(allocator);
    } else if (std.mem.eql(u8, cmd, "build")) {
        return try runBuildCommand(allocator, command_args[1..]);
    } else if (std.mem.eql(u8, cmd, "check")) {
        return try runCheckCommand(allocator, command_args[1..]);
    } else if (std.mem.eql(u8, cmd, "learn")) {
        return try runLearnCommand(allocator, command_args[1..]);
    } else if (std.mem.eql(u8, cmd, "agent")) {
        return try runAgentCommand(allocator, command_args[1..]);
    } else if (std.mem.eql(u8, cmd, "gpu")) {
        return try runGpuCommand(allocator, command_args[1..]);
    } else if (std.mem.eql(u8, cmd, "mcp")) {
        return try runMcpCommand(allocator, command_args[1..]);
    } else if (std.mem.eql(u8, cmd, "acp")) {
        return try runAcpCommand(allocator, command_args[1..]);
    } else if (std.mem.eql(u8, cmd, "info")) {
        printInfo();
    } else if (std.mem.eql(u8, cmd, "chat")) {
        if (command_args.len < 2) {
            std.debug.print("Usage: abi chat <message...>\n", .{});
            return 1;
        }
        try runChat(allocator, command_args[1..]);
    } else if (std.mem.eql(u8, cmd, "db")) {
        try runDb(allocator, command_args[1..]);
    } else if (std.mem.eql(u8, cmd, "search")) {
        try runSearch(allocator, command_args[1..]);
    } else if (std.mem.eql(u8, cmd, "lsp")) {
        try runLsp(allocator);
    } else if (std.mem.eql(u8, cmd, "discord")) {
        try cli.runDiscord(allocator, command_args[1..]);
    } else if (std.mem.eql(u8, cmd, "dashboard")) {
        try runDashboard(allocator);
    } else if (std.mem.eql(u8, cmd, "help") or std.mem.eql(u8, cmd, "--help") or std.mem.eql(u8, cmd, "-h")) {
        printHelp();
    } else {
        printUnknownCommand(cmd);
        return 1;
    }
    return 0;
}

fn isHelpArg(arg: []const u8) bool {
    return std.mem.eql(u8, arg, "help") or std.mem.eql(u8, arg, "--help") or std.mem.eql(u8, arg, "-h");
}

fn usageBuild() void {
    std.debug.print(
        \\Usage: abi build <step> [zig-build-args...]
        \\
        \\Delegates to ./build.sh.
        \\Common steps: quick, dev, ci, test, cli, mcp, check-parity, mcp-health, interop
        \\
    , .{});
}

fn usageCheck() void {
    std.debug.print(
        \\Usage: abi check [quick|ci|parity|mcp|interop]
        \\
        \\Default: abi check quick
        \\
    , .{});
}

fn usageMcp() void {
    std.debug.print(
        \\Usage: abi mcp <command>
        \\
        \\Commands:
        \\  health              Check configured MCP HA /health endpoints
        \\  endpoints           List configured MCP endpoints from mcp/servers.json
        \\  serve [stdio|sse]   Start abi-mcp through mcp/launcher.sh
        \\
    , .{});
}

fn usageLearn() void {
    std.debug.print(
        \\Usage: abi learn <command>
        \\
        \\Commands:
        \\  status                  Show learning telemetry status
        \\  report                  Show learning performance report
        \\  feedback <kind> [--note <text>]
        \\  retrain --manual        Run a manual no-auto-mutation retrain hook
        \\  export [--out <path>]   Export learning artifacts for offline training
        \\
    , .{});
}

fn usageAgent() void {
    std.debug.print(
        \\Usage: abi agent <command>
        \\
        \\Commands:
        \\  chat <name> <message...>    Chat with an ephemeral named agent
        \\  stats <name>                Show default stats for a named agent
        \\
    , .{});
}

fn usageGpu() void {
    std.debug.print(
        \\Usage: abi gpu <command>
        \\
        \\Commands:
        \\  doctor      Show GPU build and runtime fallback diagnostics
        \\  backends    List configured GPU backends
        \\
    , .{});
}

fn usageAcp() void {
    std.debug.print(
        \\Usage: abi acp <command>
        \\
        \\Commands:
        \\  serve [options]     Start the ACP HTTP server
        \\  endpoints           List/check ACP endpoints from ACP_ENDPOINTS
        \\
    , .{});
}

fn parseNoteArg(args: []const [:0]const u8, start: usize) ?[]const u8 {
    var i = start;
    while (i < args.len) : (i += 1) {
        if (std.mem.eql(u8, args[i], "--note") and i + 1 < args.len) return args[i + 1];
    }
    return null;
}

fn parseOutArg(args: []const [:0]const u8, start: usize) ?[]const u8 {
    var i = start;
    while (i < args.len) : (i += 1) {
        if (std.mem.eql(u8, args[i], "--out") and i + 1 < args.len) return args[i + 1];
    }
    return null;
}

fn printLearningReport(report: root.ai.learning.LearningReport) void {
    std.debug.print(
        \\ABI Learning
        \\  telemetry: enabled
        \\  auto retrain: {s}
        \\  stored events: {d}
        \\  interactions: {d}
        \\  average quality: {d:.3}
        \\  positive feedback: {d}
        \\  negative feedback: {d}
        \\
    , .{
        if (report.auto_retrain_enabled) "enabled" else "disabled",
        report.stored_events,
        report.total_interactions,
        report.avg_quality,
        report.positive_feedback_count,
        report.negative_feedback_count,
    });
}

fn runLearnCommand(allocator: std.mem.Allocator, args: []const [:0]const u8) !u8 {
    if (args.len == 0 or isHelpArg(args[0])) {
        usageLearn();
        return if (args.len == 0) 1 else 0;
    }

    var runtime = try root.ai.learning.LearningRuntime.init(allocator);
    defer runtime.deinit();

    const sub = args[0];
    if (std.mem.eql(u8, sub, "status") or std.mem.eql(u8, sub, "report")) {
        printLearningReport(runtime.report());
        return 0;
    }
    if (std.mem.eql(u8, sub, "feedback")) {
        if (args.len < 2) {
            std.debug.print("Usage: abi learn feedback [positive|negative|neutral] [--note <text>]\n", .{});
            return 1;
        }
        const kind = root.ai.learning.FeedbackKind.fromString(args[1]) orelse {
            std.debug.print("Unknown feedback kind: {s}\n", .{args[1]});
            return 1;
        };
        try runtime.recordFeedback(kind, parseNoteArg(args, 2));
        std.debug.print("Recorded {s} feedback.\n", .{kind.label()});
        return 0;
    }
    if (std.mem.eql(u8, sub, "retrain")) {
        if (args.len < 2 or !std.mem.eql(u8, args[1], "--manual")) {
            std.debug.print("Manual confirmation required: abi learn retrain --manual\n", .{});
            return 1;
        }
        const did_retrain = try runtime.forceRetrain();
        std.debug.print("Manual retrain hook: {s}\n", .{if (did_retrain) "completed" else "unavailable"});
        return 0;
    }
    if (std.mem.eql(u8, sub, "export")) {
        const out_path = parseOutArg(args, 1) orelse ".abi/learning/export.jsonl";
        const count = try runtime.exportArtifacts(out_path);
        std.debug.print("Exported {d} learning events to {s}\n", .{ count, out_path });
        return 0;
    }

    std.debug.print("Unknown learn command: {s}\n", .{sub});
    usageLearn();
    return 1;
}

fn runAgentCommand(allocator: std.mem.Allocator, args: []const [:0]const u8) !u8 {
    if (args.len == 0 or isHelpArg(args[0])) {
        usageAgent();
        return if (args.len == 0) 1 else 0;
    }
    if (!build_options.feat_ai) {
        std.debug.print("AI agents are disabled. Rebuild with -Dfeat-ai=true\n", .{});
        return 1;
    }

    const sub = args[0];
    if (std.mem.eql(u8, sub, "chat")) {
        if (args.len < 3) {
            std.debug.print("Usage: abi agent chat <name> <message...>\n", .{});
            return 1;
        }
        var agent = try root.ai.agents.Agent.init(allocator, .{ .name = args[1] });
        defer agent.deinit();
        const message = try cli.joinChatMessage(allocator, args[2..]);
        defer allocator.free(message);
        const response = try agent.chat(message, allocator);
        defer allocator.free(response);
        try writeToStdout(response);
        try writeToStdout("\n");
        return 0;
    }
    if (std.mem.eql(u8, sub, "stats")) {
        if (args.len < 2) {
            std.debug.print("Usage: abi agent stats <name>\n", .{});
            return 1;
        }
        var agent = try root.ai.agents.Agent.init(allocator, .{ .name = args[1] });
        defer agent.deinit();
        const stats = agent.getStats();
        std.debug.print(
            \\Agent: {s}
            \\  history: {d}
            \\  user messages: {d}
            \\  assistant messages: {d}
            \\  total characters: {d}
            \\  tokens used: {d}
            \\
        , .{ args[1], stats.history_length, stats.user_messages, stats.assistant_messages, stats.total_characters, stats.total_tokens_used });
        return 0;
    }

    std.debug.print("Unknown agent command: {s}\n", .{sub});
    usageAgent();
    return 1;
}

fn printGpuBackends() void {
    std.debug.print("ABI GPU Backends\n", .{});
    std.debug.print("  feat_gpu: {any}\n", .{build_options.feat_gpu});
    inline for (gpu_backends) |backend| {
        std.debug.print("  {s}: {s}\n", .{ backend[0], if (backend[1]) "enabled" else "disabled" });
    }
    if (build_options.feat_gpu and !build_options.gpu_stdgpu) {
        std.debug.print("  fallback: simulated/stdgpu runtime fallback remains available through backend policy\n", .{});
    }
}

fn runGpuCommand(_: std.mem.Allocator, args: []const [:0]const u8) !u8 {
    if (args.len == 0 or isHelpArg(args[0])) {
        usageGpu();
        return if (args.len == 0) 1 else 0;
    }
    const sub = args[0];
    if (std.mem.eql(u8, sub, "backends")) {
        printGpuBackends();
        return 0;
    }
    if (std.mem.eql(u8, sub, "doctor")) {
        printGpuBackends();
        std.debug.print("  safe fallback: {s}\n", .{if (build_options.feat_gpu) "stdgpu/simulated" else "disabled build"});
        std.debug.print("  hardware backends stay guarded by runtime availability checks.\n", .{});
        return 0;
    }
    std.debug.print("Unknown gpu command: {s}\n", .{sub});
    usageGpu();
    return 1;
}

fn termExitCode(term: std.process.Child.Term) u8 {
    return switch (term) {
        .exited => |code| code,
        else => 1,
    };
}

fn runArgvCaptured(allocator: std.mem.Allocator, argv: []const []const u8) !u8 {
    var io_backend = initIoBackend(allocator);
    defer io_backend.deinit();
    const result = try std.process.run(allocator, io_backend.io(), .{ .argv = argv, .environ_map = current_environ_map });
    defer allocator.free(result.stdout);
    defer allocator.free(result.stderr);
    if (result.stdout.len > 0) writeToStdout(result.stdout) catch |err| {
        std.debug.print("Warning: failed to write stdout: {}\n", .{err});
    };
    if (result.stderr.len > 0) std.debug.print("{s}", .{result.stderr});
    return termExitCode(result.term);
}

fn runArgvStreaming(allocator: std.mem.Allocator, argv: []const []const u8) !u8 {
    var io_backend = initIoBackend(allocator);
    defer io_backend.deinit();
    var child = try std.process.spawn(io_backend.io(), .{ .argv = argv, .environ_map = current_environ_map });
    return termExitCode(try child.wait(io_backend.io()));
}

fn runShellCaptured(allocator: std.mem.Allocator, command: []const u8) !u8 {
    const shell: []const u8 = if (@import("builtin").os.tag == .windows) "cmd.exe" else "/bin/sh";
    const flag: []const u8 = if (@import("builtin").os.tag == .windows) "/c" else "-c";
    return runArgvCaptured(allocator, &.{ shell, flag, command });
}

fn runShellDiagnostic(allocator: std.mem.Allocator, command: []const u8) !u8 {
    const shell: []const u8 = if (@import("builtin").os.tag == .windows) "cmd.exe" else "/bin/sh";
    const flag: []const u8 = if (@import("builtin").os.tag == .windows) "/c" else "-c";
    var io_backend = initIoBackend(allocator);
    defer io_backend.deinit();
    const result = try std.process.run(allocator, io_backend.io(), .{ .argv = &.{ shell, flag, command }, .environ_map = current_environ_map });
    defer allocator.free(result.stdout);
    defer allocator.free(result.stderr);
    if (result.stdout.len > 0) std.debug.print("{s}", .{result.stdout});
    if (result.stderr.len > 0) std.debug.print("{s}", .{result.stderr});
    return termExitCode(result.term);
}

fn runBuildStep(allocator: std.mem.Allocator, step: []const u8, extra_args: []const [:0]const u8) !u8 {
    var argv = try allocator.alloc([]const u8, extra_args.len + 2);
    defer allocator.free(argv);
    argv[0] = "./build.sh";
    argv[1] = step;
    for (extra_args, 0..) |arg, i| argv[i + 2] = arg;
    return runArgvCaptured(allocator, argv);
}

fn runBuildCommand(allocator: std.mem.Allocator, args: []const [:0]const u8) !u8 {
    if (args.len == 0 or isHelpArg(args[0])) {
        usageBuild();
        return if (args.len == 0) 1 else 0;
    }
    return runBuildStep(allocator, args[0], args[1..]);
}

fn runCheckCommand(allocator: std.mem.Allocator, args: []const [:0]const u8) !u8 {
    if (args.len > 0 and isHelpArg(args[0])) {
        usageCheck();
        return 0;
    }

    const lane = if (args.len == 0) "quick" else args[0];
    const step = if (std.mem.eql(u8, lane, "quick"))
        "quick"
    else if (std.mem.eql(u8, lane, "ci"))
        "ci"
    else if (std.mem.eql(u8, lane, "parity"))
        "check-parity"
    else if (std.mem.eql(u8, lane, "mcp"))
        "mcp-tests"
    else if (std.mem.eql(u8, lane, "interop"))
        "interop"
    else {
        std.debug.print("Unknown check lane: {s}\n", .{lane});
        usageCheck();
        return 1;
    };

    return runBuildStep(allocator, step, if (args.len > 1) args[1..] else &.{});
}

fn runMcpCommand(allocator: std.mem.Allocator, args: []const [:0]const u8) !u8 {
    if (args.len == 0 or isHelpArg(args[0])) {
        usageMcp();
        return if (args.len == 0) 1 else 0;
    }

    const sub = args[0];
    if (std.mem.eql(u8, sub, "health")) {
        return runArgvCaptured(allocator, &.{ "bash", "scripts/check-mcp-health.sh" });
    }
    if (std.mem.eql(u8, sub, "endpoints")) {
        return runShellCaptured(allocator,
            \\if command -v jq >/dev/null 2>&1; then
            \\  jq -r '.mcpServers | to_entries[] | "\(.key) http://" + .value.env.ABI_MCP_HOST + ":" + .value.env.ABI_MCP_PORT + (.value.healthCheck.path // "/health")' mcp/servers.json
            \\else
            \\  echo "jq not found; showing raw mcp/servers.json"
            \\  cat mcp/servers.json
            \\fi
        );
    }
    if (std.mem.eql(u8, sub, "serve")) {
        const mode = if (args.len >= 2) args[1] else "stdio";
        if (!std.mem.eql(u8, mode, "stdio") and !std.mem.eql(u8, mode, "sse")) {
            std.debug.print("Invalid MCP serve mode: {s}\n", .{mode});
            std.debug.print("Usage: abi mcp serve [stdio|sse]\n", .{});
            return 1;
        }
        return runArgvStreaming(allocator, &.{ "./mcp/launcher.sh", mode });
    }

    std.debug.print("Unknown MCP command: {s}\n", .{sub});
    usageMcp();
    return 1;
}

fn runAcpCommand(allocator: std.mem.Allocator, args: []const [:0]const u8) !u8 {
    if (args.len == 0 or isHelpArg(args[0])) {
        usageAcp();
        return if (args.len == 0) 1 else 0;
    }
    if (std.mem.eql(u8, args[0], "endpoints")) {
        return runArgvCaptured(allocator, &.{ "bash", "scripts/list-acp-endpoints.sh" });
    }
    std.debug.print("Unknown ACP command: {s}\n", .{args[0]});
    usageAcp();
    return 1;
}

fn printUnknownCommand(cmd: []const u8) void {
    std.debug.print("Unknown command: {s}\n", .{cmd});
    if (cli.suggestCommand(cmd)) |name| std.debug.print("Did you mean: abi {s}\n", .{name});
    std.debug.print("Example: abi doctor\nRun 'abi help' for the full command list.\n", .{});
}

/// Enhanced routing information for profile decision.
fn printRoutingInfo(decision: anytype, message: []const u8) void {
    std.debug.print(
        \\Routing Analysis:
        \\  Input length: {d} chars
        \\  Primary profile: {s}
        \\  Strategy: {s}
        \\  Confidence: {d:.0}%
        \\  Reason: {s}
        \\
    , .{
        message.len,
        decision.primary.name(),
        @tagName(decision.strategy),
        decision.confidence * 100.0,
        decision.reason,
    });
}

// ── Status (no-args) ────────────────────────────────────────────────────

pub fn printStatus() void {
    printSharedCliText(cli.writeStatus) catch |err| {
        std.debug.print("Warning: Failed to write status: {}\n", .{err});
    };
}

fn printFeatureTag(enabled: bool) void {
    if (enabled) {
        std.debug.print("[enabled]\n", .{});
    } else {
        std.debug.print("[disabled]\n", .{});
    }
}

// ── Version ─────────────────────────────────────────────────────────────

pub fn printVersion() void {
    printSharedCliText(cli.writeVersion) catch {};
}

// ── Help ────────────────────────────────────────────────────────────────

pub fn printHelp() void {
    printSharedCliText(cli.writeHelp) catch {};
}

// ── Features ────────────────────────────────────────────────────────────

pub fn printFeatures() void {
    printHeader("ABI Features — Compile-Time Feature Catalog", null);

    inline for (feature_catalog.all) |entry| {
        const enabled = @field(build_options, entry.compile_flag_field);
        const tag: []const u8 = if (enabled) "[+]" else "[-]";
        const parent_str: []const u8 = if (entry.parent != null) "  " else "";
        std.debug.print("  {s} {s}{s} — {s}\n", .{ tag, parent_str, entry.feature.name(), entry.description });
    }

    const counts = countEnabledFeatures();
    std.debug.print("\n{d}/{d} features enabled.\n", .{ counts.enabled, counts.total });
}

// ── Platform ────────────────────────────────────────────────────────────

pub fn printPlatform() void {
    const platform = root.platform;
    const info = platform.getPlatformInfo();

    printHeader("ABI Platform — System Detection", null);

    std.debug.print(
        \\OS:           {s}
        \\Architecture: {s}
        \\Description:  {s}
        \\CPU Cores:    {d}
        \\Threading:    {s}
        \\
    , .{
        @tagName(info.os),
        @tagName(info.arch),
        platform.getDescription(),
        platform.getCpuCount(),
        if (platform.supportsThreading()) "supported" else "unavailable",
    });

    std.debug.print("GPU Backends:\n", .{});
    inline for (gpu_backends) |backend| {
        const tag: []const u8 = if (backend[1]) "[+]" else "[-]";
        std.debug.print("  {s} {s}\n", .{ tag, backend[0] });
    }

    std.debug.print("\n", .{});
}

// ── Connectors ──────────────────────────────────────────────────────────

pub fn printConnectors() void {
    printHeader("ABI Connectors — LLM Provider Adapters", null);

    const connectors = [_]struct { env: [:0]const u8, fallback: ?[:0]const u8, name: []const u8 }{
        .{ .env = "ABI_OPENAI_API_KEY", .fallback = "OPENAI_API_KEY", .name = "OpenAI (GPT-4, GPT-3.5)" },
        .{ .env = "ABI_ANTHROPIC_API_KEY", .fallback = "ANTHROPIC_API_KEY", .name = "Anthropic (Claude)" },
        .{ .env = "ABI_GEMINI_API_KEY", .fallback = "GEMINI_API_KEY", .name = "Google Gemini" },
        .{ .env = "ABI_MISTRAL_API_KEY", .fallback = "MISTRAL_API_KEY", .name = "Mistral AI" },
        .{ .env = "ABI_COHERE_API_KEY", .fallback = "COHERE_API_KEY", .name = "Cohere (Chat, Embed, Rerank)" },
        .{ .env = "ABI_HF_API_TOKEN", .fallback = "HF_API_TOKEN", .name = "HuggingFace Inference API" },
        .{ .env = "ABI_OLLAMA_HOST", .fallback = "OLLAMA_HOST", .name = "Ollama (local)" },
        .{ .env = "ABI_LM_STUDIO_HOST", .fallback = null, .name = "LM Studio (local, OpenAI-compat)" },
        .{ .env = "ABI_VLLM_HOST", .fallback = null, .name = "vLLM (local, high-throughput)" },
        .{ .env = "ABI_MLX_HOST", .fallback = null, .name = "MLX (Apple Silicon)" },
        .{ .env = "ABI_LLAMA_CPP_HOST", .fallback = null, .name = "llama.cpp server" },
        .{ .env = "ABI_DISCORD_BOT_TOKEN", .fallback = "DISCORD_BOT_TOKEN", .name = "Discord bot integration" },
    };

    var configured: u32 = 0;
    for (connectors) |c| {
        const has_primary = std.c.getenv(c.env.ptr) != null;
        const has_fallback = if (c.fallback) |fb| std.c.getenv(fb.ptr) != null else false;
        const is_set = has_primary or has_fallback;
        if (is_set) configured += 1;
        const tag = if (is_set) "\x1b[32m[configured]\x1b[0m" else "\x1b[90m[not set]\x1b[0m";
        std.debug.print("  {s: <24} {s} {s}\n", .{ c.env, tag, c.name });
    }

    std.debug.print("\n  {d}/12 providers configured", .{configured});
    if (configured == 0) {
        std.debug.print(" — set env vars to enable providers", .{});
    }
    std.debug.print("\n  Legacy env vars (e.g. OPENAI_API_KEY) are supported as fallbacks.\n", .{});
    std.debug.print("  Use 'abi chat' to test routing.\n\n", .{});
}

// ── Plugins ─────────────────────────────────────────────────────────────

pub fn runPlugins(allocator: std.mem.Allocator) !void {
    const path = try llm_plugin_manifest.defaultPath(allocator);
    defer allocator.free(path);

    var manifest = try llm_plugin_manifest.loadDefault(allocator);
    defer manifest.deinit();

    std.debug.print("ABI LLM Plugins\n", .{});
    std.debug.print("Manifest: {s}\n", .{path});
    std.debug.print("Configured: {d}\n", .{manifest.entries.items.len});

    if (manifest.entries.items.len == 0) {
        std.debug.print("No LLM provider plugins configured.\n", .{});
        return;
    }

    for (manifest.entries.items) |entry| {
        std.debug.print("  {s} [{s}] {s}", .{
            entry.id,
            entry.kind.label(),
            if (entry.enabled) "enabled" else "disabled",
        });
        if (entry.model) |model| std.debug.print(" model={s}", .{model});
        if (entry.base_url) |base_url| std.debug.print(" base_url={s}", .{base_url});
        if (entry.library_path) |library_path| std.debug.print(" library={s}", .{library_path});
        std.debug.print("\n", .{});
    }
}

// ── Skills ──────────────────────────────────────────────────────────────

pub fn runSkills(allocator: std.mem.Allocator) !void {
    var io_backend = initIoBackend(allocator);
    defer io_backend.deinit();
    const io = io_backend.io();

    const skills_dir = "zig-abi-plugin/skills";
    var dir = std.Io.Dir.cwd().openDir(io, skills_dir, .{ .iterate = true }) catch |err| {
        std.debug.print("ABI Skills\nDirectory: {s}\nError: {s}\n", .{ skills_dir, @errorName(err) });
        return;
    };
    defer dir.close(io);

    std.debug.print("ABI Skills\nDirectory: {s}\n", .{skills_dir});

    var count: usize = 0;
    var iter = dir.iterate();
    while (true) {
        const maybe_entry = iter.next(io) catch break;
        if (maybe_entry) |entry| {
            if (entry.kind != .directory) continue;

            const skill_path = try std.fs.path.join(allocator, &.{ skills_dir, entry.name, "SKILL.md" });
            defer allocator.free(skill_path);

            std.Io.Dir.cwd().access(io, skill_path, .{}) catch continue;
            std.debug.print("  {s} ({s})\n", .{ entry.name, skill_path });
            count += 1;
        } else break;
    }

    std.debug.print("Configured: {d}\n", .{count});
}

// ── Info ────────────────────────────────────────────────────────────────

pub fn printInfo() void {
    printHeader("ABI Framework — Architecture Summary", null);

    std.debug.print(
        \\Profiles:
        \\  Abbey  — Empathetic Polymath (warm, technical, adaptive)
        \\  Aviva  — Direct Expert (concise, factual, efficient)
        \\  Abi    — Adaptive Moderator (routing, policy, blending)
        \\
        \\Pipeline:
        \\  User Input → Abi Analysis → Modulation → Routing
        \\  → Execution → Constitution → WDBX Memory → Response
        \\
        \\Storage:
        \\  WDBX — Vector database with HNSW, DiskANN, ScaNN
        \\  Block chain — Cryptographic conversation memory (SHA-256)
        \\
        \\Inference:
        \\  Backends: demo | connector | local
        \\  16 connectors: OpenAI, Anthropic, Claude, Gemini, Mistral,
        \\    Cohere, HuggingFace, Ollama, LM Studio, vLLM, MLX,
        \\    llama.cpp, Codex, OpenCode, Discord, local-scheduler
        \\
        \\Features: 20 feature directories, {d} in catalog (mod/stub pattern)
        \\GPU backends: Metal, CUDA, Vulkan, WebGPU, OpenGL, stdgpu, FPGA, TPU
        \\Protocols: MCP, LSP, ACP, HA
        \\
        \\Spec: docs/spec/ABBEY-SPEC.md
        \\
    , .{feature_catalog.feature_count});
}

pub fn isServeInvocation(args: []const [:0]const u8) bool {
    if (args.len == 0) return false;

    const cmd = std.mem.sliceTo(args[0], 0);
    if (std.mem.eql(u8, cmd, "serve")) return true;
    if (!std.mem.eql(u8, cmd, "acp") or args.len < 2) return false;
    return std.mem.eql(u8, std.mem.sliceTo(args[1], 0), "serve");
}

fn wantsHelp(args: []const [:0]const u8) bool {
    for (args) |arg| {
        const s = std.mem.sliceTo(arg, 0);
        if (std.mem.eql(u8, s, "help") or std.mem.eql(u8, s, "--help") or std.mem.eql(u8, s, "-h")) {
            return true;
        }
    }
    return false;
}

fn printServeUsage() void {
    std.debug.print(
        \\Usage: abi serve [--addr <host:port>] [--port <port>]
        \\       abi acp serve [--addr <host:port>] [--port <port>]
        \\
    , .{});
}

pub fn parseServeAddress(allocator: std.mem.Allocator, args: []const [:0]const u8) ![]u8 {
    var address: ?[]const u8 = null;
    var port: ?u16 = null;

    var i: usize = 0;
    while (i < args.len) {
        const arg = std.mem.sliceTo(args[i], 0);
        if (std.mem.eql(u8, arg, "--addr") or std.mem.eql(u8, arg, "--address")) {
            if (i + 1 < args.len) {
                i += 1;
                address = std.mem.sliceTo(args[i], 0);
            }
        } else if (std.mem.eql(u8, arg, "--port")) {
            if (i + 1 < args.len) {
                i += 1;
                port = std.fmt.parseInt(u16, std.mem.sliceTo(args[i], 0), 10) catch 8080;
            }
        }
        i += 1;
    }

    if (address) |addr| {
        return allocator.dupe(u8, addr);
    }

    return std.fmt.allocPrint(allocator, "127.0.0.1:{d}", .{port orelse 8080});
}

pub fn runServe(allocator: std.mem.Allocator, args: []const [:0]const u8) !void {
    if (wantsHelp(args)) {
        printServeUsage();
        return;
    }

    var io_backend = std.Io.Threaded.init(allocator, .{ .environ = std.process.Environ.empty });
    defer io_backend.deinit();
    const io = io_backend.io();

    const address = try parseServeAddress(allocator, args);
    defer allocator.free(address);

    const card_url = try std.fmt.allocPrint(allocator, "http://{s}", .{address});
    defer allocator.free(card_url);

    const card = root.acp.AgentCard{
        .name = "abi",
        .description = "ABI Agent Communication Protocol server",
        .version = build_options.package_version,
        .url = card_url,
        .capabilities = .{
            .streaming = false,
            .pushNotifications = false,
            .stateTransitionHistory = true,
            .extensions = true,
        },
    };

    try root.acp.serveHttp(allocator, io, address, card);
}

fn envIsSet(name: [:0]const u8) bool {
    return std.c.getenv(name.ptr) != null;
}

fn printProbe(allocator: std.mem.Allocator, label: []const u8, command: []const u8) void {
    std.debug.print("{s}: ", .{label});
    const code = runShellCaptured(allocator, command) catch {
        std.debug.print("unavailable\n", .{});
        return;
    };
    if (code != 0) std.debug.print("  warning: probe exited with code {d}\n", .{code});
}

fn connectorConfigured(primary: [:0]const u8, fallback: ?[:0]const u8) bool {
    if (envIsSet(primary)) return true;
    if (fallback) |fb| return envIsSet(fb);
    return false;
}

// ── Check Env ──────────────────────────────────────────────────────────

pub fn runCheckEnv() !u8 {
    printHeader("ABI Environment Check", null);

    const env_vars = [_]struct { name: [:0]const u8, desc: []const u8 }{
        .{ .name = "ABI_OPENAI_API_KEY", .desc = "OpenAI API access" },
        .{ .name = "ABI_ANTHROPIC_API_KEY", .desc = "Anthropic API access" },
        .{ .name = "ABI_GEMINI_API_KEY", .desc = "Google Gemini API access" },
        .{ .name = "ABI_MISTRAL_API_KEY", .desc = "Mistral AI API access" },
        .{ .name = "ABI_COHERE_API_KEY", .desc = "Cohere API access" },
        .{ .name = "ABI_HF_API_TOKEN", .desc = "HuggingFace API access" },
        .{ .name = "ABI_DISCORD_BOT_TOKEN", .desc = "Discord bot authentication" },
        .{ .name = "ABI_OLLAMA_HOST", .desc = "Local Ollama server address" },
        .{ .name = "ABI_LM_STUDIO_HOST", .desc = "Local LM Studio server address" },
    };

    var missing_count: usize = 0;
    for (env_vars) |v| {
        const is_set = envIsSet(v.name) or (std.mem.startsWith(u8, v.name, "ABI_") and envIsSet(v.name[4..]));
        const status = if (is_set) "SET" else "NOT SET";
        const color = if (is_set) "\x1b[32m" else "\x1b[31m";
        const reset = "\x1b[0m";

        std.debug.print("  {s:<24} : {s}{s:<8}{s} ({s})\n", .{ v.name, color, status, reset, v.desc });
        if (!is_set) missing_count += 1;
    }

    if (missing_count > 0) {
        std.debug.print("\n\x1b[33mNote: {d} recommended environment variables are not set.\x1b[0m\n", .{missing_count});
    } else {
        std.debug.print("\n\x1b[32mAll critical environment variables are configured.\x1b[0m\n", .{});
    }

    return 0;
}

// ── Doctor ──────────────────────────────────────────────────────────────

pub fn runDoctor(allocator: std.mem.Allocator) !void {
    const version = build_options.package_version;

    std.debug.print(
        \\ABI Doctor — Developer Readiness Report
        \\════════════════════════════════════════
        \\
        \\Version: {s}
        \\Zig build version: {s}
        \\
        \\Toolchain:
        \\
    , .{ version, build_options.zig_version });

    std.debug.print("  zig version: {s}\n", .{build_options.zig_version});
    std.debug.print("  resolver: ./build.sh --status\n", .{});
    const zigly_status = runShellCaptured(allocator, "test -x tools/zigly") catch 1;
    std.debug.print("  zigly: {s}\n", .{if (zigly_status == 0) "present" else "missing or not executable"});

    std.debug.print("\nFeature Flags:\n", .{});

    inline for (feature_catalog.all) |entry| {
        if (entry.parent != null) continue;
        const enabled = @field(build_options, entry.compile_flag_field);
        std.debug.print("  {s} = {any}\n", .{ entry.compile_flag_field, enabled });
    }

    std.debug.print(
        \\
        \\GPU Backends:
        \\
    , .{});

    inline for (gpu_backends) |backend| {
        std.debug.print("  gpu_{s} = {any}\n", .{ backend[0], backend[1] });
    }

    var connector_count: u32 = 0;
    if (connectorConfigured("ABI_OPENAI_API_KEY", "OPENAI_API_KEY")) connector_count += 1;
    if (connectorConfigured("ABI_ANTHROPIC_API_KEY", "ANTHROPIC_API_KEY")) connector_count += 1;
    if (connectorConfigured("ABI_GEMINI_API_KEY", "GEMINI_API_KEY")) connector_count += 1;
    if (connectorConfigured("ABI_MISTRAL_API_KEY", "MISTRAL_API_KEY")) connector_count += 1;
    if (connectorConfigured("ABI_COHERE_API_KEY", "COHERE_API_KEY")) connector_count += 1;
    if (connectorConfigured("ABI_HF_API_TOKEN", "HF_API_TOKEN")) connector_count += 1;
    if (connectorConfigured("ABI_OLLAMA_HOST", "OLLAMA_HOST")) connector_count += 1;
    if (connectorConfigured("ABI_LM_STUDIO_HOST", null)) connector_count += 1;
    if (connectorConfigured("ABI_VLLM_HOST", null)) connector_count += 1;
    if (connectorConfigured("ABI_MLX_HOST", null)) connector_count += 1;
    if (connectorConfigured("ABI_LLAMA_CPP_HOST", null)) connector_count += 1;
    if (connectorConfigured("ABI_DISCORD_BOT_TOKEN", "DISCORD_BOT_TOKEN")) connector_count += 1;

    std.debug.print("\nRuntime Readiness:\n", .{});
    std.debug.print("  connectors: {d}/12 configured", .{connector_count});
    if (connector_count == 0) std.debug.print(" (optional; run 'abi connectors')", .{});
    std.debug.print("\n", .{});
    std.debug.print("  ACP_ENDPOINTS: {s}\n", .{if (envIsSet("ACP_ENDPOINTS")) "set" else "not set (optional unless checking remote ACP)"});
    const jq_status = runShellCaptured(allocator, "command -v jq >/dev/null 2>&1") catch 1;
    const curl_status = runShellCaptured(allocator, "command -v curl >/dev/null 2>&1") catch 1;
    std.debug.print("  jq: {s}\n", .{if (jq_status == 0) "available" else "missing (MCP endpoint summaries fall back to raw JSON)"});
    std.debug.print("  curl: {s}\n", .{if (curl_status == 0) "available" else "missing (health checks need curl)"});

    std.debug.print("\nMCP Config:\n", .{});
    _ = runShellDiagnostic(allocator,
        \\if command -v jq >/dev/null 2>&1; then
        \\  jq -r '.mcpServers | to_entries[] | "  " + .key + " -> http://" + .value.env.ABI_MCP_HOST + ":" + .value.env.ABI_MCP_PORT + (.value.healthCheck.path // "/health")' mcp/servers.json
        \\else
        \\  echo "  mcp/servers.json present; install jq for summarized endpoints"
        \\fi
    ) catch {};

    std.debug.print("\nRepository:\n", .{});
    const dirty = runShellDiagnostic(allocator,
        \\if git rev-parse --is-inside-work-tree >/dev/null 2>&1; then
        \\  changes="$(git status --short | sed -n '1,8p')"
        \\  if [ -n "$changes" ]; then echo "$changes"; else echo "  clean"; fi
        \\else
        \\  echo "  not a git worktree"
        \\fi
    ) catch 1;
    if (dirty != 0) std.debug.print("  warning: git status probe failed\n", .{});

    std.debug.print(
        \\
        \\Next fixes:
        \\  ./build.sh quick      Fast local validation
        \\  ./build.sh dev        Typecheck + CLI + MCP build
        \\  abi mcp health        Check MCP HA health when servers are running
        \\
    , .{});
}

// ── Chat ────────────────────────────────────────────────────────────────

pub fn runChat(allocator: std.mem.Allocator, message_args: []const [:0]const u8) !void {
    if (!build_options.feat_ai) {
        std.debug.print("AI features are disabled. Rebuild with -Dfeat-ai=true\n", .{});
        return;
    }

    const message = try cli.joinChatMessage(allocator, message_args);
    defer allocator.free(message);

    const ai = root.ai;
    var registry = ai.profile.ProfileRegistry.init(allocator, .{});
    defer registry.deinit();

    var router = ai.profile.MultiProfileRouter.init(allocator, &registry, .{});
    defer router.deinit();

    const decision = router.route(message);

    printHeader("ABI Chat — Profile Pipeline", null);

    std.debug.print(
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
    , .{
        message,
        decision.primary.name(),
        @tagName(decision.strategy),
        decision.confidence * 100.0,
        decision.reason,
        decision.weights.abbey * 100.0,
        decision.weights.aviva * 100.0,
        decision.weights.abi * 100.0,
    });

    std.debug.print("\nExecution:\n", .{});

    const inference = root.inference;
    const policy = ai.internal_api_policy;
    const model_resolution = policy.resolveModel(
        &.{
            "ollama/abbeycode",
            "ollama/llama3",
            "llama_cpp/qwen2.5",
        },
        .{ .allow_trusted_fallback = true },
    ) catch |err| {
        std.debug.print("  Internal API policy rejected model selection: {s}\n", .{@errorName(err)});
        return;
    };
    if (model_resolution.used_fallback) {
        std.debug.print("  Internal API fallback: {s} ({s})\n", .{ model_resolution.selected_model, model_resolution.reason });
    }

    var engine = inference.Engine.init(allocator, .{
        .backend = .connector,
        .model_id = model_resolution.selected_model,
        .kv_cache_pages = 100,
        .page_size = 16,
        .num_layers = 1,
        .num_heads = 1,
        .head_dim = 4,
        .max_batch_size = 8,
    }) catch |err| {
        std.debug.print("  Engine init failed: {s}\n", .{@errorName(err)});
        std.debug.print("\nHint: Ensure Ollama is running with abbeycode model.\n", .{});
        if (err == error.LocalBackendNotAvailable) {
            std.debug.print("      Local backend requires -Dfeat-llm=true build flag.\n", .{});
        }
        return;
    };
    defer engine.deinit();

    var result = engine.generate(.{
        .id = 1,
        .prompt = message,
        .max_tokens = 256,
        .temperature = 0.7,
        .top_p = 0.9,
        .top_k = 40,
        .profile_id = @intFromEnum(decision.primary),
    }) catch |err| {
        std.debug.print("  Inference failed: {s}\n", .{@errorName(err)});
        std.debug.print("\nHint: run 'abi connectors' to see required environment variables.\n", .{});
        if (err == error.EmptyPrompt) {
            std.debug.print("      The message was empty after processing.\n", .{});
        }
        return;
    };
    defer result.deinit(allocator);

    // Detect demo vs local backend response
    const is_demo = std.mem.startsWith(u8, result.text, "[");
    const backend_label: []const u8 = if (is_demo) "demo" else "local";

    std.debug.print("  [{s}] {s} | {d} tokens | {d:.1}ms\n", .{
        backend_label,
        engine.config.model_id,
        result.completion_tokens,
        result.latency_ms,
    });

    var wdbx_block_id: ?u64 = null;
    if (comptime build_options.feat_database) {
        var memory = root.ai.profile.ConversationMemory.init(allocator, "abi-chat");
        defer memory.deinit();
        const response_copy = try allocator.dupe(u8, result.text);
        var profile_response = root.ai.profile.ProfileResponse{
            .profile = decision.primary,
            .content = response_copy,
            .confidence = decision.confidence,
            .allocator = allocator,
        };
        defer profile_response.deinit();
        wdbx_block_id = memory.recordInteraction(decision, message, profile_response, null) catch |err| blk: {
            std.debug.print("  Warning: Failed to record WDBX memory block: {s}\n", .{@errorName(err)});
            break :blk null;
        };
    }

    var learning_runtime: ?root.ai.learning.LearningRuntime = null;
    if (root.ai.learning.LearningRuntime.init(allocator)) |runtime| {
        learning_runtime = runtime;
    } else |err| {
        std.debug.print("  Warning: Learning runtime init failed: {s}\n", .{@errorName(err)});
    }
    if (learning_runtime) |*rt| {
        defer rt.deinit();
        rt.recordInteraction(.{
            .prompt = message,
            .response = result.text,
            .profile = decision.primary.name(),
            .backend = backend_label,
            .latency_ms = result.latency_ms,
            .selected_model = engine.config.model_id,
            .wdbx_block_id = wdbx_block_id,
            .route_reason = decision.reason,
            .constitution_passed = true,
            .used_fallback_provider = model_resolution.used_fallback,
        }) catch |err| {
            std.debug.print("  Warning: Failed to record interaction: {s}\n", .{@errorName(err)});
        };
    }

    // Write the actual response to stdout so it can be piped.
    // Metadata goes to stderr (via std.debug.print above); response to stdout.
    try writeToStdout("\n");
    try writeToStdout(result.text);
    try writeToStdout("\n");
}

// ── Database ────────────────────────────────────────────────────────────

pub fn runDb(allocator: std.mem.Allocator, args: []const [:0]const u8) !void {
    const db_cli = root.database;
    if (comptime @hasDecl(db_cli, "cli")) {
        try db_cli.cli.run(allocator, args);
    } else {
        std.debug.print("Database is disabled. Rebuild with -Dfeat-database=true\n", .{});
    }
}

// ── Search ──────────────────────────────────────────────────────────────

pub fn runSearch(allocator: std.mem.Allocator, args: []const [:0]const u8) !void {
    const search = root.search;
    if (!comptime build_options.feat_search) {
        std.debug.print("Search is disabled. Rebuild with -Dfeat-search=true\n", .{});
        return;
    }

    search.init(allocator, .{}) catch |err| {
        std.debug.print("Failed to initialize search: {s}\n", .{@errorName(err)});
        return;
    };
    defer search.deinit();

    if (args.len == 0) {
        printSearchHelp();
        return;
    }

    const subcmd = args[0];
    if (std.mem.eql(u8, subcmd, "create")) {
        if (args.len < 2) {
            std.debug.print("Usage: abi search create <index_name>\n", .{});
            return;
        }
        _ = search.createIndex(allocator, args[1]) catch |err| {
            std.debug.print("Error: {s}\n", .{@errorName(err)});
            return;
        };
        std.debug.print("Index '{s}' created.\n", .{args[1]});
    } else if (std.mem.eql(u8, subcmd, "index")) {
        if (args.len < 4) {
            std.debug.print("Usage: abi search index <index_name> <doc_id> <content...>\n", .{});
            return;
        }
        // Join remaining args as content
        var content_buf: [4096]u8 = undefined;
        var content_len: usize = 0;
        for (args[3..]) |arg| {
            if (content_len > 0 and content_len < content_buf.len) {
                content_buf[content_len] = ' ';
                content_len += 1;
            }
            const copy_len = @min(arg.len, content_buf.len - content_len);
            @memcpy(content_buf[content_len..][0..copy_len], arg[0..copy_len]);
            content_len += copy_len;
        }
        if (content_len >= content_buf.len) {
            std.debug.print("Warning: content truncated to {d} bytes\n", .{content_buf.len});
        }
        search.indexDocument(args[1], args[2], content_buf[0..content_len]) catch |err| {
            std.debug.print("Error: {s}\n", .{@errorName(err)});
            return;
        };
        std.debug.print("Document '{s}' indexed in '{s}'.\n", .{ args[2], args[1] });
    } else if (std.mem.eql(u8, subcmd, "query")) {
        if (args.len < 3) {
            std.debug.print("Usage: abi search query <index_name> <query_text...>\n", .{});
            return;
        }
        var query_buf: [2048]u8 = undefined;
        var query_len: usize = 0;
        for (args[2..]) |arg| {
            if (query_len > 0 and query_len < query_buf.len) {
                query_buf[query_len] = ' ';
                query_len += 1;
            }
            const copy_len = @min(arg.len, query_buf.len - query_len);
            @memcpy(query_buf[query_len..][0..copy_len], arg[0..copy_len]);
            query_len += copy_len;
        }
        if (query_len >= query_buf.len) {
            std.debug.print("Warning: query truncated to {d} bytes\n", .{query_buf.len});
        }
        const results = search.query(allocator, args[1], query_buf[0..query_len]) catch |err| {
            std.debug.print("Error: {s}\n", .{@errorName(err)});
            return;
        };
        defer allocator.free(results);
        if (results.len == 0) {
            std.debug.print("No results found.\n", .{});
        } else {
            for (results, 0..) |result, i| {
                std.debug.print("{d}. [{d:.3}] {s}", .{ i + 1, result.score, result.doc_id });
                if (result.snippet.len > 0) {
                    std.debug.print(" — {s}", .{result.snippet});
                }
                std.debug.print("\n", .{});
            }
        }
    } else if (std.mem.eql(u8, subcmd, "delete")) {
        if (args.len < 2) {
            std.debug.print("Usage: abi search delete <index_name> [doc_id]\n", .{});
            return;
        }
        if (args.len >= 3) {
            const removed = search.deleteDocument(args[1], args[2]) catch |err| {
                std.debug.print("Error: {s}\n", .{@errorName(err)});
                return;
            };
            if (removed) {
                std.debug.print("Document '{s}' deleted from '{s}'.\n", .{ args[2], args[1] });
            } else {
                std.debug.print("Document '{s}' not found in '{s}'.\n", .{ args[2], args[1] });
            }
        } else {
            search.deleteIndex(args[1]) catch |err| {
                std.debug.print("Error: {s}\n", .{@errorName(err)});
                return;
            };
            std.debug.print("Index '{s}' deleted.\n", .{args[1]});
        }
    } else if (std.mem.eql(u8, subcmd, "stats")) {
        const s = search.stats();
        std.debug.print("Search Statistics:\n  Indexes: {d}\n  Documents: {d}\n  Terms: {d}\n", .{
            s.total_indexes, s.total_documents, s.total_terms,
        });
    } else if (std.mem.eql(u8, subcmd, "help")) {
        printSearchHelp();
    } else {
        std.debug.print("Unknown search command: {s}\n", .{subcmd});
        printSearchHelp();
    }
}

fn printSearchHelp() void {
    std.debug.print(
        \\Usage: abi search <command> [args]
        \\
        \\Commands:
        \\  create <index>                  Create a new search index
        \\  index <index> <doc_id> <text>   Add/update document in index
        \\  query <index> <query_text>      BM25 full-text search
        \\  delete <index> [doc_id]         Delete index or document
        \\  stats                           Show search index statistics
        \\  help                            Show this help
        \\
    , .{});
}

// ── LSP ─────────────────────────────────────────────────────────────────

pub fn runLsp(allocator: std.mem.Allocator) !void {
    _ = allocator;
    if (!build_options.feat_lsp) {
        std.debug.print("LSP is disabled. Rebuild with -Dfeat-lsp=true\n", .{});
        return;
    }
    // LSP client is available; server functionality is provided by external LSP servers (e.g., ZLS).
    std.debug.print("LSP client module loaded. Use 'abi lsp' with an external LSP server.\n", .{});
    std.debug.print("Note: ABI provides LSP client capabilities. For Zig language support, use ZLS directly.\n", .{});
}

// ── Dashboard ───────────────────────────────────────────────────────────

pub fn runDashboard(allocator: std.mem.Allocator) !void {
    if (build_options.feat_tui) {
        if (comptime @hasDecl(root.tui.dashboard, "run")) {
            return root.tui.dashboard.run(allocator);
        }
    }
    std.debug.print("TUI is disabled. Rebuild with -Dfeat-tui=true\n", .{});
}

// ── Tests ───────────────────────────────────────────────────────────────

test "version prints without error" {
    printVersion();
}

test "help prints without error" {
    printHelp();
}

test "status prints without error" {
    printStatus();
}

test "info prints without error" {
    printInfo();
}

test "doctor runs without error" {
    try runDoctor(std.testing.allocator);
}

test "features prints without error" {
    printFeatures();
}

test "platform prints without error" {
    printPlatform();
}

test "connectors prints without error" {
    printConnectors();
}

test "plugins command runs without error" {
    try runPlugins(std.testing.allocator);
}

test "skills command runs without error" {
    try runSkills(std.testing.allocator);
}

test "chat routes message without error" {
    const message_args = [_][:0]const u8{ "Hello,", "how", "are", "you?" };
    try runChat(std.testing.allocator, &message_args);
}

test {
    std.testing.refAllDecls(@This());
}
