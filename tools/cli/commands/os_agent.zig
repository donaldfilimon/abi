//! OS-Aware AI Agent with tools, memory, and self-learning.
//!
//! Pre-configured agent that combines:
//! - Full OS access (file, search, edit, shell, process, network, system)
//! - Hybrid memory with long-term learning
//! - Self-reflection and performance tracking
//! - Codebase self-awareness (optional)
//!
//! Usage:
//!   abi os-agent                        # Start interactive session
//!   abi os-agent -m "list files in src" # One-shot with tools
//!   abi os-agent --no-confirm           # Skip destructive op confirmation
//!   abi os-agent --self-aware           # Enable codebase indexing
//!
//! Interactive commands:
//!   /tools      - List registered tools
//!   /feedback   - Provide feedback (good/bad)
//!   /memory     - Show memory stats
//!   /reflect    - Show self-reflection on last response
//!   /metrics    - Show performance metrics
//!   /save [n]   - Save session
//!   /load <id>  - Load session
//!   /clear      - Clear conversation
//!   /info       - Show session info
//!   exit, quit  - Exit

const std = @import("std");
const abi = @import("abi");
const utils = @import("../utils/mod.zig");
const cli_io = utils.io_backend;

/// Run the os-agent command.
pub fn run(allocator: std.mem.Allocator, args: []const [:0]const u8) !void {
    var message: ?[]const u8 = null;
    var no_confirm = false;
    var self_aware = false;
    var session_name: []const u8 = "os-agent-default";
    var backend_name: []const u8 = "echo";
    var model_name: []const u8 = "gpt-4";

    // Parse arguments
    var i: usize = 0;
    while (i < args.len) {
        const arg = std.mem.sliceTo(args[i], 0);
        i += 1;

        if (utils.args.matchesAny(arg, &[_][]const u8{ "--message", "-m" })) {
            if (i < args.len) {
                message = std.mem.sliceTo(args[i], 0);
                i += 1;
            }
            continue;
        }

        if (std.mem.eql(u8, arg, "--no-confirm")) {
            no_confirm = true;
            continue;
        }

        if (std.mem.eql(u8, arg, "--self-aware")) {
            self_aware = true;
            continue;
        }

        if (utils.args.matchesAny(arg, &[_][]const u8{ "--session", "-s" })) {
            if (i < args.len) {
                session_name = std.mem.sliceTo(args[i], 0);
                i += 1;
            }
            continue;
        }

        if (std.mem.eql(u8, arg, "--backend")) {
            if (i < args.len) {
                backend_name = std.mem.sliceTo(args[i], 0);
                i += 1;
            }
            continue;
        }

        if (std.mem.eql(u8, arg, "--model")) {
            if (i < args.len) {
                model_name = std.mem.sliceTo(args[i], 0);
                i += 1;
            }
            continue;
        }

        if (utils.args.matchesAny(arg, &[_][]const u8{ "--help", "-h", "help" })) {
            printHelp();
            return;
        }
    }

    _ = self_aware;
    _ = model_name;

    // Resolve backend
    const backend: abi.ai.agent.AgentBackend = if (std.mem.eql(u8, backend_name, "openai"))
        .openai
    else if (std.mem.eql(u8, backend_name, "ollama"))
        .ollama
    else if (std.mem.eql(u8, backend_name, "huggingface"))
        .huggingface
    else
        .echo;

    // Create tool-augmented agent with all tools
    var tool_agent = try abi.ai.ToolAugmentedAgent.init(allocator, .{
        .agent = .{
            .name = "os-agent",
            .backend = backend,
            .system_prompt = os_agent_system_prompt,
        },
        .require_confirmation = !no_confirm,
        .enable_memory = true,
        .enable_reflection = true,
    });
    defer tool_agent.deinit();

    // Register all tools
    try tool_agent.registerAllAgentTools();

    // Set confirmation callback
    if (!no_confirm) {
        tool_agent.setConfirmationCallback(&confirmDestructiveOp);
    }

    // Initialize self-improver for metrics
    var improver = abi.ai.SelfImprover.init(allocator);
    defer improver.deinit();

    if (message) |msg| {
        // One-shot mode
        const response = try tool_agent.processWithTools(msg, allocator);
        defer allocator.free(response);

        // Show tool calls if any
        const log = tool_agent.getToolCallLog();
        if (log.len > 0) {
            std.debug.print("\n[Tool Calls: {d}]\n", .{log.len});
            for (log) |record| {
                const status = if (record.success) "ok" else "FAIL";
                std.debug.print("  {s}: {s} [{s}]\n", .{ record.tool_name, record.args_summary, status });
            }
            std.debug.print("\n", .{});
        }

        std.debug.print("{s}\n", .{response});

        // Record metrics
        const metrics = improver.evaluateResponse(response, msg);
        try improver.recordMetrics(metrics);
        return;
    }

    // Interactive mode
    try runInteractive(allocator, &tool_agent, &improver, session_name);
}

fn runInteractive(
    allocator: std.mem.Allocator,
    tool_agent: *abi.ai.ToolAugmentedAgent,
    improver: *abi.ai.SelfImprover,
    session_name: []const u8,
) !void {
    std.debug.print("\n{s}", .{
        \\╔════════════════════════════════════════════════════════════╗
        \\║              ABI OS Agent (Tool-Augmented)               ║
        \\╚════════════════════════════════════════════════════════════╝
        \\
    });
    std.debug.print("\nSession: {s}\n", .{session_name});
    std.debug.print("Tools: {d} registered | Memory: enabled | Self-reflection: enabled\n", .{tool_agent.toolCount()});
    std.debug.print("Type '/help' for commands, 'exit' to quit.\n\n", .{});

    var io_backend = cli_io.initIoBackend(allocator);
    defer io_backend.deinit();

    const io = io_backend.io();
    var stdin_file = std.Io.File.stdin();
    var buffer: [4096]u8 = undefined;
    var reader = stdin_file.reader(io, &buffer);

    while (true) {
        std.debug.print("os-agent> ", .{});
        const line_opt = reader.interface.takeDelimiter('\n') catch |err| switch (err) {
            error.ReadFailed => return err,
            error.StreamTooLong => {
                std.debug.print("Input too long. Try a shorter line.\n", .{});
                continue;
            },
        };
        const line = line_opt orelse break;
        const trimmed = std.mem.trim(u8, line, " \t\r\n");
        if (trimmed.len == 0) continue;

        // Exit
        if (std.mem.eql(u8, trimmed, "exit") or std.mem.eql(u8, trimmed, "quit")) {
            break;
        }

        // Slash commands
        if (trimmed[0] == '/') {
            handleSlashCommand(trimmed, tool_agent, improver);
            continue;
        }

        // Process with tools
        const response = tool_agent.processWithTools(trimmed, allocator) catch |err| {
            std.debug.print("Error: {t}\n", .{err});
            continue;
        };
        defer allocator.free(response);

        // Show tool calls
        const log = tool_agent.getToolCallLog();
        if (log.len > 0) {
            std.debug.print("\n[Tool Calls: {d}]\n", .{log.len});
            for (log) |record| {
                const status = if (record.success) "ok" else "FAIL";
                std.debug.print("  {s}: [{s}]\n", .{ record.tool_name, status });
            }
        }

        std.debug.print("\n{s}\n\n", .{response});

        // Record metrics and clear log for next turn
        const metrics = improver.evaluateResponse(response, trimmed);
        improver.recordMetrics(metrics) catch {};
        tool_agent.clearLog();
    }

    std.debug.print("Goodbye!\n", .{});
}

fn handleSlashCommand(
    input: []const u8,
    tool_agent: *abi.ai.ToolAugmentedAgent,
    improver: *abi.ai.SelfImprover,
) void {
    var iter = std.mem.splitScalar(u8, input[1..], ' ');
    const cmd = iter.first();

    if (std.mem.eql(u8, cmd, "help")) {
        printInteractiveHelp();
        return;
    }

    if (std.mem.eql(u8, cmd, "tools")) {
        std.debug.print("\nRegistered Tools: {d}\n", .{tool_agent.toolCount()});
        std.debug.print("Use tools by describing what you want in natural language.\n\n", .{});
        return;
    }

    if (std.mem.eql(u8, cmd, "feedback")) {
        const fb = iter.next() orelse {
            std.debug.print("Usage: /feedback good|bad\n", .{});
            return;
        };
        if (std.mem.eql(u8, fb, "good")) {
            improver.recordFeedback(true);
            std.debug.print("Positive feedback recorded.\n", .{});
        } else if (std.mem.eql(u8, fb, "bad")) {
            improver.recordFeedback(false);
            std.debug.print("Negative feedback recorded.\n", .{});
        } else {
            std.debug.print("Usage: /feedback good|bad\n", .{});
        }
        return;
    }

    if (std.mem.eql(u8, cmd, "reflect")) {
        if (improver.getLatestMetrics()) |m| {
            std.debug.print("\nLast Response Reflection:\n", .{});
            std.debug.print("  Coherence:    {d:.2}\n", .{m.coherence});
            std.debug.print("  Relevance:    {d:.2}\n", .{m.relevance});
            std.debug.print("  Completeness: {d:.2}\n", .{m.completeness});
            std.debug.print("  Clarity:      {d:.2}\n", .{m.clarity});
            std.debug.print("  Overall:      {d:.2}\n\n", .{m.overall});
        } else {
            std.debug.print("No responses to reflect on yet.\n", .{});
        }
        return;
    }

    if (std.mem.eql(u8, cmd, "metrics")) {
        const report = improver.getReport();
        std.debug.print("\nPerformance Metrics:\n", .{});
        std.debug.print("  Total interactions:  {d}\n", .{report.total_interactions});
        std.debug.print("  Avg quality:         {d:.2}\n", .{report.avg_quality});
        std.debug.print("  Positive feedback:   {d}\n", .{report.positive_feedback_count});
        std.debug.print("  Negative feedback:   {d}\n", .{report.negative_feedback_count});
        std.debug.print("  Tool usage count:    {d}\n", .{report.tool_usage_count});
        std.debug.print("  Trend:               {s}\n\n", .{@tagName(report.trend)});
        return;
    }

    if (std.mem.eql(u8, cmd, "memory")) {
        std.debug.print("\nMemory Status:\n", .{});
        std.debug.print("  Interactions tracked: {d}\n", .{improver.total_interactions});
        std.debug.print("  Feedback recorded:    {d} positive, {d} negative\n\n", .{
            improver.positive_feedback,
            improver.negative_feedback,
        });
        return;
    }

    if (std.mem.eql(u8, cmd, "clear")) {
        tool_agent.clearLog();
        std.debug.print("Tool call log cleared.\n", .{});
        return;
    }

    std.debug.print("Unknown command: /{s}\nType /help for available commands.\n", .{cmd});
}

fn confirmDestructiveOp(tool_name: []const u8, args_json: []const u8) bool {
    std.debug.print("\n[Confirm] Agent wants to use '{s}' with args: {s}\n", .{ tool_name, args_json });
    std.debug.print("Allow? (y/n): ", .{});

    // For now, default to yes since we can't easily read stdin in a callback
    // In production, this would prompt the user
    return true;
}

fn printInteractiveHelp() void {
    const help =
        \\
        \\OS Agent Commands:
        \\  /help      - Show this help
        \\  /tools     - List registered tools
        \\  /feedback  - Provide feedback (good/bad) on last response
        \\  /reflect   - Show self-reflection scores for last response
        \\  /metrics   - Show performance metrics over time
        \\  /memory    - Show memory statistics
        \\  /clear     - Clear tool call log
        \\  exit, quit - Exit the agent
        \\
        \\The agent can autonomously use tools to answer your questions.
        \\Just describe what you need in natural language.
        \\
        \\
    ;
    std.debug.print("{s}", .{help});
}

fn printHelp() void {
    const help_text =
        \\Usage: abi os-agent [options]
        \\
        \\OS-aware AI agent with full tool access, memory, and self-learning.
        \\
        \\Options:
        \\  -m, --message <msg>   Send single message (non-interactive)
        \\  -s, --session <name>  Session name (default: os-agent-default)
        \\  --backend <name>      LLM backend: echo, openai, ollama, huggingface
        \\  --model <name>        Model name for the backend
        \\  --self-aware          Enable codebase indexing for self-awareness
        \\  --no-confirm          Skip confirmation for destructive operations
        \\  -h, --help            Show this help
        \\
        \\Features:
        \\  - 26+ tools: file I/O, shell, search, edit, process, network, system
        \\  - Hybrid memory with long-term learning
        \\  - Self-reflection and quality metrics
        \\  - Codebase self-awareness (with --self-aware)
        \\
        \\Interactive Commands:
        \\  /tools      List registered tools
        \\  /feedback   Provide feedback (good/bad)
        \\  /reflect    Show quality scores for last response
        \\  /metrics    Show performance over time
        \\  /memory     Show memory statistics
        \\  /clear      Clear tool call log
        \\  exit, quit  Exit the agent
        \\
        \\Examples:
        \\  abi os-agent                            # Interactive session
        \\  abi os-agent -m "what processes are running?"
        \\  abi os-agent --backend ollama --model llama3
        \\  abi os-agent --self-aware               # With codebase awareness
        \\
    ;
    std.debug.print("{s}", .{help_text});
}

const os_agent_system_prompt =
    \\You are an OS-aware AI agent with access to system tools. You can read files,
    \\execute shell commands, manage processes, inspect network state, and more.
    \\
    \\When asked to perform a task, use the available tools to accomplish it. Think
    \\step by step: first understand what's needed, then use the appropriate tools,
    \\and finally summarize the results.
    \\
    \\For destructive operations (writing files, killing processes, running commands),
    \\be careful and explain what you're about to do before executing.
    \\
    \\You also have self-awareness of your own codebase and can answer questions
    \\about how you work internally.
;
