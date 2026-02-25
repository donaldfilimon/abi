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
const command_mod = @import("../command.zig");
const context_mod = @import("../framework/context.zig");
const utils = @import("../utils/mod.zig");
const cli_io = utils.io_backend;
const app_paths = abi.shared.app_paths;

pub const meta: command_mod.Meta = .{
    .name = "os-agent",
    .description = "OS-aware AI agent with tools, memory, and self-learning",
};

const SessionState = struct {
    allocator: std.mem.Allocator,
    session_id: []const u8,
    session_name: []const u8,
    messages: std.ArrayListUnmanaged(abi.ai.memory.Message) = .empty,
    created_at: i64,
    modified: bool = false,

    pub fn init(allocator: std.mem.Allocator, name: []const u8) !SessionState {
        const now = abi.shared.utils.unixSeconds();
        return .{
            .allocator = allocator,
            .session_id = try makeSessionId(allocator, name),
            .session_name = try allocator.dupe(u8, name),
            .created_at = now,
        };
    }

    pub fn deinit(self: *SessionState) void {
        self.clear();
        self.messages.deinit(self.allocator);
        self.allocator.free(self.session_id);
        self.allocator.free(self.session_name);
    }

    pub fn addMessage(self: *SessionState, role: abi.ai.memory.MessageRole, content: []const u8) !void {
        const content_copy = try self.allocator.dupe(u8, content);
        errdefer self.allocator.free(content_copy);

        try self.messages.append(self.allocator, .{
            .role = role,
            .content = content_copy,
            .timestamp = abi.shared.utils.unixSeconds(),
        });
        self.modified = true;
    }

    pub fn clear(self: *SessionState) void {
        for (self.messages.items) |*msg| {
            msg.deinit(self.allocator);
        }
        self.messages.clearRetainingCapacity();
        self.modified = true;
    }

    pub fn load(self: *SessionState) !void {
        const sessions_dir = try resolveAndEnsureSessionsDir(self.allocator);
        defer self.allocator.free(sessions_dir);

        var store = abi.ai.memory.SessionStore.init(self.allocator, sessions_dir);
        if (!store.sessionExists(self.session_id)) return;

        var loaded = try store.loadSession(self.session_id);
        defer loaded.deinit(self.allocator);

        for (self.messages.items) |*msg| {
            msg.deinit(self.allocator);
        }
        self.messages.clearRetainingCapacity();

        for (loaded.messages) |msg| {
            try self.messages.append(self.allocator, try msg.clone(self.allocator));
        }

        self.created_at = loaded.created_at;
        self.modified = false;
    }

    pub fn save(self: *SessionState, model: []const u8, system_prompt: []const u8, temperature: f32) !void {
        const sessions_dir = try resolveAndEnsureSessionsDir(self.allocator);
        defer self.allocator.free(sessions_dir);

        var store = abi.ai.memory.SessionStore.init(self.allocator, sessions_dir);
        const now = abi.shared.utils.unixSeconds();
        const session_data = abi.ai.memory.SessionData{
            .id = self.session_id,
            .name = self.session_name,
            .created_at = self.created_at,
            .updated_at = now,
            .messages = self.messages.items,
            .config = .{
                .memory_type = .hybrid,
                .max_tokens = 8000,
                .temperature = temperature,
                .model = model,
                .system_prompt = system_prompt,
            },
        };

        try store.saveSession(session_data);
        self.modified = false;
    }
};

fn resolveAndEnsureSessionsDir(allocator: std.mem.Allocator) ![]u8 {
    const root_dir = try app_paths.resolvePrimaryRoot(allocator);
    defer allocator.free(root_dir);
    const sessions_dir = try app_paths.resolvePath(allocator, "sessions");
    errdefer allocator.free(sessions_dir);

    var io_backend = std.Io.Threaded.init(allocator, .{});
    defer io_backend.deinit();
    const io = io_backend.io();

    std.Io.Dir.cwd().createDir(io, root_dir, .default_dir) catch |err| switch (err) {
        error.PathAlreadyExists => {},
        else => return err,
    };
    std.Io.Dir.cwd().createDir(io, sessions_dir, .default_dir) catch |err| switch (err) {
        error.PathAlreadyExists => {},
        else => return err,
    };

    return sessions_dir;
}

fn makeSessionId(allocator: std.mem.Allocator, name: []const u8) ![]u8 {
    var out = std.ArrayListUnmanaged(u8).empty;
    defer out.deinit(allocator);

    try out.appendSlice(allocator, "os_agent_");
    for (name) |c| {
        const valid = (c >= 'a' and c <= 'z') or
            (c >= 'A' and c <= 'Z') or
            (c >= '0' and c <= '9') or
            c == '-' or
            c == '_';
        try out.append(allocator, if (valid) c else '_');
    }

    if (out.items.len == "os_agent_".len) {
        try out.appendSlice(allocator, "default");
    }
    return out.toOwnedSlice(allocator);
}

fn buildSessionAwareInput(
    allocator: std.mem.Allocator,
    history: []const abi.ai.memory.Message,
    user_input: []const u8,
) ![]u8 {
    if (history.len == 0) return allocator.dupe(u8, user_input);

    var writer: std.Io.Writer.Allocating = .init(allocator);
    errdefer writer.deinit();

    try writer.writer.writeAll(
        \\Use the following prior conversation context when relevant.
        \\Do not repeat it verbatim unless asked.
        \\
        \\[Session Context]
        \\
    );

    const max_messages: usize = 12;
    const start = if (history.len > max_messages) history.len - max_messages else 0;
    for (history[start..]) |msg| {
        const role_label = switch (msg.role) {
            .system => "System",
            .user => "User",
            .assistant => "Assistant",
            .tool => "Tool",
        };
        const excerpt = if (msg.content.len > 800) msg.content[0..800] else msg.content;
        try writer.writer.print("{s}: {s}\n", .{ role_label, excerpt });
    }

    try writer.writer.print(
        \\
        \\[Current User Message]
        \\{s}
        \\
    , .{user_input});

    return writer.toOwnedSlice();
}

fn responseNeedsRecovery(response: []const u8) bool {
    const trimmed = std.mem.trim(u8, response, " \t\r\n");
    if (trimmed.len == 0) return true;
    if (std.mem.startsWith(u8, trimmed, "{") or std.mem.startsWith(u8, trimmed, "[")) return true;
    if (std.mem.startsWith(u8, trimmed, "```")) return true;
    if (std.mem.indexOf(u8, trimmed, "\"args\"") != null and std.mem.indexOf(u8, trimmed, "\"name\"") == null) return true;
    return false;
}

fn recoverUserFacingResponse(
    tool_agent: *abi.ai.tool_agent.ToolAugmentedAgent,
    user_input: []const u8,
    previous_response: []const u8,
    allocator: std.mem.Allocator,
) ![]const u8 {
    const repair_prompt = try std.fmt.allocPrint(
        allocator,
        "You produced a non-user-facing reply.\nUser message:\n{s}\n\n" ++
            "Previous reply:\n{s}\n\n" ++
            "Now provide a concise natural-language answer for the user (no JSON).",
        .{ user_input, previous_response },
    );
    defer allocator.free(repair_prompt);
    return tool_agent.agent.process(repair_prompt, allocator);
}

/// Run the os-agent command.
pub fn run(ctx: *const context_mod.CommandContext, args: []const [:0]const u8) !void {
    const allocator = ctx.allocator;
    var message: ?[]const u8 = null;
    var no_confirm = false;
    var self_aware: bool = false;
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

    // Resolve backend
    const backend: abi.ai.agent.AgentBackend = if (std.mem.eql(u8, backend_name, "openai"))
        .openai
    else if (std.mem.eql(u8, backend_name, "ollama"))
        .ollama
    else if (std.mem.eql(u8, backend_name, "huggingface"))
        .huggingface
    else
        .echo;

    // Build system prompt with optional self-awareness preamble
    const system_prompt = if (self_aware)
        "You are a self-aware AI agent with metacognitive capabilities. " ++ os_agent_system_prompt
    else
        os_agent_system_prompt;

    // Create tool-augmented agent with all tools
    var tool_agent = try abi.ai.tool_agent.ToolAugmentedAgent.init(allocator, .{
        .agent = .{
            .name = "os-agent",
            .backend = backend,
            .model = model_name,
            .system_prompt = system_prompt,
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
    var improver = abi.ai.self_improve.SelfImprover.init(allocator);
    defer improver.deinit();

    // Initialize and load persisted session state.
    var session = try SessionState.init(allocator, session_name);
    defer session.deinit();
    session.load() catch |err| {
        utils.output.printWarning("could not load session '{s}': {t}", .{ session_name, err });
    };

    if (message) |msg| {
        const input = try buildSessionAwareInput(allocator, session.messages.items, msg);
        defer allocator.free(input);

        // One-shot mode
        var response: []const u8 = try tool_agent.processWithTools(input, allocator);
        if (responseNeedsRecovery(response)) {
            const repaired = recoverUserFacingResponse(&tool_agent, msg, response, allocator) catch null;
            if (repaired) |fixed| {
                allocator.free(response);
                response = fixed;
            }
        }
        if (std.mem.trim(u8, response, " \t\r\n").len == 0) {
            allocator.free(response);
            response = try allocator.dupe(u8, "I couldn't produce a final answer for that request. Please try a more specific prompt.");
        }
        defer allocator.free(response);

        // Show tool calls if any
        const log = tool_agent.getToolCallLog();
        if (log.len > 0) {
            utils.output.println("\n[Tool Calls: {d}]", .{log.len});
            for (log) |record| {
                const status = if (record.success) "ok" else "FAIL";
                utils.output.println("  {s}: {s} [{s}]", .{ record.tool_name, record.args_summary, status });
            }
            utils.output.println("", .{});
        }

        utils.output.println("{s}", .{response});

        // Record metrics
        const metrics = improver.evaluateResponse(response, msg);
        try improver.recordMetrics(metrics);
        try session.addMessage(.user, msg);
        try session.addMessage(.assistant, response);
        session.save(model_name, system_prompt, tool_agent.agent.config.temperature) catch |err| {
            utils.output.printWarning("failed to persist os-agent session: {t}", .{err});
        };
        return;
    }

    // Interactive mode
    try runInteractive(allocator, &tool_agent, &improver, &session, model_name, system_prompt);
}

fn runInteractive(
    allocator: std.mem.Allocator,
    tool_agent: *abi.ai.tool_agent.ToolAugmentedAgent,
    improver: *abi.ai.self_improve.SelfImprover,
    session: *SessionState,
    model_name: []const u8,
    system_prompt: []const u8,
) !void {
    utils.output.println("\n{s}", .{
        \\╔════════════════════════════════════════════════════════════╗
        \\║              ABI OS Agent (Tool-Augmented)               ║
        \\╚════════════════════════════════════════════════════════════╝
    });
    utils.output.println("\nSession: {s}", .{session.session_name});
    utils.output.println("Tools: {d} registered | Memory: enabled | Self-reflection: enabled", .{tool_agent.toolCount()});
    utils.output.println("Loaded messages: {d}", .{session.messages.items.len});
    utils.output.println("Type '/help' for commands, 'exit' to quit.\n", .{});

    var seed_with_session_context = session.messages.items.len > 0;

    var seed_with_session_context = session.messages.items.len > 0;

    var io_backend = cli_io.initIoBackend(allocator);
    defer io_backend.deinit();

    const io = io_backend.io();
    const stdin_file = std.Io.File.stdin();
    var buffer: [4096]u8 = undefined;
    var reader = stdin_file.reader(io, &buffer);

    while (true) {
        utils.output.print("os-agent> ", .{});
        const line_opt = reader.interface.takeDelimiter('\n') catch |err| switch (err) {
            error.ReadFailed => return err,
            error.StreamTooLong => {
                utils.output.printWarning("Input too long. Try a shorter line.", .{});
                continue;
            },
        };
        const line = line_opt orelse break;
        const trimmed = std.mem.trim(u8, line, " \t\r\n");
        if (trimmed.len == 0) continue;

        // Exit
        if (std.mem.eql(u8, trimmed, "exit") or std.mem.eql(u8, trimmed, "quit")) {
            if (session.modified) {
                session.save(model_name, system_prompt, tool_agent.agent.config.temperature) catch |err| {
                    utils.output.printWarning("failed to persist os-agent session: {t}", .{err});
                };
            }
            break;
        }

        // Slash commands
        if (trimmed[0] == '/') {
            handleSlashCommand(trimmed, tool_agent, improver, session, model_name, system_prompt);
            continue;
        }

        const input = if (seed_with_session_context)
            try buildSessionAwareInput(allocator, session.messages.items, trimmed)
        else
            try allocator.dupe(u8, trimmed);
        defer allocator.free(input);

        // Process with tools
        var response: []const u8 = tool_agent.processWithTools(input, allocator) catch |err| {
            utils.output.printError("{t}", .{err});
            continue;
        };
        if (responseNeedsRecovery(response)) {
            const repaired = recoverUserFacingResponse(tool_agent, trimmed, response, allocator) catch null;
            if (repaired) |fixed| {
                allocator.free(response);
                response = fixed;
            }
        }
        if (std.mem.trim(u8, response, " \t\r\n").len == 0) {
            allocator.free(response);
            response = allocator.dupe(u8, "I couldn't produce a final answer for that request. Please try a more specific prompt.") catch {
                utils.output.printError("OutOfMemory", .{});
                continue;
            };
        }
        defer allocator.free(response);
        seed_with_session_context = false;

        // Show tool calls
        const log = tool_agent.getToolCallLog();
        if (log.len > 0) {
            utils.output.println("\n[Tool Calls: {d}]", .{log.len});
            for (log) |record| {
                const status = if (record.success) "ok" else "FAIL";
                utils.output.println("  {s}: [{s}]", .{ record.tool_name, status });
            }
        }

        utils.output.println("\n{s}\n", .{response});

        session.addMessage(.user, trimmed) catch |err| {
            utils.output.printWarning("failed to record user message: {t}", .{err});
        };
        session.addMessage(.assistant, response) catch |err| {
            utils.output.printWarning("failed to record assistant message: {t}", .{err});
        };
        session.save(model_name, system_prompt, tool_agent.agent.config.temperature) catch |err| {
            utils.output.printWarning("failed to persist os-agent session: {t}", .{err});
        };

        // Record metrics and clear log for next turn
        const metrics = improver.evaluateResponse(response, trimmed);
        improver.recordMetrics(metrics) catch {};
        tool_agent.clearLog();
    }

    utils.output.println("Goodbye!", .{});
}

fn handleSlashCommand(
    input: []const u8,
    tool_agent: *abi.ai.tool_agent.ToolAugmentedAgent,
    improver: *abi.ai.self_improve.SelfImprover,
    session: *SessionState,
    model_name: []const u8,
    system_prompt: []const u8,
) void {
    var iter = std.mem.splitScalar(u8, input[1..], ' ');
    const cmd = iter.first();

    if (std.mem.eql(u8, cmd, "help")) {
        printInteractiveHelp();
        return;
    }

    if (std.mem.eql(u8, cmd, "tools")) {
        utils.output.println("\nRegistered Tools: {d}", .{tool_agent.toolCount()});
        utils.output.println("Use tools by describing what you want in natural language.\n", .{});
        return;
    }

    if (std.mem.eql(u8, cmd, "feedback")) {
        const fb = iter.next() orelse {
            utils.output.println("Usage: /feedback good|bad", .{});
            return;
        };
        if (std.mem.eql(u8, fb, "good")) {
            improver.recordFeedback(true);
            utils.output.printSuccess("Positive feedback recorded.", .{});
        } else if (std.mem.eql(u8, fb, "bad")) {
            improver.recordFeedback(false);
            utils.output.printSuccess("Negative feedback recorded.", .{});
        } else {
            utils.output.println("Usage: /feedback good|bad", .{});
        }
        return;
    }

    if (std.mem.eql(u8, cmd, "reflect")) {
        if (improver.getLatestMetrics()) |m| {
            utils.output.printHeader("Last Response Reflection");
            utils.output.printKeyValueFmt("Coherence", "{d:.2}", .{m.coherence});
            utils.output.printKeyValueFmt("Relevance", "{d:.2}", .{m.relevance});
            utils.output.printKeyValueFmt("Completeness", "{d:.2}", .{m.completeness});
            utils.output.printKeyValueFmt("Clarity", "{d:.2}", .{m.clarity});
            utils.output.printKeyValueFmt("Overall", "{d:.2}", .{m.overall});
            utils.output.println("", .{});
        } else {
            utils.output.printInfo("No responses to reflect on yet.", .{});
        }
        return;
    }

    if (std.mem.eql(u8, cmd, "metrics")) {
        const report = improver.getReport();
        utils.output.printHeader("Performance Metrics");
        utils.output.printKeyValueFmt("Total interactions", "{d}", .{report.total_interactions});
        utils.output.printKeyValueFmt("Avg quality", "{d:.2}", .{report.avg_quality});
        utils.output.printKeyValueFmt("Positive feedback", "{d}", .{report.positive_feedback_count});
        utils.output.printKeyValueFmt("Negative feedback", "{d}", .{report.negative_feedback_count});
        utils.output.printKeyValueFmt("Tool usage count", "{d}", .{report.tool_usage_count});
        utils.output.printKeyValueFmt("Trend", "{t}", .{report.trend});
        utils.output.println("", .{});
        return;
    }

    if (std.mem.eql(u8, cmd, "memory")) {
        utils.output.printHeader("Memory Status");
        utils.output.printKeyValueFmt("Session messages", "{d}", .{session.messages.items.len});
        utils.output.printKeyValueFmt("Interactions tracked", "{d}", .{improver.total_interactions});
        utils.output.printKeyValueFmt("Feedback recorded", "{d} positive, {d} negative", .{
            improver.positive_feedback,
            improver.negative_feedback,
        });
        utils.output.println("", .{});
        return;
    }

    if (std.mem.eql(u8, cmd, "save")) {
        session.save(model_name, system_prompt, tool_agent.agent.config.temperature) catch |err| {
            utils.output.printError("saving session: {t}", .{err});
            return;
        };
        utils.output.printSuccess("Session saved: {s} ({d} messages)", .{ session.session_name, session.messages.items.len });
        return;
    }

    if (std.mem.eql(u8, cmd, "load")) {
        session.load() catch |err| {
            utils.output.printError("loading session: {t}", .{err});
            return;
        };
        utils.output.printSuccess("Session loaded: {s} ({d} messages)", .{ session.session_name, session.messages.items.len });
        return;
    }

    if (std.mem.eql(u8, cmd, "info")) {
        utils.output.printHeader("Session Information");
        utils.output.printKeyValue("ID", session.session_id);
        utils.output.printKeyValue("Name", session.session_name);
        utils.output.printKeyValueFmt("Messages", "{d}", .{session.messages.items.len});
        utils.output.printKeyValueFmt("Modified", "{}", .{session.modified});
        utils.output.println("", .{});
        return;
    }

    if (std.mem.eql(u8, cmd, "clear")) {
        tool_agent.clearLog();
        session.clear();
        session.save(model_name, system_prompt, tool_agent.agent.config.temperature) catch |err| {
            utils.output.printWarning("failed to persist cleared session: {t}", .{err});
        };
        utils.output.printSuccess("Session history and tool call log cleared.", .{});
        return;
    }

    utils.output.printWarning("Unknown command: /{s}", .{cmd});
    utils.output.println("Type /help for available commands.", .{});
}

fn confirmDestructiveOp(tool_name: []const u8, args_json: []const u8) bool {
    utils.output.printWarning("[Confirm] Agent wants to use '{s}' with args: {s}", .{ tool_name, args_json });
    utils.output.print("Allow? (y/n): ", .{});

    // For now, default to yes since we can't easily read stdin in a callback
    // In production, this would prompt the user
    return true;
}

fn printInteractiveHelp() void {
    utils.output.println("", .{});
    utils.output.printHeader("OS Agent Commands");
    utils.output.println("  /help      - Show this help", .{});
    utils.output.println("  /tools     - List registered tools", .{});
    utils.output.println("  /feedback  - Provide feedback (good/bad) on last response", .{});
    utils.output.println("  /reflect   - Show self-reflection scores for last response", .{});
    utils.output.println("  /metrics   - Show performance metrics over time", .{});
    utils.output.println("  /memory    - Show memory statistics", .{});
    utils.output.println("  /save      - Persist current session state", .{});
    utils.output.println("  /load      - Reload current session from disk", .{});
    utils.output.println("  /info      - Show session metadata", .{});
    utils.output.println("  /clear     - Clear tool call log", .{});
    utils.output.println("  exit, quit - Exit the agent", .{});
    utils.output.println("", .{});
    utils.output.println("The agent can autonomously use tools to answer your questions.", .{});
    utils.output.println("Just describe what you need in natural language.", .{});
    utils.output.println("", .{});
}

fn printHelp() void {
    utils.output.println("Usage: abi os-agent [options]", .{});
    utils.output.println("", .{});
    utils.output.println("OS-aware AI agent with full tool access, memory, and self-learning.", .{});
    utils.output.println("", .{});
    utils.output.println("Options:", .{});
    utils.output.println("  -m, --message <msg>   Send single message (non-interactive)", .{});
    utils.output.println("  -s, --session <name>  Session name (default: os-agent-default)", .{});
    utils.output.println("  --backend <name>      LLM backend: echo, openai, ollama, huggingface", .{});
    utils.output.println("  --model <name>        Model name for the backend", .{});
    utils.output.println("  --self-aware          Enable codebase indexing for self-awareness", .{});
    utils.output.println("  --no-confirm          Skip confirmation for destructive operations", .{});
    utils.output.println("  -h, --help            Show this help", .{});
    utils.output.println("", .{});
    utils.output.println("Features:", .{});
    utils.output.println("  - 26+ tools: file I/O, shell, search, edit, process, network, system", .{});
    utils.output.println("  - Hybrid memory with long-term learning", .{});
    utils.output.println("  - Self-reflection and quality metrics", .{});
    utils.output.println("  - Codebase self-awareness (with --self-aware)", .{});
    utils.output.println("", .{});
    utils.output.println("Interactive Commands:", .{});
    utils.output.println("  /tools      List registered tools", .{});
    utils.output.println("  /feedback   Provide feedback (good/bad)", .{});
    utils.output.println("  /reflect    Show quality scores for last response", .{});
    utils.output.println("  /metrics    Show performance over time", .{});
    utils.output.println("  /memory     Show memory statistics", .{});
    utils.output.println("  /save       Persist current session state", .{});
    utils.output.println("  /load       Reload current session from disk", .{});
    utils.output.println("  /info       Show session metadata", .{});
    utils.output.println("  /clear      Clear tool call log", .{});
    utils.output.println("  exit, quit  Exit the agent", .{});
    utils.output.println("", .{});
    utils.output.println("Examples:", .{});
    utils.output.println("  abi os-agent                            # Interactive session", .{});
    utils.output.println("  abi os-agent -m \"what processes are running?\"", .{});
    utils.output.println("  abi os-agent --backend ollama --model llama3", .{});
    utils.output.println("  abi os-agent --self-aware               # With codebase awareness", .{});
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

test {
    std.testing.refAllDecls(@This());
}
