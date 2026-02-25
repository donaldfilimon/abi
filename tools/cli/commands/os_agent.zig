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
        std.debug.print("Warning: could not load session '{s}': {t}\n", .{ session_name, err });
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
        try session.addMessage(.user, msg);
        try session.addMessage(.assistant, response);
        session.save(model_name, system_prompt, tool_agent.agent.config.temperature) catch |err| {
            std.debug.print("Warning: failed to persist os-agent session: {t}\n", .{err});
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
    std.debug.print("\n{s}", .{
        \\╔════════════════════════════════════════════════════════════╗
        \\║              ABI OS Agent (Tool-Augmented)               ║
        \\╚════════════════════════════════════════════════════════════╝
        \\
    });
    std.debug.print("\nSession: {s}\n", .{session.session_name});
    std.debug.print("Tools: {d} registered | Memory: enabled | Self-reflection: enabled\n", .{tool_agent.toolCount()});
    std.debug.print("Loaded messages: {d}\n", .{session.messages.items.len});
    std.debug.print("Type '/help' for commands, 'exit' to quit.\n\n", .{});

    var seed_with_session_context = session.messages.items.len > 0;

    var io_backend = cli_io.initIoBackend(allocator);
    defer io_backend.deinit();

    const io = io_backend.io();
    const stdin_file = std.Io.File.stdin();
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
            if (session.modified) {
                session.save(model_name, system_prompt, tool_agent.agent.config.temperature) catch |err| {
                    std.debug.print("Warning: failed to persist os-agent session: {t}\n", .{err});
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
            std.debug.print("Error: {t}\n", .{err});
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
                std.debug.print("Error: OutOfMemory\n", .{});
                continue;
            };
        }
        defer allocator.free(response);
        seed_with_session_context = false;

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

        session.addMessage(.user, trimmed) catch |err| {
            std.debug.print("Warning: failed to record user message: {t}\n", .{err});
        };
        session.addMessage(.assistant, response) catch |err| {
            std.debug.print("Warning: failed to record assistant message: {t}\n", .{err});
        };
        session.save(model_name, system_prompt, tool_agent.agent.config.temperature) catch |err| {
            std.debug.print("Warning: failed to persist os-agent session: {t}\n", .{err});
        };

        // Record metrics and clear log for next turn
        const metrics = improver.evaluateResponse(response, trimmed);
        improver.recordMetrics(metrics) catch {};
        tool_agent.clearLog();
    }

    std.debug.print("Goodbye!\n", .{});
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
        std.debug.print("  Trend:               {t}\n\n", .{report.trend});
        return;
    }

    if (std.mem.eql(u8, cmd, "memory")) {
        std.debug.print("\nMemory Status:\n", .{});
        std.debug.print("  Session messages:     {d}\n", .{session.messages.items.len});
        std.debug.print("  Interactions tracked: {d}\n", .{improver.total_interactions});
        std.debug.print("  Feedback recorded:    {d} positive, {d} negative\n\n", .{
            improver.positive_feedback,
            improver.negative_feedback,
        });
        return;
    }

    if (std.mem.eql(u8, cmd, "save")) {
        session.save(model_name, system_prompt, tool_agent.agent.config.temperature) catch |err| {
            std.debug.print("Error saving session: {t}\n", .{err});
            return;
        };
        std.debug.print("Session saved: {s} ({d} messages)\n", .{ session.session_name, session.messages.items.len });
        return;
    }

    if (std.mem.eql(u8, cmd, "load")) {
        session.load() catch |err| {
            std.debug.print("Error loading session: {t}\n", .{err});
            return;
        };
        std.debug.print("Session loaded: {s} ({d} messages)\n", .{ session.session_name, session.messages.items.len });
        return;
    }

    if (std.mem.eql(u8, cmd, "info")) {
        std.debug.print("\nSession Information:\n", .{});
        std.debug.print("  ID: {s}\n", .{session.session_id});
        std.debug.print("  Name: {s}\n", .{session.session_name});
        std.debug.print("  Messages: {d}\n", .{session.messages.items.len});
        std.debug.print("  Modified: {}\n\n", .{session.modified});
        return;
    }

    if (std.mem.eql(u8, cmd, "clear")) {
        tool_agent.clearLog();
        session.clear();
        session.save(model_name, system_prompt, tool_agent.agent.config.temperature) catch |err| {
            std.debug.print("Warning: failed to persist cleared session: {t}\n", .{err});
        };
        std.debug.print("Session history and tool call log cleared.\n", .{});
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
        \\  /save      - Persist current session state
        \\  /load      - Reload current session from disk
        \\  /info      - Show session metadata
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
        \\  /save       Persist current session state
        \\  /load       Reload current session from disk
        \\  /info       Show session metadata
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
