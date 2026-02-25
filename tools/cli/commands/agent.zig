//! AI agent command with session management and prompt debugging.
//!
//! Commands:
//! - agent - Start interactive session
//! - agent --message "text" - One-shot message
//! - agent --session <name> - Use named session
//! - agent --persona <type> - Use specific persona (assistant, coder, writer, etc.)
//! - agent --show-prompt - Display the full prompt before sending
//! - agent --list-sessions - List available sessions
//! - agent --list-personas - List available persona types
//!
//! Interactive commands:
//! - /save [name] - Save current session
//! - /load <name> - Load a session
//! - /sessions - List available sessions
//! - /clear - Clear current conversation
//! - /info - Show session info
//! - /prompt - Show current prompt (for debugging)
//! - exit, quit - Exit the agent

const std = @import("std");
const abi = @import("abi");
const command_mod = @import("../command.zig");
const context_mod = @import("../framework/context.zig");
const utils = @import("../utils/mod.zig");
const cli_io = utils.io_backend;
const super_ = @import("ralph/super.zig");
const app_paths = abi.shared.app_paths;

pub const meta: command_mod.Meta = .{
    .name = "agent",
    .description = "Run AI agent (interactive or one-shot)",
};

/// Session state for the interactive agent.
const SessionState = struct {
    allocator: std.mem.Allocator,
    session_id: []const u8,
    session_name: []const u8,
    messages: std.ArrayListUnmanaged(abi.ai.memory.Message),
    modified: bool,

    pub fn init(allocator: std.mem.Allocator, name: []const u8) !SessionState {
        // Generate session ID using monotonic time
        var id_buf: [32]u8 = undefined;
        const timestamp = getUnixSeconds();
        const id_slice = std.fmt.bufPrint(&id_buf, "session_{d}", .{timestamp}) catch {
            // Fallback to a simple ID
            @memcpy(id_buf[0..8], "sess_000");
            return error.OutOfMemory;
        };
        const id_len = id_slice.len;

        // Allocate strings - propagate errors to avoid freeing string literals
        const session_id = try allocator.dupe(u8, id_buf[0..id_len]);
        errdefer allocator.free(session_id);
        const session_name = try allocator.dupe(u8, name);

        return .{
            .allocator = allocator,
            .session_id = session_id,
            .session_name = session_name,
            .messages = .empty,
            .modified = false,
        };
    }

    pub fn deinit(self: *SessionState) void {
        for (self.messages.items) |*msg| {
            self.allocator.free(msg.content);
            if (msg.name) |n| self.allocator.free(n);
        }
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
            .timestamp = getUnixSeconds(),
        });
        self.modified = true;
    }

    pub fn save(self: *SessionState, name: ?[]const u8) !void {
        // Update name if provided
        if (name) |n| {
            self.allocator.free(self.session_name);
            self.session_name = try self.allocator.dupe(u8, n);
        }

        const now = getUnixSeconds();
        const session_data = abi.ai.memory.SessionData{
            .id = self.session_id,
            .name = self.session_name,
            .created_at = now,
            .updated_at = now,
            .messages = self.messages.items,
            .config = .{},
        };

        const sessions_dir = try app_paths.resolvePath(self.allocator, "sessions");
        defer self.allocator.free(sessions_dir);

        var primary_store = abi.ai.memory.SessionStore.init(self.allocator, sessions_dir);
        try primary_store.saveSession(session_data);
        self.modified = false;
    }

    pub fn load(self: *SessionState, session_id: []const u8) !void {
        const sessions_dir = try app_paths.resolvePath(self.allocator, "sessions");
        defer self.allocator.free(sessions_dir);

        var primary_store = abi.ai.memory.SessionStore.init(self.allocator, sessions_dir);
        var data = try primary_store.loadSession(session_id);
        defer data.deinit(self.allocator);

        // Clear current messages
        for (self.messages.items) |*msg| {
            self.allocator.free(msg.content);
            if (msg.name) |n| self.allocator.free(n);
        }
        self.messages.clearRetainingCapacity();

        // Copy messages from loaded session
        for (data.messages) |msg| {
            try self.addMessage(msg.role, msg.content);
        }

        // Update session info
        self.allocator.free(self.session_id);
        self.allocator.free(self.session_name);
        self.session_id = try self.allocator.dupe(u8, data.id);
        self.session_name = try self.allocator.dupe(u8, data.name);
        self.modified = false;
    }

    pub fn clear(self: *SessionState) void {
        for (self.messages.items) |*msg| {
            self.allocator.free(msg.content);
            if (msg.name) |n| self.allocator.free(n);
        }
        self.messages.clearRetainingCapacity();
        self.modified = true;
    }
};

fn runRalph(allocator: std.mem.Allocator, args: []const [:0]const u8) !void {
    var task_value_present = false;
    var i: usize = 0;
    while (i < args.len) : (i += 1) {
        const arg_text = std.mem.sliceTo(args[i], 0);
        if (!(std.mem.eql(u8, arg_text, "--task") or std.mem.eql(u8, arg_text, "-t"))) {
            continue;
        }

        if (i + 1 < args.len) {
            const next = std.mem.sliceTo(args[i + 1], 0);
            if (next.len > 0 and next[0] != '-') {
                task_value_present = true;
                break;
            }
        }
    }

    if (!task_value_present) {
        utils.output.printError("--task argument is required for Ralph mode.", .{});
        utils.output.println("Usage: abi agent ralph --task \"Your task description\"", .{});
        utils.output.println("       abi agent ralph --task \"...\" --store-skill \"Lesson learned\"", .{});
        utils.output.println("       abi agent ralph --task \"...\" --auto-skill  # LLM extracts and stores a lesson", .{});
        return;
    }

    return super_.runSuper(allocator, args);
}

/// Run the agent command with the provided arguments.
pub fn run(ctx: *const context_mod.CommandContext, args: []const [:0]const u8) !void {
    const allocator = ctx.allocator;
    // Check for "ralph" subcommand
    if (args.len > 0 and std.mem.eql(u8, std.mem.sliceTo(args[0], 0), "ralph")) {
        return runRalph(allocator, args[1..]);
    }

    const agent_mod = abi.ai.agent;
    const prompts = abi.ai.prompts;

    var name: []const u8 = "cli-agent";
    var message: ?[]const u8 = null;
    var session_name: []const u8 = "default";
    var load_session: ?[]const u8 = null;
    var list_sessions = false;
    var list_personas = false;
    var show_prompt = false;
    var persona_type: prompts.PersonaType = .assistant;

    // Tool and learning flags
    var enable_all_tools = false;
    var enable_os_tools = false;
    var enable_file_tools = false;
    var enable_learn = false;
    var no_confirm = false;

    var i: usize = 0;
    while (i < args.len) {
        const arg = args[i];
        i += 1;

        if (std.mem.eql(u8, std.mem.sliceTo(arg, 0), "--name")) {
            if (i < args.len) {
                name = std.mem.sliceTo(args[i], 0);
                i += 1;
            }
            continue;
        }

        if (utils.args.matchesAny(arg, &[_][]const u8{ "--message", "-m" })) {
            if (i < args.len) {
                message = std.mem.sliceTo(args[i], 0);
                i += 1;
            }
            continue;
        }

        if (utils.args.matchesAny(arg, &[_][]const u8{ "--session", "-s" })) {
            if (i < args.len) {
                session_name = std.mem.sliceTo(args[i], 0);
                i += 1;
            }
            continue;
        }

        if (std.mem.eql(u8, std.mem.sliceTo(arg, 0), "--load")) {
            if (i < args.len) {
                load_session = std.mem.sliceTo(args[i], 0);
                i += 1;
            }
            continue;
        }

        if (std.mem.eql(u8, std.mem.sliceTo(arg, 0), "--persona")) {
            if (i < args.len) {
                const persona_name = std.mem.sliceTo(args[i], 0);
                persona_type = parsePersonaType(persona_name);
                i += 1;
            }
            continue;
        }

        if (std.mem.eql(u8, std.mem.sliceTo(arg, 0), "--show-prompt")) {
            show_prompt = true;
            continue;
        }

        if (std.mem.eql(u8, std.mem.sliceTo(arg, 0), "--list-sessions")) {
            list_sessions = true;
            continue;
        }

        if (std.mem.eql(u8, std.mem.sliceTo(arg, 0), "--list-personas")) {
            list_personas = true;
            continue;
        }

        // Tool and learning flags
        if (std.mem.eql(u8, std.mem.sliceTo(arg, 0), "--all-tools")) {
            enable_all_tools = true;
            continue;
        }
        if (std.mem.eql(u8, std.mem.sliceTo(arg, 0), "--os-tools")) {
            enable_os_tools = true;
            continue;
        }
        if (std.mem.eql(u8, std.mem.sliceTo(arg, 0), "--file-tools")) {
            enable_file_tools = true;
            continue;
        }
        if (std.mem.eql(u8, std.mem.sliceTo(arg, 0), "--learn")) {
            enable_learn = true;
            continue;
        }
        if (std.mem.eql(u8, std.mem.sliceTo(arg, 0), "--no-confirm")) {
            no_confirm = true;
            continue;
        }

        if (utils.args.matchesAny(arg, &[_][]const u8{ "--help", "-h", "help" })) {
            printHelp();
            return;
        }
    }

    // Handle list sessions command
    if (list_sessions) {
        return listSessions(allocator);
    }

    // Handle list personas command
    if (list_personas) {
        return listPersonas();
    }

    // Get persona for system prompt
    const persona = prompts.getPersona(persona_type);

    const use_tools = enable_all_tools or enable_os_tools or enable_file_tools;

    // Initialize session state
    var session = try SessionState.init(allocator, session_name);
    defer session.deinit();

    // Load session if requested
    if (load_session) |sid| {
        session.load(sid) catch |err| {
            utils.output.printError("Could not load session '{s}': {t}", .{ sid, err });
            const sessions_dir = app_paths.resolvePath(allocator, "sessions") catch "~/.abi/sessions";
            utils.output.printInfo("Session directory: {s}", .{sessions_dir});
            utils.output.printInfo("Use --list-sessions to see available sessions, or start a new session without --load.", .{});
            return;
        };
    }

    // Check for AI provider configuration before proceeding
    const has_openai = std.c.getenv("ABI_OPENAI_API_KEY") != null;
    const has_anthropic = std.c.getenv("ABI_ANTHROPIC_API_KEY") != null;
    const has_ollama = std.c.getenv("ABI_OLLAMA_HOST") != null;
    if (!has_openai and !has_anthropic and !has_ollama) {
        utils.output.printWarning("No AI provider configured.", .{});
        utils.output.printInfo("Set one of: ABI_OPENAI_API_KEY, ABI_ANTHROPIC_API_KEY, or ABI_OLLAMA_HOST", .{});
        utils.output.printInfo("Without a provider, the agent will use built-in response generation.", .{});
        utils.output.println("", .{});
    }

    if (use_tools) {
        // Tool-augmented agent mode
        var tool_agent = try abi.ai.tool_agent.ToolAugmentedAgent.init(allocator, .{
            .agent = .{
                .name = name,
                .system_prompt = persona.system_prompt,
                .temperature = persona.suggested_temperature,
            },
            .require_confirmation = !no_confirm,
            .enable_memory = enable_learn,
        });
        defer tool_agent.deinit();

        // Register requested tools
        if (enable_all_tools) {
            tool_agent.registerAllAgentTools() catch |err| {
                utils.output.printWarning("Some agent tools failed to register: {t}", .{err});
                utils.output.printInfo("Agent will continue with available tools.", .{});
            };
        } else {
            if (enable_os_tools) {
                abi.ai.tools.os_tools.registerAll(&tool_agent.tool_registry) catch |err| {
                    utils.output.printWarning("OS tools registration failed: {t}", .{err});
                };
            }
            if (enable_file_tools) {
                abi.ai.tools.file_tools.registerAll(&tool_agent.tool_registry) catch |err| {
                    utils.output.printWarning("File tools registration failed: {t}", .{err});
                };
                abi.ai.tools.search_tools.registerAll(&tool_agent.tool_registry) catch |err| {
                    utils.output.printWarning("Search tools registration failed: {t}", .{err});
                };
                abi.ai.tools.edit_tools.registerAll(&tool_agent.tool_registry) catch |err| {
                    utils.output.printWarning("Edit tools registration failed: {t}", .{err});
                };
            }
        }

        if (message) |msg| {
            if (show_prompt) {
                var builder = prompts.PromptBuilder.init(allocator, persona_type);
                defer builder.deinit();
                try builder.addUserMessage(msg);
                const exported = try builder.exportDebug();
                defer allocator.free(exported);
                utils.output.println("{s}", .{exported});
            }

            const response = try tool_agent.processWithTools(msg, allocator);
            defer allocator.free(response);

            // Show tool calls
            const log = tool_agent.getToolCallLog();
            if (log.len > 0) {
                utils.output.println("\n[Tool Calls: {d}]", .{log.len});
                for (log) |record| {
                    const status = if (record.success) "ok" else "FAIL";
                    utils.output.println("  {s}: [{s}]", .{ record.tool_name, status });
                }
                utils.output.println("", .{});
            }

            utils.output.println("User: {s}", .{msg});
            utils.output.println("Agent: {s}", .{response});

            try session.addMessage(.user, msg);
            try session.addMessage(.assistant, response);
            return;
        }

        // Interactive with tools — use basic agent path for now
        // (tool-augmented interactive REPL uses the os-agent command)
        utils.output.println("Tool-augmented interactive mode. Tools: {d} registered.", .{tool_agent.toolCount()});
        utils.output.printInfo("Tip: Use 'abi os-agent' for the full interactive tool experience.\n", .{});

        // Fall through to standard interactive with the inner agent
        try runInteractive(allocator, &tool_agent.agent, &session, persona_type, show_prompt);
    } else {
        // Standard agent mode (no tools)
        var agent = try agent_mod.Agent.init(allocator, .{
            .name = name,
            .system_prompt = persona.system_prompt,
            .temperature = persona.suggested_temperature,
        });
        defer agent.deinit();

        if (message) |msg| {
            if (show_prompt) {
                var builder = prompts.PromptBuilder.init(allocator, persona_type);
                defer builder.deinit();
                try builder.addUserMessage(msg);
                const exported = try builder.exportDebug();
                defer allocator.free(exported);
                utils.output.println("{s}", .{exported});
            }

            const response = try agent.process(msg, allocator);
            defer allocator.free(response);
            utils.output.println("User: {s}", .{msg});
            utils.output.println("Agent: {s}", .{response});

            try session.addMessage(.user, msg);
            try session.addMessage(.assistant, response);
            return;
        }

        try runInteractive(allocator, &agent, &session, persona_type, show_prompt);
    }
}

fn runInteractive(
    allocator: std.mem.Allocator,
    agent: *abi.ai.agent.Agent,
    session: *SessionState,
    persona_type: abi.ai.prompts.PersonaType,
    show_prompt: bool,
) !void {
    const persona = abi.ai.prompts.getPersona(persona_type);
    utils.output.println("\n╔════════════════════════════════════════════════════════════╗", .{});
    utils.output.println("║                    ABI AI Agent                            ║", .{});
    utils.output.println("╚════════════════════════════════════════════════════════════╝\n", .{});
    utils.output.println("Session: {s} ({s})", .{ session.session_name, session.session_id });
    utils.output.println("Persona: {s} - {s}", .{ persona.name, persona.description });
    utils.output.println("Type '/help' for commands, 'exit' to quit.\n", .{});

    var io_backend = cli_io.initIoBackend(allocator);
    defer io_backend.deinit();

    const io = io_backend.io();
    var stdin_file = std.Io.File.stdin();
    var buffer: [4096]u8 = undefined;
    var reader = stdin_file.reader(io, &buffer);

    while (true) {
        utils.output.print("> ", .{});
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

        // Handle exit commands
        if (std.mem.eql(u8, trimmed, "exit") or std.mem.eql(u8, trimmed, "quit")) {
            if (session.modified) {
                utils.output.print("Session has unsaved changes. Save before exit? (y/n): ", .{});
                const save_opt = reader.interface.takeDelimiter('\n') catch continue;
                const save_line = save_opt orelse continue;
                const save_trimmed = std.mem.trim(u8, save_line, " \t\r\n");
                if (save_trimmed.len > 0 and (save_trimmed[0] == 'y' or save_trimmed[0] == 'Y')) {
                    session.save(null) catch |err| {
                        utils.output.printError("saving: {t}", .{err});
                    };
                }
            }
            break;
        }

        // Handle slash commands
        if (trimmed[0] == '/') {
            handleSlashCommand(allocator, session, trimmed, persona_type) catch |err| {
                utils.output.printError("Command error: {t}", .{err});
            };
            continue;
        }

        // Show prompt if requested
        if (show_prompt) {
            var builder = abi.ai.prompts.PromptBuilder.init(allocator, persona_type);
            defer builder.deinit();
            // Add history
            for (session.messages.items) |msg| {
                const role: abi.ai.prompts.Role = switch (msg.role) {
                    .user => .user,
                    .assistant => .assistant,
                    .system => .system,
                    .tool => .tool,
                };
                try builder.addMessage(role, msg.content);
            }
            try builder.addUserMessage(trimmed);
            const exported = try builder.exportDebug();
            defer allocator.free(exported);
            utils.output.println("{s}", .{exported});
        }

        // Process message with agent
        const response = try agent.process(trimmed, allocator);
        defer allocator.free(response);

        // Add to session history
        try session.addMessage(.user, trimmed);
        try session.addMessage(.assistant, response);

        utils.output.println("Agent: {s}\n", .{response});
    }

    utils.output.println("Goodbye!", .{});
}

fn handleSlashCommand(
    allocator: std.mem.Allocator,
    session: *SessionState,
    input: []const u8,
    persona_type: abi.ai.prompts.PersonaType,
) !void {
    var iter = std.mem.splitScalar(u8, input[1..], ' ');
    const cmd = iter.first();

    if (std.mem.eql(u8, cmd, "help")) {
        printInteractiveHelp();
        return;
    }

    if (std.mem.eql(u8, cmd, "prompt")) {
        // Show the current prompt with all history
        var builder = abi.ai.prompts.PromptBuilder.init(allocator, persona_type);
        defer builder.deinit();
        for (session.messages.items) |msg| {
            const role: abi.ai.prompts.Role = switch (msg.role) {
                .user => .user,
                .assistant => .assistant,
                .system => .system,
                .tool => .tool,
            };
            try builder.addMessage(role, msg.content);
        }
        const exported = try builder.exportDebug();
        defer allocator.free(exported);
        utils.output.println("{s}", .{exported});
        return;
    }

    if (std.mem.eql(u8, cmd, "persona")) {
        const persona = abi.ai.prompts.getPersona(persona_type);
        utils.output.printHeader("Current Persona");
        utils.output.printKeyValue("Name", persona.name);
        utils.output.printKeyValue("Description", persona.description);
        utils.output.printKeyValueFmt("Temperature", "{d:.1}", .{persona.suggested_temperature});
        utils.output.println("\nSystem Prompt:\n{s}\n", .{persona.system_prompt});
        return;
    }

    if (std.mem.eql(u8, cmd, "personas")) {
        listPersonas();
        return;
    }

    if (std.mem.eql(u8, cmd, "save")) {
        const name = iter.next();
        session.save(name) catch |err| {
            utils.output.printError("saving session: {t}", .{err});
            return;
        };
        utils.output.printSuccess("Session saved: {s}", .{session.session_name});
        return;
    }

    if (std.mem.eql(u8, cmd, "load")) {
        const session_id = iter.next() orelse {
            utils.output.println("Usage: /load <session_id>", .{});
            return;
        };
        session.load(session_id) catch |err| {
            utils.output.printError("loading session: {t}", .{err});
            return;
        };
        utils.output.printSuccess("Session loaded: {s} ({d} messages)", .{ session.session_name, session.messages.items.len });
        return;
    }

    if (std.mem.eql(u8, cmd, "sessions")) {
        listSessions(allocator) catch |err| {
            utils.output.printError("listing sessions: {t}", .{err});
        };
        return;
    }

    if (std.mem.eql(u8, cmd, "clear")) {
        session.clear();
        utils.output.printSuccess("Conversation cleared.", .{});
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

    if (std.mem.eql(u8, cmd, "history")) {
        utils.output.printHeader("Conversation History");
        for (session.messages.items, 0..) |msg, idx| {
            const role_str = switch (msg.role) {
                .user => "You",
                .assistant => "Agent",
                .system => "System",
                .tool => "Tool",
            };
            utils.output.println("[{d}] {s}: {s}", .{ idx + 1, role_str, msg.content });
        }
        utils.output.println("", .{});
        return;
    }

    utils.output.printWarning("Unknown command: /{s}", .{cmd});
    utils.output.println("Type /help for available commands.", .{});
}

fn listSessions(allocator: std.mem.Allocator) !void {
    const sessions_dir = app_paths.resolvePath(allocator, "sessions") catch |err| {
        utils.output.printError("resolving session directories: {t}", .{err});
        return;
    };
    defer allocator.free(sessions_dir);

    var sessions = std.ArrayListUnmanaged(abi.ai.memory.SessionMeta).empty;
    defer {
        for (sessions.items) |*session_meta| session_meta.deinit(allocator);
        sessions.deinit(allocator);
    }

    var primary_store = abi.ai.memory.SessionStore.init(allocator, sessions_dir);
    var primary_sessions: ?[]abi.ai.memory.SessionMeta = null;
    primary_sessions = primary_store.listSessions() catch |err| switch (err) {
        error.SessionNotFound => null,
        else => {
            utils.output.printError("listing sessions: {t}", .{err});
            return;
        },
    };
    defer if (primary_sessions) |sessions_list| allocator.free(sessions_list);

    if (primary_sessions) |primary_session_list| {
        for (primary_session_list) |session_meta| {
            try sessions.append(allocator, session_meta);
        }
    }

    if (sessions.items.len == 0) {
        utils.output.printInfo("No saved sessions found.", .{});
        return;
    }

    utils.output.printHeader("Saved Sessions");
    utils.output.println("{s:<20} {s:<20} {s:<10} {s:<10}", .{ "ID", "Name", "Messages", "Updated" });
    utils.output.println("─────────────────────────────────────────────────────────────", .{});

    for (sessions.items) |sess_meta| {
        utils.output.println("{s:<20} {s:<20} {d:<10} {d:<10}", .{
            sess_meta.id,
            sess_meta.name,
            sess_meta.message_count,
            sess_meta.updated_at,
        });
    }
    utils.output.println("", .{});
}

fn sessionMetaExistsById(list: []const abi.ai.memory.SessionMeta, session_id: []const u8) bool {
    for (list) |session_meta| {
        if (std.mem.eql(u8, session_meta.id, session_id)) return true;
    }
    return false;
}

fn printInteractiveHelp() void {
    const help =
        \\
        \\Interactive Commands:
        \\  /help      - Show this help
        \\  /save [n]  - Save session (optionally with name)
        \\  /load <id> - Load a saved session
        \\  /sessions  - List saved sessions
        \\  /clear     - Clear conversation history
        \\  /info      - Show session information
        \\  /history   - Show conversation history
        \\  /prompt    - Show the full prompt (for debugging)
        \\  /persona   - Show current persona details
        \\  /personas  - List available personas
        \\  exit, quit - Exit the agent
        \\
        \\
    ;
    utils.output.print("{s}", .{help});
}

fn listPersonas() void {
    const all_personas = abi.ai.prompts.listPersonas();
    utils.output.printHeader("Available Personas");
    utils.output.println("{s:<12} {s:<15} {s}", .{ "Name", "Temperature", "Description" });
    utils.output.println("─────────────────────────────────────────────────────────────", .{});
    for (all_personas) |pt| {
        const p = abi.ai.prompts.getPersona(pt);
        utils.output.println("{s:<12} {d:<15.1} {s}", .{ p.name, p.suggested_temperature, p.description });
    }
    utils.output.println("\nUse --persona <name> to select a persona.\n", .{});
}

fn parsePersonaType(name: []const u8) abi.ai.prompts.PersonaType {
    if (std.mem.eql(u8, name, "coder")) return .coder;
    if (std.mem.eql(u8, name, "writer")) return .writer;
    if (std.mem.eql(u8, name, "analyst")) return .analyst;
    if (std.mem.eql(u8, name, "companion")) return .companion;
    if (std.mem.eql(u8, name, "docs")) return .docs;
    if (std.mem.eql(u8, name, "reviewer")) return .reviewer;
    if (std.mem.eql(u8, name, "minimal")) return .minimal;
    if (std.mem.eql(u8, name, "abbey")) return .abbey;
    if (std.mem.eql(u8, name, "ralph")) return .ralph;
    return .assistant; // Default
}

fn printHelp() void {
    const help_text =
        \\Usage: abi agent [options]
        \\
        \\Run the AI agent with optional session management and persona selection.
        \\
        \\Options:
        \\  --name <name>       Agent name (default: cli-agent)
        \\  -m, --message <msg> Send single message (non-interactive)
        \\  -s, --session <n>   Session name to use
        \\  --load <id>         Load existing session by ID
        \\  --persona <type>    Use specific persona (assistant, coder, writer, etc.)
        \\  --show-prompt       Display the full prompt before sending
        \\  --list-sessions     List all saved sessions
        \\  --list-personas     List available persona types
        \\  -h, --help          Show this help
        \\
        \\Tool & Learning Options:
        \\  --all-tools         Enable all tools (file, search, edit, OS, process, network)
        \\  --os-tools          Enable OS tools only (shell, env, clipboard, etc.)
        \\  --file-tools        Enable file tools only (read, write, search, edit)
        \\  --learn             Enable hybrid memory with long-term learning
        \\  --no-confirm        Skip confirmation for destructive operations
        \\
        \\Personas:
        \\  assistant  - General-purpose helpful assistant (default)
        \\  coder      - Programming and code-focused assistant
        \\  writer     - Creative writing assistant
        \\  analyst    - Data analysis and research assistant
        \\  reviewer   - Code review specialist
        \\  docs       - Technical documentation helper
        \\  companion  - Friendly conversational companion
        \\  minimal    - Direct, concise response mode
        \\  abbey      - Opinionated, emotionally intelligent AI
        \\
        \\Interactive Commands:
        \\  /save [name]   Save current session
        \\  /load <id>     Load a saved session
        \\  /sessions      List available sessions
        \\  /clear         Clear conversation
        \\  /info          Show session info
        \\  /history       Show conversation history
        \\  /prompt        Show the full prompt (for debugging)
        \\  /persona       Show current persona details
        \\  exit, quit     Exit the agent
        \\
        \\Examples:
        \\  abi agent                          # Start interactive session
        \\  abi agent --persona coder          # Use coder persona
        \\  abi agent --show-prompt -m "Hi"    # Show prompt, then send
        \\  abi agent --session "project-x"    # Named session
        \\  abi agent --list-personas          # Show all personas
        \\  abi agent --all-tools -m "ls src"  # One-shot with tools
        \\  abi agent --all-tools --learn      # Interactive with tools + learning
        \\
        \\Ralph Mode (Iterative Loop, auto-training):
        \\  abi agent ralph --task "..."      # Run a task through Super Ralph flow
        \\  --iterations <n>                   # Max iterations (overrides ralph.yml)
        \\  --store-skill "..."                # Store a skill for future runs
        \\  --auto-skill                       # LLM extracts a lesson from this run and stores it (model self-improves)
        \\
    ;
    utils.output.print("{s}", .{help_text});
}

/// Get a pseudo-unique timestamp for session IDs.
/// Uses nanosecond counter for reasonable uniqueness.
fn getUnixSeconds() i64 {
    // Use a simple counter based on timer - the actual value doesn't matter
    // as long as it's reasonably unique within a session
    var timer = abi.shared.time.Timer.start() catch return 1000;
    // Return nanoseconds divided down to make more manageable IDs
    // The timer value combined with nanoTimestamp gives reasonable uniqueness
    const base = timer.read();
    return @intCast(base / 1_000_000); // Convert to milliseconds for shorter IDs
}
