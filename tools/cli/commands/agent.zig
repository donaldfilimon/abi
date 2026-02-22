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
const utils = @import("../utils/mod.zig");
const cli_io = utils.io_backend;
const super_ = @import("ralph/super.zig");

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
    store: ?abi.ai.memory.SessionStore,

    pub fn init(allocator: std.mem.Allocator, name: []const u8) !SessionState {
        // Use cross-platform path separator
        const sessions_dir = ".abi" ++ std.fs.path.sep_str ++ "sessions";
        const store: ?abi.ai.memory.SessionStore = abi.ai.memory.SessionStore.init(allocator, sessions_dir);

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
            .store = store,
        };
    }

    pub fn deinit(self: *SessionState) void {
        for (self.messages.items) |*msg| {
            self.allocator.free(msg.content);
            if (msg.name) |n| self.allocator.free(n);
        }
        self.messages.deinit(self.allocator);
        // SessionStore doesn't need explicit cleanup
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
        var store = self.store orelse return error.PersistenceDisabled;

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

        try store.saveSession(session_data);
        self.modified = false;
    }

    pub fn load(self: *SessionState, session_id: []const u8) !void {
        var store = self.store orelse return error.PersistenceDisabled;

        var data = try store.loadSession(session_id);
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
        std.debug.print("Error: --task argument is required for Ralph mode.\n", .{});
        std.debug.print("Usage: abi agent ralph --task \"Your task description\"\n", .{});
        std.debug.print("       abi agent ralph --task \"...\" --store-skill \"Lesson learned\"\n", .{});
        std.debug.print("       abi agent ralph --task \"...\" --auto-skill  # LLM extracts and stores a lesson\n", .{});
        return;
    }

    return super_.runSuper(allocator, args);
}

/// Run the agent command with the provided arguments.
pub fn run(allocator: std.mem.Allocator, args: []const [:0]const u8) !void {
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
            std.debug.print("Warning: Could not load session '{s}': {t}\n", .{ sid, err });
        };
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
            try tool_agent.registerAllAgentTools();
        } else {
            if (enable_os_tools) {
                abi.ai.tools.os_tools.registerAll(&tool_agent.tool_registry) catch {};
            }
            if (enable_file_tools) {
                abi.ai.tools.file_tools.registerAll(&tool_agent.tool_registry) catch {};
                abi.ai.tools.search_tools.registerAll(&tool_agent.tool_registry) catch {};
                abi.ai.tools.edit_tools.registerAll(&tool_agent.tool_registry) catch {};
            }
        }

        if (message) |msg| {
            if (show_prompt) {
                var builder = prompts.PromptBuilder.init(allocator, persona_type);
                defer builder.deinit();
                try builder.addUserMessage(msg);
                const exported = try builder.exportDebug();
                defer allocator.free(exported);
                std.debug.print("{s}\n", .{exported});
            }

            const response = try tool_agent.processWithTools(msg, allocator);
            defer allocator.free(response);

            // Show tool calls
            const log = tool_agent.getToolCallLog();
            if (log.len > 0) {
                std.debug.print("\n[Tool Calls: {d}]\n", .{log.len});
                for (log) |record| {
                    const status = if (record.success) "ok" else "FAIL";
                    std.debug.print("  {s}: [{s}]\n", .{ record.tool_name, status });
                }
                std.debug.print("\n", .{});
            }

            std.debug.print("User: {s}\n", .{msg});
            std.debug.print("Agent: {s}\n", .{response});

            try session.addMessage(.user, msg);
            try session.addMessage(.assistant, response);
            return;
        }

        // Interactive with tools — use basic agent path for now
        // (tool-augmented interactive REPL uses the os-agent command)
        std.debug.print("Tool-augmented interactive mode. Tools: {d} registered.\n", .{tool_agent.toolCount()});
        std.debug.print("Tip: Use 'abi os-agent' for the full interactive tool experience.\n\n", .{});

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
                std.debug.print("{s}\n", .{exported});
            }

            const response = try agent.process(msg, allocator);
            defer allocator.free(response);
            std.debug.print("User: {s}\n", .{msg});
            std.debug.print("Agent: {s}\n", .{response});

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
    std.debug.print("\n╔════════════════════════════════════════════════════════════╗\n", .{});
    std.debug.print("║                    ABI AI Agent                            ║\n", .{});
    std.debug.print("╚════════════════════════════════════════════════════════════╝\n\n", .{});
    std.debug.print("Session: {s} ({s})\n", .{ session.session_name, session.session_id });
    std.debug.print("Persona: {s} - {s}\n", .{ persona.name, persona.description });
    std.debug.print("Type '/help' for commands, 'exit' to quit.\n\n", .{});

    var io_backend = cli_io.initIoBackend(allocator);
    defer io_backend.deinit();

    const io = io_backend.io();
    var stdin_file = std.Io.File.stdin();
    var buffer: [4096]u8 = undefined;
    var reader = stdin_file.reader(io, &buffer);

    while (true) {
        std.debug.print("> ", .{});
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

        // Handle exit commands
        if (std.mem.eql(u8, trimmed, "exit") or std.mem.eql(u8, trimmed, "quit")) {
            if (session.modified) {
                std.debug.print("Session has unsaved changes. Save before exit? (y/n): ", .{});
                const save_opt = reader.interface.takeDelimiter('\n') catch continue;
                const save_line = save_opt orelse continue;
                const save_trimmed = std.mem.trim(u8, save_line, " \t\r\n");
                if (save_trimmed.len > 0 and (save_trimmed[0] == 'y' or save_trimmed[0] == 'Y')) {
                    session.save(null) catch |err| {
                        std.debug.print("Error saving: {t}\n", .{err});
                    };
                }
            }
            break;
        }

        // Handle slash commands
        if (trimmed[0] == '/') {
            handleSlashCommand(allocator, session, trimmed, persona_type) catch |err| {
                std.debug.print("Command error: {t}\n", .{err});
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
            std.debug.print("{s}\n", .{exported});
        }

        // Process message with agent
        const response = try agent.process(trimmed, allocator);
        defer allocator.free(response);

        // Add to session history
        try session.addMessage(.user, trimmed);
        try session.addMessage(.assistant, response);

        std.debug.print("Agent: {s}\n\n", .{response});
    }

    std.debug.print("Goodbye!\n", .{});
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
        std.debug.print("{s}\n", .{exported});
        return;
    }

    if (std.mem.eql(u8, cmd, "persona")) {
        const persona = abi.ai.prompts.getPersona(persona_type);
        std.debug.print("\nCurrent Persona: {s}\n", .{persona.name});
        std.debug.print("Description: {s}\n", .{persona.description});
        std.debug.print("Temperature: {d:.1}\n\n", .{persona.suggested_temperature});
        std.debug.print("System Prompt:\n{s}\n\n", .{persona.system_prompt});
        return;
    }

    if (std.mem.eql(u8, cmd, "personas")) {
        listPersonas();
        return;
    }

    if (std.mem.eql(u8, cmd, "save")) {
        const name = iter.next();
        session.save(name) catch |err| {
            std.debug.print("Error saving session: {t}\n", .{err});
            return;
        };
        std.debug.print("Session saved: {s}\n", .{session.session_name});
        return;
    }

    if (std.mem.eql(u8, cmd, "load")) {
        const session_id = iter.next() orelse {
            std.debug.print("Usage: /load <session_id>\n", .{});
            return;
        };
        session.load(session_id) catch |err| {
            std.debug.print("Error loading session: {t}\n", .{err});
            return;
        };
        std.debug.print("Session loaded: {s} ({d} messages)\n", .{ session.session_name, session.messages.items.len });
        return;
    }

    if (std.mem.eql(u8, cmd, "sessions")) {
        listSessions(allocator) catch |err| {
            std.debug.print("Error listing sessions: {t}\n", .{err});
        };
        return;
    }

    if (std.mem.eql(u8, cmd, "clear")) {
        session.clear();
        std.debug.print("Conversation cleared.\n", .{});
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

    if (std.mem.eql(u8, cmd, "history")) {
        std.debug.print("\nConversation History:\n", .{});
        std.debug.print("─────────────────────────────────────────\n", .{});
        for (session.messages.items, 0..) |msg, idx| {
            const role_str = switch (msg.role) {
                .user => "You",
                .assistant => "Agent",
                .system => "System",
                .tool => "Tool",
            };
            std.debug.print("[{d}] {s}: {s}\n", .{ idx + 1, role_str, msg.content });
        }
        std.debug.print("─────────────────────────────────────────\n\n", .{});
        return;
    }

    std.debug.print("Unknown command: /{s}\nType /help for available commands.\n", .{cmd});
}

fn listSessions(allocator: std.mem.Allocator) !void {
    const sessions_dir = ".abi" ++ std.fs.path.sep_str ++ "sessions";
    var store = abi.ai.memory.SessionStore.init(allocator, sessions_dir);

    const sessions = store.listSessions() catch |err| {
        std.debug.print("Error listing sessions: {t}\n", .{err});
        return;
    };
    defer allocator.free(sessions);

    if (sessions.len == 0) {
        std.debug.print("No saved sessions found.\n", .{});
        return;
    }

    std.debug.print("\nSaved Sessions:\n", .{});
    std.debug.print("─────────────────────────────────────────────────────────────\n", .{});
    std.debug.print("{s:<20} {s:<20} {s:<10} {s:<10}\n", .{ "ID", "Name", "Messages", "Updated" });
    std.debug.print("─────────────────────────────────────────────────────────────\n", .{});

    for (sessions) |sess_meta| {
        defer {
            allocator.free(sess_meta.id);
            allocator.free(sess_meta.name);
        }
        std.debug.print("{s:<20} {s:<20} {d:<10} {d:<10}\n", .{
            sess_meta.id,
            sess_meta.name,
            sess_meta.message_count,
            sess_meta.updated_at,
        });
    }
    std.debug.print("\n", .{});
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
    std.debug.print("{s}", .{help});
}

fn listPersonas() void {
    const all_personas = abi.ai.prompts.listPersonas();
    std.debug.print("\nAvailable Personas:\n", .{});
    std.debug.print("─────────────────────────────────────────────────────────────\n", .{});
    std.debug.print("{s:<12} {s:<15} {s}\n", .{ "Name", "Temperature", "Description" });
    std.debug.print("─────────────────────────────────────────────────────────────\n", .{});
    for (all_personas) |pt| {
        const p = abi.ai.prompts.getPersona(pt);
        std.debug.print("{s:<12} {d:<15.1} {s}\n", .{ p.name, p.suggested_temperature, p.description });
    }
    std.debug.print("\nUse --persona <name> to select a persona.\n\n", .{});
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
    std.debug.print("{s}", .{help_text});
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
