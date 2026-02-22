const std = @import("std");
const context_mod = @import("../../framework/context.zig");
const abi = @import("abi");
const utils = @import("../../utils/mod.zig");
const cli_io = utils.io_backend;
const run_cmd = @import("run.zig");

const ChatMessage = abi.ai.llm.providers.ChatMessage;
const ProviderId = abi.ai.llm.providers.ProviderId;

pub fn runSession(ctx: *const context_mod.CommandContext, args: []const [:0]const u8) !void {
    const allocator = ctx.allocator;
    if (utils.args.containsHelpArgs(args)) {
        printSessionHelp();
        return;
    }

    var options = try run_cmd.parseRunArgs(allocator, args);
    defer allocator.free(options.fallback);

    if (options.model == null) {
        std.debug.print("Error: --model is required.\n\n", .{});
        printSessionHelp();
        return;
    }

    var system_prompt: []const u8 = "You are a concise, practical assistant.";
    var i: usize = 0;
    while (i < args.len) {
        const arg = std.mem.sliceTo(args[i], 0);
        i += 1;
        if (std.mem.eql(u8, arg, "--system") and i < args.len) {
            system_prompt = std.mem.sliceTo(args[i], 0);
            i += 1;
        }
    }

    std.debug.print("LLM session started (model: {s}).\n", .{options.model.?});
    std.debug.print("Commands: /quit, /exit, /clear, /help, /providers, /backend <id>, /model <id>\n\n", .{});

    // Track heap-allocated model name from /model command to avoid dangling pointers.
    var model_switched: ?[]u8 = null;
    defer if (model_switched) |m| allocator.free(m);

    var io_backend = cli_io.initIoBackend(allocator);
    defer io_backend.deinit();
    const io = io_backend.io();

    const stdin_file = std.Io.File.stdin();
    var read_buffer: [8192]u8 = undefined;
    var reader = stdin_file.reader(io, &read_buffer);

    // Structured message history for multi-turn conversations.
    var history: std.ArrayListUnmanaged(ChatMessage) = .empty;
    defer history.deinit(allocator);

    // Owned copies of message text so we can free them independently.
    var history_text: std.ArrayListUnmanaged([]u8) = .empty;
    defer {
        for (history_text.items) |text| allocator.free(text);
        history_text.deinit(allocator);
    }

    while (true) {
        std.debug.print("You> ", .{});
        const line = reader.interface.takeDelimiter('\n') catch |err| {
            std.debug.print("\nInput error: {t}\n", .{err});
            continue;
        } orelse break;

        const trimmed = std.mem.trim(u8, line, " \t\r\n");
        if (trimmed.len == 0) continue;

        // ── Slash commands ─────────────────────────────────────────
        if (std.mem.eql(u8, trimmed, "/quit") or std.mem.eql(u8, trimmed, "/exit")) {
            break;
        }
        if (std.mem.eql(u8, trimmed, "/help")) {
            printSlashHelp();
            continue;
        }
        if (std.mem.eql(u8, trimmed, "/clear")) {
            for (history_text.items) |text| allocator.free(text);
            history_text.clearRetainingCapacity();
            history.clearRetainingCapacity();
            std.debug.print("Session context cleared.\n\n", .{});
            continue;
        }
        if (std.mem.eql(u8, trimmed, "/providers")) {
            printProviderStatus(allocator);
            continue;
        }
        if (std.mem.startsWith(u8, trimmed, "/backend")) {
            const rest = std.mem.trim(u8, trimmed["/backend".len..], " \t");
            if (rest.len == 0) {
                if (options.backend) |b| {
                    std.debug.print("Current backend: {s}\n\n", .{b.label()});
                } else {
                    std.debug.print("No backend pinned (using auto-routing).\n\n", .{});
                }
                continue;
            }
            if (parseProviderId(rest)) |new_backend| {
                options.backend = new_backend;
                std.debug.print("Backend switched to: {s}\n\n", .{new_backend.label()});
            } else {
                std.debug.print("Unknown backend: {s}\n", .{rest});
                std.debug.print("Available: local_gguf, llama_cpp, mlx, ollama, lm_studio, vllm, anthropic, openai, plugin_http, plugin_native\n\n", .{});
            }
            continue;
        }
        if (std.mem.startsWith(u8, trimmed, "/model")) {
            const rest = std.mem.trim(u8, trimmed["/model".len..], " \t");
            if (rest.len == 0) {
                std.debug.print("Current model: {s}\n\n", .{options.model.?});
                continue;
            }
            // Dupe into heap so it survives read buffer reuse
            const model_copy = allocator.dupe(u8, rest) catch {
                std.debug.print("Error: out of memory\n", .{});
                continue;
            };
            // Free previous model if it was heap-allocated via /model
            if (model_switched) |prev| allocator.free(prev);
            model_switched = model_copy;
            options.model = model_copy;
            std.debug.print("Model switched to: {s}\n\n", .{model_copy});
            continue;
        }

        // ── Append user message to structured history ──────────────
        const user_text = try allocator.dupe(u8, trimmed);
        try history_text.append(allocator, user_text);
        try history.append(allocator, .{ .role = "user", .content = user_text });

        // ── Generate with structured messages ──────────────────────
        var result = abi.ai.llm.providers.generate(allocator, .{
            .model = options.model.?,
            .prompt = trimmed,
            .messages = history.items,
            .system_prompt = system_prompt,
            .backend = options.backend,
            .fallback = options.fallback,
            .strict_backend = options.strict_backend,
            .plugin_id = options.plugin_id,
            .max_tokens = options.max_tokens,
            .temperature = options.temperature,
            .top_p = options.top_p,
            .top_k = options.top_k,
            .repetition_penalty = options.repetition_penalty,
        }) catch |err| {
            std.debug.print("LLM session call failed: {t}\n\n", .{err});
            continue;
        };
        defer result.deinit(allocator);

        // ── Append assistant response to structured history ────────
        const assistant_text = try allocator.dupe(u8, result.content);
        try history_text.append(allocator, assistant_text);
        try history.append(allocator, .{ .role = "assistant", .content = assistant_text });

        std.debug.print("[{s}] Assistant> {s}\n\n", .{ result.provider.label(), result.content });
    }

    std.debug.print("Session ended.\n", .{});
}

fn printProviderStatus(allocator: std.mem.Allocator) void {
    std.debug.print("\nProvider status:\n", .{});
    inline for (abi.ai.llm.providers.registry.all_providers) |provider| {
        const available = abi.ai.llm.providers.health.isAvailable(allocator, provider, null);
        std.debug.print("  {s:16} {s}\n", .{ provider.label(), if (available) "[OK]" else "[ ]" });
    }
    std.debug.print("\n", .{});
}

fn parseProviderId(value: []const u8) ?ProviderId {
    if (ProviderId.fromString(value)) |provider| return provider;

    if (std.mem.eql(u8, value, "llama-cpp")) return .llama_cpp;
    if (std.mem.eql(u8, value, "lm-studio")) return .lm_studio;
    if (std.mem.eql(u8, value, "plugin-http")) return .plugin_http;
    if (std.mem.eql(u8, value, "plugin-native")) return .plugin_native;
    if (std.mem.eql(u8, value, "local-gguf")) return .local_gguf;

    return null;
}

fn printSlashHelp() void {
    std.debug.print(
        "\nSession commands:\n" ++
            "  /quit, /exit           Exit session\n" ++
            "  /clear                 Clear conversation history\n" ++
            "  /help                  Show this help\n" ++
            "  /providers             Show available providers\n" ++
            "  /backend <id>          Switch backend (e.g. /backend anthropic)\n" ++
            "  /model <id>            Switch model (e.g. /model llama3)\n\n",
        .{},
    );
}

pub fn printSessionHelp() void {
    std.debug.print(
        "Usage: abi llm session --model <id|path> [options]\n\n" ++
            "Interactive LLM session using the same provider router as 'llm run'.\n" ++
            "Maintains structured multi-turn conversation history.\n\n" ++
            "Options:\n" ++
            "  -m, --model <id|path>   Model id or local file path\n" ++
            "  --backend <id>          Pin backend\n" ++
            "  --fallback <csv>        Comma-separated fallback chain\n" ++
            "  --strict-backend        Disable fallback\n" ++
            "  --plugin <id>           Pin plugin id\n" ++
            "  --system <text>         System prompt\n" ++
            "  -n, --max-tokens <n>    Max tokens (default: 256)\n" ++
            "  -t, --temperature <f>   Temperature (default: 0.7)\n\n" ++
            "Session commands:\n" ++
            "  /quit /exit             Exit session\n" ++
            "  /clear                  Clear conversation history\n" ++
            "  /help                   Show command help\n" ++
            "  /providers              Show available providers\n" ++
            "  /backend <id>           Switch backend mid-session\n" ++
            "  /model <id>             Switch model mid-session\n",
        .{},
    );
}
