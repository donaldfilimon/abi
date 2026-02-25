const std = @import("std");
const context_mod = @import("../../framework/context.zig");
const abi = @import("abi");
const utils = @import("../../utils/mod.zig");
const cli_io = utils.io_backend;
const run_cmd = @import("run.zig");

const ChatMessage = abi.ai.llm.providers.ChatMessage;
const ProviderId = abi.ai.llm.providers.ProviderId;
const provider_parser = abi.ai.llm.providers.parser;

const SyncModelEntry = struct {
    provider: ProviderId,
    model: []u8,
};

const SyncState = struct {
    enabled: bool = false,
    chain: []ProviderId = &.{},
    owns_chain: bool = false,
    next_index: usize = 0,
    model_map: std.ArrayListUnmanaged(SyncModelEntry) = .empty,

    fn deinit(self: *SyncState, allocator: std.mem.Allocator) void {
        if (self.owns_chain) allocator.free(self.chain);
        for (self.model_map.items) |entry| allocator.free(entry.model);
        self.model_map.deinit(allocator);
        self.* = undefined;
    }

    fn modelForProvider(self: *const SyncState, provider: ProviderId) ?[]const u8 {
        for (self.model_map.items) |entry| {
            if (entry.provider == provider) return entry.model;
        }
        return null;
    }

    fn setModel(self: *SyncState, allocator: std.mem.Allocator, provider: ProviderId, model: []const u8) !void {
        for (self.model_map.items) |*entry| {
            if (entry.provider == provider) {
                allocator.free(entry.model);
                entry.model = try allocator.dupe(u8, model);
                return;
            }
        }
        try self.model_map.append(allocator, .{
            .provider = provider,
            .model = try allocator.dupe(u8, model),
        });
    }
};

pub fn runSession(ctx: *const context_mod.CommandContext, args: []const [:0]const u8) !void {
    const allocator = ctx.allocator;
    if (utils.args.containsHelpArgs(args)) {
        printSessionHelp();
        return;
    }

    var options = try run_cmd.parseRunArgs(allocator, args);
    defer allocator.free(options.fallback);

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

    var sync = try parseSyncArgs(allocator, args);
    defer sync.deinit(allocator);

    if (options.model == null and !sync.enabled) {
        utils.output.printError("--model is required.", .{});
        utils.output.println("", .{});
        printSessionHelp();
        return;
    }
    if (options.model == null and sync.enabled) {
        options.model = "sync";
    }

    utils.output.printSuccess("LLM session started (model: {s}).", .{options.model.?});
    utils.output.println("Commands: /quit, /exit, /clear, /help, /providers, /backend <id>, /model <id>, /sync", .{});
    utils.output.println("", .{});

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
        utils.output.print("You> ", .{});
        const line = reader.interface.takeDelimiter('\n') catch |err| {
            utils.output.printError("Input error: {t}", .{err});
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
            utils.output.printSuccess("Session context cleared.", .{});
            utils.output.println("", .{});
            continue;
        }
        if (std.mem.eql(u8, trimmed, "/providers")) {
            printProviderStatus(allocator);
            continue;
        }
        if (std.mem.eql(u8, trimmed, "/sync")) {
            printSyncStatus(&sync);
            continue;
        }
        if (std.mem.startsWith(u8, trimmed, "/sync")) {
            const rest = std.mem.trim(u8, trimmed["/sync".len..], " \t");
            if (rest.len == 0) {
                printSyncStatus(&sync);
            } else if (std.mem.eql(u8, rest, "on")) {
                sync.enabled = true;
                utils.output.printSuccess("Sync mode enabled.", .{});
                utils.output.println("", .{});
            } else if (std.mem.eql(u8, rest, "off")) {
                sync.enabled = false;
                utils.output.println("Sync mode disabled.", .{});
                utils.output.println("", .{});
            } else {
                utils.output.println("Usage: /sync [on|off]", .{});
                utils.output.println("", .{});
            }
            continue;
        }
        if (std.mem.startsWith(u8, trimmed, "/backend")) {
            const rest = std.mem.trim(u8, trimmed["/backend".len..], " \t");
            if (rest.len == 0) {
                if (options.backend) |b| {
                    utils.output.printKeyValueFmt("Current backend", "{s}", .{b.label()});
                    utils.output.println("", .{});
                } else {
                    utils.output.println("No backend pinned (using auto-routing).", .{});
                    utils.output.println("", .{});
                }
                continue;
            }
            if (parseProviderId(rest)) |new_backend| {
                options.backend = new_backend;
                if (sync.enabled) {
                    sync.enabled = false;
                    utils.output.printSuccess("Backend switched to: {s} (sync disabled)", .{new_backend.label()});
                    utils.output.println("", .{});
                } else {
                    utils.output.printSuccess("Backend switched to: {s}", .{new_backend.label()});
                    utils.output.println("", .{});
                }
            } else {
                utils.output.printError("Unknown backend: {s}", .{rest});
                utils.output.println("Available: local_gguf, llama_cpp, mlx, ollama, ollama_passthrough, lm_studio, vllm, anthropic, openai, codex, opencode, claude, gemini, plugin_http, plugin_native", .{});
                utils.output.println("", .{});
            }
            continue;
        }
        if (std.mem.startsWith(u8, trimmed, "/model")) {
            const rest = std.mem.trim(u8, trimmed["/model".len..], " \t");
            if (rest.len == 0) {
                utils.output.printKeyValueFmt("Current model", "{s}", .{options.model.?});
                utils.output.println("", .{});
                continue;
            }
            // Dupe into heap so it survives read buffer reuse
            const model_copy = allocator.dupe(u8, rest) catch {
                utils.output.printError("out of memory", .{});
                continue;
            };
            // Free previous model if it was heap-allocated via /model
            if (model_switched) |prev| allocator.free(prev);
            model_switched = model_copy;
            options.model = model_copy;
            utils.output.printSuccess("Model switched to: {s}", .{model_copy});
            utils.output.println("", .{});
            continue;
        }

        // ── Append user message to structured history ──────────────
        const user_text = try allocator.dupe(u8, trimmed);
        try history_text.append(allocator, user_text);
        try history.append(allocator, .{ .role = "user", .content = user_text });

        // ── Generate with structured messages ──────────────────────
        var result: abi.ai.llm.providers.GenerateResult = undefined;
        if (sync.enabled and sync.chain.len > 0) {
            var got_result = false;
            var attempts: usize = 0;
            while (attempts < sync.chain.len) : (attempts += 1) {
                const idx = (sync.next_index + attempts) % sync.chain.len;
                const provider = sync.chain[idx];
                const model = sync.modelForProvider(provider) orelse {
                    utils.output.printWarning("[{s}] skipped: no model configured", .{provider.label()});
                    continue;
                };

                result = abi.ai.llm.providers.generate(allocator, .{
                    .model = model,
                    .prompt = trimmed,
                    .messages = history.items,
                    .system_prompt = system_prompt,
                    .backend = provider,
                    .fallback = &.{},
                    .strict_backend = true,
                    .plugin_id = options.plugin_id,
                    .max_tokens = options.max_tokens,
                    .temperature = options.temperature,
                    .top_p = options.top_p,
                    .top_k = options.top_k,
                    .repetition_penalty = options.repetition_penalty,
                }) catch |err| {
                    utils.output.printError("[{s}] failed: {t}", .{ provider.label(), err });
                    continue;
                };

                sync.next_index = (idx + 1) % sync.chain.len;
                got_result = true;
                break;
            }
            if (!got_result) {
                utils.output.printError("All sync providers failed for this turn.", .{});
                utils.output.println("", .{});
                continue;
            }
        } else {
            result = abi.ai.llm.providers.generate(allocator, .{
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
                utils.output.printError("LLM session call failed: {t}", .{err});
                utils.output.println("", .{});
                continue;
            };
        }
        defer result.deinit(allocator);

        // ── Append assistant response to structured history ────────
        const assistant_text = try allocator.dupe(u8, result.content);
        try history_text.append(allocator, assistant_text);
        try history.append(allocator, .{ .role = "assistant", .content = assistant_text });

        utils.output.println("[{s}] Assistant> {s}", .{ result.provider.label(), result.content });
        utils.output.println("", .{});
    }

    utils.output.println("Session ended.", .{});
}

fn printProviderStatus(allocator: std.mem.Allocator) void {
    utils.output.printHeader("Provider status");
    inline for (abi.ai.llm.providers.registry.all_providers) |provider| {
        const available = abi.ai.llm.providers.health.isAvailable(allocator, provider, null);
        utils.output.printStatusLineFmt("{s:16}", .{provider.label()}, available);
    }
    utils.output.println("", .{});
}

fn parseProviderId(value: []const u8) ?ProviderId {
    return provider_parser.parseProviderId(value);
}

fn parseSyncArgs(allocator: std.mem.Allocator, args: []const [:0]const u8) !SyncState {
    var sync = SyncState{};
    errdefer sync.deinit(allocator);

    var sync_enabled = false;
    var providers_csv: ?[]const u8 = null;
    var model_map_csv: ?[]const u8 = null;

    var i: usize = 0;
    while (i < args.len) {
        const arg = std.mem.sliceTo(args[i], 0);
        i += 1;

        if (std.mem.eql(u8, arg, "--sync")) {
            sync_enabled = true;
            continue;
        }
        if (std.mem.eql(u8, arg, "--sync-providers") and i < args.len) {
            sync_enabled = true;
            providers_csv = std.mem.sliceTo(args[i], 0);
            i += 1;
            continue;
        }
        if (std.mem.eql(u8, arg, "--sync-models") and i < args.len) {
            sync_enabled = true;
            model_map_csv = std.mem.sliceTo(args[i], 0);
            i += 1;
            continue;
        }
    }

    if (!sync_enabled) return sync;

    sync.enabled = true;
    sync.next_index = 0;

    if (providers_csv) |csv| {
        var chain_builder = std.ArrayListUnmanaged(ProviderId).empty;
        defer chain_builder.deinit(allocator);

        var it = std.mem.splitScalar(u8, csv, ',');
        while (it.next()) |piece| {
            const trimmed = std.mem.trim(u8, piece, " \t\r\n");
            if (trimmed.len == 0) continue;
            const provider = provider_parser.parseProviderId(trimmed) orelse return error.InvalidBackend;
            var exists = false;
            for (chain_builder.items) |existing| {
                if (existing == provider) {
                    exists = true;
                    break;
                }
            }
            if (!exists) try chain_builder.append(allocator, provider);
        }

        sync.chain = try chain_builder.toOwnedSlice(allocator);
        sync.owns_chain = true;
    } else {
        sync.chain = try allocator.dupe(ProviderId, abi.ai.llm.providers.registry.sync_round_robin_chain[0..]);
        sync.owns_chain = true;
    }

    for (sync.chain) |provider| {
        if (try defaultModelForProvider(allocator, provider)) |model| {
            defer allocator.free(model);
            try sync.setModel(allocator, provider, model);
        }
    }

    if (model_map_csv) |csv| {
        var it = std.mem.splitScalar(u8, csv, ',');
        while (it.next()) |piece| {
            const trimmed = std.mem.trim(u8, piece, " \t\r\n");
            if (trimmed.len == 0) continue;

            const eq_idx = std.mem.indexOfScalar(u8, trimmed, '=') orelse return error.InvalidBackend;
            const provider_text = std.mem.trim(u8, trimmed[0..eq_idx], " \t");
            const model_text = std.mem.trim(u8, trimmed[eq_idx + 1 ..], " \t");
            if (provider_text.len == 0 or model_text.len == 0) return error.InvalidBackend;

            const provider = provider_parser.parseProviderId(provider_text) orelse return error.InvalidBackend;
            try sync.setModel(allocator, provider, model_text);
        }
    }

    return sync;
}

fn defaultModelForProvider(allocator: std.mem.Allocator, provider: ProviderId) !?[]u8 {
    switch (provider) {
        .codex => {
            if (try abi.connectors.tryLoadCodex(allocator)) |cfg| {
                var config = cfg;
                defer config.deinit(allocator);
                return @as(?[]u8, try allocator.dupe(u8, config.model));
            }
            return null;
        },
        .opencode => {
            if (try abi.connectors.tryLoadOpenCode(allocator)) |cfg| {
                var config = cfg;
                defer config.deinit(allocator);
                return @as(?[]u8, try allocator.dupe(u8, config.model));
            }
            return null;
        },
        .claude => {
            if (try abi.connectors.tryLoadClaude(allocator)) |cfg| {
                var config = cfg;
                defer config.deinit(allocator);
                return @as(?[]u8, try allocator.dupe(u8, config.model));
            }
            return null;
        },
        .gemini => {
            if (try abi.connectors.tryLoadGemini(allocator)) |cfg| {
                var config = cfg;
                defer config.deinit(allocator);
                return @as(?[]u8, try allocator.dupe(u8, config.model));
            }
            return null;
        },
        .ollama_passthrough => {
            if (try abi.connectors.tryLoadOllamaPassthrough(allocator)) |cfg| {
                var config = cfg;
                defer config.deinit(allocator);
                return @as(?[]u8, try allocator.dupe(u8, config.model));
            }
            return null;
        },
        .ollama => {
            if (try abi.connectors.tryLoadOllama(allocator)) |cfg| {
                var config = cfg;
                defer config.deinit(allocator);
                return @as(?[]u8, try allocator.dupe(u8, config.model));
            }
            return null;
        },
        .anthropic => {
            if (try abi.connectors.tryLoadAnthropic(allocator)) |cfg| {
                var config = cfg;
                defer config.deinit(allocator);
                return @as(?[]u8, try allocator.dupe(u8, config.model));
            }
            return null;
        },
        .openai => {
            if (try abi.connectors.tryLoadOpenAI(allocator)) |cfg| {
                var config = cfg;
                defer config.deinit(allocator);
                return @as(?[]u8, try allocator.dupe(u8, config.model));
            }
            return null;
        },
        .llama_cpp => {
            if (try abi.connectors.tryLoadLlamaCpp(allocator)) |cfg| {
                var config = cfg;
                defer config.deinit(allocator);
                return @as(?[]u8, try allocator.dupe(u8, config.model));
            }
            return null;
        },
        .mlx => {
            if (try abi.connectors.tryLoadMLX(allocator)) |cfg| {
                var config = cfg;
                defer config.deinit(allocator);
                return @as(?[]u8, try allocator.dupe(u8, config.model));
            }
            return null;
        },
        .lm_studio => {
            if (try abi.connectors.tryLoadLMStudio(allocator)) |cfg| {
                var config = cfg;
                defer config.deinit(allocator);
                return @as(?[]u8, try allocator.dupe(u8, config.model));
            }
            return null;
        },
        .vllm => {
            if (try abi.connectors.tryLoadVLLM(allocator)) |cfg| {
                var config = cfg;
                defer config.deinit(allocator);
                return @as(?[]u8, try allocator.dupe(u8, config.model));
            }
            return null;
        },
        else => return null,
    }
}

fn printSyncStatus(sync: *const SyncState) void {
    utils.output.printKeyValueFmt("Sync", "{s}", .{if (sync.enabled) "enabled" else "disabled"});
    if (sync.chain.len == 0) {
        utils.output.println("Sync chain: (empty)", .{});
        utils.output.println("", .{});
        return;
    }
    utils.output.print("  Sync chain: ", .{});
    for (sync.chain, 0..) |provider, idx| {
        if (idx != 0) utils.output.print(" -> ", .{});
        utils.output.print("{s}", .{provider.label()});
    }
    utils.output.println("", .{});
    utils.output.printKeyValueFmt("Next provider", "{s}", .{sync.chain[sync.next_index % sync.chain.len].label()});
    utils.output.println("", .{});
}

fn printSlashHelp() void {
    utils.output.print(
        "\nSession commands:\n" ++
            "  /quit, /exit           Exit session\n" ++
            "  /clear                 Clear conversation history\n" ++
            "  /help                  Show this help\n" ++
            "  /providers             Show available providers\n" ++
            "  /backend <id>          Switch backend (e.g. /backend anthropic)\n" ++
            "  /model <id>            Switch model (e.g. /model llama3)\n" ++
            "  /sync [on|off]         Show or toggle sync mode\n\n",
        .{},
    );
}

pub fn printSessionHelp() void {
    utils.output.print(
        "Usage: abi llm session --model <id|path> [options]\n\n" ++
            "Interactive LLM session using the same provider router as 'llm run'.\n" ++
            "Maintains structured multi-turn conversation history.\n\n" ++
            "Options:\n" ++
            "  -m, --model <id|path>   Model id or local file path\n" ++
            "  --backend <id>          Pin backend\n" ++
            "  --fallback <csv>        Comma-separated fallback chain\n" ++
            "  --strict-backend        Disable fallback\n" ++
            "  --plugin <id>           Pin plugin id\n" ++
            "  --sync                  Enable round-robin sync mode\n" ++
            "  --sync-providers <csv>  Override sync provider chain\n" ++
            "  --sync-models <csv>     Provider model map (provider=model,...)\n" ++
            "  --system <text>         System prompt\n" ++
            "  -n, --max-tokens <n>    Max tokens (default: 256)\n" ++
            "  -t, --temperature <f>   Temperature (default: 0.7)\n\n" ++
            "Session commands:\n" ++
            "  /quit /exit             Exit session\n" ++
            "  /clear                  Clear conversation history\n" ++
            "  /help                   Show command help\n" ++
            "  /providers              Show available providers\n" ++
            "  /backend <id>           Switch backend mid-session\n" ++
            "  /model <id>             Switch model mid-session\n" ++
            "  /sync [on|off]          Show or toggle sync mode\n",
        .{},
    );
}

test "parseSyncArgs parses sync providers and model overrides" {
    const allocator = std.testing.allocator;
    const args = [_][:0]const u8{
        "--sync",
        "--sync-providers",
        "codex,claude,gemini",
        "--sync-models",
        "codex=gpt-5-codex,claude=claude-3-5-sonnet-20241022",
    };

    var sync = try parseSyncArgs(allocator, &args);
    defer sync.deinit(allocator);

    try std.testing.expect(sync.enabled);
    try std.testing.expectEqual(@as(usize, 3), sync.chain.len);
    try std.testing.expect(sync.chain[0] == .codex);
    try std.testing.expect(sync.chain[1] == .claude);
    try std.testing.expect(sync.chain[2] == .gemini);
    try std.testing.expect(sync.modelForProvider(.codex) != null);
    try std.testing.expect(sync.modelForProvider(.claude) != null);
}

test {
    std.testing.refAllDecls(@This());
}
