const std = @import("std");
const abi = @import("abi");
const utils = @import("../../utils/mod.zig");
const cli_io = utils.io_backend;
const run_cmd = @import("run.zig");

pub fn runSession(allocator: std.mem.Allocator, args: []const [:0]const u8) !void {
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
    std.debug.print("Commands: /quit, /exit, /clear, /help\n\n", .{});

    var io_backend = cli_io.initIoBackend(allocator);
    defer io_backend.deinit();
    const io = io_backend.io();

    const stdin_file = std.Io.File.stdin();
    var read_buffer: [8192]u8 = undefined;
    var reader = stdin_file.reader(io, &read_buffer);

    var history = std.ArrayListUnmanaged(u8).empty;
    defer history.deinit(allocator);

    while (true) {
        std.debug.print("You> ", .{});
        const line = reader.interface.takeDelimiter('\n') catch |err| {
            std.debug.print("\nInput error: {t}\n", .{err});
            continue;
        } orelse break;

        const trimmed = std.mem.trim(u8, line, " \t\r\n");
        if (trimmed.len == 0) continue;

        if (std.mem.eql(u8, trimmed, "/quit") or std.mem.eql(u8, trimmed, "/exit")) {
            break;
        }
        if (std.mem.eql(u8, trimmed, "/help")) {
            std.debug.print("/quit, /exit, /clear, /help\n\n", .{});
            continue;
        }
        if (std.mem.eql(u8, trimmed, "/clear")) {
            history.clearRetainingCapacity();
            std.debug.print("Session context cleared.\n\n", .{});
            continue;
        }

        const prompt = try buildPrompt(allocator, &history, system_prompt, trimmed);
        defer allocator.free(prompt);

        var result = abi.ai.llm.providers.generate(allocator, .{
            .model = options.model.?,
            .prompt = prompt,
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

        try appendTurn(allocator, &history, trimmed, result.content);

        std.debug.print("[{s}] Assistant> {s}\n\n", .{ result.provider.label(), result.content });
    }

    std.debug.print("Session ended.\n", .{});
}

fn buildPrompt(
    allocator: std.mem.Allocator,
    history: *const std.ArrayListUnmanaged(u8),
    system_prompt: []const u8,
    user_input: []const u8,
) ![]u8 {
    var out = std.ArrayListUnmanaged(u8).empty;
    errdefer out.deinit(allocator);

    try out.appendSlice(allocator, system_prompt);
    try out.appendSlice(allocator, "\n\n");
    if (history.items.len > 0) {
        try out.appendSlice(allocator, history.items);
        try out.appendSlice(allocator, "\n");
    }
    try out.appendSlice(allocator, "User: ");
    try out.appendSlice(allocator, user_input);
    try out.appendSlice(allocator, "\nAssistant: ");

    return out.toOwnedSlice(allocator);
}

fn appendTurn(
    allocator: std.mem.Allocator,
    history: *std.ArrayListUnmanaged(u8),
    user_input: []const u8,
    answer: []const u8,
) !void {
    if (history.items.len > 0) {
        try history.appendSlice(allocator, "\n");
    }
    try history.appendSlice(allocator, "User: ");
    try history.appendSlice(allocator, user_input);
    try history.appendSlice(allocator, "\nAssistant: ");
    try history.appendSlice(allocator, answer);
}

pub fn printSessionHelp() void {
    std.debug.print(
        "Usage: abi llm session --model <id|path> [options]\\n\\n" ++
            "Interactive LLM session using the same provider router as 'llm run'.\\n\\n" ++
            "Options:\\n" ++
            "  -m, --model <id|path>   Model id or local file path\\n" ++
            "  --backend <id>          Pin backend\\n" ++
            "  --fallback <csv>        Comma-separated fallback chain\\n" ++
            "  --strict-backend        Disable fallback\\n" ++
            "  --plugin <id>           Pin plugin id\\n" ++
            "  --system <text>         System prompt\\n" ++
            "  -n, --max-tokens <n>    Max tokens (default: 256)\\n" ++
            "  -t, --temperature <f>   Temperature (default: 0.7)\\n\\n" ++
            "Session commands:\\n" ++
            "  /quit /exit             Exit session\\n" ++
            "  /clear                  Clear in-memory conversation context\\n" ++
            "  /help                   Show command help\\n",
        .{},
    );
}
