//! LLM chat subcommand - Interactive chat mode.

const std = @import("std");
const context_mod = @import("../../framework/context.zig");
const abi = @import("abi");
const utils = @import("../../utils/mod.zig");
const cli_io = utils.io_backend;
const mod = @import("mod.zig");

pub fn runChat(ctx: *const context_mod.CommandContext, args: []const [:0]const u8) !void {
    const allocator = ctx.allocator;
    if (utils.args.containsHelpArgs(args)) {
        mod.printHelp();
        return;
    }

    if (args.len == 0) {
        utils.output.println("Usage: abi llm chat <model>", .{});
        utils.output.println("\nStarts an interactive chat session with the specified LLM model.", .{});
        utils.output.println("Note: Requires a real terminal for interactive input.", .{});
        utils.output.println("\nChat commands:", .{});
        utils.output.println("  /quit, /exit  - Exit chat", .{});
        utils.output.println("  /clear        - Clear conversation history", .{});
        utils.output.println("  /system       - Show/set system prompt", .{});
        utils.output.println("  /help         - Show available commands", .{});
        utils.output.println("  /stats        - Show generation statistics", .{});
        return;
    }

    var model_path: ?[]const u8 = null;
    var allow_ollama_fallback: bool = true;
    var ollama_model: ?[]const u8 = null;

    var i: usize = 0;
    while (i < args.len) {
        const arg = std.mem.sliceTo(args[i], 0);
        i += 1;

        if (std.mem.eql(u8, arg, "--no-ollama-fallback")) {
            allow_ollama_fallback = false;
            continue;
        }

        if (std.mem.eql(u8, arg, "--ollama-model")) {
            if (i < args.len) {
                ollama_model = std.mem.sliceTo(args[i], 0);
                i += 1;
            }
            continue;
        }

        if (model_path == null) {
            model_path = arg;
        }
    }

    if (model_path == null) {
        utils.output.println("Usage: abi llm chat <model>", .{});
        return;
    }

    utils.output.println("Loading model: {s}...", .{model_path.?});

    // Create inference engine
    var engine = abi.ai.llm.Engine.init(allocator, .{
        .max_new_tokens = 512,
        .temperature = 0.7,
        .top_p = 0.9,
        .allow_ollama_fallback = allow_ollama_fallback,
        .ollama_model = ollama_model,
    });
    defer engine.deinit();

    // Load model
    engine.loadModel(model_path.?) catch |err| {
        utils.output.printError("Loading model: {t}", .{err});
        if (err == error.UnsupportedArchitecture) {
            utils.output.println("\nThis GGUF architecture is not yet supported by ABI local inference.", .{});
            utils.output.println("Current local engine targets LLaMA-compatible transformer layouts.", .{});
            utils.output.printInfo("Tip: remove `--no-ollama-fallback` to run this model via Ollama.", .{});
            return;
        }
        utils.output.println("\nNote: GGUF model loading requires a valid GGUF file.", .{});
        utils.output.println("You can download models from: https://huggingface.co/TheBloke", .{});
        return;
    };

    const backend = engine.getBackend();
    utils.output.printKeyValueFmt("Backend", "{s}", .{backend.label()});
    if (backend == .ollama) {
        if (engine.getBackendModelName()) |name| {
            utils.output.printKeyValueFmt("Ollama model", "{s}", .{name});
        }
    }

    utils.output.printHeader("Interactive Chat Mode");
    utils.output.println("Type '/quit' to exit, '/help' for commands.\n", .{});

    // Set up Zig 0.16 I/O backend
    var io_backend = cli_io.initIoBackend(allocator);
    defer io_backend.deinit();
    const io = io_backend.io();

    const stdin_file = std.Io.File.stdin();
    var read_buffer: [4096]u8 = undefined;
    var reader = stdin_file.reader(io, &read_buffer);

    // Conversation history (simplified - could be improved with proper context management)
    var conversation = std.ArrayListUnmanaged(u8).empty;
    defer conversation.deinit(allocator);

    const system_prompt: []const u8 = "You are a helpful assistant.";

    while (true) {
        utils.output.print("You> ", .{});

        // Read user input
        const input = reader.interface.takeDelimiter('\n') catch |err| {
            utils.output.println("", .{});
            utils.output.printError("Input error: {t}", .{err});
            continue;
        } orelse break;

        // Trim whitespace
        const trimmed = std.mem.trim(u8, input, " \t\r\n");
        if (trimmed.len == 0) continue;

        // Handle commands
        if (std.mem.startsWith(u8, trimmed, "/")) {
            if (std.mem.eql(u8, trimmed, "/quit") or std.mem.eql(u8, trimmed, "/exit")) {
                utils.output.println("Goodbye!", .{});
                break;
            }

            if (std.mem.eql(u8, trimmed, "/clear")) {
                conversation.clearRetainingCapacity();
                utils.output.printSuccess("Conversation history cleared.", .{});
                utils.output.println("", .{});
                continue;
            }

            if (std.mem.eql(u8, trimmed, "/help")) {
                utils.output.println("\nChat Commands:", .{});
                utils.output.println("  /quit, /exit  - Exit the chat", .{});
                utils.output.println("  /clear        - Clear conversation history", .{});
                utils.output.println("  /system       - Show current system prompt", .{});
                utils.output.println("  /stats        - Show generation statistics", .{});
                utils.output.println("  /help         - Show this help\n", .{});
                continue;
            }

            if (std.mem.eql(u8, trimmed, "/system")) {
                utils.output.printKeyValueFmt("System prompt", "{s}", .{system_prompt});
                utils.output.println("", .{});
                continue;
            }

            if (std.mem.eql(u8, trimmed, "/stats")) {
                const stats = engine.getStats();
                utils.output.println("\nGeneration Statistics:", .{});
                utils.output.printKeyValueFmt("Prefill", "{d:.1} tokens/sec", .{stats.prefillTokensPerSecond()});
                utils.output.printKeyValueFmt("Decode", "{d:.1} tokens/sec", .{stats.decodeTokensPerSecond()});
                utils.output.println("", .{});
                continue;
            }

            utils.output.printWarning("Unknown command: {s}", .{trimmed});
            utils.output.println("Type '/help' for available commands.\n", .{});
            continue;
        }

        // Build prompt with conversation context
        conversation.clearRetainingCapacity();
        try conversation.appendSlice(allocator, system_prompt);
        try conversation.appendSlice(allocator, "\n\nUser: ");
        try conversation.appendSlice(allocator, trimmed);
        try conversation.appendSlice(allocator, "\n\nAssistant: ");

        // Generate response
        utils.output.print("\nAssistant> ", .{});
        const response = engine.generate(allocator, conversation.items) catch |err| {
            utils.output.printError("Generating response: {t}", .{err});
            utils.output.println("", .{});
            continue;
        };
        defer allocator.free(response);

        utils.output.println("{s}\n", .{response});
    }
}

test {
    std.testing.refAllDecls(@This());
}
