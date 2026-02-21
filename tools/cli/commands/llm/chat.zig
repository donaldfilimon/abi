//! LLM chat subcommand - Interactive chat mode.

const std = @import("std");
const abi = @import("abi");
const utils = @import("../../utils/mod.zig");
const cli_io = utils.io_backend;
const mod = @import("mod.zig");

pub fn runChat(allocator: std.mem.Allocator, args: []const [:0]const u8) !void {
    if (utils.args.containsHelpArgs(args)) {
        mod.printHelp();
        return;
    }

    if (args.len == 0) {
        std.debug.print("Usage: abi llm chat <model>\n", .{});
        std.debug.print("\nStarts an interactive chat session with the specified LLM model.\n", .{});
        std.debug.print("Note: Requires a real terminal for interactive input.\n", .{});
        std.debug.print("\nChat commands:\n", .{});
        std.debug.print("  /quit, /exit  - Exit chat\n", .{});
        std.debug.print("  /clear        - Clear conversation history\n", .{});
        std.debug.print("  /system       - Show/set system prompt\n", .{});
        std.debug.print("  /help         - Show available commands\n", .{});
        std.debug.print("  /stats        - Show generation statistics\n", .{});
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
        std.debug.print("Usage: abi llm chat <model>\n", .{});
        return;
    }

    std.debug.print("Loading model: {s}...\n", .{model_path.?});

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
        std.debug.print("Error loading model: {t}\n", .{err});
        if (err == error.UnsupportedArchitecture) {
            std.debug.print("\nThis GGUF architecture is not yet supported by ABI local inference.\n", .{});
            std.debug.print("Current local engine targets LLaMA-compatible transformer layouts.\n", .{});
            std.debug.print("Tip: remove `--no-ollama-fallback` to run this model via Ollama.\n", .{});
            return;
        }
        std.debug.print("\nNote: GGUF model loading requires a valid GGUF file.\n", .{});
        std.debug.print("You can download models from: https://huggingface.co/TheBloke\n", .{});
        return;
    };

    const backend = engine.getBackend();
    std.debug.print("Backend: {s}\n", .{backend.label()});
    if (backend == .ollama) {
        if (engine.getBackendModelName()) |name| {
            std.debug.print("Ollama model: {s}\n", .{name});
        }
    }

    std.debug.print("\nInteractive Chat Mode\n", .{});
    std.debug.print("=====================\n", .{});
    std.debug.print("Type '/quit' to exit, '/help' for commands.\n\n", .{});

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
        std.debug.print("You> ", .{});

        // Read user input
        const input = reader.interface.takeDelimiter('\n') catch |err| {
            std.debug.print("\nInput error: {t}\n", .{err});
            continue;
        } orelse break;

        // Trim whitespace
        const trimmed = std.mem.trim(u8, input, " \t\r\n");
        if (trimmed.len == 0) continue;

        // Handle commands
        if (std.mem.startsWith(u8, trimmed, "/")) {
            if (std.mem.eql(u8, trimmed, "/quit") or std.mem.eql(u8, trimmed, "/exit")) {
                std.debug.print("Goodbye!\n", .{});
                break;
            }

            if (std.mem.eql(u8, trimmed, "/clear")) {
                conversation.clearRetainingCapacity();
                std.debug.print("Conversation history cleared.\n\n", .{});
                continue;
            }

            if (std.mem.eql(u8, trimmed, "/help")) {
                std.debug.print("\nChat Commands:\n", .{});
                std.debug.print("  /quit, /exit  - Exit the chat\n", .{});
                std.debug.print("  /clear        - Clear conversation history\n", .{});
                std.debug.print("  /system       - Show current system prompt\n", .{});
                std.debug.print("  /stats        - Show generation statistics\n", .{});
                std.debug.print("  /help         - Show this help\n\n", .{});
                continue;
            }

            if (std.mem.eql(u8, trimmed, "/system")) {
                std.debug.print("System prompt: {s}\n\n", .{system_prompt});
                continue;
            }

            if (std.mem.eql(u8, trimmed, "/stats")) {
                const stats = engine.getStats();
                std.debug.print("\nGeneration Statistics:\n", .{});
                std.debug.print("  Prefill: {d:.1} tokens/sec\n", .{stats.prefillTokensPerSecond()});
                std.debug.print("  Decode:  {d:.1} tokens/sec\n\n", .{stats.decodeTokensPerSecond()});
                continue;
            }

            std.debug.print("Unknown command: {s}\n", .{trimmed});
            std.debug.print("Type '/help' for available commands.\n\n", .{});
            continue;
        }

        // Build prompt with conversation context
        conversation.clearRetainingCapacity();
        try conversation.appendSlice(allocator, system_prompt);
        try conversation.appendSlice(allocator, "\n\nUser: ");
        try conversation.appendSlice(allocator, trimmed);
        try conversation.appendSlice(allocator, "\n\nAssistant: ");

        // Generate response
        std.debug.print("\nAssistant> ", .{});
        const response = engine.generate(allocator, conversation.items) catch |err| {
            std.debug.print("Error generating response: {t}\n\n", .{err});
            continue;
        };
        defer allocator.free(response);

        std.debug.print("{s}\n\n", .{response});
    }
}
