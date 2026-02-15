//! LLM command for local model inference.
//!
//! Commands:
//! - llm info <model> - Show model information
//! - llm generate <model> --prompt <text> - Generate text
//! - llm chat <model> - Interactive chat mode
//! - llm bench <model> - Benchmark model performance

const std = @import("std");
const abi = @import("abi");
const utils = @import("../utils/mod.zig");
const cli_io = utils.io_backend;

const llm_subcommands = [_][]const u8{
    "info",
    "generate",
    "chat",
    "bench",
    "list",
    "list-local",
    "demo",
    "download",
    "serve",
};

fn lInfo(a: std.mem.Allocator, p: *utils.args.ArgParser) !void {
    try runInfo(a, p.remaining());
}
fn lGenerate(a: std.mem.Allocator, p: *utils.args.ArgParser) !void {
    try runGenerate(a, p.remaining());
}
fn lChat(a: std.mem.Allocator, p: *utils.args.ArgParser) !void {
    try runChat(a, p.remaining());
}
fn lBench(a: std.mem.Allocator, p: *utils.args.ArgParser) !void {
    try runBench(a, p.remaining());
}
fn lList(_: std.mem.Allocator, p: *utils.args.ArgParser) !void {
    if (p.containsHelp()) {
        printHelp();
        return;
    }
    runList();
}
fn lListLocal(a: std.mem.Allocator, p: *utils.args.ArgParser) !void {
    if (p.containsHelp()) {
        printHelp();
        return;
    }
    runListLocal(a, p.remaining());
}
fn lDemo(a: std.mem.Allocator, p: *utils.args.ArgParser) !void {
    try runDemo(a, p.remaining());
}
fn lDownload(a: std.mem.Allocator, p: *utils.args.ArgParser) !void {
    try runDownload(a, p.remaining());
}
fn lServe(a: std.mem.Allocator, p: *utils.args.ArgParser) !void {
    try runServe(a, p.remaining());
}
fn lUnknown(cmd: []const u8) void {
    std.debug.print("Unknown llm command: {s}\n", .{cmd});
    if (utils.args.suggestCommand(cmd, &llm_subcommands)) |suggestion| {
        std.debug.print("Did you mean: {s}\n", .{suggestion});
    }
}
fn lHelp(_: std.mem.Allocator) void {
    printHelp();
}

const llm_commands = [_]utils.subcommand.Command{
    .{ .names = &.{"info"}, .run = lInfo },
    .{ .names = &.{"generate"}, .run = lGenerate },
    .{ .names = &.{"chat"}, .run = lChat },
    .{ .names = &.{"bench"}, .run = lBench },
    .{ .names = &.{"list"}, .run = lList },
    .{ .names = &.{"list-local"}, .run = lListLocal },
    .{ .names = &.{"demo"}, .run = lDemo },
    .{ .names = &.{"download"}, .run = lDownload },
    .{ .names = &.{"serve"}, .run = lServe },
};

/// Run the LLM command with the provided arguments.
pub fn run(allocator: std.mem.Allocator, args: []const [:0]const u8) !void {
    var parser = utils.args.ArgParser.init(allocator, args);
    try utils.subcommand.runSubcommand(allocator, &parser, &llm_commands, null, lHelp, lUnknown);
}

fn runInfo(allocator: std.mem.Allocator, args: []const [:0]const u8) !void {
    if (utils.args.containsHelpArgs(args)) {
        printHelp();
        return;
    }

    if (args.len == 0) {
        std.debug.print("Usage: abi llm info <model-path>\n", .{});
        return;
    }

    const model_path = std.mem.sliceTo(args[0], 0);

    std.debug.print("Loading model: {s}\n", .{model_path});

    // Try to open as GGUF
    var gguf_file = abi.ai.llm.io.GgufFile.open(allocator, model_path) catch |err| {
        std.debug.print("Error: Failed to open model file: {t}\n", .{err});
        if (err == error.FileTooLarge) {
            printModelFileSizeHint(allocator, model_path);
        }
        return;
    };
    defer gguf_file.deinit();

    // Print summary
    gguf_file.printSummaryDebug();

    // Print additional info
    std.debug.print("\n", .{});

    // Estimate memory requirements
    const config = abi.ai.llm.model.LlamaConfig.fromGguf(&gguf_file);
    const mem_estimate = config.estimateMemory();
    const param_estimate = config.estimateParameters();

    std.debug.print("Estimated Parameters: {d:.2}B\n", .{@as(f64, @floatFromInt(param_estimate)) / 1e9});
    std.debug.print("Estimated Memory: {d:.2} GB\n", .{@as(f64, @floatFromInt(mem_estimate)) / (1024 * 1024 * 1024)});
    std.debug.print("Attention dims: q={d}, kv={d}, v={d}\n", .{ config.queryDim(), config.kvDim(), config.valueDim() });
    std.debug.print("Head dims: q={d}, kv={d}, v={d}\n", .{ config.queryHeadDim(), config.keyHeadDim(), config.valueHeadDim() });
    std.debug.print("Local LLaMA layout: {s}\n", .{if (config.supportsLlamaAttentionLayout()) "compatible" else "unsupported"});

    // List some tensors
    std.debug.print("\nTensors:\n", .{});
    var count: u32 = 0;
    var iter = gguf_file.tensors.iterator();
    while (iter.next()) |entry| {
        if (count >= 10) {
            std.debug.print("  ... and more\n", .{});
            break;
        }
        const info = entry.value_ptr.*;
        std.debug.print("  {s}: [{d}", .{ info.name, info.dims[0] });
        for (1..info.n_dims) |d| {
            std.debug.print(", {d}", .{info.dims[d]});
        }
        std.debug.print("] ({t})\n", .{info.tensor_type});
        count += 1;
    }
}

fn runGenerate(allocator: std.mem.Allocator, args: []const [:0]const u8) !void {
    if (utils.args.containsHelpArgs(args)) {
        printHelp();
        return;
    }

    var model_path: ?[]const u8 = null;
    var prompt: ?[]const u8 = null;
    var max_tokens: u32 = 256;
    var temperature: f32 = 0.7;
    var top_p: f32 = 0.9;
    var top_k: u32 = 40;
    // New generation options
    var repeat_penalty: f32 = 1.1;
    var seed: ?u64 = null;
    var stream: bool = false;
    var allow_ollama_fallback: bool = true;
    var ollama_model: ?[]const u8 = null;
    var stop_sequences = std.ArrayListUnmanaged([]const u8).empty;
    defer stop_sequences.deinit(allocator);
    // Advanced sampling options (llama.cpp parity)
    var tfs_z: f32 = 1.0; // tail-free sampling (1.0 = disabled)
    var mirostat: u8 = 0; // 0 = disabled, 1 = v1, 2 = v2
    var mirostat_tau: f32 = 5.0;
    var mirostat_eta: f32 = 0.1;

    var i: usize = 0;
    while (i < args.len) {
        const arg = args[i];
        i += 1;

        if (utils.args.matchesAny(arg, &[_][]const u8{ "--model", "-m" })) {
            if (i < args.len) {
                model_path = std.mem.sliceTo(args[i], 0);
                i += 1;
            }
            continue;
        }

        if (utils.args.matchesAny(arg, &[_][]const u8{ "--prompt", "-p" })) {
            if (i < args.len) {
                prompt = std.mem.sliceTo(args[i], 0);
                i += 1;
            }
            continue;
        }

        if (utils.args.matchesAny(arg, &[_][]const u8{ "--max-tokens", "-n" })) {
            if (i < args.len) {
                const val = std.mem.sliceTo(args[i], 0);
                max_tokens = std.fmt.parseInt(u32, val, 10) catch 256;
                i += 1;
            }
            continue;
        }

        if (utils.args.matchesAny(arg, &[_][]const u8{ "--temperature", "-t" })) {
            if (i < args.len) {
                const val = std.mem.sliceTo(args[i], 0);
                temperature = std.fmt.parseFloat(f32, val) catch 0.7;
                i += 1;
            }
            continue;
        }

        if (std.mem.eql(u8, std.mem.sliceTo(arg, 0), "--top-p")) {
            if (i < args.len) {
                const val = std.mem.sliceTo(args[i], 0);
                top_p = std.fmt.parseFloat(f32, val) catch 0.9;
                i += 1;
            }
            continue;
        }

        // New options
        if (std.mem.eql(u8, std.mem.sliceTo(arg, 0), "--repeat-penalty")) {
            if (i < args.len) {
                const val = std.mem.sliceTo(args[i], 0);
                repeat_penalty = std.fmt.parseFloat(f32, val) catch 1.1;
                i += 1;
            }
            continue;
        }

        if (std.mem.eql(u8, std.mem.sliceTo(arg, 0), "--seed")) {
            if (i < args.len) {
                const val = std.mem.sliceTo(args[i], 0);
                seed = std.fmt.parseInt(u64, val, 10) catch null;
                i += 1;
            }
            continue;
        }

        if (std.mem.eql(u8, std.mem.sliceTo(arg, 0), "--stop")) {
            if (i < args.len) {
                const val = std.mem.sliceTo(args[i], 0);
                try stop_sequences.append(allocator, val);
                i += 1;
            }
            continue;
        }

        if (std.mem.eql(u8, std.mem.sliceTo(arg, 0), "--stream")) {
            stream = true;
            continue;
        }

        if (std.mem.eql(u8, std.mem.sliceTo(arg, 0), "--no-ollama-fallback")) {
            allow_ollama_fallback = false;
            continue;
        }

        if (std.mem.eql(u8, std.mem.sliceTo(arg, 0), "--ollama-model")) {
            if (i < args.len) {
                ollama_model = std.mem.sliceTo(args[i], 0);
                i += 1;
            }
            continue;
        }

        if (std.mem.eql(u8, std.mem.sliceTo(arg, 0), "--top-k")) {
            if (i < args.len) {
                const val = std.mem.sliceTo(args[i], 0);
                top_k = std.fmt.parseInt(u32, val, 10) catch 40;
                i += 1;
            }
            continue;
        }

        if (std.mem.eql(u8, std.mem.sliceTo(arg, 0), "--tfs")) {
            if (i < args.len) {
                const val = std.mem.sliceTo(args[i], 0);
                tfs_z = std.fmt.parseFloat(f32, val) catch 1.0;
                i += 1;
            }
            continue;
        }

        if (std.mem.eql(u8, std.mem.sliceTo(arg, 0), "--mirostat")) {
            if (i < args.len) {
                const val = std.mem.sliceTo(args[i], 0);
                mirostat = std.fmt.parseInt(u8, val, 10) catch 0;
                i += 1;
            }
            continue;
        }

        if (std.mem.eql(u8, std.mem.sliceTo(arg, 0), "--mirostat-tau")) {
            if (i < args.len) {
                const val = std.mem.sliceTo(args[i], 0);
                mirostat_tau = std.fmt.parseFloat(f32, val) catch 5.0;
                i += 1;
            }
            continue;
        }

        if (std.mem.eql(u8, std.mem.sliceTo(arg, 0), "--mirostat-eta")) {
            if (i < args.len) {
                const val = std.mem.sliceTo(args[i], 0);
                mirostat_eta = std.fmt.parseFloat(f32, val) catch 0.1;
                i += 1;
            }
            continue;
        }

        // Positional: model path or prompt
        if (model_path == null) {
            model_path = std.mem.sliceTo(arg, 0);
        } else if (prompt == null) {
            prompt = std.mem.sliceTo(arg, 0);
        }
    }

    if (model_path == null) {
        std.debug.print("Error: Model path required\n", .{});
        std.debug.print("Usage: abi llm generate <model> --prompt <text>\n", .{});
        return;
    }

    if (prompt == null) {
        std.debug.print("Error: Prompt required\n", .{});
        std.debug.print("Usage: abi llm generate <model> --prompt <text>\n", .{});
        return;
    }

    std.debug.print("Loading model: {s}\n", .{model_path.?});
    std.debug.print("Prompt: {s}\n", .{prompt.?});
    std.debug.print("Max tokens: {d}, Temperature: {d:.2}, Top-p: {d:.2}, Top-k: {d}, Repeat penalty: {d:.2}\n", .{ max_tokens, temperature, top_p, top_k, repeat_penalty });
    if (seed) |s| {
        std.debug.print("Seed: {d}\n", .{s});
    }
    if (tfs_z < 1.0) {
        std.debug.print("Tail-free sampling: z={d:.2}\n", .{tfs_z});
    }
    if (mirostat > 0) {
        std.debug.print("Mirostat v{d}: tau={d:.2}, eta={d:.2}\n", .{ mirostat, mirostat_tau, mirostat_eta });
    }
    if (stream) {
        std.debug.print("Streaming: enabled\n", .{});
    }
    if (!allow_ollama_fallback) {
        std.debug.print("Ollama fallback: disabled\n", .{});
    } else if (ollama_model) |name| {
        std.debug.print("Ollama model override: {s}\n", .{name});
    }
    std.debug.print("\nGenerating...\n\n", .{});

    // Create inference engine
    var engine = abi.ai.llm.Engine.init(allocator, .{
        .max_new_tokens = max_tokens,
        .temperature = temperature,
        .top_p = top_p,
        .top_k = top_k,
        .repetition_penalty = repeat_penalty,
        .allow_ollama_fallback = allow_ollama_fallback,
        .ollama_model = ollama_model,
    });
    defer engine.deinit();

    // Load model
    engine.loadModel(model_path.?) catch |err| {
        std.debug.print("Error loading model: {t}\n", .{err});
        if (err == error.FileTooLarge) {
            printModelFileSizeHint(allocator, model_path.?);
            return;
        }
        if (err == error.UnsupportedArchitecture) {
            std.debug.print("\nThis GGUF architecture is not yet supported by ABI local inference.\n", .{});
            std.debug.print("Current local engine targets LLaMA-compatible transformer layouts.\n", .{});
            std.debug.print("Tip: remove `--no-ollama-fallback` to run this model via Ollama.\n", .{});
            printUnsupportedLayoutSummary(allocator, model_path.?);
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
    std.debug.print("\n", .{});

    // Generate
    const output = engine.generate(allocator, prompt.?) catch |err| {
        std.debug.print("Error during generation: {t}\n", .{err});
        return;
    };
    defer allocator.free(output);

    std.debug.print("{s}\n", .{output});

    // Print stats
    const stats = engine.getStats();
    std.debug.print("\n---\n", .{});
    std.debug.print("Stats: {d:.1} tok/s prefill, {d:.1} tok/s decode\n", .{
        stats.prefillTokensPerSecond(),
        stats.decodeTokensPerSecond(),
    });
}

fn printUnsupportedLayoutSummary(allocator: std.mem.Allocator, model_path: []const u8) void {
    var gguf_file = abi.ai.llm.io.GgufFile.open(allocator, model_path) catch return;
    defer gguf_file.deinit();

    const config = abi.ai.llm.model.LlamaConfig.fromGguf(&gguf_file);
    std.debug.print("Detected architecture: {s}\n", .{config.arch});
    std.debug.print("Detected dims: hidden={d}, q={d}, kv={d}, v={d}\n", .{
        config.dim,
        config.queryDim(),
        config.kvDim(),
        config.valueDim(),
    });
    std.debug.print("Detected heads: q={d}, kv={d} (head dims q/kv/v={d}/{d}/{d})\n", .{
        config.n_heads,
        config.n_kv_heads,
        config.queryHeadDim(),
        config.keyHeadDim(),
        config.valueHeadDim(),
    });
}

fn printModelFileSizeHint(allocator: std.mem.Allocator, model_path: []const u8) void {
    var io_backend = cli_io.initIoBackend(allocator);
    defer io_backend.deinit();
    const io = io_backend.io();

    var file = blk: {
        if (std.fs.path.isAbsolute(model_path)) {
            break :blk std.Io.Dir.openFileAbsolute(io, model_path, .{}) catch return;
        }
        break :blk std.Io.Dir.cwd().openFile(io, model_path, .{}) catch return;
    };
    defer file.close(io);

    const stat = file.stat(io) catch return;
    if (stat.size == 0) {
        std.debug.print("Model file is empty (0 bytes). Re-download or use the real Ollama blob path.\n", .{});
    }
}

fn runChat(allocator: std.mem.Allocator, args: []const [:0]const u8) !void {
    if (utils.args.containsHelpArgs(args)) {
        printHelp();
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

fn runBench(allocator: std.mem.Allocator, args: []const [:0]const u8) !void {
    if (utils.args.containsHelpArgs(args)) {
        printHelp();
        return;
    }

    var model_path: ?[]const u8 = null;
    var prompt_tokens: u32 = 128;
    var gen_tokens: u32 = 64;
    var runtime_runs: u32 = 3;
    var prompt_text: ?[]const u8 = null;
    var compare_ollama: bool = false;
    var ollama_model: ?[]const u8 = null;
    var compare_mlx: bool = false;
    var mlx_model: ?[]const u8 = null;
    var compare_vllm: bool = false;
    var vllm_model: ?[]const u8 = null;
    var compare_lmstudio: bool = false;
    var lmstudio_model: ?[]const u8 = null;
    var json_output: bool = false;
    var wdbx_out: ?[]const u8 = null;

    var i: usize = 0;
    while (i < args.len) {
        const arg = args[i];
        i += 1;

        if (std.mem.eql(u8, std.mem.sliceTo(arg, 0), "--prompt-tokens")) {
            if (i < args.len) {
                const val = std.mem.sliceTo(args[i], 0);
                prompt_tokens = std.fmt.parseInt(u32, val, 10) catch 128;
                i += 1;
            }
            continue;
        }

        if (std.mem.eql(u8, std.mem.sliceTo(arg, 0), "--gen-tokens")) {
            if (i < args.len) {
                const val = std.mem.sliceTo(args[i], 0);
                gen_tokens = std.fmt.parseInt(u32, val, 10) catch 64;
                i += 1;
            }
            continue;
        }

        if (std.mem.eql(u8, std.mem.sliceTo(arg, 0), "--runs")) {
            if (i < args.len) {
                const val = std.mem.sliceTo(args[i], 0);
                runtime_runs = std.fmt.parseInt(u32, val, 10) catch 3;
                i += 1;
            }
            continue;
        }

        if (std.mem.eql(u8, std.mem.sliceTo(arg, 0), "--prompt")) {
            if (i < args.len) {
                prompt_text = std.mem.sliceTo(args[i], 0);
                i += 1;
            }
            continue;
        }

        if (std.mem.eql(u8, std.mem.sliceTo(arg, 0), "--compare-ollama")) {
            compare_ollama = true;
            continue;
        }

        if (std.mem.eql(u8, std.mem.sliceTo(arg, 0), "--ollama-model")) {
            if (i < args.len) {
                ollama_model = std.mem.sliceTo(args[i], 0);
                i += 1;
            }
            continue;
        }

        if (std.mem.eql(u8, std.mem.sliceTo(arg, 0), "--compare-mlx")) {
            compare_mlx = true;
            continue;
        }

        if (std.mem.eql(u8, std.mem.sliceTo(arg, 0), "--mlx-model")) {
            if (i < args.len) {
                mlx_model = std.mem.sliceTo(args[i], 0);
                i += 1;
            }
            continue;
        }

        if (std.mem.eql(u8, std.mem.sliceTo(arg, 0), "--compare-vllm")) {
            compare_vllm = true;
            continue;
        }

        if (std.mem.eql(u8, std.mem.sliceTo(arg, 0), "--vllm-model")) {
            if (i < args.len) {
                vllm_model = std.mem.sliceTo(args[i], 0);
                i += 1;
            }
            continue;
        }

        if (std.mem.eql(u8, std.mem.sliceTo(arg, 0), "--compare-lmstudio")) {
            compare_lmstudio = true;
            continue;
        }

        if (std.mem.eql(u8, std.mem.sliceTo(arg, 0), "--lmstudio-model")) {
            if (i < args.len) {
                lmstudio_model = std.mem.sliceTo(args[i], 0);
                i += 1;
            }
            continue;
        }

        if (std.mem.eql(u8, std.mem.sliceTo(arg, 0), "--compare-all")) {
            compare_ollama = true;
            compare_mlx = true;
            compare_vllm = true;
            compare_lmstudio = true;
            continue;
        }

        if (std.mem.eql(u8, std.mem.sliceTo(arg, 0), "--json")) {
            json_output = true;
            continue;
        }

        if (std.mem.eql(u8, std.mem.sliceTo(arg, 0), "--wdbx-out")) {
            if (i < args.len) {
                wdbx_out = std.mem.sliceTo(args[i], 0);
                i += 1;
            }
            continue;
        }

        if (model_path == null) {
            model_path = std.mem.sliceTo(arg, 0);
        }
    }

    if (model_path == null and !compare_ollama and !compare_mlx and !compare_vllm and !compare_lmstudio) {
        std.debug.print("Usage: abi llm bench <model> [--prompt-tokens N] [--gen-tokens N] [--runs N] [--compare-ollama] [--compare-mlx] [--compare-vllm] [--compare-lmstudio] [--compare-all] [--json]\n", .{});
        return;
    }

    runtime_runs = @max(runtime_runs, 1);

    const bench_prompt = if (prompt_text) |p| p else try buildBenchmarkPrompt(allocator, prompt_tokens);
    defer if (prompt_text == null) allocator.free(@constCast(bench_prompt));

    std.debug.print("LLM Benchmark\n", .{});
    std.debug.print("=============\n", .{});
    if (model_path) |path| {
        std.debug.print("Local model: {s}\n", .{path});
    }
    std.debug.print("Prompt: {s}\n", .{bench_prompt});
    std.debug.print("Prompt tokens: {d}\n", .{prompt_tokens});
    std.debug.print("Generation tokens: {d}\n", .{gen_tokens});
    std.debug.print("Runtime runs: {d}\n", .{runtime_runs});
    if (compare_ollama) {
        std.debug.print("Compare with Ollama: enabled\n", .{});
    }
    if (ollama_model) |name| {
        std.debug.print("Ollama model override: {s}\n", .{name});
    }
    if (compare_mlx) {
        std.debug.print("Compare with MLX: enabled\n", .{});
    }
    if (mlx_model) |name| {
        std.debug.print("MLX model override: {s}\n", .{name});
    }
    if (compare_vllm) {
        std.debug.print("Compare with vLLM: enabled\n", .{});
    }
    if (vllm_model) |name| {
        std.debug.print("vLLM model override: {s}\n", .{name});
    }
    if (compare_lmstudio) {
        std.debug.print("Compare with LM Studio: enabled\n", .{});
    }
    if (lmstudio_model) |name| {
        std.debug.print("LM Studio model override: {s}\n", .{name});
    }
    if (json_output) {
        std.debug.print("Output: JSON\n", .{});
    }
    if (wdbx_out) |path| {
        std.debug.print("WDBX output: {s}\n", .{path});
    }
    std.debug.print("\n", .{});

    // Synthetic compute benchmark (backend-agnostic)
    const bench_result = runComputeBenchmark(allocator, prompt_tokens, gen_tokens);

    std.debug.print("Synthetic Compute Benchmark\n", .{});
    std.debug.print("---------------------------\n", .{});
    std.debug.print("  Dimensions: {d}x{d} @ {d}x{d}\n", .{ bench_result.m, bench_result.k, bench_result.k, bench_result.n });
    std.debug.print("  Time: {d:.2} ms\n", .{bench_result.time_ms});
    std.debug.print("  GFLOPS: {d:.2}\n", .{bench_result.gflops});
    std.debug.print("Estimated local throughput from matmul:\n", .{});
    std.debug.print("  Prefill: ~{d:.0} tokens/sec\n", .{bench_result.est_prefill_tok_s});
    std.debug.print("  Decode:  ~{d:.0} tokens/sec\n", .{bench_result.est_decode_tok_s});

    var local_runtime: ?RuntimeBenchResult = null;
    if (model_path) |path| {
        local_runtime = runLocalRuntimeBenchmark(allocator, path, bench_prompt, gen_tokens, runtime_runs) catch |err| blk: {
            std.debug.print("\nLocal runtime benchmark unavailable: {t}\n", .{err});
            break :blk null;
        };
    }
    defer if (local_runtime) |*res| res.deinit(allocator);

    var ollama_runtime: ?OllamaRuntimeBenchResult = null;
    if (compare_ollama) {
        ollama_runtime = runOllamaRuntimeBenchmark(allocator, bench_prompt, gen_tokens, ollama_model, runtime_runs) catch |err| blk: {
            std.debug.print("\nOllama benchmark unavailable: {t}\n", .{err});
            break :blk null;
        };
    }
    defer if (ollama_runtime) |*res| res.deinit(allocator);

    var mlx_runtime: ?MlxRuntimeBenchResult = null;
    if (compare_mlx) {
        mlx_runtime = runMlxRuntimeBenchmark(allocator, bench_prompt, gen_tokens, mlx_model, runtime_runs) catch |err| blk: {
            std.debug.print("\nMLX benchmark unavailable: {t}\n", .{err});
            break :blk null;
        };
    }
    defer if (mlx_runtime) |*res| res.deinit(allocator);

    var vllm_runtime: ?VllmRuntimeBenchResult = null;
    if (compare_vllm) {
        vllm_runtime = runVllmRuntimeBenchmark(allocator, bench_prompt, gen_tokens, vllm_model, runtime_runs) catch |err| blk: {
            std.debug.print("\nvLLM benchmark unavailable: {t}\n", .{err});
            break :blk null;
        };
    }
    defer if (vllm_runtime) |*res| res.deinit(allocator);

    var lmstudio_runtime: ?LmStudioRuntimeBenchResult = null;
    if (compare_lmstudio) {
        lmstudio_runtime = runLmStudioRuntimeBenchmark(allocator, bench_prompt, gen_tokens, lmstudio_model, runtime_runs) catch |err| blk: {
            std.debug.print("\nLM Studio benchmark unavailable: {t}\n", .{err});
            break :blk null;
        };
    }
    defer if (lmstudio_runtime) |*res| res.deinit(allocator);

    if (local_runtime) |local| {
        std.debug.print("\nLocal Runtime Benchmark\n", .{});
        std.debug.print("-----------------------\n", .{});
        std.debug.print("  Backend: {s}\n", .{local.backend.label()});
        std.debug.print("  Runs: {d}\n", .{local.runs.len});
        std.debug.print("  Prompt tokens (mean): {d}\n", .{local.summary.prompt_tokens_mean});
        std.debug.print("  Generated tokens (mean): {d}\n", .{local.summary.generated_tokens_mean});
        printRuntimeStats("Wall time ms", local.summary.elapsed_ms);
        printRuntimeStats("Prefill tok/s", local.summary.prefill_tok_s);
        printRuntimeStats("Decode tok/s", local.summary.decode_tok_s);
    }

    if (ollama_runtime) |res| {
        std.debug.print("\nOllama Runtime Benchmark\n", .{});
        std.debug.print("------------------------\n", .{});
        std.debug.print("  Backend: ollama\n", .{});
        std.debug.print("  Model: {s}\n", .{res.model_name});
        std.debug.print("  Runs: {d}\n", .{res.runs.len});
        std.debug.print("  Prompt tokens (mean): {d}\n", .{res.summary.prompt_tokens_mean});
        std.debug.print("  Generated tokens (mean): {d}\n", .{res.summary.generated_tokens_mean});
        printRuntimeStats("Wall time ms", res.summary.elapsed_ms);
        printRuntimeStats("Prefill tok/s", res.summary.prefill_tok_s);
        printRuntimeStats("Decode tok/s", res.summary.decode_tok_s);
    }

    if (mlx_runtime) |res| {
        std.debug.print("\nMLX Runtime Benchmark\n", .{});
        std.debug.print("---------------------\n", .{});
        std.debug.print("  Backend: mlx\n", .{});
        std.debug.print("  Model: {s}\n", .{res.model_name});
        std.debug.print("  Runs: {d}\n", .{res.runs.len});
        std.debug.print("  Prompt tokens (mean): {d}\n", .{res.summary.prompt_tokens_mean});
        std.debug.print("  Generated tokens (mean): {d}\n", .{res.summary.generated_tokens_mean});
        printRuntimeStats("Wall time ms", res.summary.elapsed_ms);
        printRuntimeStats("Prefill tok/s", res.summary.prefill_tok_s);
        printRuntimeStats("Decode tok/s", res.summary.decode_tok_s);
    }

    if (vllm_runtime) |res| {
        std.debug.print("\nvLLM Runtime Benchmark\n", .{});
        std.debug.print("----------------------\n", .{});
        std.debug.print("  Backend: vllm\n", .{});
        std.debug.print("  Model: {s}\n", .{res.model_name});
        std.debug.print("  Runs: {d}\n", .{res.runs.len});
        std.debug.print("  Prompt tokens (mean): {d}\n", .{res.summary.prompt_tokens_mean});
        std.debug.print("  Generated tokens (mean): {d}\n", .{res.summary.generated_tokens_mean});
        printRuntimeStats("Wall time ms", res.summary.elapsed_ms);
        printRuntimeStats("Decode tok/s", res.summary.decode_tok_s);
    }

    if (lmstudio_runtime) |res| {
        std.debug.print("\nLM Studio Runtime Benchmark\n", .{});
        std.debug.print("---------------------------\n", .{});
        std.debug.print("  Backend: lm-studio\n", .{});
        std.debug.print("  Model: {s}\n", .{res.model_name});
        std.debug.print("  Runs: {d}\n", .{res.runs.len});
        std.debug.print("  Prompt tokens (mean): {d}\n", .{res.summary.prompt_tokens_mean});
        std.debug.print("  Generated tokens (mean): {d}\n", .{res.summary.generated_tokens_mean});
        printRuntimeStats("Wall time ms", res.summary.elapsed_ms);
        printRuntimeStats("Decode tok/s", res.summary.decode_tok_s);
    }

    // Comparison summary table
    {
        const BackendEntry = struct {
            name: []const u8,
            status: []const u8,
            decode_mean: f64,
            decode_p50: f64,
            prefill_mean: f64,
            wall_mean: f64,
        };

        var entries: [5]BackendEntry = undefined;
        var entry_count: usize = 0;

        if (local_runtime) |local| {
            entries[entry_count] = .{
                .name = "local-gguf",
                .status = "OK",
                .decode_mean = local.summary.decode_tok_s.mean,
                .decode_p50 = local.summary.decode_tok_s.p50,
                .prefill_mean = local.summary.prefill_tok_s.mean,
                .wall_mean = local.summary.elapsed_ms.mean,
            };
            entry_count += 1;
        } else if (model_path != null) {
            entries[entry_count] = .{ .name = "local-gguf", .status = "FAIL", .decode_mean = 0, .decode_p50 = 0, .prefill_mean = 0, .wall_mean = 0 };
            entry_count += 1;
        }
        if (compare_ollama) {
            if (ollama_runtime) |res| {
                entries[entry_count] = .{
                    .name = "ollama",
                    .status = "OK",
                    .decode_mean = res.summary.decode_tok_s.mean,
                    .decode_p50 = res.summary.decode_tok_s.p50,
                    .prefill_mean = res.summary.prefill_tok_s.mean,
                    .wall_mean = res.summary.elapsed_ms.mean,
                };
            } else {
                entries[entry_count] = .{ .name = "ollama", .status = "SKIP", .decode_mean = 0, .decode_p50 = 0, .prefill_mean = 0, .wall_mean = 0 };
            }
            entry_count += 1;
        }
        if (compare_mlx) {
            if (mlx_runtime) |res| {
                entries[entry_count] = .{
                    .name = "mlx",
                    .status = "OK",
                    .decode_mean = res.summary.decode_tok_s.mean,
                    .decode_p50 = res.summary.decode_tok_s.p50,
                    .prefill_mean = res.summary.prefill_tok_s.mean,
                    .wall_mean = res.summary.elapsed_ms.mean,
                };
            } else {
                entries[entry_count] = .{ .name = "mlx", .status = "SKIP", .decode_mean = 0, .decode_p50 = 0, .prefill_mean = 0, .wall_mean = 0 };
            }
            entry_count += 1;
        }
        if (compare_vllm) {
            if (vllm_runtime) |res| {
                entries[entry_count] = .{
                    .name = "vllm",
                    .status = "OK",
                    .decode_mean = res.summary.decode_tok_s.mean,
                    .decode_p50 = res.summary.decode_tok_s.p50,
                    .prefill_mean = res.summary.prefill_tok_s.mean,
                    .wall_mean = res.summary.elapsed_ms.mean,
                };
            } else {
                entries[entry_count] = .{ .name = "vllm", .status = "SKIP", .decode_mean = 0, .decode_p50 = 0, .prefill_mean = 0, .wall_mean = 0 };
            }
            entry_count += 1;
        }
        if (compare_lmstudio) {
            if (lmstudio_runtime) |res| {
                entries[entry_count] = .{
                    .name = "lm-studio",
                    .status = "OK",
                    .decode_mean = res.summary.decode_tok_s.mean,
                    .decode_p50 = res.summary.decode_tok_s.p50,
                    .prefill_mean = res.summary.prefill_tok_s.mean,
                    .wall_mean = res.summary.elapsed_ms.mean,
                };
            } else {
                entries[entry_count] = .{ .name = "lm-studio", .status = "SKIP", .decode_mean = 0, .decode_p50 = 0, .prefill_mean = 0, .wall_mean = 0 };
            }
            entry_count += 1;
        }

        if (entry_count >= 2) {
            std.debug.print("\nBackend Comparison\n", .{});
            std.debug.print("==================\n", .{});
            std.debug.print("  {s:<14} {s:>14} {s:>14} {s:>12} {s:>6}\n", .{ "Backend", "Decode tok/s", "Prefill tok/s", "Wall ms", "Status" });
            std.debug.print("  {s:<14} {s:>14} {s:>14} {s:>12} {s:>6}\n", .{ "-" ** 14, "-" ** 14, "-" ** 14, "-" ** 12, "-" ** 6 });
            for (entries[0..entry_count]) |e| {
                if (std.mem.eql(u8, e.status, "OK")) {
                    if (e.prefill_mean > 0) {
                        std.debug.print("  {s:<14} {d:>7.1} ({d:.1}) {d:>14.1} {d:>12.1} {s:>6}\n", .{ e.name, e.decode_mean, e.decode_p50, e.prefill_mean, e.wall_mean, e.status });
                    } else {
                        std.debug.print("  {s:<14} {d:>7.1} ({d:.1}) {s:>14} {d:>12.1} {s:>6}\n", .{ e.name, e.decode_mean, e.decode_p50, "N/A", e.wall_mean, e.status });
                    }
                } else {
                    std.debug.print("  {s:<14} {s:>14} {s:>14} {s:>12} {s:>6}\n", .{ e.name, "-", "-", "-", e.status });
                }
            }

            // Pairwise decode speed ratios
            std.debug.print("\n  Decode Speed Ratios (mean):\n", .{});
            for (entries[0..entry_count], 0..) |a, ai| {
                if (a.decode_mean <= 0) continue;
                for (entries[0..entry_count], 0..) |b, bi| {
                    if (bi <= ai) continue;
                    if (b.decode_mean <= 0) continue;
                    const ratio = b.decode_mean / a.decode_mean;
                    std.debug.print("    {s} / {s}: {d:.2}x\n", .{ b.name, a.name, ratio });
                }
            }
        }
    }

    // JSON export
    if (json_output) {
        printBenchJson(
            allocator,
            prompt_tokens,
            gen_tokens,
            runtime_runs,
            bench_result,
            local_runtime,
            ollama_runtime,
            mlx_runtime,
            vllm_runtime,
            lmstudio_runtime,
        );
    }

    if (wdbx_out) |path| {
        appendBenchRecordToWdbx(
            allocator,
            path,
            bench_prompt,
            prompt_tokens,
            gen_tokens,
            runtime_runs,
            bench_result,
            local_runtime,
            ollama_runtime,
        ) catch |err| {
            std.debug.print("\nWDBX write failed: {t}\n", .{err});
        };
    }

    if (!json_output) {
        std.debug.print("\nNote: Local runtime requires ABI-native support for the GGUF architecture.\n", .{});
        std.debug.print("When unsupported, use Ollama fallback for execution and compare decode throughput.\n", .{});
    }
}

const BenchResult = struct {
    m: u32,
    k: u32,
    n: u32,
    time_ms: f64,
    gflops: f64,
    est_prefill_tok_s: f64,
    est_decode_tok_s: f64,
};

const RuntimeBenchSample = struct {
    prompt_tokens: u32,
    generated_tokens: u32,
    elapsed_ms: f64,
    prefill_tok_s: f64,
    decode_tok_s: f64,
};

const RuntimeStatsSummary = struct {
    mean: f64 = 0.0,
    p50: f64 = 0.0,
    p90: f64 = 0.0,
    p95: f64 = 0.0,
};

const RuntimeSampleSummary = struct {
    prompt_tokens_mean: u32 = 0,
    generated_tokens_mean: u32 = 0,
    elapsed_ms: RuntimeStatsSummary = .{},
    prefill_tok_s: RuntimeStatsSummary = .{},
    decode_tok_s: RuntimeStatsSummary = .{},
};

const RuntimeBenchResult = struct {
    backend: abi.ai.llm.EngineBackend,
    runs: []RuntimeBenchSample,
    summary: RuntimeSampleSummary,

    pub fn deinit(self: *RuntimeBenchResult, allocator: std.mem.Allocator) void {
        allocator.free(self.runs);
        self.* = undefined;
    }
};

const OllamaRuntimeBenchResult = struct {
    model_name: []u8,
    runs: []RuntimeBenchSample,
    summary: RuntimeSampleSummary,

    pub fn deinit(self: *OllamaRuntimeBenchResult, allocator: std.mem.Allocator) void {
        allocator.free(self.model_name);
        allocator.free(self.runs);
        self.* = undefined;
    }
};

fn printRuntimeStats(label: []const u8, stats: RuntimeStatsSummary) void {
    std.debug.print("  {s}: mean={d:.2}, p50={d:.2}, p90={d:.2}, p95={d:.2}\n", .{
        label,
        stats.mean,
        stats.p50,
        stats.p90,
        stats.p95,
    });
}

fn summarizeRuntimeSamples(allocator: std.mem.Allocator, runs: []const RuntimeBenchSample) !RuntimeSampleSummary {
    if (runs.len == 0) return .{};

    const elapsed_values = try allocator.alloc(f64, runs.len);
    defer allocator.free(elapsed_values);

    const prefill_values = try allocator.alloc(f64, runs.len);
    defer allocator.free(prefill_values);

    const decode_values = try allocator.alloc(f64, runs.len);
    defer allocator.free(decode_values);

    var prompt_total: u64 = 0;
    var generated_total: u64 = 0;
    for (runs, 0..) |sample, idx| {
        elapsed_values[idx] = sample.elapsed_ms;
        prefill_values[idx] = sample.prefill_tok_s;
        decode_values[idx] = sample.decode_tok_s;
        prompt_total += sample.prompt_tokens;
        generated_total += sample.generated_tokens;
    }

    const run_count_u64: u64 = @intCast(runs.len);

    return .{
        .prompt_tokens_mean = @intCast(prompt_total / run_count_u64),
        .generated_tokens_mean = @intCast(generated_total / run_count_u64),
        .elapsed_ms = try summarizeF64(allocator, elapsed_values),
        .prefill_tok_s = try summarizeF64(allocator, prefill_values),
        .decode_tok_s = try summarizeF64(allocator, decode_values),
    };
}

fn summarizeF64(allocator: std.mem.Allocator, values: []const f64) !RuntimeStatsSummary {
    if (values.len == 0) return .{};

    const sorted = try allocator.dupe(f64, values);
    defer allocator.free(sorted);

    std.mem.sort(f64, sorted, {}, std.sort.asc(f64));

    var total: f64 = 0.0;
    for (values) |v| total += v;
    const count = @as(f64, @floatFromInt(values.len));

    return .{
        .mean = total / count,
        .p50 = percentileFromSorted(sorted, 0.50),
        .p90 = percentileFromSorted(sorted, 0.90),
        .p95 = percentileFromSorted(sorted, 0.95),
    };
}

fn percentileFromSorted(sorted: []const f64, quantile: f64) f64 {
    if (sorted.len == 0) return 0.0;

    const q = std.math.clamp(quantile, 0.0, 1.0);
    const last_index = sorted.len - 1;
    const position = q * @as(f64, @floatFromInt(last_index));
    const lower_idx: usize = @intFromFloat(@floor(position));
    const upper_idx: usize = @intFromFloat(@ceil(position));

    if (upper_idx >= sorted.len or lower_idx == upper_idx) {
        return sorted[lower_idx];
    }

    const fraction = position - @as(f64, @floatFromInt(lower_idx));
    return sorted[lower_idx] + (sorted[upper_idx] - sorted[lower_idx]) * fraction;
}

fn runComputeBenchmark(allocator: std.mem.Allocator, prompt_tokens: u32, gen_tokens: u32) BenchResult {
    // Typical 7B model dimensions: hidden_size=4096, intermediate_size=11008
    const m: u32 = @max(prompt_tokens, 1);
    const k: u32 = 4096; // hidden_size
    const n: u32 = 4096; // hidden_size (for attention projection)

    // Allocate test matrices
    const a = allocator.alloc(f32, @as(usize, m) * k) catch {
        return BenchResult{
            .m = m,
            .k = k,
            .n = n,
            .time_ms = 0,
            .gflops = 0,
            .est_prefill_tok_s = 0,
            .est_decode_tok_s = 0,
        };
    };
    defer allocator.free(a);

    const b = allocator.alloc(f32, @as(usize, k) * n) catch {
        return BenchResult{
            .m = m,
            .k = k,
            .n = n,
            .time_ms = 0,
            .gflops = 0,
            .est_prefill_tok_s = 0,
            .est_decode_tok_s = 0,
        };
    };
    defer allocator.free(b);

    const c = allocator.alloc(f32, @as(usize, m) * n) catch {
        return BenchResult{
            .m = m,
            .k = k,
            .n = n,
            .time_ms = 0,
            .gflops = 0,
            .est_prefill_tok_s = 0,
            .est_decode_tok_s = 0,
        };
    };
    defer allocator.free(c);

    // Initialize with random-ish values
    for (a, 0..) |*v, idx| {
        v.* = @as(f32, @floatFromInt(idx % 100)) / 100.0 - 0.5;
    }
    for (b, 0..) |*v, idx| {
        v.* = @as(f32, @floatFromInt(idx % 100)) / 100.0 - 0.5;
    }

    // Warmup
    abi.ai.llm.ops.matrixMultiply(a, b, c, m, k, n);

    // Benchmark
    const iterations: u32 = 5;
    var timer = abi.shared.time.Timer.start() catch {
        return BenchResult{
            .m = m,
            .k = k,
            .n = n,
            .time_ms = 0,
            .gflops = 0,
            .est_prefill_tok_s = 0,
            .est_decode_tok_s = 0,
        };
    };

    for (0..iterations) |_| {
        abi.ai.llm.ops.matrixMultiply(a, b, c, m, k, n);
    }

    const elapsed_ns = timer.read();
    const time_ms = @as(f64, @floatFromInt(elapsed_ns)) / (1_000_000.0 * @as(f64, @floatFromInt(iterations)));

    // Calculate GFLOPS: 2*M*N*K operations per matmul
    const flops: f64 = 2.0 * @as(f64, @floatFromInt(m)) * @as(f64, @floatFromInt(n)) * @as(f64, @floatFromInt(k));
    const gflops = (flops / (time_ms / 1000.0)) / 1_000_000_000.0;

    // Estimate tokens/sec based on typical 7B model compute
    // A 7B model has ~32 transformer layers, each with multiple matmuls
    // Prefill processes all tokens in parallel, decode processes 1 token at a time
    const layers: f64 = 32.0;
    const matmuls_per_layer: f64 = 4.0; // Q,K,V projection + output projection
    const total_flops_per_token = flops * layers * matmuls_per_layer;

    const est_prefill_tok_s = (gflops * 1_000_000_000.0) / total_flops_per_token * @as(f64, @floatFromInt(prompt_tokens));
    const est_decode_tok_s = (gflops * 1_000_000_000.0) / (total_flops_per_token / @as(f64, @floatFromInt(gen_tokens)));

    return BenchResult{
        .m = m,
        .k = k,
        .n = n,
        .time_ms = time_ms,
        .gflops = gflops,
        .est_prefill_tok_s = @min(est_prefill_tok_s, 10000.0),
        .est_decode_tok_s = @min(est_decode_tok_s, 100.0),
    };
}

fn runLocalRuntimeBenchmark(
    allocator: std.mem.Allocator,
    model_path: []const u8,
    prompt: []const u8,
    gen_tokens: u32,
    runs: u32,
) !RuntimeBenchResult {
    const run_count: usize = @intCast(@max(runs, @as(u32, 1)));
    const samples = try allocator.alloc(RuntimeBenchSample, run_count);
    errdefer allocator.free(samples);

    var engine = abi.ai.llm.Engine.init(allocator, .{
        .max_new_tokens = gen_tokens,
        .allow_ollama_fallback = false,
    });
    defer engine.deinit();

    try engine.loadModel(model_path);
    if (engine.getBackend() != .local_gguf) return error.UnsupportedArchitecture;

    for (samples) |*sample| {
        var timer = try abi.shared.time.Timer.start();
        const output = try engine.generate(allocator, prompt);
        defer allocator.free(output);
        const elapsed_ms = @as(f64, @floatFromInt(timer.read())) / 1_000_000.0;

        const stats = engine.getStats();
        const prompt_count = if (stats.prompt_tokens > 0) stats.prompt_tokens else estimateTokenCount(prompt);
        const generated_count = if (stats.generated_tokens > 0) stats.generated_tokens else estimateTokenCount(output);

        var decode_tok_s = stats.decodeTokensPerSecond();
        if (decode_tok_s <= 0 and elapsed_ms > 0) {
            decode_tok_s = @as(f64, @floatFromInt(generated_count)) / (elapsed_ms / 1000.0);
        }

        sample.* = .{
            .prompt_tokens = prompt_count,
            .generated_tokens = generated_count,
            .elapsed_ms = elapsed_ms,
            .prefill_tok_s = stats.prefillTokensPerSecond(),
            .decode_tok_s = decode_tok_s,
        };
    }

    const summary = try summarizeRuntimeSamples(allocator, samples);
    return .{
        .backend = engine.getBackend(),
        .runs = samples,
        .summary = summary,
    };
}

fn runOllamaRuntimeBenchmark(
    allocator: std.mem.Allocator,
    prompt: []const u8,
    gen_tokens: u32,
    model_override: ?[]const u8,
    runs: u32,
) !OllamaRuntimeBenchResult {
    const run_count: usize = @intCast(@max(runs, @as(u32, 1)));
    const samples = try allocator.alloc(RuntimeBenchSample, run_count);
    errdefer allocator.free(samples);

    var client = try abi.connectors.ollama.createClient(allocator);
    defer client.deinit();

    if (model_override) |model_name| {
        if (model_name.len > 0) {
            if (client.config.model_owned) {
                allocator.free(@constCast(client.config.model));
            }
            client.config.model = try allocator.dupe(u8, model_name);
            client.config.model_owned = true;
        }
    }

    const model_name_copy = try allocator.dupe(u8, client.config.model);
    errdefer allocator.free(model_name_copy);

    for (samples) |*sample| {
        var timer = try abi.shared.time.Timer.start();
        var res = try client.generate(.{
            .model = client.config.model,
            .prompt = prompt,
            .stream = false,
            .options = .{
                .temperature = 0.7,
                .num_predict = gen_tokens,
                .top_p = 0.9,
                .top_k = 40,
            },
        });
        defer res.deinit(allocator);

        const elapsed_ms = @as(f64, @floatFromInt(timer.read())) / 1_000_000.0;
        const prompt_count = res.prompt_eval_count orelse estimateTokenCount(prompt);
        const generated_count = res.eval_count orelse estimateTokenCount(res.response);

        const prefill_tok_s = blk: {
            if (res.prompt_eval_count) |count| {
                if (res.prompt_eval_duration_ns) |dur| {
                    if (dur > 0) {
                        break :blk @as(f64, @floatFromInt(count)) / (@as(f64, @floatFromInt(dur)) / 1_000_000_000.0);
                    }
                }
            }
            break :blk 0.0;
        };

        var decode_tok_s = blk: {
            if (res.eval_count) |count| {
                if (res.eval_duration_ns) |dur| {
                    if (dur > 0) {
                        break :blk @as(f64, @floatFromInt(count)) / (@as(f64, @floatFromInt(dur)) / 1_000_000_000.0);
                    }
                }
            }
            break :blk 0.0;
        };
        if (decode_tok_s <= 0 and elapsed_ms > 0) {
            decode_tok_s = @as(f64, @floatFromInt(generated_count)) / (elapsed_ms / 1000.0);
        }

        sample.* = .{
            .prompt_tokens = prompt_count,
            .generated_tokens = generated_count,
            .elapsed_ms = elapsed_ms,
            .prefill_tok_s = prefill_tok_s,
            .decode_tok_s = decode_tok_s,
        };
    }

    const summary = try summarizeRuntimeSamples(allocator, samples);
    return .{
        .model_name = model_name_copy,
        .runs = samples,
        .summary = summary,
    };
}

const MlxRuntimeBenchResult = struct {
    model_name: []u8,
    runs: []RuntimeBenchSample,
    summary: RuntimeSampleSummary,

    pub fn deinit(self: *MlxRuntimeBenchResult, allocator: std.mem.Allocator) void {
        allocator.free(self.model_name);
        allocator.free(self.runs);
        self.* = undefined;
    }
};

const VllmRuntimeBenchResult = struct {
    model_name: []u8,
    runs: []RuntimeBenchSample,
    summary: RuntimeSampleSummary,

    pub fn deinit(self: *VllmRuntimeBenchResult, allocator: std.mem.Allocator) void {
        allocator.free(self.model_name);
        allocator.free(self.runs);
        self.* = undefined;
    }
};

const LmStudioRuntimeBenchResult = struct {
    model_name: []u8,
    runs: []RuntimeBenchSample,
    summary: RuntimeSampleSummary,

    pub fn deinit(self: *LmStudioRuntimeBenchResult, allocator: std.mem.Allocator) void {
        allocator.free(self.model_name);
        allocator.free(self.runs);
        self.* = undefined;
    }
};

fn runMlxRuntimeBenchmark(
    allocator: std.mem.Allocator,
    prompt: []const u8,
    gen_tokens: u32,
    model_override: ?[]const u8,
    runs: u32,
) !MlxRuntimeBenchResult {
    const run_count: usize = @intCast(@max(runs, @as(u32, 1)));
    const samples = try allocator.alloc(RuntimeBenchSample, run_count);
    errdefer allocator.free(samples);

    var client = try abi.connectors.mlx.createClient(allocator);
    defer client.deinit();

    if (model_override) |model_name| {
        if (model_name.len > 0) {
            if (client.config.model_owned) {
                allocator.free(@constCast(client.config.model));
            }
            client.config.model = try allocator.dupe(u8, model_name);
            client.config.model_owned = true;
        }
    }

    const model_name_copy = try allocator.dupe(u8, client.config.model);
    errdefer allocator.free(model_name_copy);

    for (samples) |*sample| {
        var timer = try abi.shared.time.Timer.start();

        // MLX uses OpenAI-compatible chat completions; use generate() for simplicity
        const response_text = try client.generate(prompt, gen_tokens);
        defer allocator.free(response_text);

        const elapsed_ms = @as(f64, @floatFromInt(timer.read())) / 1_000_000.0;

        // MLX server returns usage in the chat response but generate() consumes it.
        // Estimate token counts from text length as fallback.
        const prompt_count = estimateTokenCount(prompt);
        const generated_count = estimateTokenCount(response_text);

        // Derive throughput from wall time since MLX chat API doesn't expose
        // per-phase timing like Ollama's eval_duration_ns fields.
        const decode_tok_s = if (elapsed_ms > 0)
            @as(f64, @floatFromInt(generated_count)) / (elapsed_ms / 1000.0)
        else
            0.0;

        sample.* = .{
            .prompt_tokens = prompt_count,
            .generated_tokens = generated_count,
            .elapsed_ms = elapsed_ms,
            .prefill_tok_s = 0.0, // Not available from chat completions API
            .decode_tok_s = decode_tok_s,
        };
    }

    const summary = try summarizeRuntimeSamples(allocator, samples);
    return .{
        .model_name = model_name_copy,
        .runs = samples,
        .summary = summary,
    };
}

fn printBenchJson(
    _: std.mem.Allocator,
    prompt_tokens: u32,
    gen_tokens: u32,
    runs: u32,
    compute: BenchResult,
    local_runtime: ?RuntimeBenchResult,
    ollama_runtime: ?OllamaRuntimeBenchResult,
    mlx_runtime: ?MlxRuntimeBenchResult,
    vllm_runtime: ?VllmRuntimeBenchResult,
    lmstudio_runtime: ?LmStudioRuntimeBenchResult,
) void {
    std.debug.print("{{\n", .{});
    std.debug.print("  \"prompt_tokens\": {d},\n", .{prompt_tokens});
    std.debug.print("  \"gen_tokens\": {d},\n", .{gen_tokens});
    std.debug.print("  \"runs\": {d},\n", .{runs});
    std.debug.print("  \"compute\": {{ \"gflops\": {d:.2}, \"est_prefill_tok_s\": {d:.1}, \"est_decode_tok_s\": {d:.1} }},\n", .{
        compute.gflops,
        compute.est_prefill_tok_s,
        compute.est_decode_tok_s,
    });
    std.debug.print("  \"backends\": {{\n", .{});

    var printed_any = false;

    if (local_runtime) |local| {
        if (printed_any) std.debug.print(",\n", .{});
        std.debug.print("    \"local_gguf\": {{ \"status\": \"ok\", \"decode_tok_s\": {{ \"mean\": {d:.2}, \"p50\": {d:.2}, \"p90\": {d:.2} }}, \"prefill_tok_s\": {{ \"mean\": {d:.2} }}, \"wall_ms\": {{ \"mean\": {d:.2} }} }}", .{
            local.summary.decode_tok_s.mean,
            local.summary.decode_tok_s.p50,
            local.summary.decode_tok_s.p90,
            local.summary.prefill_tok_s.mean,
            local.summary.elapsed_ms.mean,
        });
        printed_any = true;
    }
    if (ollama_runtime) |res| {
        if (printed_any) std.debug.print(",\n", .{});
        std.debug.print("    \"ollama\": {{ \"status\": \"ok\", \"model\": \"{s}\", \"decode_tok_s\": {{ \"mean\": {d:.2}, \"p50\": {d:.2}, \"p90\": {d:.2} }}, \"prefill_tok_s\": {{ \"mean\": {d:.2} }}, \"wall_ms\": {{ \"mean\": {d:.2} }} }}", .{
            res.model_name,
            res.summary.decode_tok_s.mean,
            res.summary.decode_tok_s.p50,
            res.summary.decode_tok_s.p90,
            res.summary.prefill_tok_s.mean,
            res.summary.elapsed_ms.mean,
        });
        printed_any = true;
    }
    if (mlx_runtime) |res| {
        if (printed_any) std.debug.print(",\n", .{});
        std.debug.print("    \"mlx\": {{ \"status\": \"ok\", \"model\": \"{s}\", \"decode_tok_s\": {{ \"mean\": {d:.2}, \"p50\": {d:.2}, \"p90\": {d:.2} }}, \"wall_ms\": {{ \"mean\": {d:.2} }} }}", .{
            res.model_name,
            res.summary.decode_tok_s.mean,
            res.summary.decode_tok_s.p50,
            res.summary.decode_tok_s.p90,
            res.summary.elapsed_ms.mean,
        });
        printed_any = true;
    }
    if (vllm_runtime) |res| {
        if (printed_any) std.debug.print(",\n", .{});
        std.debug.print("    \"vllm\": {{ \"status\": \"ok\", \"model\": \"{s}\", \"decode_tok_s\": {{ \"mean\": {d:.2}, \"p50\": {d:.2}, \"p90\": {d:.2} }}, \"wall_ms\": {{ \"mean\": {d:.2} }} }}", .{
            res.model_name,
            res.summary.decode_tok_s.mean,
            res.summary.decode_tok_s.p50,
            res.summary.decode_tok_s.p90,
            res.summary.elapsed_ms.mean,
        });
        printed_any = true;
    }
    if (lmstudio_runtime) |res| {
        if (printed_any) std.debug.print(",\n", .{});
        std.debug.print("    \"lm_studio\": {{ \"status\": \"ok\", \"model\": \"{s}\", \"decode_tok_s\": {{ \"mean\": {d:.2}, \"p50\": {d:.2}, \"p90\": {d:.2} }}, \"wall_ms\": {{ \"mean\": {d:.2} }} }}", .{
            res.model_name,
            res.summary.decode_tok_s.mean,
            res.summary.decode_tok_s.p50,
            res.summary.decode_tok_s.p90,
            res.summary.elapsed_ms.mean,
        });
        printed_any = true;
    }

    std.debug.print("\n  }}\n", .{});
    std.debug.print("}}\n", .{});
}

fn runVllmRuntimeBenchmark(
    allocator: std.mem.Allocator,
    prompt: []const u8,
    gen_tokens: u32,
    model_override: ?[]const u8,
    runs: u32,
) !VllmRuntimeBenchResult {
    const run_count: usize = @intCast(@max(runs, @as(u32, 1)));
    const samples = try allocator.alloc(RuntimeBenchSample, run_count);
    errdefer allocator.free(samples);

    var client = try abi.connectors.vllm.createClient(allocator);
    defer client.deinit();

    if (model_override) |model_name| {
        if (model_name.len > 0) {
            if (client.config.model_owned) {
                allocator.free(@constCast(client.config.model));
            }
            client.config.model = try allocator.dupe(u8, model_name);
            client.config.model_owned = true;
        }
    }

    const model_name_copy = try allocator.dupe(u8, client.config.model);
    errdefer allocator.free(model_name_copy);

    for (samples) |*sample| {
        var timer = try abi.shared.time.Timer.start();

        var res = try client.chatCompletion(.{
            .model = client.config.model,
            .messages = &.{
                .{ .role = "user", .content = prompt },
            },
            .max_tokens = gen_tokens,
            .temperature = 0.7,
        });

        const elapsed_ms = @as(f64, @floatFromInt(timer.read())) / 1_000_000.0;

        const prompt_count = res.usage.prompt_tokens;
        const generated_count = res.usage.completion_tokens;

        // Free response strings
        allocator.free(res.id);
        allocator.free(res.model);
        for (res.choices) |choice| {
            allocator.free(choice.message.role);
            allocator.free(choice.message.content);
            allocator.free(choice.finish_reason);
        }
        allocator.free(res.choices);

        const decode_tok_s = if (elapsed_ms > 0)
            @as(f64, @floatFromInt(generated_count)) / (elapsed_ms / 1000.0)
        else
            0.0;

        sample.* = .{
            .prompt_tokens = prompt_count,
            .generated_tokens = generated_count,
            .elapsed_ms = elapsed_ms,
            .prefill_tok_s = 0.0,
            .decode_tok_s = decode_tok_s,
        };
    }

    const summary = try summarizeRuntimeSamples(allocator, samples);
    return .{
        .model_name = model_name_copy,
        .runs = samples,
        .summary = summary,
    };
}

fn runLmStudioRuntimeBenchmark(
    allocator: std.mem.Allocator,
    prompt: []const u8,
    gen_tokens: u32,
    model_override: ?[]const u8,
    runs: u32,
) !LmStudioRuntimeBenchResult {
    const run_count: usize = @intCast(@max(runs, @as(u32, 1)));
    const samples = try allocator.alloc(RuntimeBenchSample, run_count);
    errdefer allocator.free(samples);

    var client = try abi.connectors.lm_studio.createClient(allocator);
    defer client.deinit();

    if (model_override) |model_name| {
        if (model_name.len > 0) {
            if (client.config.model_owned) {
                allocator.free(@constCast(client.config.model));
            }
            client.config.model = try allocator.dupe(u8, model_name);
            client.config.model_owned = true;
        }
    }

    const model_name_copy = try allocator.dupe(u8, client.config.model);
    errdefer allocator.free(model_name_copy);

    for (samples) |*sample| {
        var timer = try abi.shared.time.Timer.start();

        var res = try client.chatCompletion(.{
            .model = client.config.model,
            .messages = &.{
                .{ .role = "user", .content = prompt },
            },
            .max_tokens = gen_tokens,
            .temperature = 0.7,
        });

        const elapsed_ms = @as(f64, @floatFromInt(timer.read())) / 1_000_000.0;

        const prompt_count = res.usage.prompt_tokens;
        const generated_count = res.usage.completion_tokens;

        // Free response strings
        allocator.free(res.id);
        allocator.free(res.model);
        for (res.choices) |choice| {
            allocator.free(choice.message.role);
            allocator.free(choice.message.content);
            allocator.free(choice.finish_reason);
        }
        allocator.free(res.choices);

        const decode_tok_s = if (elapsed_ms > 0)
            @as(f64, @floatFromInt(generated_count)) / (elapsed_ms / 1000.0)
        else
            0.0;

        sample.* = .{
            .prompt_tokens = prompt_count,
            .generated_tokens = generated_count,
            .elapsed_ms = elapsed_ms,
            .prefill_tok_s = 0.0,
            .decode_tok_s = decode_tok_s,
        };
    }

    const summary = try summarizeRuntimeSamples(allocator, samples);
    return .{
        .model_name = model_name_copy,
        .runs = samples,
        .summary = summary,
    };
}

fn buildBenchmarkPrompt(allocator: std.mem.Allocator, prompt_tokens: u32) ![]u8 {
    var out = std.ArrayListUnmanaged(u8).empty;
    errdefer out.deinit(allocator);

    const words = @max(prompt_tokens, 16);
    for (0..words) |idx| {
        if (idx != 0) try out.append(allocator, ' ');
        const token = switch (idx % 8) {
            0 => "analyze",
            1 => "latency",
            2 => "throughput",
            3 => "architecture",
            4 => "token",
            5 => "benchmark",
            6 => "inference",
            else => "runtime",
        };
        try out.appendSlice(allocator, token);
    }

    return out.toOwnedSlice(allocator);
}

fn estimateTokenCount(text: []const u8) u32 {
    if (text.len == 0) return 0;

    var count: u32 = 0;
    var in_word = false;
    for (text) |c| {
        const is_space = std.ascii.isWhitespace(c);
        if (is_space) {
            in_word = false;
            continue;
        }
        if (!in_word) {
            count += 1;
            in_word = true;
        }
    }
    return count;
}

const RuntimeRunRecordJson = struct {
    wall_ms: f64,
    prefill_tps: f64,
    decode_tps: f64,
    prompt_tokens: u32,
    generated_tokens: u32,
};

const RuntimeSummaryJson = struct {
    backend: []const u8,
    runs: usize,
    prompt_tokens_mean: u32,
    generated_tokens_mean: u32,
    wall_ms: RuntimeStatsSummary,
    prefill_tps: RuntimeStatsSummary,
    decode_tps: RuntimeStatsSummary,
    per_run: []const RuntimeRunRecordJson,
};

const OllamaRuntimeSummaryJson = struct {
    backend: []const u8,
    model: []const u8,
    runs: usize,
    prompt_tokens_mean: u32,
    generated_tokens_mean: u32,
    wall_ms: RuntimeStatsSummary,
    prefill_tps: RuntimeStatsSummary,
    decode_tps: RuntimeStatsSummary,
    per_run: []const RuntimeRunRecordJson,
};

fn mapRuntimeRunsForJson(allocator: std.mem.Allocator, runs: []const RuntimeBenchSample) ![]RuntimeRunRecordJson {
    const mapped = try allocator.alloc(RuntimeRunRecordJson, runs.len);
    for (runs, 0..) |sample, idx| {
        mapped[idx] = .{
            .wall_ms = sample.elapsed_ms,
            .prefill_tps = sample.prefill_tok_s,
            .decode_tps = sample.decode_tok_s,
            .prompt_tokens = sample.prompt_tokens,
            .generated_tokens = sample.generated_tokens,
        };
    }
    return mapped;
}

fn appendBenchRecordToWdbx(
    allocator: std.mem.Allocator,
    output_path: []const u8,
    prompt: []const u8,
    prompt_tokens_target: u32,
    gen_tokens_target: u32,
    runtime_runs: u32,
    compute: BenchResult,
    local_runtime: ?RuntimeBenchResult,
    ollama_runtime: ?OllamaRuntimeBenchResult,
) !void {
    var handle = try abi.database.wdbx.createDatabaseWithConfig(allocator, output_path, .{
        .cache_norms = false,
        .initial_capacity = 0,
        .use_vector_pool = false,
        .thread_safe = false,
    });
    defer abi.database.wdbx.closeDatabase(&handle);

    abi.database.wdbx.restore(&handle, output_path) catch {};

    const now_ms = abi.shared.time.unixMs();
    const stats = abi.database.wdbx.getStats(&handle);
    const record_id: u64 = stats.count + 1;

    var local_runs_json: []RuntimeRunRecordJson = &.{};
    defer if (local_runs_json.len > 0) allocator.free(local_runs_json);
    if (local_runtime) |local| {
        local_runs_json = try mapRuntimeRunsForJson(allocator, local.runs);
    }

    var ollama_runs_json: []RuntimeRunRecordJson = &.{};
    defer if (ollama_runs_json.len > 0) allocator.free(ollama_runs_json);
    if (ollama_runtime) |ollama| {
        ollama_runs_json = try mapRuntimeRunsForJson(allocator, ollama.runs);
    }

    const local_payload: ?RuntimeSummaryJson = if (local_runtime) |local| .{
        .backend = local.backend.label(),
        .runs = local.runs.len,
        .prompt_tokens_mean = local.summary.prompt_tokens_mean,
        .generated_tokens_mean = local.summary.generated_tokens_mean,
        .wall_ms = local.summary.elapsed_ms,
        .prefill_tps = local.summary.prefill_tok_s,
        .decode_tps = local.summary.decode_tok_s,
        .per_run = local_runs_json,
    } else null;

    const ollama_payload: ?OllamaRuntimeSummaryJson = if (ollama_runtime) |ollama| .{
        .backend = "ollama",
        .model = ollama.model_name,
        .runs = ollama.runs.len,
        .prompt_tokens_mean = ollama.summary.prompt_tokens_mean,
        .generated_tokens_mean = ollama.summary.generated_tokens_mean,
        .wall_ms = ollama.summary.elapsed_ms,
        .prefill_tps = ollama.summary.prefill_tok_s,
        .decode_tps = ollama.summary.decode_tok_s,
        .per_run = ollama_runs_json,
    } else null;

    const comparison_payload: ?struct {
        ollama_over_local_decode_mean: f64,
        ollama_over_local_decode_p50: f64,
    } = blk: {
        if (local_runtime) |local| {
            if (ollama_runtime) |ollama| {
                if (local.summary.decode_tok_s.mean > 0 and local.summary.decode_tok_s.p50 > 0) {
                    break :blk .{
                        .ollama_over_local_decode_mean = ollama.summary.decode_tok_s.mean / local.summary.decode_tok_s.mean,
                        .ollama_over_local_decode_p50 = ollama.summary.decode_tok_s.p50 / local.summary.decode_tok_s.p50,
                    };
                }
            }
        }
        break :blk null;
    };

    const payload = struct {
        schema_version: []const u8,
        ts_ms: i64,
        config: struct {
            prompt: []const u8,
            prompt_tokens_target: u32,
            gen_tokens_target: u32,
            runs: u32,
        },
        compute: struct {
            m: u32,
            k: u32,
            n: u32,
            time_ms: f64,
            gflops: f64,
            est_prefill_tok_s: f64,
            est_decode_tok_s: f64,
        },
        local: ?RuntimeSummaryJson,
        ollama: ?OllamaRuntimeSummaryJson,
        comparison: @TypeOf(comparison_payload),
    }{
        .schema_version = "abi.llm.bench.v2",
        .ts_ms = now_ms,
        .config = .{
            .prompt = prompt,
            .prompt_tokens_target = prompt_tokens_target,
            .gen_tokens_target = gen_tokens_target,
            .runs = runtime_runs,
        },
        .compute = .{
            .m = compute.m,
            .k = compute.k,
            .n = compute.n,
            .time_ms = compute.time_ms,
            .gflops = compute.gflops,
            .est_prefill_tok_s = compute.est_prefill_tok_s,
            .est_decode_tok_s = compute.est_decode_tok_s,
        },
        .local = local_payload,
        .ollama = ollama_payload,
        .comparison = comparison_payload,
    };

    var metadata_writer: std.Io.Writer.Allocating = .init(allocator);
    defer metadata_writer.deinit();
    try std.json.Stringify.value(payload, .{}, &metadata_writer.writer);
    const metadata = try metadata_writer.toOwnedSlice();
    defer allocator.free(metadata);

    try abi.database.wdbx.insertVector(&handle, record_id, &[_]f32{}, metadata);
    try abi.database.wdbx.backup(&handle, output_path);
    std.debug.print("\nWDBX benchmark record appended: {s}\n", .{output_path});
}

fn runList() void {
    std.debug.print("Supported Model Formats\n", .{});
    std.debug.print("=======================\n\n", .{});
    std.debug.print("Recommended Default Model:\n", .{});
    std.debug.print("  GPT-2 (124M parameters) - Open source, no authentication required\n", .{});
    std.debug.print("  Download: https://huggingface.co/TheBloke/gpt2-GGUF\n", .{});
    std.debug.print("\n", .{});
    std.debug.print("GGUF (llama.cpp format)\n", .{});
    std.debug.print("  - GPT-2 (124M, 355M, 774M, 1.5B) - Recommended for local training\n", .{});
    std.debug.print("  - LLaMA 2 (7B, 13B, 70B)\n", .{});
    std.debug.print("  - Mistral (7B)\n", .{});
    std.debug.print("  - Mixtral (8x7B MoE)\n", .{});
    std.debug.print("  - Phi-2, Phi-3\n", .{});
    std.debug.print("  - Qwen, Yi\n", .{});
    std.debug.print("\n", .{});
    std.debug.print("Quantization Types:\n", .{});
    std.debug.print("  - F32, F16, BF16 (full precision)\n", .{});
    std.debug.print("  - Q8_0 (8-bit quantization)\n", .{});
    std.debug.print("  - Q4_0, Q4_1 (4-bit quantization)\n", .{});
    std.debug.print("  - Q5_0, Q5_1 (5-bit quantization)\n", .{});
    std.debug.print("  - K-quants (Q2_K, Q3_K, Q4_K, Q5_K, Q6_K)\n", .{});
    std.debug.print("\n", .{});
    std.debug.print("Where to download:\n", .{});
    std.debug.print("  https://huggingface.co/TheBloke\n", .{});
    std.debug.print("  https://huggingface.co/models?other=gguf\n", .{});
}

fn runListLocal(allocator: std.mem.Allocator, args: []const [:0]const u8) void {
    if (utils.args.containsHelpArgs(args)) {
        printHelp();
        return;
    }

    var search_dir: []const u8 = ".";

    // Parse directory argument
    if (args.len > 0) {
        search_dir = std.mem.sliceTo(args[0], 0);
    }

    std.debug.print("Searching for models in: {s}\n\n", .{search_dir});

    // List .gguf files
    var io_backend = cli_io.initIoBackend(allocator);
    defer io_backend.deinit();
    const io = io_backend.io();

    var dir = std.Io.Dir.cwd().openDir(io, search_dir, .{ .iterate = true }) catch |err| {
        std.debug.print("Error: Cannot open directory {s}: {t}\n", .{ search_dir, err });
        return;
    };
    defer dir.close(io);

    var count: u32 = 0;
    var iter = dir.iterate();
    while (true) {
        const maybe_entry = iter.next(io) catch break;
        if (maybe_entry) |entry| {
            if (entry.kind == .file and std.mem.endsWith(u8, entry.name, ".gguf")) {
                std.debug.print("  {s}\n", .{entry.name});
                count += 1;
            }
        } else break;
    }

    if (count == 0) {
        std.debug.print("  No GGUF models found.\n", .{});
        std.debug.print("\nDownload models from:\n", .{});
        std.debug.print("  https://huggingface.co/TheBloke\n", .{});
        std.debug.print("  https://huggingface.co/models?other=gguf\n", .{});
    } else {
        std.debug.print("\nFound {d} model(s).\n", .{count});
    }
}

fn runDownload(allocator: std.mem.Allocator, args: []const [:0]const u8) !void {
    if (utils.args.containsHelpArgs(args)) {
        printHelp();
        return;
    }

    if (args.len == 0) {
        std.debug.print("Usage: abi llm download <url> [--output <path>]\n\n", .{});
        std.debug.print("Download a GGUF model from a URL.\n\n", .{});
        std.debug.print("Examples:\n", .{});
        std.debug.print("  abi llm download https://example.com/model.gguf\n", .{});
        std.debug.print("  abi llm download https://huggingface.co/TheBloke/Llama-2-7B-GGUF/resolve/main/llama-2-7b.Q4_K_M.gguf\n\n", .{});
        std.debug.print("Note: For HuggingFace models, use the 'resolve/main/' URL format.\n", .{});
        return;
    }

    var url: ?[]const u8 = null;
    var output_path: ?[]const u8 = null;

    var i: usize = 0;
    while (i < args.len) {
        const arg = args[i];
        i += 1;

        if (std.mem.eql(u8, std.mem.sliceTo(arg, 0), "--output") or std.mem.eql(u8, std.mem.sliceTo(arg, 0), "-o")) {
            if (i < args.len) {
                output_path = std.mem.sliceTo(args[i], 0);
                i += 1;
            }
            continue;
        }

        if (url == null) {
            url = std.mem.sliceTo(arg, 0);
        }
    }

    if (url == null) {
        std.debug.print("Error: URL required\n", .{});
        return;
    }

    // Extract filename from URL if no output path specified
    const final_path = output_path orelse blk: {
        // Find last '/' in URL
        if (std.mem.lastIndexOf(u8, url.?, "/")) |idx| {
            break :blk url.?[idx + 1 ..];
        }
        break :blk "model.gguf";
    };

    std.debug.print("Downloading: {s}\n", .{url.?});
    std.debug.print("Output: {s}\n\n", .{final_path});

    // Initialize I/O backend for HTTP download
    var io_backend = cli_io.initIoBackend(allocator);
    defer io_backend.deinit();
    const io = io_backend.io();

    // Initialize downloader
    var downloader = abi.ai.models.Downloader.init(allocator);
    defer downloader.deinit();

    const ProgressState = struct {
        var last_percent: u8 = 255;
    };

    const progress_callback = struct {
        fn callback(progress: abi.ai.models.DownloadProgress) void {
            if (progress.percent == ProgressState.last_percent) return;
            ProgressState.last_percent = progress.percent;

            const downloaded_mb = @as(f64, @floatFromInt(progress.downloaded_bytes)) / (1024 * 1024);
            const total_mb = @as(f64, @floatFromInt(progress.total_bytes)) / (1024 * 1024);
            const speed_mb = @as(f64, @floatFromInt(progress.speed_bytes_per_sec)) / (1024 * 1024);

            if (progress.total_bytes > 0) {
                std.debug.print("\r{d}% ({d:.1}/{d:.1} MB) {d:.1} MB/s", .{
                    progress.percent,
                    downloaded_mb,
                    total_mb,
                    speed_mb,
                });
            } else {
                std.debug.print("\r{d:.1} MB {d:.1} MB/s", .{
                    downloaded_mb,
                    speed_mb,
                });
            }

            if (progress.percent >= 100) {
                std.debug.print("\n", .{});
            }
        }
    }.callback;

    const result = downloader.downloadWithIo(io, url.?, .{
        .output_path = final_path,
        .progress_callback = progress_callback,
        .resume_download = true,
    });

    if (result) |download_result| {
        defer allocator.free(download_result.path);
        const size_mb = @as(f64, @floatFromInt(download_result.bytes_downloaded)) / (1024 * 1024);
        std.debug.print("Download complete: {s}\n", .{download_result.path});
        std.debug.print("Size: {d:.2} MB\n", .{size_mb});
        std.debug.print("SHA256: {s}\n", .{&download_result.checksum});
        if (download_result.was_resumed) {
            std.debug.print("Note: Download resumed from partial file.\n", .{});
        }
        if (download_result.checksum_verified) {
            std.debug.print("Checksum verified.\n", .{});
        }
    } else |err| {
        std.debug.print("\nDownload failed: {t}\n\n", .{err});
        std.debug.print("Manual download options:\n", .{});
        std.debug.print("  curl -L -o {s} \"{s}\"\n", .{ final_path, url.? });
        std.debug.print("  wget -O {s} \"{s}\"\n", .{ final_path, url.? });
    }
}

fn printHelp() void {
    const help_text =
        "Usage: abi llm <command> [options]\n\n" ++
        "Run local LLM inference with GGUF models (llama.cpp compatible).\n\n" ++
        "Commands:\n" ++
        "  info <model>       Show model information\n" ++
        "  generate <model>   Generate text from a prompt\n" ++
        "  chat <model>       Interactive chat mode\n" ++
        "  bench <model>      Benchmark model performance\n" ++
        "  serve              Start streaming inference HTTP server\n" ++
        "  list               List supported models and formats\n" ++
        "  list-local [dir]   List GGUF models in directory\n" ++
        "  download <url>     Download a model from URL\n" ++
        "  help               Show this help message\n\n" ++
        "Generate options:\n" ++
        "  -m, --model <path>      Path to GGUF model file\n" ++
        "  -p, --prompt <text>     Text prompt for generation\n" ++
        "  -n, --max-tokens <n>    Maximum tokens to generate (default: 256)\n" ++
        "  -t, --temperature <f>   Temperature for sampling (default: 0.7)\n" ++
        "  --top-p <f>             Top-p nucleus sampling (default: 0.9)\n" ++
        "  --top-k <n>             Top-k filtering (default: 40)\n" ++
        "  --repeat-penalty <f>    Repetition penalty (default: 1.1)\n" ++
        "  --seed <n>              Random seed for reproducibility\n" ++
        "  --stop <text>           Stop sequence (can specify multiple)\n" ++
        "  --stream                Enable streaming output\n" ++
        "  --ollama-model <name>   Override Ollama model for fallback execution\n" ++
        "  --no-ollama-fallback    Disable Ollama fallback for unsupported GGUF\n\n" ++
        "Advanced sampling (llama.cpp parity):\n" ++
        "  --tfs <f>               Tail-free sampling parameter (default: 1.0 = disabled)\n" ++
        "  --mirostat <n>          Mirostat mode (0=off, 1=v1, 2=v2, default: 0)\n" ++
        "  --mirostat-tau <f>      Mirostat target entropy (default: 5.0)\n" ++
        "  --mirostat-eta <f>      Mirostat learning rate (default: 0.1)\n\n" ++
        "Benchmark options:\n" ++
        "  --prompt-tokens <n>     Number of prompt tokens (default: 128)\n" ++
        "  --gen-tokens <n>        Number of tokens to generate (default: 64)\n" ++
        "  --runs <n>              Number of runtime runs for percentile stats (default: 3)\n" ++
        "  --prompt <text>         Prompt text for runtime benchmark\n" ++
        "  --compare-ollama        Run Ollama benchmark and compare decode speed\n" ++
        "  --ollama-model <name>   Override Ollama model for compare run\n" ++
        "  --compare-mlx           Run MLX benchmark and compare decode speed\n" ++
        "  --mlx-model <name>      Override MLX model for compare run\n" ++
        "  --compare-vllm          Run vLLM benchmark and compare decode speed\n" ++
        "  --vllm-model <name>     Override vLLM model for compare run\n" ++
        "  --compare-lmstudio      Run LM Studio benchmark and compare decode speed\n" ++
        "  --lmstudio-model <name> Override LM Studio model for compare run\n" ++
        "  --compare-all           Compare all available backends\n" ++
        "  --json                  Output results in JSON format\n" ++
        "  --wdbx-out <path>       Append benchmark record to WDBX database\n\n" ++
        "Serve options:\n" ++
        "  -m, --model <path>      Path to GGUF model file (required for local inference)\n" ++
        "  -a, --address <addr>    Listen address (default: 127.0.0.1:8080)\n" ++
        "  --auth-token <token>    Bearer token for authentication (optional)\n" ++
        "  --preload               Pre-load model on startup (reduces first-request latency)\n\n" ++
        "Examples:\n" ++
        "  abi llm info ./llama-7b.gguf\n" ++
        "  abi llm generate ./llama-7b.gguf -p \"Hello, how are you?\"\n" ++
        "  abi llm generate ./mistral-7b.gguf -p \"Write a poem\" -n 100 -t 0.8 --stream\n" ++
        "  abi llm generate ./model.gguf -p \"Complete this:\" --seed 42 --repeat-penalty 1.2\n" ++
        "  abi llm generate ./model.gguf -p \"Story:\" --mirostat 2 --mirostat-tau 3.0\n" ++
        "  abi llm generate ./model.gguf -p \"Code:\" --tfs 0.95 --top-k 50\n" ++
        "  abi llm generate ./gpt-oss-20b.gguf -p \"Hi\" --ollama-model gpt-oss\n" ++
        "  abi llm bench ./llama-7b.gguf --gen-tokens 128 --runs 5 --compare-ollama\n" ++
        "  abi llm bench --compare-ollama --compare-mlx --runs 5 --gen-tokens 128\n" ++
        "  abi llm bench --compare-all --runs 5 --json\n" ++
        "  abi llm bench --compare-vllm --compare-lmstudio --runs 3\n" ++
        "  abi llm bench ./llama-7b.gguf --compare-ollama --runs 7 --wdbx-out ./bench.wdbx\n" ++
        "  abi llm serve -m ./llama-7b.gguf --preload\n" ++
        "  abi llm serve -m ./model.gguf -a 0.0.0.0:8000 --auth-token secret\n" ++
        "  abi llm list-local ./models\n" ++
        "  abi llm download https://huggingface.co/.../model.gguf -o my-model.gguf\n";
    std.debug.print("{s}", .{help_text});
}

fn runDemo(allocator: std.mem.Allocator, args: []const [:0]const u8) !void {
    if (utils.args.containsHelpArgs(args)) {
        printHelp();
        return;
    }

    _ = allocator; // Not used in demo mode
    var prompt: ?[]const u8 = null;
    var max_tokens: u32 = 100;

    // Parse arguments
    var i: usize = 0;
    while (i < args.len) {
        const arg = args[i];
        i += 1;

        if (utils.args.matchesAny(arg, &[_][]const u8{ "--prompt", "-p" })) {
            if (i < args.len) {
                prompt = std.mem.sliceTo(args[i], 0);
                i += 1;
            }
            continue;
        }

        if (utils.args.matchesAny(arg, &[_][]const u8{ "--max-tokens", "-n" })) {
            if (i < args.len) {
                const val = std.mem.sliceTo(args[i], 0);
                max_tokens = std.fmt.parseInt(u32, val, 10) catch 100;
                i += 1;
            }
            continue;
        }

        // Positional: prompt
        if (prompt == null) {
            prompt = std.mem.sliceTo(arg, 0);
        }
    }

    if (prompt == null) {
        prompt = "Hello, can you tell me about the ABI framework?";
    }

    std.debug.print("🤖 ABI LLM Demo Mode\n", .{});
    std.debug.print("==================\n", .{});
    std.debug.print("Prompt: {s}\n", .{prompt.?});
    std.debug.print("Max tokens: {d}\n\n", .{max_tokens});

    std.debug.print("Generating response...\n\n", .{});

    // Simulate generation with a demo response
    const response =
        "Hello! I'm the ABI framework's demo LLM assistant. While I don't have a real language model loaded right now, I can still help you understand how the system works!\n\n" ++
        "The ABI framework is designed for modular AI services, GPU compute, and vector databases. It supports:\n\n" ++
        "• Multiple LLM formats (GGUF, PyTorch, etc.)\n" ++
        "• GPU acceleration (CUDA, Vulkan, Metal)\n" ++
        "• Vector databases with HNSW indexing\n" ++
        "• Distributed computing with Raft consensus\n" ++
        "• Real-time observability and monitoring\n\n" ++
        "To use real models, download GGUF files from https://huggingface.co/TheBloke\n" ++
        "For example: abi llm download https://huggingface.co/.../gpt2.gguf\n\n" ++
        "This demo shows the interface works correctly - the actual model loading happens when you provide a real GGUF file path.";

    const truncated_response = if (response.len > max_tokens * 4) response[0 .. max_tokens * 4] else response;

    std.debug.print("{s}\n", .{truncated_response});

    std.debug.print("\n---\n", .{});
    std.debug.print("Demo Stats: 25.0 tok/s prefill, 15.0 tok/s decode\n", .{});
    std.debug.print("💡 Tip: Use 'abi llm list' to see supported model formats\n", .{});
}

/// Start the streaming inference HTTP server
///
/// Provides OpenAI-compatible endpoints for LLM inference:
/// - POST /v1/chat/completions (streaming with SSE)
/// - GET /health
/// - GET /v1/models
fn runServe(allocator: std.mem.Allocator, args: []const [:0]const u8) !void {
    if (utils.args.containsHelpArgs(args)) {
        printServeHelp();
        return;
    }

    // Parse arguments
    var model_path: ?[]const u8 = null;
    var address: []const u8 = "127.0.0.1:8080";
    var auth_token: ?[]const u8 = null;
    var preload: bool = false;

    var i: usize = 0;
    while (i < args.len) {
        const arg = std.mem.sliceTo(args[i], 0);
        i += 1;

        if (utils.args.matchesAny(arg, &[_][]const u8{ "--model", "-m" })) {
            if (i < args.len) {
                model_path = std.mem.sliceTo(args[i], 0);
                i += 1;
            }
            continue;
        }

        if (utils.args.matchesAny(arg, &[_][]const u8{ "--address", "-a" })) {
            if (i < args.len) {
                address = std.mem.sliceTo(args[i], 0);
                i += 1;
            }
            continue;
        }

        if (std.mem.eql(u8, arg, "--auth-token")) {
            if (i < args.len) {
                auth_token = std.mem.sliceTo(args[i], 0);
                i += 1;
            }
            continue;
        }

        if (std.mem.eql(u8, arg, "--preload")) {
            preload = true;
            continue;
        }

        // First positional argument is model path
        if (model_path == null) {
            model_path = arg;
        }
    }

    // Create server configuration
    const server_config = abi.ai.streaming.ServerConfig{
        .address = address,
        .auth_token = auth_token,
        .default_model_path = model_path,
        .preload_model = preload,
        .default_backend = .local,
        .enable_openai_compat = true,
        .enable_websocket = true,
    };

    // Print startup banner
    std.debug.print("\n", .{});
    std.debug.print("╔═══════════════════════════════════════════════════════════╗\n", .{});
    std.debug.print("║          ABI Streaming Inference Server                   ║\n", .{});
    std.debug.print("╚═══════════════════════════════════════════════════════════╝\n", .{});
    std.debug.print("\n", .{});

    if (model_path) |path| {
        std.debug.print("  Model: {s}\n", .{path});
        if (preload) {
            std.debug.print("  Mode:  Pre-loading model on startup...\n", .{});
        } else {
            std.debug.print("  Mode:  Lazy loading (model loads on first request)\n", .{});
        }
    } else {
        std.debug.print("  Model: None configured\n", .{});
        std.debug.print("  Note:  Use -m <path> to specify a GGUF model\n", .{});
    }

    std.debug.print("  Address: {s}\n", .{address});
    if (auth_token != null) {
        std.debug.print("  Auth: Bearer token required\n", .{});
    } else {
        std.debug.print("  Auth: Disabled (open access)\n", .{});
    }
    std.debug.print("\n", .{});

    // Initialize server
    var server = abi.ai.streaming.StreamingServer.init(allocator, server_config) catch |err| {
        std.debug.print("Failed to initialize server: {t}\n", .{err});
        return err;
    };
    defer server.deinit();

    // Print endpoints
    std.debug.print("Endpoints:\n", .{});
    std.debug.print("  POST /v1/chat/completions  - OpenAI-compatible chat (stream=true for SSE)\n", .{});
    std.debug.print("  POST /api/stream           - ABI streaming endpoint\n", .{});
    std.debug.print("  GET  /api/stream/ws        - WebSocket streaming\n", .{});
    std.debug.print("  GET  /v1/models            - List available models\n", .{});
    std.debug.print("  GET  /health               - Health check\n", .{});
    std.debug.print("\n", .{});

    // Print usage example
    std.debug.print("Test with:\n", .{});
    if (auth_token) |token| {
        std.debug.print("  curl -X POST http://{s}/v1/chat/completions \\\n", .{address});
        std.debug.print("    -H \"Content-Type: application/json\" \\\n", .{});
        std.debug.print("    -H \"Authorization: Bearer {s}\" \\\n", .{token});
        std.debug.print("    -d '{{\"messages\":[{{\"role\":\"user\",\"content\":\"Hello\"}}],\"stream\":true}}'\n", .{});
    } else {
        std.debug.print("  curl -X POST http://{s}/v1/chat/completions \\\n", .{address});
        std.debug.print("    -H \"Content-Type: application/json\" \\\n", .{});
        std.debug.print("    -d '{{\"messages\":[{{\"role\":\"user\",\"content\":\"Hello\"}}],\"stream\":true}}'\n", .{});
    }
    std.debug.print("\n", .{});
    std.debug.print("Press Ctrl+C to stop the server.\n", .{});
    std.debug.print("─────────────────────────────────────────────────────────────\n", .{});

    // Start serving (blocking)
    server.serve() catch |err| {
        std.debug.print("Server error: {t}\n", .{err});
        return err;
    };
}

fn printServeHelp() void {
    const help_text =
        "Usage: abi llm serve [options]\n\n" ++
        "Start an HTTP server for streaming LLM inference.\n\n" ++
        "The server provides OpenAI-compatible endpoints, allowing you to use\n" ++
        "standard OpenAI SDKs and clients with local GGUF models.\n\n" ++
        "Options:\n" ++
        "  -m, --model <path>      Path to GGUF model file\n" ++
        "  -a, --address <addr>    Listen address (default: 127.0.0.1:8080)\n" ++
        "  --auth-token <token>    Bearer token for authentication (optional)\n" ++
        "  --preload               Pre-load model on startup\n" ++
        "  -h, --help              Show this help message\n\n" ++
        "Endpoints:\n" ++
        "  POST /v1/chat/completions  OpenAI-compatible chat completions\n" ++
        "  POST /api/stream           Custom ABI streaming endpoint\n" ++
        "  GET  /api/stream/ws        WebSocket upgrade for streaming\n" ++
        "  GET  /v1/models            List available models\n" ++
        "  GET  /health               Health check (no auth required)\n\n" ++
        "Examples:\n" ++
        "  abi llm serve -m ./llama-7b.gguf\n" ++
        "  abi llm serve -m ./model.gguf -a 0.0.0.0:8000 --preload\n" ++
        "  abi llm serve -m ./model.gguf --auth-token my-secret-token\n\n" ++
        "Testing:\n" ++
        "  # Health check\n" ++
        "  curl http://127.0.0.1:8080/health\n\n" ++
        "  # Streaming chat completion\n" ++
        "  curl -N http://127.0.0.1:8080/v1/chat/completions \\\n" ++
        "    -H \"Content-Type: application/json\" \\\n" ++
        "    -d '{\"messages\":[{\"role\":\"user\",\"content\":\"Hello\"}],\"stream\":true}'\n";
    std.debug.print("{s}", .{help_text});
}
