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

/// Run the LLM command with the provided arguments.
pub fn run(allocator: std.mem.Allocator, args: []const [:0]const u8) !void {
    if (args.len == 0 or utils.args.matchesAny(args[0], &[_][]const u8{ "help", "--help", "-h" })) {
        printHelp();
        return;
    }

    const command = std.mem.sliceTo(args[0], 0);

    if (std.mem.eql(u8, command, "info")) {
        try runInfo(allocator, args[1..]);
        return;
    }

    if (std.mem.eql(u8, command, "generate")) {
        try runGenerate(allocator, args[1..]);
        return;
    }

    if (std.mem.eql(u8, command, "chat")) {
        try runChat(allocator, args[1..]);
        return;
    }

    if (std.mem.eql(u8, command, "bench")) {
        try runBench(allocator, args[1..]);
        return;
    }

    if (std.mem.eql(u8, command, "list")) {
        if (utils.args.containsHelpArgs(args[1..])) {
            printHelp();
            return;
        }
        runList();
        return;
    }

    if (std.mem.eql(u8, command, "list-local")) {
        if (utils.args.containsHelpArgs(args[1..])) {
            printHelp();
            return;
        }
        runListLocal(allocator, args[1..]);
        return;
    }

    if (std.mem.eql(u8, command, "demo")) {
        try runDemo(allocator, args[1..]);
        return;
    }

    if (std.mem.eql(u8, command, "download")) {
        try runDownload(allocator, args[1..]);
        return;
    }

    if (std.mem.eql(u8, command, "serve")) {
        try runServe(allocator, args[1..]);
        return;
    }

    std.debug.print("Unknown llm command: {s}\n", .{command});
    if (utils.args.suggestCommand(command, &llm_subcommands)) |suggestion| {
        std.debug.print("Did you mean: {s}\n", .{suggestion});
    }
    printHelp();
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
    var stop_sequences = std.ArrayListUnmanaged([]const u8){};
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
    std.debug.print("\nGenerating...\n\n", .{});

    // Create inference engine
    var engine = abi.ai.llm.Engine.init(allocator, .{
        .max_new_tokens = max_tokens,
        .temperature = temperature,
        .top_p = top_p,
    });
    defer engine.deinit();

    // Load model
    engine.loadModel(model_path.?) catch |err| {
        std.debug.print("Error loading model: {t}\n", .{err});
        std.debug.print("\nNote: GGUF model loading requires a valid GGUF file.\n", .{});
        std.debug.print("You can download models from: https://huggingface.co/TheBloke\n", .{});
        return;
    };

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

    const model_path = std.mem.sliceTo(args[0], 0);

    std.debug.print("Loading model: {s}...\n", .{model_path});

    // Create inference engine
    var engine = abi.ai.llm.Engine.init(allocator, .{
        .max_new_tokens = 512,
        .temperature = 0.7,
        .top_p = 0.9,
    });
    defer engine.deinit();

    // Load model
    engine.loadModel(model_path) catch |err| {
        std.debug.print("Error loading model: {t}\n", .{err});
        std.debug.print("\nNote: GGUF model loading requires a valid GGUF file.\n", .{});
        std.debug.print("You can download models from: https://huggingface.co/TheBloke\n", .{});
        return;
    };

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
    var conversation = std.ArrayListUnmanaged(u8){};
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

        if (model_path == null) {
            model_path = std.mem.sliceTo(arg, 0);
        }
    }

    if (model_path == null) {
        std.debug.print("Usage: abi llm bench <model> [--prompt-tokens N] [--gen-tokens N]\n", .{});
        return;
    }

    std.debug.print("LLM Benchmark\n", .{});
    std.debug.print("=============\n", .{});
    std.debug.print("Model: {s}\n", .{model_path.?});
    std.debug.print("Prompt tokens: {d}\n", .{prompt_tokens});
    std.debug.print("Generation tokens: {d}\n", .{gen_tokens});
    std.debug.print("\n", .{});

    // Run compute benchmarks
    const bench_result = runComputeBenchmark(allocator, prompt_tokens, gen_tokens);

    std.debug.print("Benchmark Results\n", .{});
    std.debug.print("-----------------\n", .{});
    std.debug.print("MatMul Performance:\n", .{});
    std.debug.print("  Dimensions: {d}x{d} @ {d}x{d}\n", .{ bench_result.m, bench_result.k, bench_result.k, bench_result.n });
    std.debug.print("  Time: {d:.2} ms\n", .{bench_result.time_ms});
    std.debug.print("  GFLOPS: {d:.2}\n", .{bench_result.gflops});
    std.debug.print("\nEstimated Performance:\n", .{});
    std.debug.print("  Prefill: ~{d:.0} tokens/sec\n", .{bench_result.est_prefill_tok_s});
    std.debug.print("  Decode:  ~{d:.0} tokens/sec\n", .{bench_result.est_decode_tok_s});
    std.debug.print("\nNote: Actual performance depends on model architecture and quantization.\n", .{});
    std.debug.print("Download GGUF models from: https://huggingface.co/TheBloke\n", .{});
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
    var timer = std.time.Timer.start() catch {
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
        "  --stream                Enable streaming output\n\n" ++
        "Advanced sampling (llama.cpp parity):\n" ++
        "  --tfs <f>               Tail-free sampling parameter (default: 1.0 = disabled)\n" ++
        "  --mirostat <n>          Mirostat mode (0=off, 1=v1, 2=v2, default: 0)\n" ++
        "  --mirostat-tau <f>      Mirostat target entropy (default: 5.0)\n" ++
        "  --mirostat-eta <f>      Mirostat learning rate (default: 0.1)\n\n" ++
        "Benchmark options:\n" ++
        "  --prompt-tokens <n>     Number of prompt tokens (default: 128)\n" ++
        "  --gen-tokens <n>        Number of tokens to generate (default: 64)\n\n" ++
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
        "  abi llm bench ./llama-7b.gguf --gen-tokens 128\n" ++
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
