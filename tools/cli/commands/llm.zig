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
        runList();
        return;
    }

    std.debug.print("Unknown llm command: {s}\n", .{command});
    printHelp();
}

fn runInfo(allocator: std.mem.Allocator, args: []const [:0]const u8) !void {
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
    var model_path: ?[]const u8 = null;
    var prompt: ?[]const u8 = null;
    var max_tokens: u32 = 256;
    var temperature: f32 = 0.7;
    var top_p: f32 = 0.9;

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
    std.debug.print("Max tokens: {d}, Temperature: {d:.2}, Top-p: {d:.2}\n", .{ max_tokens, temperature, top_p });
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
    const output = engine.generate(prompt.?) catch |err| {
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

fn runChat(_: std.mem.Allocator, _: []const [:0]const u8) !void {
    std.debug.print("Interactive chat mode\n", .{});
    std.debug.print("=====================\n\n", .{});
    std.debug.print("Note: Full chat mode requires a loaded model.\n", .{});
    std.debug.print("Usage: abi llm chat <model>\n", .{});
    std.debug.print("\nChat commands:\n", .{});
    std.debug.print("  /quit - Exit chat\n", .{});
    std.debug.print("  /clear - Clear conversation history\n", .{});
    std.debug.print("  /system <prompt> - Set system prompt\n", .{});
}

fn runBench(allocator: std.mem.Allocator, args: []const [:0]const u8) !void {
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
    std.debug.print("GGUF (llama.cpp format)\n", .{});
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

fn printHelp() void {
    const help_text =
        "Usage: abi llm <command> [options]\n\n" ++
        "Run local LLM inference with GGUF models.\n\n" ++
        "Commands:\n" ++
        "  info <model>       Show model information\n" ++
        "  generate <model>   Generate text from a prompt\n" ++
        "  chat <model>       Interactive chat mode\n" ++
        "  bench <model>      Benchmark model performance\n" ++
        "  list               List supported models and formats\n" ++
        "  help               Show this help message\n\n" ++
        "Generate options:\n" ++
        "  -m, --model <path>      Path to GGUF model file\n" ++
        "  -p, --prompt <text>     Text prompt for generation\n" ++
        "  -n, --max-tokens <n>    Maximum tokens to generate (default: 256)\n" ++
        "  -t, --temperature <f>   Temperature for sampling (default: 0.7)\n" ++
        "  --top-p <f>             Top-p nucleus sampling (default: 0.9)\n\n" ++
        "Benchmark options:\n" ++
        "  --prompt-tokens <n>     Number of prompt tokens (default: 128)\n" ++
        "  --gen-tokens <n>        Number of tokens to generate (default: 64)\n\n" ++
        "Examples:\n" ++
        "  abi llm info ./llama-7b.gguf\n" ++
        "  abi llm generate ./llama-7b.gguf -p \"Hello, how are you?\"\n" ++
        "  abi llm generate ./mistral-7b.gguf -p \"Write a poem\" -n 100 -t 0.8\n" ++
        "  abi llm bench ./llama-7b.gguf --gen-tokens 128\n" ++
        "  abi llm list\n";
    std.debug.print("{s}", .{help_text});
}
