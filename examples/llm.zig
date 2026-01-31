//! LLM Inference Example
//!
//! Demonstrates local LLM inference with:
//! - GGUF model loading
//! - Tokenization (BPE/SentencePiece)
//! - Text generation with sampling
//! - Streaming output

const std = @import("std");
const abi = @import("abi");

pub fn main(init: std.process.Init.Minimal) !void {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();
    const allocator = gpa.allocator();

    std.debug.print("=== ABI LLM Inference Example ===\n\n", .{});

    if (!abi.ai.isEnabled()) {
        std.debug.print("AI feature is disabled. Enable with -Denable-ai=true\n", .{});
        return;
    }

    // Initialize framework
    var framework = abi.initWithConfig(allocator, .{
        .ai = .{ .llm = .{} },
    }) catch |err| {
        std.debug.print("Framework initialization failed: {}\n", .{err});
        return err;
    };
    defer framework.deinit();

    // === Model Loading Demo ===
    std.debug.print("--- Model Loading ---\n", .{});

    // Check for model file argument or use default
    const model_path = blk: {
        var args_it = init.args.iterateAllocator(allocator) catch |err| {
            std.debug.print("Failed to read args: {t}\n", .{err});
            break :blk "model.gguf";
        };
        defer args_it.deinit();
        _ = args_it.next(); // Skip executable name.
        if (args_it.next()) |arg| break :blk arg[0..arg.len];
        break :blk "model.gguf";
    };

    std.debug.print("Looking for model: {s}\n", .{model_path});

    // Try to load model (will fail gracefully if not found)
    var model = abi.ai.llm.Model.load(allocator, model_path) catch |err| {
        std.debug.print("\nModel not found or failed to load: {}\n", .{err});
        std.debug.print("\nTo use this example:\n", .{});
        std.debug.print("  1. Download a GGUF model (e.g., from HuggingFace)\n", .{});
        std.debug.print("  2. Run: zig build run-llm -- path/to/model.gguf\n", .{});
        std.debug.print("\nShowing API demo without model...\n\n", .{});

        // Demo the API structure without actual model
        demoApiStructure();
        return;
    };
    defer model.deinit();

    // === Model Info ===
    std.debug.print("\n--- Model Information ---\n", .{});
    const info = model.info();
    std.debug.print("Architecture: {s}\n", .{info.architecture});
    std.debug.print("Hidden dim: {d}\n", .{info.dim});
    std.debug.print("Layers: {d}\n", .{info.n_layers});
    std.debug.print("Heads: {d} (KV: {d})\n", .{ info.n_heads, info.n_kv_heads });
    std.debug.print("Vocab size: {d}\n", .{info.vocab_size});
    std.debug.print("Max context: {d}\n", .{info.max_seq_len});
    std.debug.print("KV cache: {B}\n", .{info.kv_cache_memory});
    std.debug.print("Weights: {B}\n", .{info.weights_memory});

    // === Tokenization Demo ===
    std.debug.print("\n--- Tokenization ---\n", .{});
    const test_text = "Hello, world! How are you today?";
    const tokens = model.encode(test_text) catch |err| {
        std.debug.print("Tokenization failed: {}\n", .{err});
        return err;
    };
    defer allocator.free(tokens);

    std.debug.print("Input: \"{s}\"\n", .{test_text});
    std.debug.print("Tokens ({d}): ", .{tokens.len});
    for (tokens) |tok| {
        std.debug.print("{d} ", .{tok});
    }
    std.debug.print("\n", .{});

    // Decode back
    const decoded = model.decode(tokens) catch |err| {
        std.debug.print("Detokenization failed: {}\n", .{err});
        return err;
    };
    defer allocator.free(decoded);
    std.debug.print("Decoded: \"{s}\"\n", .{decoded});

    // === Generation Demo ===
    std.debug.print("\n--- Text Generation ---\n", .{});

    const prompt = "Once upon a time";
    std.debug.print("Prompt: \"{s}\"\n", .{prompt});
    std.debug.print("Generating...\n\n", .{});

    // Configure generation
    const gen_config = abi.ai.llm.generation.GeneratorConfig{
        .temperature = 0.8,
        .top_k = 40,
        .top_p = 0.95,
        .repetition_penalty = 1.1,
        .max_tokens = 50,
    };

    const prompt_tokens = model.encode(prompt) catch |err| {
        std.debug.print("Prompt encoding failed: {}\n", .{err});
        return err;
    };
    defer allocator.free(prompt_tokens);

    var timer = std.time.Timer.start() catch null;
    const output_tokens = model.generate(prompt_tokens, gen_config) catch |err| {
        std.debug.print("Generation failed: {}\n", .{err});
        return err;
    };
    const elapsed_ns = if (timer) |*t| t.read() else 0;
    defer allocator.free(output_tokens);

    const output_text = model.decode(output_tokens) catch |err| {
        std.debug.print("Decoding failed: {}\n", .{err});
        return err;
    };
    defer allocator.free(output_text);

    std.debug.print("{s}{s}\n\n", .{ prompt, output_text });

    // === Generation Stats ===
    std.debug.print("--- Generation Stats ---\n", .{});
    const token_count = output_tokens.len;
    std.debug.print("Tokens generated: {d}\n", .{token_count});
    if (elapsed_ns > 0) {
        const tokens_per_second = @as(f64, @floatFromInt(token_count)) /
            (@as(f64, @floatFromInt(elapsed_ns)) / 1_000_000_000.0);
        std.debug.print("Tokens/second: {d:.2}\n", .{tokens_per_second});
        std.debug.print("Generation time: {d:.2}ms\n", .{@as(f64, @floatFromInt(elapsed_ns)) / 1_000_000.0});
    }

    std.debug.print("\n=== LLM Example Complete ===\n", .{});
}

/// Demo API structure when model is not available
fn demoApiStructure() void {
    std.debug.print("--- LLM API Structure Demo ---\n\n", .{});

    std.debug.print("1. Load Model:\n", .{});
    std.debug.print("   var model = try abi.ai.llm.Model.load(allocator, \"model.gguf\");\n", .{});
    std.debug.print("   defer model.deinit();\n\n", .{});

    std.debug.print("2. Tokenize Text:\n", .{});
    std.debug.print("   const tokens = try model.encode(\"Hello world\");\n", .{});
    std.debug.print("   defer allocator.free(tokens);\n\n", .{});

    std.debug.print("3. Configure Sampling:\n", .{});
    std.debug.print("   const config = abi.ai.llm.generation.GeneratorConfig{{\n", .{});
    std.debug.print("       .temperature = 0.8,\n", .{});
    std.debug.print("       .top_k = 40,\n", .{});
    std.debug.print("       .top_p = 0.95,\n", .{});
    std.debug.print("   }};\n\n", .{});

    std.debug.print("4. Generate Text:\n", .{});
    std.debug.print("   const prompt_tokens = try model.encode(\"Once upon a time\");\n", .{});
    std.debug.print("   defer allocator.free(prompt_tokens);\n", .{});
    std.debug.print("   const output_tokens = try model.generate(prompt_tokens, config);\n", .{});
    std.debug.print("   defer allocator.free(output_tokens);\n", .{});
    std.debug.print("   const output = try model.decode(output_tokens);\n", .{});
    std.debug.print("   defer allocator.free(output);\n", .{});
    std.debug.print("   std.debug.print(\"Once upon a time{{s}}\", .{{output}});\n\n", .{});

    std.debug.print("5. CLI Commands:\n", .{});
    std.debug.print("   zig build run -- llm info model.gguf\n", .{});
    std.debug.print("   zig build run -- llm generate model.gguf --prompt \"Hello\"\n", .{});
    std.debug.print("   zig build run -- llm chat model.gguf\n", .{});
    std.debug.print("   zig build run -- llm bench model.gguf\n\n", .{});

    std.debug.print("Supported formats: GGUF (Q4_0, Q4_1, Q5_0, Q5_1, Q8_0)\n", .{});
    std.debug.print("Supported tokenizers: BPE, SentencePiece\n", .{});
}
