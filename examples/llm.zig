//! LLM Inference Example
//!
//! Demonstrates local LLM inference with:
//! - GGUF model loading
//! - Tokenization (BPE/SentencePiece)
//! - Text generation with sampling
//! - Streaming output

const std = @import("std");
const abi = @import("abi");

pub fn main() !void {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();
    const allocator = gpa.allocator();

    std.debug.print("=== ABI LLM Inference Example ===\n\n", .{});

    if (!abi.ai.isEnabled()) {
        std.debug.print("AI feature is disabled. Enable with -Denable-ai=true\n", .{});
        return;
    }

    // Initialize framework
    var framework = abi.init(allocator, abi.FrameworkOptions{
        .enable_ai = true,
        .enable_llm = true,
        .enable_gpu = false,
    }) catch |err| {
        std.debug.print("Framework initialization failed: {}\n", .{err});
        return err;
    };
    defer abi.shutdown(&framework);

    // === Model Loading Demo ===
    std.debug.print("--- Model Loading ---\n", .{});

    // Check for model file argument or use default
    const model_path = if (std.process.argsWithAllocator(allocator)) |args| blk: {
        defer {
            for (args) |arg| allocator.free(arg);
            allocator.free(args);
        }
        if (args.len > 1) break :blk args[1];
        break :blk "model.gguf";
    } else |_| "model.gguf";

    std.debug.print("Looking for model: {s}\n", .{model_path});

    // Try to load model (will fail gracefully if not found)
    var model = abi.llm.loadModel(allocator, model_path, .{
        .context_length = 2048,
        .batch_size = 512,
        .use_mmap = true,
    }) catch |err| {
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
    const info = model.getInfo();
    std.debug.print("Architecture: {s}\n", .{info.architecture});
    std.debug.print("Parameters: {d}M\n", .{info.parameter_count / 1_000_000});
    std.debug.print("Context length: {d}\n", .{info.context_length});
    std.debug.print("Embedding size: {d}\n", .{info.embedding_size});
    std.debug.print("Vocab size: {d}\n", .{info.vocab_size});
    std.debug.print("Quantization: {t}\n", .{info.quantization});

    // === Tokenization Demo ===
    std.debug.print("\n--- Tokenization ---\n", .{});
    const test_text = "Hello, world! How are you today?";
    const tokens = model.tokenize(allocator, test_text) catch |err| {
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
    const decoded = model.detokenize(allocator, tokens) catch |err| {
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

    // Configure sampling
    const sampler_config = abi.llm.SamplerConfig{
        .temperature = 0.8,
        .top_k = 40,
        .top_p = 0.95,
        .repeat_penalty = 1.1,
        .max_tokens = 50,
    };

    // Generate with streaming
    var generator = model.createGenerator(sampler_config) catch |err| {
        std.debug.print("Failed to create generator: {}\n", .{err});
        return err;
    };
    defer generator.deinit();

    generator.setPrompt(prompt) catch |err| {
        std.debug.print("Failed to set prompt: {}\n", .{err});
        return err;
    };

    std.debug.print("{s}", .{prompt});
    var token_count: usize = 0;
    while (generator.next()) |token_text| {
        std.debug.print("{s}", .{token_text});
        token_count += 1;
    }
    std.debug.print("\n\n", .{});

    // === Generation Stats ===
    std.debug.print("--- Generation Stats ---\n", .{});
    const gen_stats = generator.getStats();
    std.debug.print("Tokens generated: {d}\n", .{token_count});
    std.debug.print("Tokens/second: {d:.2}\n", .{gen_stats.tokens_per_second});
    std.debug.print("Prompt eval time: {d:.2}ms\n", .{gen_stats.prompt_eval_time_ms});
    std.debug.print("Generation time: {d:.2}ms\n", .{gen_stats.generation_time_ms});

    std.debug.print("\n=== LLM Example Complete ===\n", .{});
}

/// Demo API structure when model is not available
fn demoApiStructure() void {
    std.debug.print("--- LLM API Structure Demo ---\n\n", .{});

    std.debug.print("1. Load Model:\n", .{});
    std.debug.print("   var model = try abi.llm.loadModel(allocator, \"model.gguf\", .{{}});\n", .{});
    std.debug.print("   defer model.deinit();\n\n", .{});

    std.debug.print("2. Tokenize Text:\n", .{});
    std.debug.print("   const tokens = try model.tokenize(allocator, \"Hello world\");\n", .{});
    std.debug.print("   defer allocator.free(tokens);\n\n", .{});

    std.debug.print("3. Configure Sampling:\n", .{});
    std.debug.print("   const config = abi.llm.SamplerConfig{{\n", .{});
    std.debug.print("       .temperature = 0.8,\n", .{});
    std.debug.print("       .top_k = 40,\n", .{});
    std.debug.print("       .top_p = 0.95,\n", .{});
    std.debug.print("   }};\n\n", .{});

    std.debug.print("4. Generate Text:\n", .{});
    std.debug.print("   var gen = try model.createGenerator(config);\n", .{});
    std.debug.print("   defer gen.deinit();\n", .{});
    std.debug.print("   try gen.setPrompt(\"Once upon a time\");\n", .{});
    std.debug.print("   while (gen.next()) |token| {{\n", .{});
    std.debug.print("       std.debug.print(\"{{s}}\", .{{token}});\n", .{});
    std.debug.print("   }}\n\n", .{});

    std.debug.print("5. CLI Commands:\n", .{});
    std.debug.print("   zig build run -- llm info model.gguf\n", .{});
    std.debug.print("   zig build run -- llm generate model.gguf --prompt \"Hello\"\n", .{});
    std.debug.print("   zig build run -- llm chat model.gguf\n", .{});
    std.debug.print("   zig build run -- llm bench model.gguf\n\n", .{});

    std.debug.print("Supported formats: GGUF (Q4_0, Q4_1, Q5_0, Q5_1, Q8_0)\n", .{});
    std.debug.print("Supported tokenizers: BPE, SentencePiece\n", .{});
}
