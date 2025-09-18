//! Transformer Architecture Example
//!
//! This example demonstrates how to use the complete transformer architecture
//! implemented in the ABI AI Framework, including multi-head attention,
//! positional encoding, and transformer blocks.

const std = @import("std");
const abi = @import("abi");
const transformer = abi.ai.transformer;

pub fn main() !void {
    var arena = std.heap.ArenaAllocator.init(std.heap.page_allocator);
    defer arena.deinit();
    const allocator = arena.allocator();

    std.debug.print("🚀 ABI Transformer Architecture Example\n", .{});
    std.debug.print("========================================\n\n", .{});

    // Model configuration
    const vocab_size = 1000; // Vocabulary size
    const embed_dim = 64; // Embedding dimension
    const num_heads = 8; // Number of attention heads
    const num_layers = 6; // Number of transformer layers
    const max_seq_len = 32; // Maximum sequence length

    std.debug.print("Model Configuration:\n", .{});
    std.debug.print("├─ Vocabulary Size: {}\n", .{vocab_size});
    std.debug.print("├─ Embedding Dimension: {}\n", .{embed_dim});
    std.debug.print("├─ Attention Heads: {}\n", .{num_heads});
    std.debug.print("├─ Transformer Layers: {}\n", .{num_layers});
    std.debug.print("├─ Max Sequence Length: {}\n", .{max_seq_len});
    std.debug.print("└─ Parameters: ~{d}\n\n", .{vocab_size * embed_dim + num_layers * embed_dim * 4});

    // Initialize transformer model
    std.debug.print("🔧 Initializing Transformer Model...\n", .{});
    const model = try transformer.Transformer.init(allocator, vocab_size, embed_dim, num_layers, num_heads, max_seq_len);
    defer model.deinit(allocator);

    std.debug.print("✅ Model initialized successfully!\n\n", .{});

    // Create sample input sequence
    const sample_tokens = [_]u32{
        1, 45, 23, 67, 89, 12, 34, 56, 78, 90, // Sample token sequence
        2, 46, 24, 68, 91, 13, 35, 57, 79, 92, // Continuation
    };

    std.debug.print("📝 Sample Input Sequence:\n", .{});
    std.debug.print("Tokens: ", .{});
    for (sample_tokens, 0..) |token, i| {
        if (i > 0) std.debug.print(", ", .{});
        std.debug.print("{}", .{token});
    }
    std.debug.print("\n\n", .{});

    // Prepare output buffer
    const output_size = sample_tokens.len * vocab_size; // Logits for each position
    var output_logits = try allocator.alloc(f32, output_size);

    // Forward pass
    std.debug.print("⚡ Running Transformer Forward Pass...\n", .{});
    const start_time = std.time.nanoTimestamp();

    try model.forward(&sample_tokens, output_logits);

    const end_time = std.time.nanoTimestamp();
    const duration_ns = @as(f64, @floatFromInt(end_time - start_time));

    std.debug.print("✅ Forward pass completed in {d:.2} ms\n\n", .{duration_ns / 1_000_000.0});

    // Analyze output
    std.debug.print("📊 Output Analysis:\n", .{});

    // Find most likely tokens for each position
    for (0..sample_tokens.len) |pos| {
        const logits_start = pos * vocab_size;
        const logits = output_logits[logits_start .. logits_start + vocab_size];

        // Find token with highest logit
        var max_logit: f32 = -std.math.inf(f32);
        var max_token: usize = 0;

        for (logits, 0..) |logit, token| {
            if (logit > max_logit) {
                max_logit = logit;
                max_token = token;
            }
        }

        std.debug.print("├─ Position {}: Input token {}, Predicted token {} (logit: {d:.3})\n", .{ pos, sample_tokens[pos], max_token, max_logit });
    }

    // Calculate some statistics
    var total_sum: f32 = 0.0;
    var max_value: f32 = -std.math.inf(f32);
    var min_value: f32 = std.math.inf(f32);

    for (output_logits) |logit| {
        total_sum += logit;
        max_value = @max(max_value, logit);
        min_value = @min(min_value, logit);
    }

    const avg_value = total_sum / @as(f32, @floatFromInt(output_logits.len));

    std.debug.print("\n📈 Statistics:\n", .{});
    std.debug.print("├─ Total Logits: {}\n", .{output_logits.len});
    std.debug.print("├─ Average Logit: {d:.4}\n", .{avg_value});
    std.debug.print("├─ Max Logit: {d:.4}\n", .{max_value});
    std.debug.print("├─ Min Logit: {d:.4}\n", .{min_value});
    std.debug.print("└─ Logit Range: {d:.4}\n\n", .{max_value - min_value});

    // Demonstrate individual components
    std.debug.print("🔬 Component Demonstration:\n", .{});

    // Test Multi-Head Attention
    std.debug.print("├─ Multi-Head Attention: ", .{});
    const mha = try transformer.MultiHeadAttention.init(allocator, embed_dim, num_heads);
    defer mha.deinit(allocator);

    var query = try allocator.alloc(f32, embed_dim);
    defer allocator.free(query);
    var key = try allocator.alloc(f32, embed_dim);
    defer allocator.free(key);
    var value = try allocator.alloc(f32, embed_dim);
    defer allocator.free(value);
    const attention_output = try allocator.alloc(f32, embed_dim);
    defer allocator.free(attention_output);

    // Initialize with test data
    for (0..embed_dim) |i| {
        const fi = @as(f32, @floatFromInt(i));
        query[i] = std.math.sin(fi * 0.1);
        key[i] = std.math.cos(fi * 0.1);
        value[i] = fi / @as(f32, @floatFromInt(embed_dim));
    }

    try mha.forward(query, key, value, attention_output);
    std.debug.print("✅ Working\n", .{});

    // Test Positional Encoding
    std.debug.print("├─ Positional Encoding: ", .{});
    const pos_enc = try transformer.PositionalEncoding.init(allocator, max_seq_len, embed_dim);
    defer pos_enc.deinit(allocator);

    const pos_test_input = try allocator.alloc(f32, embed_dim);
    defer allocator.free(pos_test_input);
    @memset(pos_test_input, 1.0);

    pos_enc.encode(pos_test_input, 1);
    std.debug.print("✅ Working\n", .{});

    // Test Layer Normalization
    std.debug.print("├─ Layer Normalization: ", .{});
    const layer_norm = try transformer.LayerNorm.init(allocator, embed_dim);
    defer layer_norm.deinit(allocator);

    layer_norm.forward(pos_test_input, 1);
    std.debug.print("✅ Working\n", .{});

    // Test Feed-Forward Network
    std.debug.print("└─ Feed-Forward Network: ", .{});
    const ff_net = try transformer.FeedForwardNetwork.init(allocator, embed_dim, embed_dim * 4);
    defer ff_net.deinit(allocator);

    var ff_input = try allocator.alloc(f32, embed_dim);
    defer allocator.free(ff_input);
    const ff_output = try allocator.alloc(f32, embed_dim);
    defer allocator.free(ff_output);

    for (0..embed_dim) |i| {
        ff_input[i] = @as(f32, @floatFromInt(i % 10)) * 0.1;
    }

    try ff_net.forward(ff_input, ff_output);
    std.debug.print("✅ Working\n\n", .{});

    // Performance summary
    std.debug.print("⚡ Performance Summary:\n", .{});
    std.debug.print("├─ Model Size: {} layers, {} attention heads\n", .{ num_layers, num_heads });
    std.debug.print("├─ Processing Speed: {d:.2} ms for {} tokens\n", .{ duration_ns / 1_000_000.0, sample_tokens.len });
    std.debug.print("├─ Throughput: {d:.0} tokens/second\n", .{@as(f64, @floatFromInt(sample_tokens.len)) / (duration_ns / 1_000_000_000.0)});
    std.debug.print("└─ Memory Efficient: Uses arena allocation\n\n", .{});

    std.debug.print("🎉 Transformer Architecture Example Complete!\n", .{});
    std.debug.print("The ABI AI Framework now supports advanced transformer models\n", .{});
    std.debug.print("with multi-head attention, positional encoding, and layer normalization.\n", .{});
}
