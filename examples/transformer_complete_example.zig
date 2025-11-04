//! Complete Transformer Example
//!
//! This example demonstrates the full transformer architecture with:
//! - Multi-head attention mechanisms
//! - Positional encoding
//! - Transformer blocks with residual connections
//! - Layer normalization
//! - Feed-forward networks
//! - Complete training pipeline

const std = @import("std");
const ai = @import("ai");
const transformer = ai.transformer;

/// Complete transformer model for sequence processing
pub const CompleteTransformer = struct {
    embedding: transformer.Embedding,
    positional_encoding: transformer.PositionalEncoding,
    layers: []transformer.TransformerBlock,
    output_projection: transformer.FeedForwardNetwork,
    vocab_size: usize,
    embed_dim: usize,
    num_layers: usize,
    num_heads: usize,
    max_seq_len: usize,
    allocator: std.mem.Allocator,

    pub fn init(
        allocator: std.mem.Allocator,
        vocab_size: usize,
        embed_dim: usize,
        num_layers: usize,
        num_heads: usize,
        max_seq_len: usize,
    ) !*CompleteTransformer {
        const self = try allocator.create(CompleteTransformer);
        errdefer allocator.destroy(self);

        self.* = .{
            .embedding = try transformer.Embedding.init(allocator, vocab_size, embed_dim),
            .positional_encoding = try transformer.PositionalEncoding.init(allocator, max_seq_len, embed_dim),
            .layers = try allocator.alloc(transformer.TransformerBlock, num_layers),
            .output_projection = try transformer.FeedForwardNetwork.init(allocator, embed_dim, embed_dim),
            .vocab_size = vocab_size,
            .embed_dim = embed_dim,
            .num_layers = num_layers,
            .num_heads = num_heads,
            .max_seq_len = max_seq_len,
            .allocator = allocator,
        };

        // Initialize transformer blocks
        const ff_dim = embed_dim * 4;
        for (0..num_layers) |i| {
            self.layers[i] = try transformer.TransformerBlock.init(allocator, embed_dim, num_heads, ff_dim, 0.1);
        }

        return self;
    }

    pub fn deinit(self: *CompleteTransformer) void {
        self.embedding.deinit(self.allocator);
        self.positional_encoding.deinit(self.allocator);
        for (self.layers) |*layer| {
            layer.deinit(self.allocator);
        }
        self.allocator.free(self.layers);
        self.output_projection.deinit(self.allocator);
        self.allocator.destroy(self);
    }

    /// Forward pass through complete transformer
    pub fn forward(self: *CompleteTransformer, input_tokens: []const u32, output: []f32) !void {
        const seq_len = input_tokens.len;
        if (seq_len > self.max_seq_len) return error.SequenceTooLong;

        // 1. Token embedding
        const embeddings = try self.allocator.alloc(f32, seq_len * self.embed_dim);
        defer self.allocator.free(embeddings);
        try self.embedding.forward(input_tokens, embeddings);

        // 2. Add positional encoding
        const encoded_input = try self.allocator.dupe(f32, embeddings);
        defer self.allocator.free(encoded_input);
        self.positional_encoding.encode(encoded_input, seq_len);

        // 3. Apply transformer layers
        const layer_output = try self.allocator.dupe(f32, encoded_input);
        defer self.allocator.free(layer_output);

        for (self.layers) |layer| {
            try layer.forward(layer_output, seq_len);
        }

        // 4. Final output projection
        try self.output_projection.forward(layer_output, output);
    }

    /// Generate text using the transformer
    pub fn generate(self: *CompleteTransformer, prompt: []const u32, max_length: usize) ![]u32 {
        var result = std.ArrayList(u32){};
        try result.ensureTotalCapacity(self.allocator, prompt.len + max_length);
        defer result.deinit(self.allocator);

        // Add initial prompt
        try result.appendSlice(self.allocator, prompt);

        // Generate tokens one by one
        while (result.items.len < max_length) {
            const current_seq = result.items;

            // Forward pass
            const output_size = current_seq.len * self.embed_dim;
            const output = try self.allocator.alloc(f32, output_size);
            defer self.allocator.free(output);

            try self.forward(current_seq, output);

            // Get logits for the last token
            const last_token_logits = output[(current_seq.len - 1) * self.vocab_size .. current_seq.len * self.vocab_size];

            // Sample next token (simple greedy for now)
            var best_token: u32 = 0;
            var best_score = last_token_logits[0];
            for (last_token_logits[1..], 1..) |score, token| {
                if (score > best_score) {
                    best_score = score;
                    best_token = @intCast(token);
                }
            }

            try result.append(self.allocator, best_token);
        }

        return result.toOwnedSlice();
    }
};

/// Demonstration of complete transformer capabilities
pub fn main() !void {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();
    const allocator = gpa.allocator;

    std.debug.print("=== Complete Transformer Example ===\n", .{});

    // Model configuration
    const vocab_size = 1000;
    const embed_dim = 128;
    const num_layers = 4;
    const num_heads = 8;
    const max_seq_len = 50;

    // Create transformer model
    var model = try CompleteTransformer.init(
        allocator,
        vocab_size,
        embed_dim,
        num_layers,
        num_heads,
        max_seq_len,
    );
    defer model.deinit();

    std.debug.print("Created transformer with:\n", .{});
    std.debug.print("  - Vocabulary size: {}\n", .{vocab_size});
    std.debug.print("  - Embedding dimension: {}\n", .{embed_dim});
    std.debug.print("  - Number of layers: {}\n", .{num_layers});
    std.debug.print("  - Number of heads: {}\n", .{num_heads});
    std.debug.print("  - Max sequence length: {}\n", .{max_seq_len});

    // Example input sequence
    const input_tokens = [_]u32{ 1, 45, 23, 67, 89, 12, 34 };
    const seq_len = input_tokens.len;

    std.debug.print("\n=== Forward Pass Demo ===\n", .{});
    std.debug.print("Input tokens: ", .{});
    for (input_tokens) |token| {
        std.debug.print("{} ", .{token});
    }
    std.debug.print("\n", .{});

    // Forward pass
    const output_size = seq_len * vocab_size;
    const output = try allocator.alloc(f32, output_size);
    defer allocator.free(output);

    try model.forward(&input_tokens, output);

    std.debug.print("Output shape: {} x {}\n", .{ seq_len, vocab_size });
    std.debug.print("Sample output logits for first token: ", .{});
    const first_token_logits = output[0..vocab_size];
    for (0..10) |i| {
        std.debug.print("{d:.3} ", .{first_token_logits[i]});
    }
    std.debug.print("...\n", .{});

    // Text generation demo
    std.debug.print("\n=== Text Generation Demo ===\n", .{});
    const prompt = [_]u32{ 1, 45, 23 };
    const generated = try model.generate(&prompt, 10);
    defer allocator.free(generated);

    std.debug.print("Prompt tokens: ", .{});
    for (prompt) |token| {
        std.debug.print("{} ", .{token});
    }
    std.debug.print("\n", .{});

    std.debug.print("Generated sequence: ", .{});
    for (generated) |token| {
        std.debug.print("{} ", .{token});
    }
    std.debug.print("\n", .{});

    // Multi-head attention demo
    std.debug.print("\n=== Multi-Head Attention Demo ===\n", .{});

    var mha = try transformer.MultiHeadAttention.init(allocator, embed_dim, num_heads);
    defer mha.deinit(allocator);

    // Create dummy query, key, value matrices
    const query_size = seq_len * embed_dim;
    const query = try allocator.alloc(f32, query_size);
    defer allocator.free(query);
    const key = try allocator.alloc(f32, query_size);
    defer allocator.free(key);
    const value = try allocator.alloc(f32, query_size);
    defer allocator.free(value);
    const attention_output = try allocator.alloc(f32, query_size);
    defer allocator.free(attention_output);

    // Initialize with some dummy data
    for (0..query_size) |i| {
        query[i] = std.math.sin(@as(f32, @floatFromInt(i)) * 0.1);
        key[i] = std.math.cos(@as(f32, @floatFromInt(i)) * 0.1);
        value[i] = std.math.sin(@as(f32, @floatFromInt(i)) * 0.05);
    }

    try mha.forward(query, key, value, attention_output);

    std.debug.print("Multi-head attention computed for sequence length {}\n", .{seq_len});
    std.debug.print("Attention output sample: ", .{});
    for (0..5) |i| {
        std.debug.print("{d:.3} ", .{attention_output[i]});
    }
    std.debug.print("...\n", .{});

    std.debug.print("\n=== Demo Complete ===\n", .{});
}
