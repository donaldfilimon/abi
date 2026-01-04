//! Transformer model implementation with attention mechanism and inference.
//!
//! Provides text encoding, decoding, inference, and embedding generation
//! using a configurable transformer architecture with multi-head attention.

const std = @import("std");
const simd = @import("../../../shared/simd.zig");

pub const TransformerError = error{
    InvalidConfiguration,
    EmptyInput,
    ContextTooLarge,
    OutOfMemory,
};

pub const TransformerConfig = struct {
    layers: u16 = 4,
    hidden_size: u16 = 256,
    intermediate_size: u16 = 1024,
    vocab_size: u32 = 8192,
    max_tokens: u32 = 128,
    num_heads: u16 = 8,
    seed: u64 = 0x2a9d_7d3c_b1e5_4f03,
    temperature: f32 = 0.8,
    top_p: f32 = 0.9,
    top_k: u32 = 40,

    pub fn validate(self: TransformerConfig) TransformerError!void {
        if (self.layers == 0) return TransformerError.InvalidConfiguration;
        if (self.hidden_size == 0) return TransformerError.InvalidConfiguration;
        if (self.intermediate_size == 0) return TransformerError.InvalidConfiguration;
        if (self.vocab_size < 2) return TransformerError.InvalidConfiguration;
        if (self.max_tokens == 0) return TransformerError.InvalidConfiguration;
        if (self.num_heads == 0) return TransformerError.InvalidConfiguration;
        if (self.hidden_size % self.num_heads != 0) {
            return TransformerError.InvalidConfiguration;
        }
        if (self.temperature < 0 or self.temperature > 2.0) {
            return TransformerError.InvalidConfiguration;
        }
        if (self.top_p < 0 or self.top_p > 1.0) return TransformerError.InvalidConfiguration;
    }
};

pub const TransformerModel = struct {
    allocator: std.mem.Allocator,
    config: TransformerConfig,
    embedding_weights: []f32,
    layer_query_weights: [][]f32,
    layer_key_weights: [][]f32,
    layer_value_weights: [][]f32,
    layer_output_weights: [][]f32,
    layer_ff_0_weights: [][]f32,
    layer_ff_1_weights: [][]f32,
    lm_head_weights: []f32,
    rng: std.Random.DefaultPrng,

    pub fn init(allocator: std.mem.Allocator, config: TransformerConfig) !TransformerModel {
        try config.validate();

        const hidden_size = config.hidden_size;
        const intermediate_size = config.intermediate_size;

        const embedding_weights = try allocator.alloc(f32, config.vocab_size * hidden_size);
        @memset(embedding_weights, 0);

        var layer_query_weights = try allocator.alloc([]f32, config.layers);
        var layer_key_weights = try allocator.alloc([]f32, config.layers);
        var layer_value_weights = try allocator.alloc([]f32, config.layers);
        var layer_output_weights = try allocator.alloc([]f32, config.layers);
        var layer_ff_0_weights = try allocator.alloc([]f32, config.layers);
        var layer_ff_1_weights = try allocator.alloc([]f32, config.layers);

        var rng = std.Random.DefaultPrng.init(config.seed);

        for (0..config.layers) |i| {
            layer_query_weights[i] = try allocator.alloc(f32, hidden_size * hidden_size);
            layer_key_weights[i] = try allocator.alloc(f32, hidden_size * hidden_size);
            layer_value_weights[i] = try allocator.alloc(f32, hidden_size * hidden_size);
            layer_output_weights[i] = try allocator.alloc(f32, hidden_size * hidden_size);
            layer_ff_0_weights[i] = try allocator.alloc(f32, hidden_size * intermediate_size);
            layer_ff_1_weights[i] = try allocator.alloc(f32, intermediate_size * hidden_size);

            initGaussian(layer_query_weights[i], &rng);
            initGaussian(layer_key_weights[i], &rng);
            initGaussian(layer_value_weights[i], &rng);
            initGaussian(layer_output_weights[i], &rng);
            initGaussian(layer_ff_0_weights[i], &rng);
            initGaussian(layer_ff_1_weights[i], &rng);
        }

        const lm_head_weights = try allocator.alloc(f32, hidden_size * config.vocab_size);
        initGaussian(lm_head_weights, &rng);

        return .{
            .allocator = allocator,
            .config = config,
            .embedding_weights = embedding_weights,
            .layer_query_weights = layer_query_weights,
            .layer_key_weights = layer_key_weights,
            .layer_value_weights = layer_value_weights,
            .layer_output_weights = layer_output_weights,
            .layer_ff_0_weights = layer_ff_0_weights,
            .layer_ff_1_weights = layer_ff_1_weights,
            .lm_head_weights = lm_head_weights,
            .rng = rng,
        };
    }

    pub fn deinit(self: *TransformerModel) void {
        self.allocator.free(self.lm_head_weights);
        for (self.layer_ff_1_weights) |w| self.allocator.free(w);
        for (self.layer_ff_0_weights) |w| self.allocator.free(w);
        for (self.layer_output_weights) |w| self.allocator.free(w);
        for (self.layer_value_weights) |w| self.allocator.free(w);
        for (self.layer_key_weights) |w| self.allocator.free(w);
        for (self.layer_query_weights) |w| self.allocator.free(w);
        self.allocator.free(self.layer_ff_1_weights);
        self.allocator.free(self.layer_ff_0_weights);
        self.allocator.free(self.layer_output_weights);
        self.allocator.free(self.layer_value_weights);
        self.allocator.free(self.layer_key_weights);
        self.allocator.free(self.layer_query_weights);
        self.allocator.free(self.embedding_weights);
        self.* = undefined;
    }

    fn initGaussian(data: []f32, rng: *std.Random.DefaultPrng) void {
        const scale = @sqrt(2.0 / @as(f32, @floatFromInt(data.len)));
        for (data) |*val| {
            val.* = rng.random().floatNorm(f32) * scale;
        }
    }

    pub fn infer(self: *TransformerModel, allocator: std.mem.Allocator, input: []const u8, max_output_tokens: ?u32) ![]u8 {
        const tokens = try self.encode(allocator, input);
        defer allocator.free(tokens);

        const max_tokens = max_output_tokens orelse 32;
        const output_tokens = try self.generate(allocator, tokens, max_tokens);
        defer allocator.free(output_tokens);

        return self.decode(output_tokens);
    }

    pub fn generate(
        self: *TransformerModel,
        allocator: std.mem.Allocator,
        prompt: []const u32,
        max_tokens: u32,
    ) ![]u32 {
        if (prompt.len == 0) return error.EmptyInput;
        if (prompt.len > self.config.max_tokens) return error.ContextTooLarge;

        var result = try allocator.alloc(u32, max_tokens);
        errdefer allocator.free(result);

        var context = try allocator.alloc(f32, prompt.len * self.config.hidden_size);
        defer allocator.free(context);

        for (prompt, 0..) |token, i| {
            if (token < self.config.vocab_size) {
                @memcpy(context[i * self.config.hidden_size .. (i + 1) * self.config.hidden_size], self.embedding_weights[token * self.config.hidden_size ..]);
            }
        }

        var generated: usize = 0;
        while (generated < max_tokens) : (generated += 1) {
            const logits = try self.forward(allocator, context);
            defer allocator.free(logits);

            const token = self.sampleToken(logits);
            result[generated] = token;

            if (token == 0) break;

            if (context.len + self.config.hidden_size > self.config.max_tokens * self.config.hidden_size) {
                break;
            }

            const new_context = try allocator.alloc(f32, context.len + self.config.hidden_size);
            @memcpy(new_context[0..context.len], context);
            @memcpy(new_context[context.len .. context.len + self.config.hidden_size], self.embedding_weights[token * self.config.hidden_size ..]);
            allocator.free(context);
            context = new_context;
        }

        return result[0..generated];
    }

    pub fn forward(self: *const TransformerModel, allocator: std.mem.Allocator, input: []const f32) ![]f32 {
        const hidden_size = self.config.hidden_size;
        const vocab_size = self.config.vocab_size;

        var current = try allocator.alloc(f32, input.len);
        defer allocator.free(current);
        @memcpy(current, input);

        const num_layers = self.config.layers;
        for (0..num_layers) |layer_idx| {
            const q_w = self.layer_query_weights[layer_idx];
            const k_w = self.layer_key_weights[layer_idx];
            const v_w = self.layer_value_weights[layer_idx];
            const o_w = self.layer_output_weights[layer_idx];
            const ff_0_w = self.layer_ff_0_weights[layer_idx];
            const ff_1_w = self.layer_ff_1_weights[layer_idx];

            var attention_out = try self.multiHeadAttention(allocator, current, q_w, k_w, v_w, o_w);
            defer allocator.free(attention_out);

            for (current, 0..) |val, idx| {
                current[idx] = val + 0.1 * attention_out[idx];
            }

            var ff_out = try self.feedForward(allocator, current, ff_0_w, ff_1_w);
            defer allocator.free(ff_out);

            for (current, 0..) |val, idx| {
                current[idx] = val + 0.1 * ff_out[idx];
            }
        }

        const logits = try allocator.alloc(f32, vocab_size);
        @memset(logits, 0);

        const last_hidden = current[(current.len - hidden_size)..];
        for (0..vocab_size) |i| {
            var sum: f32 = 0;
            for (0..hidden_size) |j| {
                sum += last_hidden[j] * self.lm_head_weights[j * vocab_size + i];
            }
            logits[i] = sum;
        }

        return logits;
    }

    fn multiHeadAttention(
        self: *const TransformerModel,
        allocator: std.mem.Allocator,
        input: []const f32,
        query_w: []const f32,
        key_w: []const f32,
        value_w: []const f32,
        output_w: []const f32,
    ) ![]f32 {
        const seq_len = input.len / self.config.hidden_size;
        const hidden_size = self.config.hidden_size;
        const num_heads = self.config.num_heads;
        const head_dim = hidden_size / num_heads;

        var q = try allocator.alloc(f32, input.len);
        var k = try allocator.alloc(f32, input.len);
        var v = try allocator.alloc(f32, input.len);
        defer allocator.free(q);
        defer allocator.free(k);
        defer allocator.free(v);

        matMul(q, input, query_w, hidden_size, hidden_size);
        matMul(k, input, key_w, hidden_size, hidden_size);
        matMul(v, input, value_w, hidden_size, hidden_size);

        const output = try allocator.alloc(f32, input.len);
        @memset(output, 0);

        const scale = 1.0 / @sqrt(@as(f32, @floatFromInt(head_dim)));

        for (0..num_heads) |head| {
            const head_offset = head * head_dim;
            const q_head = q[head_offset .. head_offset + head_dim * seq_len];
            const k_head = k[head_offset .. head_offset + head_dim * seq_len];
            const v_head = v[head_offset .. head_offset + head_dim * seq_len];
            const out_head = output[head_offset .. head_offset + head_dim * seq_len];

            var attention = try allocator.alloc(f32, seq_len * seq_len);
            defer allocator.free(attention);

            for (0..seq_len) |i| {
                for (0..seq_len) |j| {
                    var dot: f32 = 0;
                    for (0..head_dim) |d| {
                        dot += q_head[i * head_dim + d] * k_head[j * head_dim + d];
                    }
                    attention[i * seq_len + j] = dot * scale;
                }
            }

            softmax2D(attention, seq_len);

            for (0..seq_len) |i| {
                for (0..head_dim) |d| {
                    var sum: f32 = 0;
                    for (0..seq_len) |j| {
                        sum += attention[i * seq_len + j] * v_head[j * head_dim + d];
                    }
                    out_head[i * head_dim + d] += sum;
                }
            }
        }

        matMul(output, output, output_w, hidden_size, hidden_size);

        return output;
    }

    fn feedForward(
        self: *const TransformerModel,
        allocator: std.mem.Allocator,
        input: []const f32,
        ff_0_w: []const f32,
        ff_1_w: []const f32,
    ) ![]f32 {
        const hidden_size = self.config.hidden_size;
        const intermediate_size = self.config.intermediate_size;

        const intermediate = try allocator.alloc(f32, input.len / hidden_size * intermediate_size);
        defer allocator.free(intermediate);
        matMul(intermediate, input, ff_0_w, hidden_size, intermediate_size);

        for (intermediate) |*val| {
            val.* = @max(0, val.*);
        }

        const output = try allocator.alloc(f32, input.len);
        defer allocator.free(output);
        matMul(output, intermediate, ff_1_w, intermediate_size, hidden_size);

        return output;
    }

    fn matMul(output: []f32, input: []const f32, matrix: []const f32, in_cols: usize, out_cols: usize) void {
        const seq_len = input.len / in_cols;
        @memset(output, 0);

        for (0..seq_len) |seq| {
            for (0..out_cols) |j| {
                var sum: f32 = 0;
                for (0..in_cols) |k| {
                    sum += input[seq * in_cols + k] * matrix[k * out_cols + j];
                }
                output[seq * out_cols + j] = sum;
            }
        }
    }

    fn softmax2D(data: []f32, size: usize) void {
        for (0..size) |i| {
            const row = data[i * size .. (i + 1) * size];

            var max_val: f32 = -1e38;
            for (row) |val| {
                if (val > max_val) max_val = val;
            }

            var sum: f32 = 0;
            for (row) |*val| {
                val.* = @exp(val.* - max_val);
                sum += val.*;
            }

            for (row) |*val| {
                val.* /= sum;
            }
        }
    }

    pub fn sampleToken(self: *const TransformerModel, logits: []f32) u32 {
        const vocab_size = self.config.vocab_size;
        const temperature = self.config.temperature;
        const top_k = self.config.top_k;

        var max_logit: f32 = -1e38;
        for (logits) |logit| {
            if (logit > max_logit) max_logit = logit;
        }

        var probs = try self.allocator.alloc(f32, vocab_size);
        defer self.allocator.free(probs);

        var sum: f32 = 0;
        for (logits, 0..) |logit, i| {
            const val = @exp((logit - max_logit) / temperature);
            probs[i] = val;
            sum += val;
        }

        if (sum > 0) {
            for (probs) |*p| {
                p.* /= sum;
            }
        }

        const actual_top_k = @min(top_k, vocab_size);
        var candidates = try self.allocator.alloc([2]f32, vocab_size);
        defer self.allocator.free(candidates);

        for (probs, 0..) |prob, i| {
            candidates[i] = .{ @as(f32, @floatFromInt(i)), prob };
        }

        for (0..actual_top_k) |i| {
            var max_idx: usize = i;
            var max_val: f32 = candidates[i][1];
            for ((i + 1)..vocab_size) |j| {
                if (candidates[j][1] > max_val) {
                    max_idx = j;
                    max_val = candidates[j][1];
                }
            }
            if (max_idx != i) {
                const temp = candidates[i];
                candidates[i] = candidates[max_idx];
                candidates[max_idx] = temp;
            }
        }

        var cumsum: f32 = 0;
        for (0..actual_top_k) |i| {
            cumsum += candidates[i][1];
        }

        const rand_val = self.rng.random().float(f32) * cumsum;
        cumsum = 0;
        for (0..actual_top_k) |i| {
            cumsum += candidates[i][1];
            if (rand_val <= cumsum) {
                return @intFromFloat(candidates[i][0]);
            }
        }

        return @intFromFloat(candidates[actual_top_k - 1][0]);
    }

    pub fn encode(
        self: *const TransformerModel,
        allocator: std.mem.Allocator,
        input: []const u8,
    ) ![]u32 {
        if (input.len == 0) return error.EmptyInput;

        var list = std.ArrayListUnmanaged(u32).empty;
        errdefer list.deinit(allocator);

        var it = std.mem.tokenizeAny(u8, input, " \t\r\n");
        while (it.next()) |token| {
            if (list.items.len >= self.config.max_tokens) break;
            const id = hashToken(self.config.seed, self.config.vocab_size, token);
            try list.append(allocator, id);
        }
        if (list.items.len == 0) return error.EmptyInput;
        return list.toOwnedSlice(allocator);
    }

    pub fn decode(
        self: *const TransformerModel,
        tokens: []const u32,
    ) ![]u8 {
        var output = std.ArrayListUnmanaged(u8).empty;
        errdefer output.deinit(self.allocator);

        for (tokens, 0..) |token, i| {
            if (i > 0) try output.append(self.allocator, ' ');
            const decoded = decodeToken(token);
            try output.appendSlice(self.allocator, decoded);
        }
        return output.toOwnedSlice(self.allocator);
    }

    pub fn embed(
        self: *const TransformerModel,
        allocator: std.mem.Allocator,
        input: []const u8,
    ) ![]f32 {
        const tokens = try self.encode(allocator, input);
        defer allocator.free(tokens);

        const size: usize = @intCast(self.config.hidden_size);
        var embedding = try allocator.alloc(f32, size);
        @memset(embedding, 0);

        for (tokens) |token| {
            const start = @as(usize, @intCast(token % self.config.vocab_size)) * size;
            for (0..size) |i| {
                embedding[i] += self.embedding_weights[start + i];
            }
        }

        normalizeInPlace(embedding);
        return embedding;
    }
};

fn hashToken(seed: u64, vocab_size: u32, token: []const u8) u32 {
    const hash = std.hash.Wyhash.hash(seed, token);
    return @intCast(hash % vocab_size);
}

fn decodeToken(token: u32) []const u8 {
    if (token == 0) return "<EOS>";
    if (token == 1) return "<UNK>";
    if (token < 256) {
        const buf: [1]u8 = .{@as(u8, @intCast(token))};
        return &buf;
    }
    return "<token>";
}

fn normalizeInPlace(values: []f32) void {
    const norm = simd.vectorL2Norm(values);
    if (norm == 0) return;
    for (values) |*value| {
        value.* /= norm;
    }
}

test "transformer encode and decode" {
    var rng = std.Random.DefaultPrng.init(12345);
    const allocator = std.testing.allocator;

    var model = try TransformerModel.init(allocator, .{
        .layers = 2,
        .hidden_size = 64,
        .num_heads = 4,
        .vocab_size = 512,
        .max_tokens = 16,
        .seed = rng.seed(),
    });
    defer model.deinit();

    const tokens = try model.encode(allocator, "hello world from abi");
    defer allocator.free(tokens);
    try std.testing.expect(tokens.len > 0);
    try std.testing.expect(tokens.len <= 16);

    const decoded = try model.decode(allocator, tokens);
    defer allocator.free(decoded);
    try std.testing.expect(decoded.len > 0);
}

test "transformer embeddings are normalized" {
    const allocator = std.testing.allocator;

    var model = try TransformerModel.init(allocator, .{
        .hidden_size = 32,
        .vocab_size = 256,
        .seed = 54321,
    });
    defer model.deinit();

    const embedding = try model.embed(allocator, "hello world");
    defer allocator.free(embedding);
    try std.testing.expectEqual(@as(usize, 32), embedding.len);
    const norm = simd.vectorL2Norm(embedding);
    try std.testing.expect(std.math.approxEqAbs(f32, norm, 1.0, 0.01));
}

test "transformer rejects invalid configuration" {
    const allocator = std.testing.allocator;

    try std.testing.expectError(
        TransformerError.InvalidConfiguration,
        TransformerModel.init(allocator, .{ .layers = 0 }),
    );

    try std.testing.expectError(
        TransformerError.InvalidConfiguration,
        TransformerModel.init(allocator, .{ .hidden_size = 0 }),
    );

    try std.testing.expectError(
        TransformerError.InvalidConfiguration,
        TransformerModel.init(allocator, .{ .num_heads = 0 }),
    );
}

test "transformer rejects empty input" {
    const allocator = std.testing.allocator;

    var model = try TransformerModel.init(allocator, .{});
    defer model.deinit();

    try std.testing.expectError(
        TransformerError.EmptyInput,
        model.encode(allocator, "   "),
    );
}

test "transformer generate tokens" {
    const allocator = std.testing.allocator;

    var model = try TransformerModel.init(allocator, .{
        .layers = 2,
        .hidden_size = 64,
        .num_heads = 4,
        .vocab_size = 512,
        .max_tokens = 8,
        .seed = 11111,
    });
    defer model.deinit();

    const prompt = try model.encode(allocator, "hello");
    defer allocator.free(prompt);

    const generated = try model.generate(allocator, prompt, 4);
    defer allocator.free(generated);

    try std.testing.expect(generated.len > 0);
    try std.testing.expect(generated.len <= 4);
}

test "transformer inference" {
    const allocator = std.testing.allocator;

    var model = try TransformerModel.init(allocator, .{
        .layers = 2,
        .hidden_size = 64,
        .num_heads = 4,
        .vocab_size = 512,
        .max_tokens = 8,
        .seed = 22222,
    });
    defer model.deinit();

    const output = try model.infer(allocator, "test input", 16);
    defer allocator.free(output);

    try std.testing.expect(output.len > 0);
}
