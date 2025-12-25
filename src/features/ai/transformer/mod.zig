const std = @import("std");
const simd = @import("../../../shared/simd.zig");

pub const TransformerError = error{
    InvalidConfiguration,
    EmptyInput,
};

pub const TransformerConfig = struct {
    layers: u16 = 4,
    hidden_size: u16 = 256,
    vocab_size: u32 = 8192,
    max_tokens: u32 = 128,
    seed: u64 = 0x2a9d_7d3c_b1e5_4f03,
    temperature: f32 = 0.8,
    top_p: f32 = 0.9,

    pub fn validate(self: TransformerConfig) TransformerError!void {
        if (self.layers == 0) return TransformerError.InvalidConfiguration;
        if (self.hidden_size == 0) return TransformerError.InvalidConfiguration;
        if (self.vocab_size < 2) return TransformerError.InvalidConfiguration;
        if (self.max_tokens == 0) return TransformerError.InvalidConfiguration;
        if (self.temperature < 0 or self.temperature > 2.0) {
            return TransformerError.InvalidConfiguration;
        }
        if (self.top_p < 0 or self.top_p > 1.0) return TransformerError.InvalidConfiguration;
    }
};

pub const TransformerModel = struct {
    config: TransformerConfig,

    pub fn init(config: TransformerConfig) TransformerModel {
        return .{ .config = config };
    }

    pub fn infer(self: *TransformerModel, allocator: std.mem.Allocator, input: []const u8) ![]u8 {
        const tokens = try self.encode(allocator, input);
        defer allocator.free(tokens);

        var output = std.ArrayList(u8).empty;
        errdefer output.deinit(allocator);
        try output.print(
            allocator,
            "transformer(layers={d}, hidden={d}, tokens={d}): ",
            .{ self.config.layers, self.config.hidden_size, tokens.len },
        );
        try appendTokens(&output, allocator, tokens);
        return output.toOwnedSlice(allocator);
    }

    pub fn encode(self: *const TransformerModel, allocator: std.mem.Allocator, input: []const u8) ![]u32 {
        try self.config.validate();

        var list = std.ArrayList(u32).empty;
        errdefer list.deinit(allocator);
        var it = std.mem.tokenizeAny(u8, input, " \t\r\n");
        while (it.next()) |token| {
            if (list.items.len >= self.config.max_tokens) break;
            const id = hashToken(self.config.seed, self.config.vocab_size, token);
            try list.append(allocator, id);
        }
        if (list.items.len == 0) return TransformerError.EmptyInput;
        return list.toOwnedSlice(allocator);
    }

    pub fn decode(self: *const TransformerModel, allocator: std.mem.Allocator, tokens: []const u32) ![]u8 {
        _ = self;
        var output = std.ArrayList(u8).empty;
        errdefer output.deinit(allocator);
        try appendTokens(&output, allocator, tokens);
        return output.toOwnedSlice(allocator);
    }

    pub fn embed(self: *const TransformerModel, allocator: std.mem.Allocator, input: []const u8) ![]f32 {
        const tokens = try self.encode(allocator, input);
        defer allocator.free(tokens);

        const size: usize = @intCast(self.config.hidden_size);
        const modulus: u32 = @intCast(self.config.hidden_size);
        var embedding = try allocator.alloc(f32, size);
        @memset(embedding, 0);

        for (tokens) |token| {
            const index: usize = @intCast(token % modulus);
            embedding[index] += 1.0;
        }

        normalizeInPlace(embedding);
        return embedding;
    }
};

fn hashToken(seed: u64, vocab_size: u32, token: []const u8) u32 {
    const hash = std.hash.Wyhash.hash(seed, token);
    return @intCast(hash % vocab_size);
}

fn appendTokens(list: *std.ArrayList(u8), allocator: std.mem.Allocator, tokens: []const u32) !void {
    for (tokens, 0..) |token, i| {
        if (i > 0) try list.append(allocator, ' ');
        try list.print(allocator, "tok{d}", .{token});
    }
}

fn normalizeInPlace(values: []f32) void {
    const norm = simd.VectorOps.l2Norm(values);
    if (norm == 0) return;
    for (values) |*value| {
        value.* /= norm;
    }
}

test "transformer encode and decode" {
    var model = TransformerModel.init(.{ .vocab_size = 128, .max_tokens = 4 });
    const tokens = try model.encode(std.testing.allocator, "hello world from abi");
    defer std.testing.allocator.free(tokens);
    try std.testing.expect(tokens.len <= 4);
    try std.testing.expect(tokens.len > 0);
    for (tokens) |token| {
        try std.testing.expect(token < 128);
    }

    const decoded = try model.decode(std.testing.allocator, tokens);
    defer std.testing.allocator.free(decoded);
    try std.testing.expect(std.mem.indexOf(u8, decoded, "tok") != null);
}

test "transformer embeddings are normalized" {
    var model = TransformerModel.init(.{ .hidden_size = 8, .vocab_size = 64 });
    const embedding = try model.embed(std.testing.allocator, "hello world");
    defer std.testing.allocator.free(embedding);
    try std.testing.expectEqual(@as(usize, 8), embedding.len);
    const norm = simd.VectorOps.l2Norm(embedding);
    try std.testing.expect(std.math.approxEqAbs(f32, norm, 1.0, 0.001));
}
