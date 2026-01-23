//! Batch generation for processing multiple prompts.

const std = @import("std");
const generator_mod = @import("generator.zig");
const sampler = @import("sampler.zig");

/// Batch generation configuration.
pub const BatchConfig = struct {
    /// Maximum batch size
    max_batch_size: u32 = 8,
    /// Generation config per sequence
    gen_config: generator_mod.GeneratorConfig = .{},
    /// Enable continuous batching
    continuous_batching: bool = true,
};

/// Single sequence in a batch.
pub const BatchSequence = struct {
    id: u32,
    prompt_tokens: []const u32,
    generated_tokens: std.ArrayListUnmanaged(u32),
    position: u32,
    finished: bool,
    finish_reason: ?FinishReason,

    pub const FinishReason = enum {
        max_tokens,
        stop_token,
        error_,
    };
};

/// Batch generator for multiple sequences.
pub const BatchGenerator = struct {
    allocator: std.mem.Allocator,
    config: BatchConfig,
    sequences: std.ArrayListUnmanaged(BatchSequence),
    next_id: u32,

    pub fn init(allocator: std.mem.Allocator, config: BatchConfig) BatchGenerator {
        return .{
            .allocator = allocator,
            .config = config,
            .sequences = std.ArrayListUnmanaged(BatchSequence).empty,
            .next_id = 0,
        };
    }

    pub fn deinit(self: *BatchGenerator) void {
        for (self.sequences.items) |*seq| {
            seq.generated_tokens.deinit(self.allocator);
        }
        self.sequences.deinit(self.allocator);
        self.* = undefined;
    }

    /// Add a sequence to the batch.
    pub fn addSequence(self: *BatchGenerator, prompt_tokens: []const u32) !u32 {
        if (self.sequences.items.len >= self.config.max_batch_size) {
            return error.BatchFull;
        }

        const id = self.next_id;
        self.next_id += 1;

        try self.sequences.append(self.allocator, .{
            .id = id,
            .prompt_tokens = prompt_tokens,
            .generated_tokens = std.ArrayListUnmanaged(u32).empty,
            .position = 0,
            .finished = false,
            .finish_reason = null,
        });

        return id;
    }

    /// Remove a finished sequence.
    pub fn removeSequence(self: *BatchGenerator, id: u32) void {
        for (self.sequences.items, 0..) |seq, i| {
            if (seq.id == id) {
                var removed = self.sequences.orderedRemove(i);
                removed.generated_tokens.deinit(self.allocator);
                return;
            }
        }
    }

    /// Get active (non-finished) sequence count.
    pub fn activeCount(self: *const BatchGenerator) u32 {
        var count: u32 = 0;
        for (self.sequences.items) |seq| {
            if (!seq.finished) count += 1;
        }
        return count;
    }

    /// Check if batch is empty.
    pub fn isEmpty(self: *const BatchGenerator) bool {
        return self.sequences.items.len == 0;
    }

    /// Check if batch is full.
    pub fn isFull(self: *const BatchGenerator) bool {
        return self.sequences.items.len >= self.config.max_batch_size;
    }

    /// Step all sequences forward by one token.
    pub fn step(self: *BatchGenerator, logits_batch: []const []f32, samplers: []sampler.Sampler) !void {
        for (self.sequences.items, 0..) |*seq, i| {
            if (seq.finished) continue;

            const logits = @constCast(logits_batch[i]);
            const next_token = samplers[i].sample(logits);

            // Check for stop token
            for (self.config.gen_config.stop_tokens) |stop| {
                if (next_token == stop) {
                    seq.finished = true;
                    seq.finish_reason = .stop_token;
                    break;
                }
            }

            if (!seq.finished) {
                try seq.generated_tokens.append(self.allocator, next_token);
                seq.position += 1;

                // Check max tokens
                if (seq.generated_tokens.items.len >= self.config.gen_config.max_tokens) {
                    seq.finished = true;
                    seq.finish_reason = .max_tokens;
                }
            }
        }
    }

    /// Get sequence by ID.
    pub fn getSequence(self: *const BatchGenerator, id: u32) ?*const BatchSequence {
        for (self.sequences.items) |*seq| {
            if (seq.id == id) return seq;
        }
        return null;
    }

    /// Get all generated tokens for a sequence.
    pub fn getGeneratedTokens(self: *const BatchGenerator, id: u32) ?[]const u32 {
        const seq = self.getSequence(id) orelse return null;
        return seq.generated_tokens.items;
    }

    /// Check if a sequence is finished.
    pub fn isFinished(self: *const BatchGenerator, id: u32) bool {
        const seq = self.getSequence(id) orelse return true;
        return seq.finished;
    }
};

/// Results from batch generation.
pub const BatchResults = struct {
    results: []generator_mod.GenerationResult,
    allocator: std.mem.Allocator,

    pub fn deinit(self: *BatchResults) void {
        for (self.results) |*r| {
            r.deinit(self.allocator);
        }
        self.allocator.free(self.results);
        self.* = undefined;
    }
};

test "batch generator basic" {
    const allocator = std.testing.allocator;

    var gen = BatchGenerator.init(allocator, .{ .max_batch_size = 4 });
    defer gen.deinit();

    try std.testing.expect(gen.isEmpty());
    try std.testing.expect(!gen.isFull());

    const prompt = [_]u32{ 1, 2, 3 };
    const id = try gen.addSequence(&prompt);

    try std.testing.expectEqual(@as(u32, 0), id);
    try std.testing.expect(!gen.isEmpty());
    try std.testing.expectEqual(@as(u32, 1), gen.activeCount());
}

test "batch capacity" {
    const allocator = std.testing.allocator;

    var gen = BatchGenerator.init(allocator, .{ .max_batch_size = 2 });
    defer gen.deinit();

    const prompt = [_]u32{1};
    _ = try gen.addSequence(&prompt);
    _ = try gen.addSequence(&prompt);

    try std.testing.expect(gen.isFull());
    try std.testing.expectError(error.BatchFull, gen.addSequence(&prompt));
}
