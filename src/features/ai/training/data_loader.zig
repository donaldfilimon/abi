//! Data loading infrastructure for LLM training.
//!
//! Provides:
//! - DataLoader: Batched data iteration with shuffling
//! - TokenizedDataset: Memory-mapped tokenized data
//! - SequencePacker: Pack variable-length sequences efficiently
//! - InstructionDataset: Alpaca/ShareGPT format parsing

const std = @import("std");
const time = @import("../../../services/shared/time.zig");

fn initIoBackend(allocator: std.mem.Allocator) std.Io.Threaded {
    return std.Io.Threaded.init(allocator, .{ .environ = std.process.Environ.empty });
}

/// A batch of training data.
pub const Batch = struct {
    /// Input token IDs [batch_size * seq_len]
    input_ids: []const u32,
    /// Label token IDs (shifted input) [batch_size * seq_len]
    labels: []const u32,
    /// Attention mask (1 = attend, 0 = ignore) [batch_size * seq_len]
    attention_mask: ?[]const u8,
    /// Batch size
    batch_size: u32,
    /// Sequence length
    seq_len: u32,
};

/// Iterator over batches.
pub const BatchIterator = struct {
    dataset: *const TokenizedDataset,
    batch_size: u32,
    seq_len: u32,
    current_idx: usize,
    shuffle_indices: ?[]usize,
    allocator: std.mem.Allocator,

    // Batch buffers
    input_buffer: []u32,
    label_buffer: []u32,

    pub fn init(
        allocator: std.mem.Allocator,
        dataset: *const TokenizedDataset,
        batch_size: u32,
        seq_len: u32,
        shuffle: bool,
    ) !BatchIterator {
        const batch_tokens = @as(usize, batch_size) * seq_len;
        const input_buffer = try allocator.alloc(u32, batch_tokens);
        errdefer allocator.free(input_buffer);
        const label_buffer = try allocator.alloc(u32, batch_tokens);
        errdefer allocator.free(label_buffer);

        var shuffle_indices: ?[]usize = null;
        if (shuffle) {
            const num_batches = dataset.numBatches(batch_size, seq_len);
            shuffle_indices = try allocator.alloc(usize, num_batches);
            for (shuffle_indices.?, 0..) |*idx, i| {
                idx.* = i;
            }
            // Shuffle using Fisher-Yates
            const seed = blk: {
                var timer = time.Timer.start() catch break :blk @as(u64, 0);
                break :blk timer.read();
            };
            var rng = std.Random.DefaultPrng.init(seed);
            rng.random().shuffle(usize, shuffle_indices.?);
        }

        return .{
            .dataset = dataset,
            .batch_size = batch_size,
            .seq_len = seq_len,
            .current_idx = 0,
            .shuffle_indices = shuffle_indices,
            .allocator = allocator,
            .input_buffer = input_buffer,
            .label_buffer = label_buffer,
        };
    }

    pub fn deinit(self: *BatchIterator) void {
        self.allocator.free(self.label_buffer);
        self.allocator.free(self.input_buffer);
        if (self.shuffle_indices) |indices| {
            self.allocator.free(indices);
        }
        self.* = undefined;
    }

    pub fn next(self: *BatchIterator) ?Batch {
        const num_batches = self.dataset.numBatches(self.batch_size, self.seq_len);
        if (self.current_idx >= num_batches) return null;

        const batch_idx = if (self.shuffle_indices) |indices|
            indices[self.current_idx]
        else
            self.current_idx;

        const batch_tokens = @as(usize, self.batch_size) * self.seq_len;
        const start = batch_idx * batch_tokens;

        // Copy data to buffers
        const data = self.dataset.data;
        if (start + batch_tokens + 1 > data.len) {
            self.current_idx += 1;
            return self.next();
        }

        @memcpy(self.input_buffer, data[start..][0..batch_tokens]);
        @memcpy(self.label_buffer, data[start + 1 ..][0..batch_tokens]);

        self.current_idx += 1;

        return Batch{
            .input_ids = self.input_buffer,
            .labels = self.label_buffer,
            .attention_mask = null,
            .batch_size = self.batch_size,
            .seq_len = self.seq_len,
        };
    }

    pub fn reset(self: *BatchIterator) void {
        self.current_idx = 0;
        // Reshuffle if needed
        if (self.shuffle_indices) |indices| {
            const seed = blk: {
                var timer = time.Timer.start() catch break :blk @as(u64, 0);
                break :blk timer.read();
            };
            var rng = std.Random.DefaultPrng.init(seed);
            rng.random().shuffle(usize, indices);
        }
    }

    pub fn numBatches(self: *const BatchIterator) usize {
        return self.dataset.numBatches(self.batch_size, self.seq_len);
    }
};

/// Pre-tokenized dataset loaded from binary file.
pub const TokenizedDataset = struct {
    allocator: std.mem.Allocator,
    data: []const u32,
    owns_data: bool,

    /// Load tokenized data from a binary file.
    /// Format: raw u32 token IDs in little-endian.
    pub fn load(allocator: std.mem.Allocator, path: []const u8) !TokenizedDataset {
        var io_backend = initIoBackend(allocator);
        defer io_backend.deinit();
        const io = io_backend.io();

        const file = std.Io.Dir.cwd().openFile(io, path, .{}) catch |err| {
            std.log.warn("Failed to open dataset: {s}: {t}", .{ path, err });
            return error.FileNotFound;
        };
        defer file.close(io);

        const stat = try file.stat(io);
        const size = stat.size;

        if (size % 4 != 0) return error.InvalidFormat;

        const num_tokens = size / 4;
        const buffer = try allocator.alloc(u32, num_tokens);
        errdefer allocator.free(buffer);

        const bytes = std.mem.sliceAsBytes(buffer);
        var reader = file.reader(io);
        const read = try reader.readAll(bytes);
        if (read != size) return error.ReadError;

        return .{
            .allocator = allocator,
            .data = buffer,
            .owns_data = true,
        };
    }

    /// Create dataset from existing token slice.
    pub fn fromSlice(allocator: std.mem.Allocator, data: []const u32) TokenizedDataset {
        return .{
            .allocator = allocator,
            .data = data,
            .owns_data = false,
        };
    }

    pub fn deinit(self: *TokenizedDataset) void {
        if (self.owns_data) {
            self.allocator.free(@constCast(self.data));
        }
        self.* = undefined;
    }

    /// Get number of tokens.
    pub fn len(self: *const TokenizedDataset) usize {
        return self.data.len;
    }

    /// Calculate number of batches.
    pub fn numBatches(self: *const TokenizedDataset, batch_size: u32, seq_len: u32) usize {
        const batch_tokens = @as(usize, batch_size) * seq_len;
        if (self.data.len <= batch_tokens) return 0;
        return (self.data.len - 1) / batch_tokens; // -1 for labels
    }

    /// Create batch iterator.
    pub fn batches(
        self: *const TokenizedDataset,
        allocator: std.mem.Allocator,
        batch_size: u32,
        seq_len: u32,
        shuffle: bool,
    ) !BatchIterator {
        return BatchIterator.init(allocator, self, batch_size, seq_len, shuffle);
    }
};

/// DataLoader wraps dataset with batching and shuffling.
pub const DataLoader = struct {
    allocator: std.mem.Allocator,
    dataset: TokenizedDataset,
    batch_size: u32,
    seq_len: u32,
    shuffle: bool,
    drop_last: bool,

    pub const Config = struct {
        batch_size: u32 = 4,
        seq_len: u32 = 512,
        shuffle: bool = true,
        drop_last: bool = true,
    };

    pub fn init(allocator: std.mem.Allocator, dataset: TokenizedDataset, config: Config) DataLoader {
        return .{
            .allocator = allocator,
            .dataset = dataset,
            .batch_size = config.batch_size,
            .seq_len = config.seq_len,
            .shuffle = config.shuffle,
            .drop_last = config.drop_last,
        };
    }

    pub fn deinit(self: *DataLoader) void {
        self.dataset.deinit();
        self.* = undefined;
    }

    /// Get batch iterator.
    pub fn iterator(self: *const DataLoader) !BatchIterator {
        return self.dataset.batches(self.allocator, self.batch_size, self.seq_len, self.shuffle);
    }

    /// Number of batches per epoch.
    pub fn numBatches(self: *const DataLoader) usize {
        return self.dataset.numBatches(self.batch_size, self.seq_len);
    }

    /// Total tokens in dataset.
    pub fn numTokens(self: *const DataLoader) usize {
        return self.dataset.len();
    }
};

/// Sequence packer for efficient variable-length batching.
pub const SequencePacker = struct {
    allocator: std.mem.Allocator,
    max_seq_len: u32,
    pad_token_id: u32,
    buffer: std.ArrayListUnmanaged(u32),
    lengths: std.ArrayListUnmanaged(u32),

    pub fn init(allocator: std.mem.Allocator, max_seq_len: u32, pad_token_id: u32) SequencePacker {
        return .{
            .allocator = allocator,
            .max_seq_len = max_seq_len,
            .pad_token_id = pad_token_id,
            .buffer = std.ArrayListUnmanaged(u32).empty,
            .lengths = std.ArrayListUnmanaged(u32).empty,
        };
    }

    pub fn deinit(self: *SequencePacker) void {
        self.lengths.deinit(self.allocator);
        self.buffer.deinit(self.allocator);
        self.* = undefined;
    }

    /// Add a sequence to the packer.
    pub fn addSequence(self: *SequencePacker, tokens: []const u32) !void {
        const seq_len = @min(@as(u32, @intCast(tokens.len)), self.max_seq_len);
        try self.buffer.appendSlice(self.allocator, tokens[0..seq_len]);
        try self.lengths.append(self.allocator, seq_len);
    }

    /// Pack all sequences into fixed-length batches.
    /// Returns packed token IDs and attention masks.
    pub fn pack(self: *SequencePacker, batch_size: u32) !PackedBatch {
        const num_seqs = self.lengths.items.len;
        const num_batches = (num_seqs + batch_size - 1) / batch_size;
        const total_tokens = num_batches * batch_size * self.max_seq_len;

        var tokens = try self.allocator.alloc(u32, total_tokens);
        errdefer self.allocator.free(tokens);
        var mask = try self.allocator.alloc(u8, total_tokens);
        errdefer self.allocator.free(mask);

        // Initialize with padding
        @memset(tokens, self.pad_token_id);
        @memset(mask, 0);

        // Copy sequences
        var src_offset: usize = 0;
        for (self.lengths.items, 0..) |seq_len, i| {
            const batch_idx = i / batch_size;
            const seq_idx = i % batch_size;
            const dst_offset = (batch_idx * batch_size + seq_idx) * self.max_seq_len;

            const src = self.buffer.items[src_offset..][0..seq_len];
            @memcpy(tokens[dst_offset..][0..seq_len], src);
            @memset(mask[dst_offset..][0..seq_len], 1);

            src_offset += seq_len;
        }

        return .{
            .allocator = self.allocator,
            .tokens = tokens,
            .attention_mask = mask,
            .batch_size = batch_size,
            .seq_len = self.max_seq_len,
            .num_batches = @intCast(num_batches),
        };
    }

    pub const PackedBatch = struct {
        allocator: std.mem.Allocator,
        tokens: []u32,
        attention_mask: []u8,
        batch_size: u32,
        seq_len: u32,
        num_batches: u32,

        pub fn deinit(self: *PackedBatch) void {
            self.allocator.free(self.attention_mask);
            self.allocator.free(self.tokens);
            self.* = undefined;
        }

        /// Get batch at index.
        pub fn getBatch(self: *const PackedBatch, batch_idx: u32) Batch {
            const batch_tokens = @as(usize, self.batch_size) * self.seq_len;
            const start = @as(usize, batch_idx) * batch_tokens;

            return .{
                .input_ids = self.tokens[start..][0..batch_tokens],
                .labels = self.tokens[start + 1 ..][0..batch_tokens],
                .attention_mask = self.attention_mask[start..][0..batch_tokens],
                .batch_size = self.batch_size,
                .seq_len = self.seq_len,
            };
        }
    };
};

/// Instruction tuning sample (Alpaca format).
pub const InstructionSample = struct {
    instruction: []const u8,
    input: ?[]const u8,
    output: []const u8,
};

/// Parse JSONL instruction tuning dataset.
pub fn parseInstructionDataset(
    allocator: std.mem.Allocator,
    jsonl_data: []const u8,
) !std.ArrayListUnmanaged(InstructionSample) {
    var samples = std.ArrayListUnmanaged(InstructionSample).empty;
    errdefer samples.deinit(allocator);

    var lines = std.mem.splitScalar(u8, jsonl_data, '\n');
    while (lines.next()) |line| {
        if (line.len == 0) continue;

        // Parse JSON line
        const parsed = std.json.parseFromSlice(
            struct {
                instruction: []const u8,
                input: ?[]const u8 = null,
                output: []const u8,
            },
            allocator,
            line,
            .{},
        ) catch continue;

        try samples.append(allocator, .{
            .instruction = parsed.value.instruction,
            .input = parsed.value.input,
            .output = parsed.value.output,
        });
    }

    return samples;
}

test "tokenized dataset from slice" {
    const allocator = std.testing.allocator;

    const data = [_]u32{ 1, 2, 3, 4, 5, 6, 7, 8, 9, 10 };
    var dataset = TokenizedDataset.fromSlice(allocator, &data);
    defer dataset.deinit();

    try std.testing.expectEqual(@as(usize, 10), dataset.len());
    try std.testing.expectEqual(@as(usize, 2), dataset.numBatches(2, 2));
}

test "batch iterator" {
    const allocator = std.testing.allocator;

    const data = [_]u32{ 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13 };
    var dataset = TokenizedDataset.fromSlice(allocator, &data);
    defer dataset.deinit();

    var iter = try dataset.batches(allocator, 2, 3, false);
    defer iter.deinit();

    var batch_count: usize = 0;
    while (iter.next()) |batch| {
        try std.testing.expectEqual(@as(u32, 2), batch.batch_size);
        try std.testing.expectEqual(@as(u32, 3), batch.seq_len);
        batch_count += 1;
    }

    try std.testing.expectEqual(@as(usize, 2), batch_count);
}

test "sequence packer" {
    const allocator = std.testing.allocator;

    var packer = SequencePacker.init(allocator, 8, 0);
    defer packer.deinit();

    try packer.addSequence(&[_]u32{ 1, 2, 3 });
    try packer.addSequence(&[_]u32{ 4, 5, 6, 7, 8 });
    try packer.addSequence(&[_]u32{ 9, 10 });

    var packed_batch = try packer.pack(2);
    defer packed_batch.deinit();

    try std.testing.expectEqual(@as(u32, 2), packed_batch.num_batches);
}

test {
    std.testing.refAllDecls(@This());
}
