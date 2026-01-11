//! Batch processing for efficient embedding generation.
//!
//! Groups texts into batches for efficient model inference,
//! with configurable batch size and parallel processing.

const std = @import("std");

/// Batch processing configuration.
pub const BatchConfig = struct {
    /// Maximum batch size.
    batch_size: usize = 32,
    /// Embedding dimension.
    dimension: u32 = 384,
    /// Whether to normalize output vectors.
    normalize: bool = true,
    /// Maximum sequence length (truncate longer inputs).
    max_seq_length: usize = 512,
    /// Pad shorter sequences.
    pad_to_max_length: bool = false,
};

/// Result of batch processing.
pub const BatchResult = struct {
    /// Batch index.
    batch_index: usize,
    /// Number of texts in this batch.
    batch_size: usize,
    /// Embeddings for this batch.
    embeddings: [][]f32,
    /// Processing time for this batch in nanoseconds.
    processing_time_ns: u64,
};

/// Batch processor for text embeddings.
pub const BatchProcessor = struct {
    allocator: std.mem.Allocator,
    config: BatchConfig,

    pub fn init(allocator: std.mem.Allocator, config: BatchConfig) BatchProcessor {
        return .{
            .allocator = allocator,
            .config = config,
        };
    }

    pub fn deinit(self: *BatchProcessor) void {
        self.* = undefined;
    }

    /// Process texts and return embeddings.
    pub fn process(self: *BatchProcessor, texts: []const []const u8) ![][]f32 {
        var results = std.ArrayListUnmanaged([]f32){};
        errdefer {
            for (results.items) |result| {
                self.allocator.free(result);
            }
            results.deinit(self.allocator);
        }

        // Process in batches
        var offset: usize = 0;
        while (offset < texts.len) {
            const end = @min(offset + self.config.batch_size, texts.len);
            const batch = texts[offset..end];

            const batch_embeddings = try self.processBatch(batch);
            defer self.allocator.free(batch_embeddings);

            for (batch_embeddings) |emb| {
                try results.append(self.allocator, emb);
            }

            offset = end;
        }

        return results.toOwnedSlice(self.allocator);
    }

    /// Process a single batch of texts.
    fn processBatch(self: *BatchProcessor, texts: []const []const u8) ![][]f32 {
        var embeddings = try self.allocator.alloc([]f32, texts.len);
        errdefer self.allocator.free(embeddings);

        for (texts, 0..) |text, i| {
            embeddings[i] = try self.generateEmbedding(text);
        }

        return embeddings;
    }

    /// Generate embedding for a single text.
    /// This is a placeholder that generates deterministic pseudo-embeddings.
    fn generateEmbedding(self: *BatchProcessor, text: []const u8) ![]f32 {
        const dim = self.config.dimension;
        var embedding = try self.allocator.alloc(f32, dim);
        errdefer self.allocator.free(embedding);

        // Generate deterministic pseudo-embedding based on text content
        // This simulates what a real embedding model would do
        const hash = std.hash.Wyhash.hash(0, text);
        var prng = std.Random.DefaultPrng.init(hash);
        const random = prng.random();

        for (embedding) |*v| {
            v.* = random.float(f32) * 2.0 - 1.0;
        }

        // Incorporate text features
        if (text.len > 0) {
            // Use text length to influence first dimension
            embedding[0] = @as(f32, @floatFromInt(@min(text.len, 1000))) / 1000.0;

            // Use first character to influence second dimension
            if (dim > 1) {
                embedding[1] = @as(f32, @floatFromInt(text[0])) / 255.0;
            }

            // Use last character to influence third dimension
            if (dim > 2) {
                embedding[2] = @as(f32, @floatFromInt(text[text.len - 1])) / 255.0;
            }
        }

        // Normalize if configured
        if (self.config.normalize) {
            normalizeVector(embedding);
        }

        return embedding;
    }

    /// Process with callback for progress reporting.
    pub fn processWithProgress(
        self: *BatchProcessor,
        texts: []const []const u8,
        callback: *const fn (usize, usize) void,
    ) ![][]f32 {
        var results = std.ArrayListUnmanaged([]f32){};
        errdefer {
            for (results.items) |result| {
                self.allocator.free(result);
            }
            results.deinit(self.allocator);
        }

        var processed: usize = 0;
        var offset: usize = 0;

        while (offset < texts.len) {
            const end = @min(offset + self.config.batch_size, texts.len);
            const batch = texts[offset..end];

            const batch_embeddings = try self.processBatch(batch);
            defer self.allocator.free(batch_embeddings);

            for (batch_embeddings) |emb| {
                try results.append(self.allocator, emb);
            }

            processed += batch.len;
            callback(processed, texts.len);

            offset = end;
        }

        return results.toOwnedSlice(self.allocator);
    }
};

/// Normalize a vector to unit length.
fn normalizeVector(vector: []f32) void {
    var norm: f32 = 0;
    for (vector) |v| {
        norm += v * v;
    }
    norm = @sqrt(norm);

    if (norm > 0) {
        for (vector) |*v| {
            v.* /= norm;
        }
    }
}

/// Compute mean vector from multiple embeddings.
pub fn meanPooling(allocator: std.mem.Allocator, embeddings: []const []const f32) ![]f32 {
    if (embeddings.len == 0) return error.EmptyInput;

    const dim = embeddings[0].len;
    const mean = try allocator.alloc(f32, dim);
    @memset(mean, 0);

    for (embeddings) |emb| {
        if (emb.len != dim) {
            allocator.free(mean);
            return error.DimensionMismatch;
        }
        for (mean, emb) |*m, e| {
            m.* += e;
        }
    }

    const count = @as(f32, @floatFromInt(embeddings.len));
    for (mean) |*m| {
        m.* /= count;
    }

    return mean;
}

/// Compute max pooling across embeddings.
pub fn maxPooling(allocator: std.mem.Allocator, embeddings: []const []const f32) ![]f32 {
    if (embeddings.len == 0) return error.EmptyInput;

    const dim = embeddings[0].len;
    const result = try allocator.alloc(f32, dim);

    // Initialize with first embedding
    @memcpy(result, embeddings[0]);

    for (embeddings[1..]) |emb| {
        if (emb.len != dim) {
            allocator.free(result);
            return error.DimensionMismatch;
        }
        for (result, emb) |*r, e| {
            r.* = @max(r.*, e);
        }
    }

    return result;
}

test "batch processor initialization" {
    const allocator = std.testing.allocator;
    var processor = BatchProcessor.init(allocator, .{});
    defer processor.deinit();

    try std.testing.expectEqual(@as(usize, 32), processor.config.batch_size);
}

test "batch processing" {
    const allocator = std.testing.allocator;
    var processor = BatchProcessor.init(allocator, .{ .dimension = 8 });
    defer processor.deinit();

    const texts = [_][]const u8{ "hello", "world", "test" };
    const embeddings = try processor.process(&texts);
    defer {
        for (embeddings) |emb| {
            allocator.free(emb);
        }
        allocator.free(embeddings);
    }

    try std.testing.expectEqual(@as(usize, 3), embeddings.len);
    try std.testing.expectEqual(@as(usize, 8), embeddings[0].len);
}

test "mean pooling" {
    const allocator = std.testing.allocator;

    const e1 = [_]f32{ 1, 2, 3 };
    const e2 = [_]f32{ 3, 4, 5 };
    const embeddings = [_][]const f32{ &e1, &e2 };

    const mean = try meanPooling(allocator, &embeddings);
    defer allocator.free(mean);

    try std.testing.expectApproxEqAbs(@as(f32, 2.0), mean[0], 0.0001);
    try std.testing.expectApproxEqAbs(@as(f32, 3.0), mean[1], 0.0001);
    try std.testing.expectApproxEqAbs(@as(f32, 4.0), mean[2], 0.0001);
}

test "normalized embeddings have unit length" {
    const allocator = std.testing.allocator;
    var processor = BatchProcessor.init(allocator, .{
        .dimension = 128,
        .normalize = true,
    });
    defer processor.deinit();

    const texts = [_][]const u8{"test text"};
    const embeddings = try processor.process(&texts);
    defer {
        for (embeddings) |emb| {
            allocator.free(emb);
        }
        allocator.free(embeddings);
    }

    // Check unit length
    var norm: f32 = 0;
    for (embeddings[0]) |v| {
        norm += v * v;
    }
    norm = @sqrt(norm);

    try std.testing.expectApproxEqAbs(@as(f32, 1.0), norm, 0.0001);
}
