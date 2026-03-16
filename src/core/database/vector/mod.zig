//! Owns semantic retrieval.

const std = @import("std");
const core = @import("../core/mod.zig");

pub const VectorConfig = struct {
    dimensions: u32,
    quantize: bool,
};

pub const VectorIndex = struct {
    allocator: std.mem.Allocator,
    config: VectorConfig,
    embeddings: std.ArrayListUnmanaged(f32),

    pub fn init(allocator: std.mem.Allocator, config: VectorConfig) VectorIndex {
        return .{
            .allocator = allocator,
            .config = config,
            .embeddings = .empty,
        };
    }

    pub fn deinit(self: *VectorIndex) void {
        self.embeddings.deinit(self.allocator);
    }

    pub fn addEmbedding(self: *VectorIndex, id: core.ids.BlockId, vector: []const f32) !void {
        _ = id;
        if (vector.len != self.config.dimensions) return error.VectorDimensionMismatch;
        try self.embeddings.appendSlice(self.allocator, vector);
    }
};
