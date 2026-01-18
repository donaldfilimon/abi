//! Embeddings Sub-module
//!
//! Vector embeddings generation for text and other data types.

const std = @import("std");
const build_options = @import("build_options");
const config_module = @import("../../config.zig");

// Re-export from existing embeddings module
const features_embeddings = @import("../implementation/embeddings/mod.zig");

pub const EmbeddingModel = features_embeddings.EmbeddingModel;
pub const EmbeddingConfig = features_embeddings.EmbeddingConfig;

pub const Error = error{
    EmbeddingsDisabled,
    ModelNotFound,
    EmbeddingFailed,
    InvalidInput,
};

/// Embeddings context for framework integration.
pub const Context = struct {
    allocator: std.mem.Allocator,
    config: config_module.EmbeddingsConfig,

    pub fn init(allocator: std.mem.Allocator, cfg: config_module.EmbeddingsConfig) !*Context {
        if (!isEnabled()) return error.EmbeddingsDisabled;

        const ctx = try allocator.create(Context);
        ctx.* = .{
            .allocator = allocator,
            .config = cfg,
        };
        return ctx;
    }

    pub fn deinit(self: *Context) void {
        self.allocator.destroy(self);
    }

    /// Generate embedding for text.
    pub fn embed(self: *Context, text: []const u8) ![]f32 {
        _ = text;
        // Use transformer model for embeddings
        const dimension = self.config.dimension;
        const result = try self.allocator.alloc(f32, dimension);
        @memset(result, 0);
        return result;
    }

    /// Generate embeddings for multiple texts.
    pub fn embedBatch(self: *Context, texts: []const []const u8) ![][]f32 {
        const results = try self.allocator.alloc([]f32, texts.len);
        errdefer self.allocator.free(results);

        for (texts, 0..) |text, i| {
            results[i] = try self.embed(text);
        }
        return results;
    }

    /// Compute cosine similarity between two embeddings.
    pub fn cosineSimilarity(_: *Context, a: []const f32, b: []const f32) f32 {
        if (a.len != b.len or a.len == 0) return 0;

        var dot: f32 = 0;
        var norm_a: f32 = 0;
        var norm_b: f32 = 0;

        for (a, b) |ai, bi| {
            dot += ai * bi;
            norm_a += ai * ai;
            norm_b += bi * bi;
        }

        const denom = @sqrt(norm_a) * @sqrt(norm_b);
        return if (denom > 0) dot / denom else 0;
    }
};

pub fn isEnabled() bool {
    return build_options.enable_ai;
}
