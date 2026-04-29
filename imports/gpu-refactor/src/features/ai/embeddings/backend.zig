//! Embedding Backend Interface
//!
//! Defines the common interface for embedding backends, enabling multiple
//! providers (OpenAI, HuggingFace, Ollama, Local) to be used interchangeably.

const std = @import("std");
const types = @import("types.zig");

// Re-export shared types from types.zig (canonical definitions).
pub const BackendError = types.BackendError;
pub const BackendType = types.BackendType;
pub const BackendConfig = types.BackendConfig;

/// Function signature for single text embedding.
pub const EmbedFn = *const fn (
    ctx: *anyopaque,
    allocator: std.mem.Allocator,
    text: []const u8,
    dimensions: usize,
) BackendError![]f32;

/// Function signature for batch text embedding.
pub const EmbedBatchFn = *const fn (
    ctx: *anyopaque,
    allocator: std.mem.Allocator,
    texts: []const []const u8,
    dimensions: usize,
) BackendError![][]f32;

/// Function signature for backend cleanup.
pub const DeinitFn = *const fn (ctx: *anyopaque) void;

// BackendType is re-exported from types.zig above.

/// Embedding backend interface using vtable pattern.
/// This allows different backend implementations to be used interchangeably.
pub const EmbeddingBackend = struct {
    /// Backend implementation context.
    ptr: *anyopaque,
    /// Function to generate embedding for single text.
    embedFn: EmbedFn,
    /// Function to generate embeddings for multiple texts.
    embedBatchFn: EmbedBatchFn,
    /// Optional cleanup function.
    deinitFn: ?DeinitFn = null,
    /// Backend type identifier.
    backend_type: BackendType,
    /// Human-readable backend name.
    name: []const u8,
    /// Model identifier (e.g., "text-embedding-3-small").
    model: []const u8 = "",
    /// Default dimensions for this backend.
    default_dimensions: usize = 384,

    /// Generate embedding for a single text.
    pub fn embed(
        self: EmbeddingBackend,
        allocator: std.mem.Allocator,
        text: []const u8,
        dimensions: usize,
    ) BackendError![]f32 {
        return self.embedFn(self.ptr, allocator, text, dimensions);
    }

    /// Generate embeddings for multiple texts.
    pub fn embedBatch(
        self: EmbeddingBackend,
        allocator: std.mem.Allocator,
        texts: []const []const u8,
        dimensions: usize,
    ) BackendError![][]f32 {
        return self.embedBatchFn(self.ptr, allocator, texts, dimensions);
    }

    /// Clean up backend resources.
    pub fn deinit(self: EmbeddingBackend) void {
        if (self.deinitFn) |deinit_fn| {
            deinit_fn(self.ptr);
        }
    }
};

// BackendConfig is re-exported from types.zig above.

// ============================================================================
// Tests
// ============================================================================

test "backend type to string" {
    try std.testing.expectEqualStrings("openai", BackendType.openai.toString());
    try std.testing.expectEqualStrings("local", BackendType.local.toString());
}

test "backend config defaults" {
    const config = BackendConfig{};
    try std.testing.expectEqual(@as(usize, 384), config.dimensions);
    try std.testing.expectEqual(@as(u32, 30_000), config.timeout_ms);
    try std.testing.expect(config.enable_cache);
}

test {
    std.testing.refAllDecls(@This());
}
