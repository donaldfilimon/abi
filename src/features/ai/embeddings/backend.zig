//! Embedding Backend Interface
//!
//! Defines the common interface for embedding backends, enabling multiple
//! providers (OpenAI, HuggingFace, Ollama, Local) to be used interchangeably.

const std = @import("std");

/// Embedding backend errors.
pub const BackendError = error{
    /// Backend not initialized or unavailable.
    BackendNotAvailable,
    /// API key or credentials missing.
    MissingCredentials,
    /// Network or API request failed.
    RequestFailed,
    /// Response could not be parsed.
    InvalidResponse,
    /// Rate limit exceeded.
    RateLimitExceeded,
    /// Input exceeds token limit.
    TokenLimitExceeded,
    /// Model not found.
    ModelNotFound,
    /// Out of memory.
    OutOfMemory,
};

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

/// Backend type enumeration.
pub const BackendType = enum {
    openai,
    huggingface,
    ollama,
    local,
    custom,

    pub fn toString(self: BackendType) []const u8 {
        return std.mem.sliceTo(@tagName(self), 0);
    }
};

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

/// Backend configuration options.
pub const BackendConfig = struct {
    /// API key or token for authentication.
    api_key: ?[]const u8 = null,
    /// Base URL for API requests.
    base_url: ?[]const u8 = null,
    /// Model identifier.
    model: []const u8 = "default",
    /// Output dimensions.
    dimensions: usize = 384,
    /// Request timeout in milliseconds.
    timeout_ms: u32 = 30_000,
    /// Maximum batch size.
    max_batch_size: usize = 100,
    /// Enable caching.
    enable_cache: bool = true,
};

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
