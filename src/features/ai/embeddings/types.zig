//! Shared types for the embeddings module.
//!
//! Used by mod.zig (via backend.zig) and stub.zig to prevent type drift
//! between enabled and disabled paths.
//!
//! Source of truth: backend.zig and mod.zig definitions.

/// Backend type enumeration.
pub const BackendType = enum {
    openai,
    huggingface,
    ollama,
    local,
    custom,

    pub fn toString(self: BackendType) []const u8 {
        return @tagName(self);
    }
};

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

/// Configuration for embedding models.
pub const EmbeddingConfig = struct {
    /// Embedding dimension.
    dimension: usize = 384,
    /// Maximum sequence length.
    max_seq_len: usize = 512,
    /// Batch size for processing.
    batch_size: usize = 32,
    /// Normalize embeddings to unit length.
    normalize: bool = true,
    /// Model identifier.
    model_id: []const u8 = "default",
};
