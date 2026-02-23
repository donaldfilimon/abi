//! Model Management Module
//!
//! Provides model download, caching, and management functionality similar to `ollama pull`.
//! Supports downloading GGUF models from HuggingFace and direct URLs.
//!
//! ## Features
//!
//! - **Download**: Download models from HuggingFace or direct URLs with progress
//! - **Cache Management**: Automatic caching in standard directories
//! - **Model Catalog**: List, search, and remove cached models
//! - **HuggingFace Integration**: Search and browse HuggingFace model hub
//! - **Resume Support**: Resume interrupted downloads via HTTP Range headers
//!
//! ## Standard Cache Directories
//!
//! Models are cached in platform-specific directories:
//! - Linux/macOS: `~/.abi/models/`
//! - Windows: `%LOCALAPPDATA%\abi\models\`
//!
//! ## Usage
//!
//! ```zig
//! const models = @import("abi").ai.models;
//!
//! // Initialize manager
//! var manager = try models.Manager.init(allocator, .{});
//! defer manager.deinit();
//!
//! // List cached models
//! const cached = manager.listModels();
//! for (cached) |model| {
//!     std.debug.print("{s}: {d} MB\n", .{model.name, model.size_bytes / 1024 / 1024});
//! }
//!
//! // Download a model from HuggingFace
//! try manager.download("TheBloke/Llama-2-7B-GGUF:Q4_K_M", .{
//!     .progress_callback = progressFn,
//! });
//! ```

const std = @import("std");
const builtin = @import("builtin");
const build_options = @import("build_options");

// Re-export submodules
pub const manager = @import("manager.zig");
pub const downloader = @import("downloader.zig");
pub const huggingface = @import("huggingface.zig");

// Re-export main types
pub const Manager = manager.Manager;
pub const ManagerConfig = manager.ManagerConfig;
pub const CachedModel = manager.CachedModel;

pub const Downloader = downloader.Downloader;
pub const DownloadConfig = downloader.DownloadConfig;
pub const DownloadProgress = downloader.DownloadProgress;
pub const DownloadResult = downloader.DownloadResult;
pub const DownloadError = downloader.DownloadError;

pub const HuggingFaceClient = huggingface.HuggingFaceClient;
pub const HuggingFaceModel = huggingface.HuggingFaceModel;
pub const HuggingFaceFile = huggingface.HuggingFaceFile;
pub const SearchResult = huggingface.SearchResult;

/// Check if model management is enabled at compile time.
pub fn isEnabled() bool {
    return build_options.enable_ai;
}

/// Error types for model management operations.
pub const Error = error{
    /// Model management feature is disabled
    ModelsDisabled,
    /// Model not found in cache
    ModelNotFound,
    /// Download failed
    DownloadFailed,
    /// Invalid model identifier
    InvalidModelId,
    /// Cache directory not accessible
    CacheNotAccessible,
    /// Network error
    NetworkError,
    /// File system error
    FileSystemError,
    /// HuggingFace API error
    HuggingFaceError,
    /// Out of memory
    OutOfMemory,
};

// ============================================================================
// Tests
// ============================================================================

test "models module compiles" {
    if (!isEnabled()) return;
    try std.testing.expect(true);
}

test "manager types available" {
    if (!isEnabled()) return;
    _ = Manager;
    _ = ManagerConfig;
    _ = CachedModel;
}

test "downloader types available" {
    if (!isEnabled()) return;
    _ = Downloader;
    _ = DownloadConfig;
    _ = DownloadProgress;
}

test "huggingface types available" {
    if (!isEnabled()) return;
    _ = HuggingFaceClient;
    _ = HuggingFaceModel;
    _ = SearchResult;
}

test {
    std.testing.refAllDecls(@This());
}
