const std = @import("std");
const core_config = @import("../../core/config/content.zig");

pub const SearchConfig = core_config.SearchConfig;

/// Errors returned by search operations.
pub const SearchError = error{
    FeatureDisabled,
    IndexNotFound,
    InvalidQuery,
    IndexCorrupted,
    OutOfMemory,
    IndexAlreadyExists,
    DocumentNotFound,
    IoError,
};

/// A single search hit with BM25 relevance score and context snippet.
pub const SearchResult = struct {
    doc_id: []const u8 = "",
    /// BM25 relevance score (higher = more relevant).
    score: f32 = 0.0,
    /// Text excerpt around the best matching region.
    snippet: []const u8 = "",
};

/// Metadata about a named full-text search index.
pub const SearchIndex = struct {
    name: []const u8 = "",
    doc_count: u64 = 0,
    size_bytes: u64 = 0,
};

/// Aggregate statistics across all search indexes.
pub const SearchStats = struct {
    total_indexes: u32 = 0,
    total_documents: u64 = 0,
    total_terms: u64 = 0,
};
