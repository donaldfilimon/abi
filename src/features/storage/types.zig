const std = @import("std");
const core_config = @import("../../core/config/platform.zig");

pub const StorageConfig = core_config.StorageConfig;
pub const StorageBackend = core_config.StorageBackend;

/// Errors returned by storage operations.
pub const StorageError = error{
    FeatureDisabled,
    ObjectNotFound,
    BucketNotFound,
    PermissionDenied,
    StorageFull,
    OutOfMemory,
    /// Key contains path traversal sequences (e.g. `../`).
    InvalidKey,
    BackendNotAvailable,
};

/// Metadata descriptor for a stored object (key, size, MIME type, mtime).
pub const StorageObject = struct {
    key: []const u8 = "",
    size: u64 = 0,
    content_type: []const u8 = "application/octet-stream",
    last_modified: u64 = 0,
};

/// Optional per-object metadata (content type and up to 4 custom key-value pairs).
pub const ObjectMetadata = struct {
    content_type: []const u8 = "application/octet-stream",
    custom: [4]MetadataEntry = [_]MetadataEntry{.{}} ** 4,
    custom_count: u8 = 0,

    pub const MetadataEntry = struct {
        key: []const u8 = "",
        value: []const u8 = "",
    };
};

/// Aggregate statistics for the storage backend.
pub const StorageStats = struct {
    total_objects: u64 = 0,
    total_bytes: u64 = 0,
    backend: StorageBackend = .memory,
};
