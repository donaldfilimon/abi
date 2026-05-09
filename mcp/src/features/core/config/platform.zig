//! Platform Configuration
//!
//! Consolidated configuration for platform infrastructure:
//! messaging, auth, cache, storage, and mobile.

// ── Messaging ──────────────────────────────────────────────────────────

pub const MessagingConfig = struct {
    max_channels: u32 = 256,
    buffer_size: u32 = 4096,
    enable_persistence: bool = false,

    pub fn defaults() MessagingConfig {
        return .{};
    }
};

// ── Auth ───────────────────────────────────────────────────────────────

pub const AuthConfig = struct {
    jwt_secret: ?[]const u8 = null,
    session_timeout_ms: u64 = 3600_000,
    max_api_keys: u32 = 100,
    enable_rbac: bool = false,
    enable_rate_limit: bool = false,

    pub fn defaults() AuthConfig {
        return .{};
    }
};

// ── Cache ──────────────────────────────────────────────────────────────

pub const EvictionPolicy = enum { lru, lfu, fifo, random };

pub const CacheConfig = struct {
    max_entries: u32 = 10_000,
    max_memory_mb: u32 = 256,
    default_ttl_ms: u64 = 300_000,
    eviction_policy: EvictionPolicy = .lru,

    pub fn defaults() CacheConfig {
        return .{};
    }
};

// ── Storage ────────────────────────────────────────────────────────────

pub const StorageBackend = enum { memory, local, s3, gcs };

pub const StorageConfig = struct {
    backend: StorageBackend = .memory,
    base_path: []const u8 = "./storage",
    max_object_size_mb: u32 = 1024,

    pub fn defaults() StorageConfig {
        return .{};
    }
};

// ── Mobile ─────────────────────────────────────────────────────────────

/// Mobile module configuration.
pub const MobileConfig = struct {
    platform: Platform = .auto,
    enable_sensors: bool = false,
    enable_notifications: bool = false,

    pub const Platform = enum { auto, ios, android };

    pub fn defaults() MobileConfig {
        return .{};
    }
};
