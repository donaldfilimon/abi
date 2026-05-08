//! Content Configuration
//!
//! Consolidated configuration for content-oriented features:
//! search, pages, and analytics.

// ── Search ─────────────────────────────────────────────────────────────

pub const SearchConfig = struct {
    max_index_size_mb: u32 = 512,
    default_result_limit: u32 = 100,
    enable_stemming: bool = true,
    enable_fuzzy: bool = true,

    pub fn defaults() SearchConfig {
        return .{};
    }
};

// ── Pages ──────────────────────────────────────────────────────────────

pub const PagesConfig = struct {
    max_pages: u32 = 256,
    default_layout: []const u8 = "default",
    enable_template_cache: bool = true,
    template_cache_size: u32 = 64,
    default_cache_ttl_ms: u64 = 0, // 0 = no caching

    pub fn defaults() PagesConfig {
        return .{};
    }
};

// ── Analytics ──────────────────────────────────────────────────────────

/// Analytics engine configuration.
pub const AnalyticsConfig = struct {
    /// Maximum events buffered before auto-flush.
    buffer_capacity: u32 = 1024,
    /// Whether to include timestamps on events.
    enable_timestamps: bool = true,
    /// Application or service identifier.
    app_id: []const u8 = "abi-app",
    /// Flush interval hint in milliseconds (0 = manual flush only).
    flush_interval_ms: u64 = 0,

    /// Default configuration.
    pub fn defaults() AnalyticsConfig {
        return .{};
    }
};
