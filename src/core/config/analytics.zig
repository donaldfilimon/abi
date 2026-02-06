//! Analytics Configuration
//!
//! Configuration for the analytics event tracking engine.

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
