//! Shared types for the analytics feature.
//!
//! Both `mod.zig` (real implementation) and `stub.zig` (disabled no-op)
//! import from here so that type definitions are not duplicated.

const std = @import("std");

// ============================================================================
// Event Types
// ============================================================================

/// A single analytics event.
pub const Event = struct {
    name: []const u8,
    timestamp_ms: u64,
    session_id: ?[]const u8 = null,
    properties: []const Property = &.{},

    pub const Property = struct {
        key: []const u8,
        value: Value,
    };

    pub const Value = union(enum) {
        string: []const u8,
        int: i64,
        float: f64,
        boolean: bool,
    };
};

/// Configuration for the analytics engine.
pub const AnalyticsConfig = struct {
    /// Maximum events buffered before auto-flush.
    buffer_capacity: u32 = 1024,
    /// Whether to include timestamps on events.
    enable_timestamps: bool = true,
    /// Application or service identifier.
    app_id: []const u8 = "abi-app",
    /// Flush interval hint in milliseconds (0 = manual flush only).
    flush_interval_ms: u64 = 0,
};

// ============================================================================
// Error Set
// ============================================================================

pub const AnalyticsError = error{
    BufferFull,
    InvalidEvent,
    FlushFailed,
    AnalyticsDisabled,
    FeatureDisabled,
    OutOfMemory,
};

// ============================================================================
// Engine Types
// ============================================================================

pub const Stats = struct {
    buffered_events: usize = 0,
    total_events: u64 = 0,
    total_sessions: u64 = 0,
};

pub const StoredEvent = struct {
    name: []const u8,
    timestamp_ms: u64,
    session_id: ?[]const u8,
};

// ============================================================================
// Funnel Step
// ============================================================================

pub const FunnelStep = struct {
    name: []const u8,
    count: std.atomic.Value(u64) = std.atomic.Value(u64).init(0),
};
