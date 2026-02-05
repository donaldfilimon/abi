//! Analytics Module
//!
//! Business event tracking, session analytics, and funnel instrumentation.
//! Unlike observability (system metrics, tracing, profiling), analytics
//! focuses on user-facing events and product usage patterns.
//!
//! ## Features
//! - Custom event tracking with typed properties
//! - Session lifecycle management
//! - Funnel step recording
//! - A/B experiment assignment tracking
//! - Thread-safe event buffer with configurable flush

const std = @import("std");
const build_options = @import("build_options");

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
// Analytics Engine
// ============================================================================

pub const AnalyticsError = error{
    BufferFull,
    InvalidEvent,
    FlushFailed,
    AnalyticsDisabled,
    OutOfMemory,
};

/// Core analytics engine. Buffers events and provides batch retrieval.
pub const Engine = struct {
    allocator: std.mem.Allocator,
    config: AnalyticsConfig,
    events: std.ArrayListUnmanaged(StoredEvent) = .empty,
    session_count: std.atomic.Value(u64) = std.atomic.Value(u64).init(0),
    event_count: std.atomic.Value(u64) = std.atomic.Value(u64).init(0),
    mutex: std.Thread.Mutex = .{},

    const StoredEvent = struct {
        name: []const u8,
        timestamp_ms: u64,
        session_id: ?[]const u8,
    };

    pub fn init(allocator: std.mem.Allocator, config: AnalyticsConfig) Engine {
        return .{
            .allocator = allocator,
            .config = config,
        };
    }

    pub fn deinit(self: *Engine) void {
        self.events.deinit(self.allocator);
        self.* = undefined;
    }

    /// Track a named event.
    pub fn track(self: *Engine, name: []const u8) AnalyticsError!void {
        return self.trackWithSession(name, null);
    }

    /// Track a named event associated with a session.
    pub fn trackWithSession(self: *Engine, name: []const u8, session_id: ?[]const u8) AnalyticsError!void {
        self.mutex.lock();
        defer self.mutex.unlock();

        if (self.events.items.len >= self.config.buffer_capacity) {
            return AnalyticsError.BufferFull;
        }

        const ts: u64 = if (self.config.enable_timestamps) timestampMs() else 0;

        self.events.append(self.allocator, .{
            .name = name,
            .timestamp_ms = ts,
            .session_id = session_id,
        }) catch return AnalyticsError.OutOfMemory;

        _ = self.event_count.fetchAdd(1, .monotonic);
    }

    /// Start a new session and return its ID.
    pub fn startSession(self: *Engine) u64 {
        return self.session_count.fetchAdd(1, .monotonic);
    }

    /// Get count of buffered events.
    pub fn bufferedCount(self: *Engine) usize {
        self.mutex.lock();
        defer self.mutex.unlock();
        return self.events.items.len;
    }

    /// Get total events tracked (including flushed).
    pub fn totalEvents(self: *const Engine) u64 {
        return self.event_count.load(.monotonic);
    }

    /// Flush all buffered events, returning the count flushed.
    pub fn flush(self: *Engine) usize {
        self.mutex.lock();
        defer self.mutex.unlock();
        const count = self.events.items.len;
        self.events.clearRetainingCapacity();
        return count;
    }

    /// Get a snapshot of current stats.
    pub fn getStats(self: *Engine) Stats {
        self.mutex.lock();
        defer self.mutex.unlock();
        return .{
            .buffered_events = self.events.items.len,
            .total_events = self.event_count.load(.monotonic),
            .total_sessions = self.session_count.load(.monotonic),
        };
    }

    pub const Stats = struct {
        buffered_events: usize,
        total_events: u64,
        total_sessions: u64,
    };
};

// ============================================================================
// Funnel Tracking
// ============================================================================

/// Track progression through a named funnel.
pub const Funnel = struct {
    name: []const u8,
    steps: std.ArrayListUnmanaged(Step) = .empty,
    allocator: std.mem.Allocator,

    pub const Step = struct {
        name: []const u8,
        count: std.atomic.Value(u64) = std.atomic.Value(u64).init(0),
    };

    pub fn init(allocator: std.mem.Allocator, name: []const u8) Funnel {
        return .{ .name = name, .allocator = allocator };
    }

    pub fn deinit(self: *Funnel) void {
        self.steps.deinit(self.allocator);
        self.* = undefined;
    }

    /// Add a step to the funnel.
    pub fn addStep(self: *Funnel, step_name: []const u8) !void {
        try self.steps.append(self.allocator, .{ .name = step_name });
    }

    /// Record a user reaching a step.
    pub fn recordStep(self: *Funnel, step_index: usize) void {
        if (step_index < self.steps.items.len) {
            _ = self.steps.items[step_index].count.fetchAdd(1, .monotonic);
        }
    }

    /// Get step counts for analysis.
    pub fn getStepCounts(self: *const Funnel, buffer: []u64) []u64 {
        const len = @min(buffer.len, self.steps.items.len);
        for (0..len) |i| {
            buffer[i] = self.steps.items[i].count.load(.monotonic);
        }
        return buffer[0..len];
    }
};

// ============================================================================
// Experiment Tracking
// ============================================================================

/// Simple A/B experiment assignment.
pub const Experiment = struct {
    name: []const u8,
    variants: []const []const u8,
    assignments: std.atomic.Value(u64) = std.atomic.Value(u64).init(0),

    /// Assign a user to a variant based on a hash of their ID.
    pub fn assign(self: *Experiment, user_id: []const u8) []const u8 {
        if (self.variants.len == 0) return "control";
        _ = self.assignments.fetchAdd(1, .monotonic);
        const hash = std.hash.Fnv1a_64.hash(user_id);
        const idx = hash % self.variants.len;
        return self.variants[idx];
    }

    /// Get total assignments.
    pub fn totalAssignments(self: *const Experiment) u64 {
        return self.assignments.load(.monotonic);
    }
};

// ============================================================================
// Module Lifecycle
// ============================================================================

/// Analytics context for Framework integration.
pub const Context = struct {
    allocator: std.mem.Allocator,
    config: AnalyticsConfig,
    engine: ?Engine = null,

    pub fn init(allocator: std.mem.Allocator, config: AnalyticsConfig) !*Context {
        const ctx = try allocator.create(Context);
        ctx.* = .{
            .allocator = allocator,
            .config = config,
            .engine = Engine.init(allocator, config),
        };
        return ctx;
    }

    pub fn deinit(self: *Context) void {
        if (self.engine) |*eng| eng.deinit();
        self.allocator.destroy(self);
    }

    pub fn getEngine(self: *Context) ?*Engine {
        return if (self.engine != null) &self.engine.? else null;
    }
};

var initialized: bool = false;

pub fn init(allocator: std.mem.Allocator) !void {
    _ = allocator;
    initialized = true;
}

pub fn deinit() void {
    initialized = false;
}

pub fn isEnabled() bool {
    return build_options.enable_analytics;
}

pub fn isInitialized() bool {
    return initialized;
}

// ============================================================================
// Helpers
// ============================================================================

/// Application start anchor for monotonic timestamps.
var app_start: ?std.time.Instant = null;

fn timestampMs() u64 {
    const start = app_start orelse blk: {
        const s = std.time.Instant.now() catch return 0;
        app_start = s;
        break :blk s;
    };
    const now = std.time.Instant.now() catch return 0;
    return now.since(start) / std.time.ns_per_ms;
}

// ============================================================================
// Tests
// ============================================================================

test "Engine tracks events" {
    const allocator = std.testing.allocator;
    var engine = Engine.init(allocator, .{ .buffer_capacity = 100 });
    defer engine.deinit();

    try engine.track("page_view");
    try engine.track("button_click");
    try engine.trackWithSession("purchase", "session-1");

    try std.testing.expectEqual(@as(usize, 3), engine.bufferedCount());
    try std.testing.expectEqual(@as(u64, 3), engine.totalEvents());
}

test "Engine respects buffer capacity" {
    const allocator = std.testing.allocator;
    var engine = Engine.init(allocator, .{ .buffer_capacity = 2 });
    defer engine.deinit();

    try engine.track("e1");
    try engine.track("e2");
    try std.testing.expectError(AnalyticsError.BufferFull, engine.track("e3"));
}

test "Engine flush clears buffer" {
    const allocator = std.testing.allocator;
    var engine = Engine.init(allocator, .{ .buffer_capacity = 100 });
    defer engine.deinit();

    try engine.track("e1");
    try engine.track("e2");
    const flushed = engine.flush();
    try std.testing.expectEqual(@as(usize, 2), flushed);
    try std.testing.expectEqual(@as(usize, 0), engine.bufferedCount());
    // Total count preserved after flush
    try std.testing.expectEqual(@as(u64, 2), engine.totalEvents());
}

test "Engine session counting" {
    const allocator = std.testing.allocator;
    var engine = Engine.init(allocator, .{});
    defer engine.deinit();

    const s1 = engine.startSession();
    const s2 = engine.startSession();
    try std.testing.expectEqual(@as(u64, 0), s1);
    try std.testing.expectEqual(@as(u64, 1), s2);

    const stats = engine.getStats();
    try std.testing.expectEqual(@as(u64, 2), stats.total_sessions);
}

test "Funnel tracks step progression" {
    const allocator = std.testing.allocator;
    var funnel = Funnel.init(allocator, "onboarding");
    defer funnel.deinit();

    try funnel.addStep("signup");
    try funnel.addStep("verify_email");
    try funnel.addStep("complete_profile");

    funnel.recordStep(0); // signup
    funnel.recordStep(0);
    funnel.recordStep(1); // verify
    funnel.recordStep(2); // complete

    var buf: [3]u64 = undefined;
    const counts = funnel.getStepCounts(&buf);
    try std.testing.expectEqual(@as(u64, 2), counts[0]);
    try std.testing.expectEqual(@as(u64, 1), counts[1]);
    try std.testing.expectEqual(@as(u64, 1), counts[2]);
}

test "Experiment assigns deterministically" {
    var exp = Experiment{
        .name = "color_test",
        .variants = &.{ "red", "blue", "green" },
    };

    const v1 = exp.assign("user-123");
    const v2 = exp.assign("user-123");
    try std.testing.expectEqualStrings(v1, v2); // same user = same variant

    try std.testing.expectEqual(@as(u64, 2), exp.totalAssignments());
}
