//! Analytics Engine
//!
//! Core analytics engine. Buffers events and provides batch retrieval.

const std = @import("std");
const time = @import("../../foundation/mod.zig").time;
const sync = @import("../../foundation/mod.zig").sync;
const types = @import("types.zig");

const Mutex = sync.Mutex;
const AnalyticsConfig = types.AnalyticsConfig;
const AnalyticsError = types.AnalyticsError;

/// Application start anchor for monotonic timestamps.
var app_start: ?time.Instant = null;

fn timestampMs() u64 {
    const start = app_start orelse blk: {
        const s = time.Instant.now() catch return 0;
        app_start = s;
        break :blk s;
    };
    const now = time.Instant.now() catch return 0;
    return now.since(start) / std.time.ns_per_ms;
}

/// Core analytics engine. Buffers events and provides batch retrieval.
pub const Engine = struct {
    allocator: std.mem.Allocator,
    config: AnalyticsConfig,
    events: std.ArrayListUnmanaged(types.StoredEvent) = .empty,
    session_count: std.atomic.Value(u64) = std.atomic.Value(u64).init(0),
    event_count: std.atomic.Value(u64) = std.atomic.Value(u64).init(0),
    mutex: Mutex = .{},

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

    pub const Stats = types.Stats;
};
