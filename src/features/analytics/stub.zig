//! Analytics stub — disabled at compile time.

const std = @import("std");
const time = @import("../../services/shared/time.zig");
const sync = @import("../../services/shared/sync.zig");

// --- Event Types ---

pub const Event = struct {
    name: []const u8,
    timestamp_ms: u64,
    session_id: ?[]const u8 = null,
    properties: []const Property = &.{},
    pub const Property = struct { key: []const u8, value: Value };
    pub const Value = union(enum) { string: []const u8, int: i64, float: f64, boolean: bool };
};

pub const AnalyticsConfig = struct {
    buffer_capacity: u32 = 1024,
    enable_timestamps: bool = true,
    app_id: []const u8 = "abi-app",
    flush_interval_ms: u64 = 0,
};

// --- Engine ---

pub const AnalyticsError = error{ BufferFull, InvalidEvent, FlushFailed, AnalyticsDisabled, OutOfMemory };

pub const Engine = struct {
    allocator: std.mem.Allocator,
    config: AnalyticsConfig,
    events: std.ArrayListUnmanaged(StoredEvent) = .empty,
    session_count: std.atomic.Value(u64) = std.atomic.Value(u64).init(0),
    event_count: std.atomic.Value(u64) = std.atomic.Value(u64).init(0),
    mutex: sync.Mutex = .{},

    const StoredEvent = struct { name: []const u8, timestamp_ms: u64, session_id: ?[]const u8 };

    pub fn init(allocator: std.mem.Allocator, config: AnalyticsConfig) Engine {
        return .{ .allocator = allocator, .config = config };
    }
    pub fn deinit(_: *Engine) void {}
    pub fn track(_: *Engine, _: []const u8) AnalyticsError!void {
        return AnalyticsError.AnalyticsDisabled;
    }
    pub fn trackWithSession(_: *Engine, _: []const u8, _: ?[]const u8) AnalyticsError!void {
        return AnalyticsError.AnalyticsDisabled;
    }
    pub fn startSession(_: *Engine) u64 {
        return 0;
    }
    pub fn bufferedCount(_: *Engine) usize {
        return 0;
    }
    pub fn totalEvents(_: *const Engine) u64 {
        return 0;
    }
    pub fn flush(_: *Engine) usize {
        return 0;
    }
    pub fn getStats(_: *Engine) Stats {
        return .{};
    }
    pub const Stats = struct { buffered_events: usize = 0, total_events: u64 = 0, total_sessions: u64 = 0 };
};

// --- Funnel ---

pub const Funnel = struct {
    name: []const u8,
    steps: std.ArrayListUnmanaged(Step) = .empty,
    allocator: std.mem.Allocator,
    pub const Step = struct { name: []const u8, count: std.atomic.Value(u64) = std.atomic.Value(u64).init(0) };
    pub fn init(allocator: std.mem.Allocator, name: []const u8) Funnel {
        return .{ .name = name, .allocator = allocator };
    }
    pub fn deinit(_: *Funnel) void {}
    pub fn addStep(_: *Funnel, _: []const u8) !void {}
    pub fn recordStep(_: *Funnel, _: usize) void {}
    pub fn getStepCounts(_: *const Funnel, buffer: []u64) []u64 {
        return buffer[0..0];
    }
};

// --- Context ---

pub const Context = struct {
    allocator: std.mem.Allocator,
    config: AnalyticsConfig,
    engine: ?Engine = null,
    pub fn init(_: std.mem.Allocator, _: AnalyticsConfig) !*Context {
        return AnalyticsError.AnalyticsDisabled;
    }
    pub fn deinit(_: *Context) void {}
    pub fn getEngine(_: *Context) ?*Engine {
        return null;
    }
};

// --- Module Lifecycle ---

var initialized: bool = false;
pub fn init(_: std.mem.Allocator) !void {
    return AnalyticsError.AnalyticsDisabled;
}
pub fn deinit() void {
    initialized = false;
}
pub fn isEnabled() bool {
    return false;
}
pub fn isInitialized() bool {
    return initialized;
}

// --- Experiment ---

pub const Experiment = struct {
    name: []const u8,
    variants: []const []const u8,
    assignments: std.atomic.Value(u64) = std.atomic.Value(u64).init(0),
    pub fn assign(_: *Experiment, _: []const u8) []const u8 {
        return "control";
    }
    pub fn totalAssignments(_: *const Experiment) u64 {
        return 0;
    }
};
