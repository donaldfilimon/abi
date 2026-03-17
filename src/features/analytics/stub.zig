//! Analytics stub — disabled at compile time.

const std = @import("std");
const stub_context = @import("../../core/stub_context.zig");
const types = @import("types.zig");

// --- Shared Types (from types.zig) ---

pub const Event = types.Event;
pub const AnalyticsConfig = types.AnalyticsConfig;
pub const AnalyticsError = types.AnalyticsError;
pub const Error = AnalyticsError;

// --- Engine ---

pub const Engine = struct {
    allocator: std.mem.Allocator,
    config: AnalyticsConfig,
    events: std.ArrayListUnmanaged(types.StoredEvent) = .empty,

    pub fn init(allocator: std.mem.Allocator, config: AnalyticsConfig) Engine {
        return .{ .allocator = allocator, .config = config };
    }
    pub fn deinit(_: *Engine) void {}
    pub fn track(_: *Engine, _: []const u8) AnalyticsError!void {
        return AnalyticsError.FeatureDisabled;
    }
    pub fn trackWithSession(_: *Engine, _: []const u8, _: ?[]const u8) AnalyticsError!void {
        return AnalyticsError.FeatureDisabled;
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
    pub const Stats = types.Stats;
};

// --- Funnel ---

pub const Funnel = struct {
    name: []const u8,
    steps: std.ArrayListUnmanaged(Step) = .empty,
    allocator: std.mem.Allocator,
    pub const Step = types.FunnelStep;
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
        return AnalyticsError.FeatureDisabled;
    }
    pub fn deinit(_: *Context) void {}
    pub fn getEngine(_: *Context) ?*Engine {
        return null;
    }
};

// --- Module Lifecycle ---

const lifecycle = stub_context.StubFeatureNoConfig(AnalyticsError);
pub const init = lifecycle.init;
pub const deinit = lifecycle.deinit;
pub const isEnabled = lifecycle.isEnabled;
pub const isInitialized = lifecycle.isInitialized;

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

test {
    std.testing.refAllDecls(@This());
}
