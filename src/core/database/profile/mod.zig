//! Defines behavioral policy overlays and profile routing.

const std = @import("std");
const context = @import("../context/mod.zig");
const core = @import("../core/mod.zig");

pub const ProfileMode = enum {
    abbey,
    aviva,
    abi,
};

pub const RequestFeatures = struct {
    urgency: f32,
    task_type: []const u8,
    requires_tools: bool,
};

pub const RoutingDecision = struct {
    mode: ProfileMode,
    tone_policy: []const u8,
    verbosity_limit: u32,
};

pub const ProfileRouter = struct {
    allocator: std.mem.Allocator,

    pub fn init(allocator: std.mem.Allocator) ProfileRouter {
        return .{ .allocator = allocator };
    }

    pub fn route(self: *ProfileRouter, features: RequestFeatures) RoutingDecision {
        _ = self;
        if (features.urgency > 0.8) {
            return .{
                .mode = .aviva,
                .tone_policy = "Direct, no padding",
                .verbosity_limit = 500,
            };
        }
        return .{
            .mode = .abbey,
            .tone_policy = "Collaborative, explanatory",
            .verbosity_limit = 2000,
        };
    }
};

pub const ActionBus = struct {
    allocator: std.mem.Allocator,

    pub fn init(allocator: std.mem.Allocator) ActionBus {
        return .{ .allocator = allocator };
    }

    pub fn dispatch(self: *ActionBus, tool_name: []const u8, args: []const u8) ![]const u8 {
        _ = self;
        _ = tool_name;
        _ = args;
        return "{}"; // JSON response stub
    }
};
