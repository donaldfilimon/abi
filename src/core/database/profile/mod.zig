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

pub const ActionHandler = *const fn ([]const u8) []const u8;

pub const ActionBus = struct {
    allocator: std.mem.Allocator,
    handlers: std.StringHashMapUnmanaged(ActionHandler) = .empty,

    pub fn init(allocator: std.mem.Allocator) ActionBus {
        return .{ .allocator = allocator };
    }

    pub fn deinit(self: *ActionBus) void {
        self.handlers.deinit(self.allocator);
    }

    /// Register a handler for a named action. Subsequent dispatch calls
    /// matching this action name will invoke the handler.
    pub fn register(self: *ActionBus, action: []const u8, handler: ActionHandler) !void {
        try self.handlers.put(self.allocator, action, handler);
    }

    /// Dispatch an action by name. If a handler is registered, it is invoked
    /// with the provided arguments and its result is returned. Otherwise,
    /// returns a JSON acknowledgment containing the action name.
    pub fn dispatch(self: *ActionBus, tool_name: []const u8, args: []const u8) ![]const u8 {
        std.log.info("ActionBus: dispatching action=\"{s}\" args_len={d}", .{ tool_name, args.len });

        if (self.handlers.get(tool_name)) |handler| {
            return handler(args);
        }

        // No registered handler — return a JSON acknowledgment.
        return std.fmt.allocPrint(
            self.allocator,
            "{{\"action\":\"{s}\",\"status\":\"accepted\",\"args_len\":{d}}}",
            .{ tool_name, args.len },
        );
    }
};
