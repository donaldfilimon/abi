const std = @import("std");
const types = @import("types.zig");

pub const AzureConfig = struct {
    port: u16 = 7071,
    function_name: ?[]const u8 = null,
    invocation_id: ?[]const u8 = null,
    website_name: ?[]const u8 = null,
    region: ?[]const u8 = null,

    pub fn fromEnvironment() AzureConfig {
        return .{};
    }
};

pub const TriggerType = enum {
    http,
    timer,
    blob,
    queue,
    event_hub,
    service_bus,
    cosmos_db,
    event_grid,
    unknown,

    pub fn fromString(_: []const u8) TriggerType {
        return .unknown;
    }
};

pub const AzureRuntime = struct {
    pub fn init(allocator: std.mem.Allocator, handler: types.CloudHandler) AzureRuntime {
        _ = allocator;
        _ = handler;
        return .{};
    }

    pub fn initWithConfig(allocator: std.mem.Allocator, handler: types.CloudHandler, config: AzureConfig) AzureRuntime {
        _ = allocator;
        _ = handler;
        _ = config;
        return .{};
    }

    pub fn run(self: *AzureRuntime) !void {
        _ = self;
        return types.Error.CloudDisabled;
    }

    pub fn processInvocation(self: *AzureRuntime, raw_request: []const u8) ![]const u8 {
        _ = self;
        _ = raw_request;
        return types.Error.CloudDisabled;
    }
};

pub fn parseInvocationRequest(allocator: std.mem.Allocator, raw_request: []const u8) !types.CloudEvent {
    _ = allocator;
    _ = raw_request;
    return types.Error.CloudDisabled;
}

pub fn parseTimerTrigger(allocator: std.mem.Allocator, raw_event: []const u8) !types.CloudEvent {
    _ = allocator;
    _ = raw_event;
    return types.Error.CloudDisabled;
}

pub fn parseBlobTrigger(allocator: std.mem.Allocator, raw_event: []const u8) !types.CloudEvent {
    _ = allocator;
    _ = raw_event;
    return types.Error.CloudDisabled;
}

pub fn formatInvocationResponse(allocator: std.mem.Allocator, response: *const types.CloudResponse) ![]const u8 {
    _ = allocator;
    _ = response;
    return types.Error.CloudDisabled;
}

pub fn runHandler(allocator: std.mem.Allocator, handler: types.CloudHandler) !void {
    _ = allocator;
    _ = handler;
    return types.Error.CloudDisabled;
}

pub fn generateFunctionJson(allocator: std.mem.Allocator, function_name: []const u8, trigger_type: TriggerType) ![]const u8 {
    _ = allocator;
    _ = function_name;
    _ = trigger_type;
    return types.Error.CloudDisabled;
}

test {
    std.testing.refAllDecls(@This());
}
