const std = @import("std");
const types = @import("types.zig");

pub const GcpConfig = struct {
    port: u16 = 8080,
    project_id: ?[]const u8 = null,
    region: ?[]const u8 = null,
    function_name: ?[]const u8 = null,

    pub fn fromEnvironment() GcpConfig {
        return .{};
    }
};

pub const GcpRuntime = struct {
    pub fn init(allocator: std.mem.Allocator, handler: types.CloudHandler) GcpRuntime {
        _ = allocator;
        _ = handler;
        return .{};
    }

    pub fn initWithConfig(allocator: std.mem.Allocator, handler: types.CloudHandler, config: GcpConfig) GcpRuntime {
        _ = allocator;
        _ = handler;
        _ = config;
        return .{};
    }

    pub fn run(self: *GcpRuntime) !void {
        _ = self;
        return types.Error.CloudDisabled;
    }

    pub fn processRequest(
        self: *GcpRuntime,
        method: []const u8,
        path: []const u8,
        headers: std.StringHashMapUnmanaged([]const u8),
        body: ?[]const u8,
    ) !types.CloudResponse {
        _ = self;
        _ = method;
        _ = path;
        _ = headers;
        _ = body;
        return types.Error.CloudDisabled;
    }
};

pub fn parseHttpRequest(
    allocator: std.mem.Allocator,
    method: []const u8,
    path: []const u8,
    headers: std.StringHashMapUnmanaged([]const u8),
    body: ?[]const u8,
    request_id: []const u8,
) !types.CloudEvent {
    _ = allocator;
    _ = method;
    _ = path;
    _ = headers;
    _ = body;
    _ = request_id;
    return types.Error.CloudDisabled;
}

pub fn parseCloudEvent(allocator: std.mem.Allocator, raw_event: []const u8) !types.CloudEvent {
    _ = allocator;
    _ = raw_event;
    return types.Error.CloudDisabled;
}

pub fn parsePubSubMessage(allocator: std.mem.Allocator, raw_event: []const u8) !types.CloudEvent {
    _ = allocator;
    _ = raw_event;
    return types.Error.CloudDisabled;
}

pub fn parseStorageEvent(allocator: std.mem.Allocator, raw_event: []const u8) !types.CloudEvent {
    _ = allocator;
    _ = raw_event;
    return types.Error.CloudDisabled;
}

pub fn formatResponse(allocator: std.mem.Allocator, response: *const types.CloudResponse) ![]const u8 {
    _ = allocator;
    _ = response;
    return types.Error.CloudDisabled;
}

pub fn runHandler(allocator: std.mem.Allocator, handler: types.CloudHandler, port: u16) !void {
    _ = allocator;
    _ = handler;
    _ = port;
    return types.Error.CloudDisabled;
}
