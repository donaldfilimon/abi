const std = @import("std");
const T = @import("mod.zig");
const http_util = @import("http_util.zig");

const Config = struct {
    base_url: []u8,
    endpoint: []u8,

    pub fn deinit(self: *Config, allocator: std.mem.Allocator) void {
        allocator.free(self.base_url);
        allocator.free(self.endpoint);
    }
};

fn init(_: std.mem.Allocator) !void {}

fn getEnvFirst(allocator: std.mem.Allocator, names: []const []const u8) !?[]u8 {
    for (names) |name| {
        const value = std.process.getEnvVarOwned(allocator, name) catch |err| switch (err) {
            error.EnvironmentVariableNotFound => continue,
            else => return err,
        };
        return value;
    }
    return null;
}

fn loadConfig(allocator: std.mem.Allocator) !Config {
    const base_url = (try getEnvFirst(allocator, &[_][]const u8{ "ABI_LOCAL_SCHEDULER_URL", "LOCAL_SCHEDULER_URL" })) orelse
        try allocator.dupe(u8, "http://127.0.0.1:8081");
    errdefer allocator.free(base_url);

    const endpoint = (try getEnvFirst(allocator, &[_][]const u8{"ABI_LOCAL_SCHEDULER_ENDPOINT"})) orelse
        try allocator.dupe(u8, "/schedule");
    errdefer allocator.free(endpoint);

    return .{ .base_url = base_url, .endpoint = endpoint };
}

fn extractText(allocator: std.mem.Allocator, root: std.json.Value) !?[]u8 {
    if (root != .object) return null;
    const obj = root.object;

    if (obj.get("content")) |val| if (val == .string) return allocator.dupe(u8, val.string);
    if (obj.get("response")) |val| if (val == .string) return allocator.dupe(u8, val.string);
    if (obj.get("output")) |val| if (val == .string) return allocator.dupe(u8, val.string);

    return null;
}

fn call(allocator: std.mem.Allocator, req: T.CallRequest) !T.CallResult {
    var config = try loadConfig(allocator);
    defer config.deinit(allocator);

    const url = try http_util.joinUrl(allocator, config.base_url, config.endpoint);
    defer allocator.free(url);

    const body = try http_util.buildJson(allocator, .{
        .model = req.model,
        .prompt = req.prompt,
        .max_tokens = req.max_tokens,
        .temperature = req.temperature,
    });
    defer allocator.free(body);

    var response = http_util.postJson(allocator, url, &.{}, body) catch |err| {
        return .{ .ok = false, .content = "", .status_code = 0, .err_msg = @errorName(err) };
    };

    const ok = response.status >= 200 and response.status < 300;
    var content = response.body;

    if (response.body.len > 0) {
        const parsed = std.json.parseFromSlice(std.json.Value, allocator, response.body, .{}) catch null;
        if (parsed) |value| {
            defer value.deinit();
            if (ok) {
                if (try extractText(allocator, value.value)) |text| {
                    allocator.free(response.body);
                    content = text;
                }
            }
        }
    }

    return .{
        .ok = ok,
        .content = content,
        .status_code = response.status,
        .err_msg = if (ok) null else "local_scheduler_request_failed",
    };
}

fn health() bool {
    return true;
}

pub fn get() T.Connector {
    return .{ .name = "local_scheduler", .init = init, .call = call, .health = health };
}
