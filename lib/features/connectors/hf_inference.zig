const std = @import("std");
const T = @import("mod.zig");
const http_util = @import("http_util.zig");

const Config = struct {
    api_token: []u8,
    base_url: []u8,

    pub fn deinit(self: *Config, allocator: std.mem.Allocator) void {
        allocator.free(self.api_token);
        allocator.free(self.base_url);
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
    const api_token = (try getEnvFirst(allocator, &[_][]const u8{ "ABI_HF_API_TOKEN", "HF_API_TOKEN", "HUGGING_FACE_HUB_TOKEN" })) orelse {
        return error.MissingApiToken;
    };
    errdefer allocator.free(api_token);

    const base_url = (try getEnvFirst(allocator, &[_][]const u8{"ABI_HF_BASE_URL"})) orelse
        try allocator.dupe(u8, "https://api-inference.huggingface.co");
    errdefer allocator.free(base_url);

    return .{ .api_token = api_token, .base_url = base_url };
}

fn extractGeneratedText(allocator: std.mem.Allocator, root: std.json.Value) !?[]u8 {
    if (root == .array and root.array.items.len > 0) {
        const first = root.array.items[0];
        if (first == .object) {
            if (first.object.get("generated_text")) |val| {
                if (val == .string) return allocator.dupe(u8, val.string);
            }
        }
    }
    if (root == .object) {
        if (root.object.get("generated_text")) |val| {
            if (val == .string) return allocator.dupe(u8, val.string);
        }
    }
    return null;
}

fn call(allocator: std.mem.Allocator, req: T.CallRequest) !T.CallResult {
    var config = loadConfig(allocator) catch |err| {
        if (err == error.MissingApiToken) {
            return .{ .ok = false, .content = "", .status_code = 401, .err_msg = "missing_hf_api_token" };
        }
        return err;
    };
    defer config.deinit(allocator);

    const model = if (req.model.len == 0) "gpt2" else req.model;
    const path = try std.fmt.allocPrint(allocator, "/models/{s}", .{model});
    defer allocator.free(path);
    const url = try http_util.joinUrl(allocator, config.base_url, path);
    defer allocator.free(url);

    const body = try http_util.buildJson(allocator, .{
        .inputs = req.prompt,
        .parameters = .{
            .max_new_tokens = req.max_tokens,
            .temperature = req.temperature,
        },
    });
    defer allocator.free(body);

    const auth_value = try std.fmt.allocPrint(allocator, "Bearer {s}", .{config.api_token});
    defer allocator.free(auth_value);
    const headers = [_]std.http.Header{
        .{ .name = "Authorization", .value = auth_value },
    };

    var response = http_util.postJson(allocator, url, &headers, body) catch |err| {
        return .{ .ok = false, .content = "", .status_code = 0, .err_msg = @errorName(err) };
    };

    const ok = response.status >= 200 and response.status < 300;
    var content = response.body;

    if (response.body.len > 0) {
        const parsed = std.json.parseFromSlice(std.json.Value, allocator, response.body, .{}) catch null;
        if (parsed) |value| {
            defer value.deinit();
            if (ok) {
                if (try extractGeneratedText(allocator, value.value)) |text| {
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
        .err_msg = if (ok) null else "hf_request_failed",
    };
}

fn health() bool {
    return true;
}

pub fn get() T.Connector {
    return .{ .name = "hf_inference", .init = init, .call = call, .health = health };
}
