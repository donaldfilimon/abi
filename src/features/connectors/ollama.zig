const std = @import("std");
const T = @import("mod.zig");
const http_util = @import("http_util.zig");

pub const Allocator = std.mem.Allocator;

const Config = struct {
    host: []u8,
    model: []u8,

    pub fn deinit(self: *Config, allocator: std.mem.Allocator) void {
        allocator.free(self.host);
        allocator.free(self.model);
    }
};

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
    const host = (try getEnvFirst(allocator, &[_][]const u8{ "ABI_OLLAMA_HOST", "OLLAMA_HOST" })) orelse
        try allocator.dupe(u8, "http://127.0.0.1:11434");
    errdefer allocator.free(host);

    const model = (try getEnvFirst(allocator, &[_][]const u8{"ABI_OLLAMA_MODEL"})) orelse
        try allocator.dupe(u8, "llama3.2");
    errdefer allocator.free(model);

    return .{ .host = host, .model = model };
}

fn init(_: std.mem.Allocator) !void {}

pub fn embedText(allocator: Allocator, host: []const u8, model: []const u8, text: []const u8) ![]f32 {
    const url = try http_util.joinUrl(allocator, host, "/api/embeddings");
    defer allocator.free(url);

    const body = try http_util.buildJson(allocator, .{
        .model = model,
        .input = text,
    });
    defer allocator.free(body);

    const response = http_util.postJson(allocator, url, &.{}, body) catch return error.NetworkError;

    if (response.status < 200 or response.status >= 300) return error.NetworkError;

    // Expected minimal shape: {"embedding":[...]} or {"data":[{"embedding":[...]}]}
    var parsed = try std.json.parseFromSlice(std.json.Value, allocator, response.body, .{});
    defer parsed.deinit();
    const root_obj = parsed.value.object;
    if (root_obj.get("embedding")) |val| {
        return parseEmbeddingArray(allocator, val);
    }
    if (root_obj.get("data")) |data_val| {
        const arr = data_val.array;
        if (arr.items.len > 0) {
            const first = arr.items[0].object;
            return parseEmbeddingArray(allocator, first.get("embedding").?);
        }
    }
    return error.InvalidResponse;
}

fn parseEmbeddingArray(allocator: Allocator, v: std.json.Value) ![]f32 {
    const arr = v.array;
    const out = try allocator.alloc(f32, arr.items.len);
    var i: usize = 0;
    while (i < out.len) : (i += 1) {
        out[i] = @floatCast(arr.items[i].float);
    }
    return out;
}

fn extractResponseText(allocator: std.mem.Allocator, root: std.json.Value) !?[]u8 {
    if (root != .object) return null;
    if (root.object.get("response")) |val| {
        if (val == .string) return allocator.dupe(u8, val.string);
    }
    return null;
}

fn parseCounts(root: std.json.Value) struct { input: u32, output: u32 } {
    var result = .{ .input = @as(u32, 0), .output = @as(u32, 0) };
    if (root != .object) return result;
    const obj = root.object;

    if (obj.get("prompt_eval_count")) |val| {
        if (val == .integer) result.input = @intCast(val.integer);
    }
    if (obj.get("eval_count")) |val| {
        if (val == .integer) result.output = @intCast(val.integer);
    }
    return result;
}

fn call(allocator: std.mem.Allocator, req: T.CallRequest) !T.CallResult {
    var config = try loadConfig(allocator);
    defer config.deinit(allocator);

    const model = if (req.model.len == 0 or std.mem.eql(u8, req.model, "default")) config.model else req.model;
    const url = try http_util.joinUrl(allocator, config.host, "/api/generate");
    defer allocator.free(url);

    const body = try http_util.buildJson(allocator, .{
        .model = model,
        .prompt = req.prompt,
        .stream = false,
        .options = .{
            .temperature = req.temperature,
            .num_predict = req.max_tokens,
        },
    });
    defer allocator.free(body);

    const response = http_util.postJson(allocator, url, &.{}, body) catch |err| {
        return .{ .ok = false, .content = "", .status_code = 0, .err_msg = @errorName(err) };
    };

    const ok = response.status >= 200 and response.status < 300;
    var content = response.body;
    var tokens_in: u32 = 0;
    var tokens_out: u32 = 0;

    if (response.body.len > 0) {
        const parsed = std.json.parseFromSlice(std.json.Value, allocator, response.body, .{}) catch null;
        if (parsed) |value| {
            defer value.deinit();
            const counts = parseCounts(value.value);
            tokens_in = counts.input;
            tokens_out = counts.output;
            if (ok) {
                if (try extractResponseText(allocator, value.value)) |text| {
                    allocator.free(response.body);
                    content = text;
                }
            }
        }
    }

    return .{
        .ok = ok,
        .content = content,
        .tokens_in = tokens_in,
        .tokens_out = tokens_out,
        .status_code = response.status,
        .err_msg = if (ok) null else "ollama_request_failed",
    };
}

fn health() bool {
    return true;
}

pub fn get() T.Connector {
    return .{ .name = "ollama", .init = init, .call = call, .health = health };
}
