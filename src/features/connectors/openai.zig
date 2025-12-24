const std = @import("std");
const T = @import("mod.zig");
const http_util = @import("http_util.zig");

const Mode = enum { responses, chat, completions };

const Config = struct {
    api_key: []u8,
    base_url: []u8,
    mode: Mode,

    pub fn deinit(self: *Config, allocator: std.mem.Allocator) void {
        allocator.free(self.api_key);
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

fn parseMode(raw: []const u8) Mode {
    if (std.ascii.eqlIgnoreCase(raw, "chat") or std.ascii.eqlIgnoreCase(raw, "chat_completions")) {
        return .chat;
    }
    if (std.ascii.eqlIgnoreCase(raw, "completions")) return .completions;
    return .responses;
}

fn loadConfig(allocator: std.mem.Allocator) !Config {
    const api_key = (try getEnvFirst(allocator, &[_][]const u8{ "ABI_OPENAI_API_KEY", "OPENAI_API_KEY" })) orelse {
        return error.MissingApiKey;
    };
    errdefer allocator.free(api_key);

    const base_url = (try getEnvFirst(allocator, &[_][]const u8{"ABI_OPENAI_BASE_URL"})) orelse
        try allocator.dupe(u8, "https://api.openai.com/v1");
    errdefer allocator.free(base_url);

    const mode_str = (try getEnvFirst(allocator, &[_][]const u8{"ABI_OPENAI_MODE"})) orelse null;
    if (mode_str) |owned| {
        defer allocator.free(owned);
        return .{ .api_key = api_key, .base_url = base_url, .mode = parseMode(owned) };
    }

    return .{ .api_key = api_key, .base_url = base_url, .mode = .responses };
}

fn endpointForMode(mode: Mode) []const u8 {
    return switch (mode) {
        .responses => "/responses",
        .chat => "/chat/completions",
        .completions => "/completions",
    };
}

fn buildRequestBody(allocator: std.mem.Allocator, mode: Mode, req: T.CallRequest) ![]u8 {
    return switch (mode) {
        .responses => http_util.buildJson(allocator, .{
            .model = req.model,
            .input = req.prompt,
            .temperature = req.temperature,
            .max_output_tokens = req.max_tokens,
        }),
        .chat => http_util.buildJson(allocator, .{
            .model = req.model,
            .messages = &[_]struct { role: []const u8, content: []const u8 }{
                .{ .role = "user", .content = req.prompt },
            },
            .temperature = req.temperature,
            .max_tokens = req.max_tokens,
        }),
        .completions => http_util.buildJson(allocator, .{
            .model = req.model,
            .prompt = req.prompt,
            .temperature = req.temperature,
            .max_tokens = req.max_tokens,
        }),
    };
}

fn extractText(allocator: std.mem.Allocator, mode: Mode, root: std.json.Value) !?[]u8 {
    if (root != .object) return null;
    const obj = root.object;

    if (mode == .responses) {
        if (obj.get("output_text")) |val| {
            if (val == .string) return allocator.dupe(u8, val.string);
        }
        if (obj.get("output")) |output_val| {
            if (output_val == .array and output_val.array.items.len > 0) {
                const first = output_val.array.items[0];
                if (first == .object) {
                    if (first.object.get("content")) |content_val| {
                        if (content_val == .array and content_val.array.items.len > 0) {
                            const content_first = content_val.array.items[0];
                            if (content_first == .object) {
                                if (content_first.object.get("text")) |text_val| {
                                    if (text_val == .string) return allocator.dupe(u8, text_val.string);
                                }
                            }
                        }
                    }
                }
            }
        }
    }

    if (obj.get("choices")) |choices_val| {
        if (choices_val == .array and choices_val.array.items.len > 0) {
            const choice = choices_val.array.items[0];
            if (choice == .object) {
                if (mode == .chat) {
                    if (choice.object.get("message")) |message_val| {
                        if (message_val == .object) {
                            if (message_val.object.get("content")) |content_val| {
                                if (content_val == .string) return allocator.dupe(u8, content_val.string);
                            }
                        }
                    }
                }
                if (choice.object.get("text")) |text_val| {
                    if (text_val == .string) return allocator.dupe(u8, text_val.string);
                }
            }
        }
    }

    return null;
}

fn parseUsage(root: std.json.Value) struct { input: u32, output: u32 } {
    var result = .{ .input = @as(u32, 0), .output = @as(u32, 0) };
    if (root != .object) return result;
    const obj = root.object;
    const usage_val = obj.get("usage") orelse return result;
    if (usage_val != .object) return result;
    const usage = usage_val.object;

    if (usage.get("input_tokens")) |val| {
        if (val == .integer) result.input = @intCast(val.integer);
    }
    if (usage.get("output_tokens")) |val| {
        if (val == .integer) result.output = @intCast(val.integer);
    }
    if (usage.get("prompt_tokens")) |val| {
        if (val == .integer) result.input = @intCast(val.integer);
    }
    if (usage.get("completion_tokens")) |val| {
        if (val == .integer) result.output = @intCast(val.integer);
    }
    return result;
}

fn call(allocator: std.mem.Allocator, req: T.CallRequest) !T.CallResult {
    var config = loadConfig(allocator) catch |err| {
        if (err == error.MissingApiKey) {
            return .{ .ok = false, .content = "", .status_code = 401, .err_msg = "missing_openai_api_key" };
        }
        return err;
    };
    defer config.deinit(allocator);

    const url = try http_util.joinUrl(allocator, config.base_url, endpointForMode(config.mode));
    defer allocator.free(url);

    const body = try buildRequestBody(allocator, config.mode, req);
    defer allocator.free(body);

    const auth_value = try std.fmt.allocPrint(allocator, "Bearer {s}", .{config.api_key});
    defer allocator.free(auth_value);

    const headers = [_]std.http.Header{
        .{ .name = "Authorization", .value = auth_value },
    };

    const response = http_util.postJson(allocator, url, &headers, body) catch |err| {
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
            tokens_in = parseUsage(value.value).input;
            tokens_out = parseUsage(value.value).output;
            if (ok) {
                if (try extractText(allocator, config.mode, value.value)) |text| {
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
        .err_msg = if (ok) null else "openai_request_failed",
    };
}

fn health() bool {
    return true;
}

pub fn embedText(allocator: std.mem.Allocator, config: T.OpenAIConfig, text: []const u8) ![]f32 {
    const url = try http_util.joinUrl(allocator, config.base_url, "/embeddings");
    defer allocator.free(url);

    const body = try http_util.buildJson(allocator, .{
        .model = config.model,
        .input = text,
    });
    defer allocator.free(body);

    const auth_value = try std.fmt.allocPrint(allocator, "Bearer {s}", .{config.api_key});
    defer allocator.free(auth_value);

    const headers = [_]std.http.Header{
        .{ .name = "Authorization", .value = auth_value },
    };

    const response = try http_util.postJson(allocator, url, &headers, body);
    errdefer allocator.free(response.body);

    if (response.status < 200 or response.status >= 300) return error.RequestFailed;

    var parsed = try std.json.parseFromSlice(std.json.Value, allocator, response.body, .{});
    defer parsed.deinit();
    allocator.free(response.body);

    const root = parsed.value.object;
    const data_val = root.get("data") orelse return error.InvalidResponse;
    if (data_val != .array or data_val.array.items.len == 0) return error.InvalidResponse;

    const first = data_val.array.items[0];
    if (first != .object) return error.InvalidResponse;
    const embedding_val = first.object.get("embedding") orelse return error.InvalidResponse;
    if (embedding_val != .array) return error.InvalidResponse;

    const arr = embedding_val.array;
    const out = try allocator.alloc(f32, arr.items.len);
    for (arr.items, 0..) |item, idx| {
        out[idx] = switch (item) {
            .float => @floatCast(item.float),
            .integer => @floatFromInt(item.integer),
            else => return error.InvalidResponse,
        };
    }
    return out;
}

pub fn get() T.Connector {
    return .{ .name = "openai", .init = init, .call = call, .health = health };
}
