//! JSON-RPC transport helpers for the LSP client.
//!
//! Provides raw request/notification sending and response reading over
//! the LSP Content-Length framing protocol.

const std = @import("std");
const jsonrpc = @import("../jsonrpc.zig");

pub const Response = struct {
    json: []u8,
    is_error: bool,
};

/// Send a JSON-RPC request and read the matching response.
pub fn requestRaw(
    allocator: std.mem.Allocator,
    stdin_writer: anytype,
    stdout_reader: anytype,
    next_id: *i64,
    max_payload_bytes: usize,
    method: []const u8,
    params_json: ?[]const u8,
) !Response {
    const id = next_id.*;
    next_id.* += 1;

    const payload = try buildMessageJson(allocator, id, method, params_json);
    defer allocator.free(payload);

    try jsonrpc.writeMessage(&stdin_writer.interface, payload);
    try stdin_writer.flush();

    var read_count: usize = 0;
    while (true) {
        const msg_opt = try jsonrpc.readMessageAlloc(
            allocator,
            &stdout_reader.interface,
            max_payload_bytes,
        );
        if (msg_opt == null) return error.EndOfStream;
        const msg = msg_opt.?;
        defer allocator.free(msg);

        const parsed = std.json.parseFromSlice(
            std.json.Value,
            allocator,
            msg,
            .{},
        ) catch |err| {
            std.log.debug("lsp: skipping malformed JSON-RPC message: {}", .{err});
            continue;
        };
        defer parsed.deinit();

        if (parsed.value != .object) continue;
        const obj = parsed.value.object;
        const id_val = obj.get("id") orelse {
            read_count += 1;
            if (read_count > 256) return error.ResponseNotFound;
            continue;
        };

        if (!idMatches(id_val, id)) {
            read_count += 1;
            if (read_count > 256) return error.ResponseNotFound;
            continue;
        }

        if (obj.get("result")) |result_val| {
            const result_json = try std.json.Stringify.valueAlloc(
                allocator,
                result_val,
                .{},
            );
            return .{ .json = result_json, .is_error = false };
        }
        if (obj.get("error")) |error_val| {
            const err_json = try std.json.Stringify.valueAlloc(
                allocator,
                error_val,
                .{},
            );
            return .{ .json = err_json, .is_error = true };
        }

        const empty = try allocator.dupe(u8, "null");
        return .{ .json = empty, .is_error = true };
    }
}

/// Send a JSON-RPC notification (no response expected).
pub fn notifyRaw(
    allocator: std.mem.Allocator,
    stdin_writer: anytype,
    method: []const u8,
    params_json: ?[]const u8,
) !void {
    const payload = try buildMessageJson(allocator, null, method, params_json);
    defer allocator.free(payload);

    try jsonrpc.writeMessage(&stdin_writer.interface, payload);
    try stdin_writer.flush();
}

/// Wait for a notification with a specific method name.
pub fn waitForNotification(
    allocator: std.mem.Allocator,
    stdout_reader: anytype,
    max_payload_bytes: usize,
    method: []const u8,
    max_messages: usize,
) !?[]u8 {
    var count: usize = 0;
    while (count < max_messages) : (count += 1) {
        const msg_opt = try jsonrpc.readMessageAlloc(
            allocator,
            &stdout_reader.interface,
            max_payload_bytes,
        );
        if (msg_opt == null) return null;
        const msg = msg_opt.?;
        defer allocator.free(msg);

        const parsed = std.json.parseFromSlice(
            std.json.Value,
            allocator,
            msg,
            .{},
        ) catch |err| {
            std.log.debug("lsp: skipping malformed notification message: {}", .{err});
            continue;
        };
        defer parsed.deinit();

        if (parsed.value != .object) continue;
        const obj = parsed.value.object;
        const method_val = obj.get("method") orelse continue;
        if (method_val != .string) continue;
        if (!std.mem.eql(u8, method_val.string, method)) continue;

        const params_val = obj.get("params") orelse std.json.Value{ .null = {} };
        return try std.json.Stringify.valueAlloc(allocator, params_val, .{});
    }
    return null;
}

/// Build a JSON-RPC message payload.
pub fn buildMessageJson(
    allocator: std.mem.Allocator,
    id: ?i64,
    method: []const u8,
    params_json: ?[]const u8,
) ![]u8 {
    var writer = std.Io.Writer.Allocating.init(allocator);
    errdefer writer.deinit();

    var jw = std.json.Stringify{ .writer = &writer.writer, .options = .{} };
    try jw.beginObject();
    try jw.objectField("jsonrpc");
    try jw.write("2.0");
    if (id) |value| {
        try jw.objectField("id");
        try jw.write(value);
    }
    try jw.objectField("method");
    try jw.write(method);
    if (params_json) |params| {
        try jw.objectField("params");
        try jw.beginWriteRaw();
        try writer.writer.writeAll(params);
        jw.endWriteRaw();
    }
    try jw.endObject();

    return writer.toOwnedSlice();
}

fn idMatches(id_val: std.json.Value, expected: i64) bool {
    return switch (id_val) {
        .integer => |v| v == expected,
        .number_string => |v| (std.fmt.parseInt(i64, v, 10) catch return false) == expected,
        .string => |v| (std.fmt.parseInt(i64, v, 10) catch return false) == expected,
        else => false,
    };
}

test {
    std.testing.refAllDecls(@This());
}
