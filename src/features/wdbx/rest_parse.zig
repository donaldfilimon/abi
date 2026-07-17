const std = @import("std");
const env = @import("../../foundation/env.zig");
const foundation = @import("../../foundation/http.zig");
const foundation_json = @import("../../foundation/json.zig");

pub const REST_TOKEN_ENV = "ABI_WDBX_REST_TOKEN";

pub const Response = struct {
    status: u16,
    body: []u8,

    pub fn deinit(self: *Response, allocator: std.mem.Allocator) void {
        allocator.free(self.body);
    }
};

pub fn json(allocator: std.mem.Allocator, status: u16, comptime fmt: []const u8, args: anytype) !Response {
    return .{ .status = status, .body = try std.fmt.allocPrint(allocator, fmt, args) };
}

pub const VectorParseError = error{ NotArray, Empty, NonNumber, OutOfMemory };

pub fn parseVectorField(allocator: std.mem.Allocator, vec_node: std.json.Value) VectorParseError![]f32 {
    const arr = switch (vec_node) {
        .array => |a| a,
        else => return error.NotArray,
    };
    if (arr.items.len == 0) return error.Empty;
    const out = try allocator.alloc(f32, arr.items.len);
    errdefer allocator.free(out);
    for (arr.items, 0..) |item, i| {
        out[i] = switch (item) {
            .float => |f| @floatCast(f),
            .integer => |n| @floatFromInt(n),
            else => return error.NonNumber,
        };
    }
    return out;
}

pub fn vectorParseErrorResponse(allocator: std.mem.Allocator, err: VectorParseError) !Response {
    return switch (err) {
        error.NotArray => json(allocator, 400, "{{\"error\":\"vector must be an array\"}}", .{}),
        error.Empty => json(allocator, 400, "{{\"error\":\"vector must be non-empty\"}}", .{}),
        error.NonNumber => json(allocator, 400, "{{\"error\":\"vector elements must be numbers\"}}", .{}),
        error.OutOfMemory => json(allocator, 500, "{{\"error\":\"oom\"}}", .{}),
    };
}

pub fn escapeJsonString(allocator: std.mem.Allocator, value: []const u8) ![]u8 {
    var out: std.ArrayListUnmanaged(u8) = .empty;
    errdefer out.deinit(allocator);
    for (value) |byte| {
        switch (byte) {
            '"' => try out.appendSlice(allocator, "\\\""),
            '\\' => try out.appendSlice(allocator, "\\\\"),
            '\n' => try out.appendSlice(allocator, "\\n"),
            '\r' => try out.appendSlice(allocator, "\\r"),
            '\t' => try out.appendSlice(allocator, "\\t"),
            0x08 => try out.appendSlice(allocator, "\\b"),
            0x0c => try out.appendSlice(allocator, "\\f"),
            0x00...0x07, 0x0b, 0x0e...0x1f => try out.print(allocator, "\\u{X:0>4}", .{byte}),
            else => try out.append(allocator, byte),
        }
    }
    return out.toOwnedSlice(allocator);
}

pub const strField = foundation_json.strField;
pub const reasonPhrase = foundation.reasonPhrase;

pub fn findBody(raw: []const u8) []const u8 {
    if (std.mem.indexOf(u8, raw, "\r\n\r\n")) |i| return raw[i + 4 ..];
    if (std.mem.indexOf(u8, raw, "\n\n")) |i| return raw[i + 2 ..];
    return "";
}

pub const HttpReadResult = foundation.HttpReadResult;
pub const readHttpRequest = foundation.readHttpRequest;

pub const requestTargetWithinBuffer = foundation.requestTargetWithinBuffer;
pub const parseContentLength = foundation.parseContentLength;
pub const headerValue = foundation.headerValue;
pub const hasBearerToken = foundation.hasBearerToken;

pub fn loadBearerToken() ?[]const u8 {
    const raw = env.get(REST_TOKEN_ENV) orelse return null;
    const token = std.mem.trim(u8, raw, " \t\r\n");
    if (token.len == 0) return null;
    return token;
}

test {
    std.testing.refAllDecls(@This());
}
