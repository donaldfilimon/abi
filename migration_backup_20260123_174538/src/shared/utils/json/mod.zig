//! JSON utilities for connector API responses

const std = @import("std");

pub fn escapeString(allocator: std.mem.Allocator, str: []const u8) ![]u8 {
    var escaped = std.ArrayListUnmanaged(u8).empty;
    errdefer escaped.deinit(allocator);

    try escaped.append(allocator, '"');

    for (str) |c| {
        switch (c) {
            '"' => try escaped.appendSlice(allocator, "\\\""),
            '\\' => try escaped.appendSlice(allocator, "\\\\"),
            '\n' => try escaped.appendSlice(allocator, "\\n"),
            '\r' => try escaped.appendSlice(allocator, "\\r"),
            '\t' => try escaped.appendSlice(allocator, "\\t"),
            else => try escaped.append(allocator, c),
        }
    }

    try escaped.append(allocator, '"');

    return escaped.toOwnedSlice(allocator);
}

/// Escape a string for JSON (without surrounding quotes).
/// Returns the escaped content that can be placed inside JSON string literals.
pub fn escapeJsonContent(allocator: std.mem.Allocator, str: []const u8) ![]u8 {
    var escaped = std.ArrayListUnmanaged(u8).empty;
    errdefer escaped.deinit(allocator);

    for (str) |c| {
        switch (c) {
            '"' => try escaped.appendSlice(allocator, "\\\""),
            '\\' => try escaped.appendSlice(allocator, "\\\\"),
            '\n' => try escaped.appendSlice(allocator, "\\n"),
            '\r' => try escaped.appendSlice(allocator, "\\r"),
            '\t' => try escaped.appendSlice(allocator, "\\t"),
            0x00...0x08, 0x0B, 0x0C, 0x0E...0x1F => {
                // Escape other control characters as \uXXXX
                var buf: [6]u8 = undefined;
                _ = std.fmt.bufPrint(&buf, "\\u{x:0>4}", .{c}) catch unreachable;
                try escaped.appendSlice(allocator, &buf);
            },
            else => try escaped.append(allocator, c),
        }
    }

    return escaped.toOwnedSlice(allocator);
}

/// Format wrapper for escaping JSON content in format strings.
/// Usage: try writer.print("\"{s}\"", .{jsonEscape(str)});
pub const JsonEscape = struct {
    content: []const u8,

    pub fn format(self: JsonEscape, comptime _: []const u8, _: std.fmt.FormatOptions, writer: anytype) !void {
        for (self.content) |c| {
            switch (c) {
                '"' => try writer.writeAll("\\\""),
                '\\' => try writer.writeAll("\\\\"),
                '\n' => try writer.writeAll("\\n"),
                '\r' => try writer.writeAll("\\r"),
                '\t' => try writer.writeAll("\\t"),
                0x00...0x1F => try writer.print("\\u{x:0>4}", .{c}),
                else => try writer.writeByte(c),
            }
        }
    }
};

/// Create a JsonEscape wrapper for use in format strings.
pub fn jsonEscape(content: []const u8) JsonEscape {
    return .{ .content = content };
}

pub const JsonError = error{
    InvalidJson,
    MissingField,
    TypeMismatch,
    ParseError,
};

pub fn parseString(allocator: std.mem.Allocator, value: std.json.Value) ![]const u8 {
    _ = allocator;
    if (value != .string) return JsonError.TypeMismatch;
    return value.string;
}

pub fn parseNumber(value: std.json.Value) !f64 {
    if (value != .float and value != .integer) return JsonError.TypeMismatch;
    return if (value == .float) value.float else @as(f64, @floatFromInt(value.integer));
}

pub fn parseInt(value: std.json.Value) !i64 {
    if (value != .integer) return JsonError.TypeMismatch;
    return value.integer;
}

pub fn parseUint(value: std.json.Value) !u64 {
    if (value != .integer) return JsonError.TypeMismatch;
    if (value.integer < 0) return JsonError.TypeMismatch;
    return @intCast(value.integer);
}

pub fn parseBool(value: std.json.Value) !bool {
    if (value != .bool) return JsonError.TypeMismatch;
    return value.bool;
}

pub fn getRequiredObject(value: std.json.Value) !std.json.ObjectMap {
    if (value != .object) return JsonError.TypeMismatch;
    return value.object;
}

pub fn getRequiredArray(value: std.json.Value) !std.json.Array {
    if (value != .array) return JsonError.TypeMismatch;
    return value.array;
}

pub fn getField(object: std.json.ObjectMap, key: []const u8) ?std.json.Value {
    return object.get(key);
}

pub fn getRequiredField(object: std.json.ObjectMap, key: []const u8) !std.json.Value {
    const value = object.get(key) orelse return JsonError.MissingField;
    return value;
}

pub fn parseStringField(object: std.json.ObjectMap, key: []const u8, allocator: std.mem.Allocator) ![]const u8 {
    const value = try getRequiredField(object, key);
    return try parseString(allocator, value);
}

pub fn parseOptionalStringField(object: std.json.ObjectMap, key: []const u8, allocator: std.mem.Allocator) !?[]const u8 {
    const value = object.get(key) orelse return null;
    return try parseString(allocator, value);
}

pub fn parseNumberField(object: std.json.ObjectMap, key: []const u8) !f64 {
    const value = try getRequiredField(object, key);
    return try parseNumber(value);
}

pub fn parseOptionalNumberField(object: std.json.ObjectMap, key: []const u8) ?f64 {
    const value = object.get(key) orelse return null;
    return parseNumber(value) catch null;
}

pub fn parseIntField(object: std.json.ObjectMap, key: []const u8) !i64 {
    const value = try getRequiredField(object, key);
    return try parseInt(value);
}

pub fn parseOptionalIntField(object: std.json.ObjectMap, key: []const u8) ?i64 {
    const value = object.get(key) orelse return null;
    return parseInt(value) catch null;
}

pub fn parseUintField(object: std.json.ObjectMap, key: []const u8) !u64 {
    const value = try getRequiredField(object, key);
    return try parseUint(value);
}

pub fn parseOptionalUintField(object: std.json.ObjectMap, key: []const u8) ?u64 {
    const value = object.get(key) orelse return null;
    return parseUint(value) catch null;
}

pub fn parseBoolField(object: std.json.ObjectMap, key: []const u8) !bool {
    const value = try getRequiredField(object, key);
    return try parseBool(value);
}

pub fn parseOptionalBoolField(object: std.json.ObjectMap, key: []const u8) ?bool {
    const value = object.get(key) orelse return null;
    return parseBool(value) catch null;
}

pub fn parseObjectField(object: std.json.ObjectMap, key: []const u8) !std.json.ObjectMap {
    const value = try getRequiredField(object, key);
    return try getRequiredObject(value);
}

pub fn parseOptionalObjectField(object: std.json.ObjectMap, key: []const u8) ?std.json.ObjectMap {
    const value = object.get(key) orelse return null;
    return getRequiredObject(value) catch null;
}

pub fn parseArrayField(object: std.json.ObjectMap, key: []const u8) !std.json.Array {
    const value = try getRequiredField(object, key);
    return try getRequiredArray(value);
}

pub fn parseOptionalArrayField(object: std.json.ObjectMap, key: []const u8) ?std.json.Array {
    const value = object.get(key) orelse return null;
    return getRequiredArray(value) catch null;
}

test "parse string field" {
    const allocator = std.testing.allocator;

    const json_text = "{\"name\":\"test\",\"value\":42}";
    const parsed = try std.json.parseFromSlice(std.json.Value, allocator, json_text, .{});
    defer parsed.deinit();

    const object = try getRequiredObject(parsed.value);
    const name = try parseStringField(object, "name", allocator);
    try std.testing.expectEqualStrings("test", name);

    const value = try parseIntField(object, "value");
    try std.testing.expectEqual(@as(i64, 42), value);
}

test "parse optional fields" {
    const allocator = std.testing.allocator;

    const json_text = "{\"required\":\"present\"}";
    const parsed = try std.json.parseFromSlice(std.json.Value, allocator, json_text, .{});
    defer parsed.deinit();

    const object = try getRequiredObject(parsed.value);

    const required = try parseStringField(object, "required", allocator);
    try std.testing.expectEqualStrings("present", required);

    const missing = parseOptionalStringField(object, "missing", allocator);
    try std.testing.expect(missing == null);
}
