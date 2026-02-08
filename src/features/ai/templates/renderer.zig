//! Template renderer that substitutes variables with values.
//!
//! Supports struct and hashmap value sources, with optional
//! strict mode that errors on missing variables.

const std = @import("std");
const parser = @import("parser.zig");
const Token = parser.Token;
const string_utils = @import("../../../services/shared/utils.zig");
const json_utils = @import("../../../services/shared/utils/json/mod.zig");

pub const RenderError = error{
    MissingVariable,
    InvalidValueType,
    OutOfMemory,
};

pub const RenderOptions = struct {
    /// Error on missing variables (otherwise use empty string).
    strict: bool = false,
    /// String to use for missing variables when not strict.
    missing_placeholder: []const u8 = "",
    /// Automatically escape HTML entities.
    auto_escape_html: bool = false,
};

pub const Renderer = struct {
    allocator: std.mem.Allocator,
    options: RenderOptions,

    pub fn init(allocator: std.mem.Allocator, options: RenderOptions) Renderer {
        return .{
            .allocator = allocator,
            .options = options,
        };
    }

    /// Render tokens with struct values.
    pub fn render(self: *Renderer, tokens: []const Token, values: anytype) RenderError![]u8 {
        var result = std.ArrayListUnmanaged(u8){};
        errdefer result.deinit(self.allocator);

        for (tokens) |token| {
            switch (token) {
                .text => |text| {
                    result.appendSlice(self.allocator, text) catch return RenderError.OutOfMemory;
                },
                .variable => |v| {
                    const value = self.getValueFromStruct(values, v.name) orelse v.default orelse {
                        if (self.options.strict) {
                            return RenderError.MissingVariable;
                        }
                        result.appendSlice(self.allocator, self.options.missing_placeholder) catch return RenderError.OutOfMemory;
                        continue;
                    };

                    // Apply filters
                    var processed = value;
                    var owned = false;

                    for (v.filters) |filter| {
                        const new_value = self.applyFilter(processed, filter) catch return RenderError.OutOfMemory;
                        if (owned) {
                            self.allocator.free(@constCast(processed));
                        }
                        processed = new_value;
                        owned = true;
                    }

                    if (self.options.auto_escape_html) {
                        const escaped = self.escapeHtml(processed) catch return RenderError.OutOfMemory;
                        if (owned) {
                            self.allocator.free(@constCast(processed));
                        }
                        result.appendSlice(self.allocator, escaped) catch {
                            self.allocator.free(escaped);
                            return RenderError.OutOfMemory;
                        };
                        self.allocator.free(escaped);
                    } else {
                        result.appendSlice(self.allocator, processed) catch {
                            if (owned) self.allocator.free(@constCast(processed));
                            return RenderError.OutOfMemory;
                        };
                        if (owned) {
                            self.allocator.free(@constCast(processed));
                        }
                    }
                },
            }
        }

        return result.toOwnedSlice(self.allocator) catch return RenderError.OutOfMemory;
    }

    /// Render tokens with a string hashmap.
    pub fn renderWithMap(self: *Renderer, tokens: []const Token, values: std.StringHashMapUnmanaged([]const u8)) RenderError![]u8 {
        var result = std.ArrayListUnmanaged(u8){};
        errdefer result.deinit(self.allocator);

        for (tokens) |token| {
            switch (token) {
                .text => |text| {
                    result.appendSlice(self.allocator, text) catch return RenderError.OutOfMemory;
                },
                .variable => |v| {
                    const value = values.get(v.name) orelse v.default orelse {
                        if (self.options.strict) {
                            return RenderError.MissingVariable;
                        }
                        result.appendSlice(self.allocator, self.options.missing_placeholder) catch return RenderError.OutOfMemory;
                        continue;
                    };

                    result.appendSlice(self.allocator, value) catch return RenderError.OutOfMemory;
                },
            }
        }

        return result.toOwnedSlice(self.allocator) catch return RenderError.OutOfMemory;
    }

    fn getValueFromStruct(self: *Renderer, values: anytype, name: []const u8) ?[]const u8 {
        _ = self;
        const T = @TypeOf(values);
        const type_info = @typeInfo(T);

        if (type_info != .@"struct") {
            return null;
        }

        inline for (type_info.@"struct".fields) |field| {
            if (std.mem.eql(u8, field.name, name)) {
                const value = @field(values, field.name);
                const FieldType = @TypeOf(value);

                if (FieldType == []const u8 or FieldType == []u8) {
                    return value;
                } else if (@typeInfo(FieldType) == .pointer) {
                    const child = @typeInfo(FieldType).pointer.child;
                    if (child == u8) {
                        return value;
                    }
                } else if (@typeInfo(FieldType) == .optional) {
                    if (value) |v| {
                        const InnerType = @TypeOf(v);
                        if (InnerType == []const u8 or InnerType == []u8) {
                            return v;
                        }
                    }
                    return null;
                }
                return null;
            }
        }

        return null;
    }

    fn applyFilter(self: *Renderer, value: []const u8, filter: Token.Filter) ![]const u8 {
        return switch (filter) {
            .upper => try self.toUpper(value),
            .lower => try self.toLower(value),
            .trim => try self.allocator.dupe(u8, std.mem.trim(u8, value, " \t\n\r")),
            .escape_html => try self.escapeHtml(value),
            .escape_json => try self.escapeJson(value),
        };
    }

    fn toUpper(self: *Renderer, value: []const u8) ![]const u8 {
        return string_utils.toUpperAscii(self.allocator, value);
    }

    fn toLower(self: *Renderer, value: []const u8) ![]const u8 {
        return string_utils.toLowerAscii(self.allocator, value);
    }

    fn escapeHtml(self: *Renderer, value: []const u8) ![]const u8 {
        var result = std.ArrayListUnmanaged(u8){};
        errdefer result.deinit(self.allocator);

        for (value) |c| {
            switch (c) {
                '<' => try result.appendSlice(self.allocator, "&lt;"),
                '>' => try result.appendSlice(self.allocator, "&gt;"),
                '&' => try result.appendSlice(self.allocator, "&amp;"),
                '"' => try result.appendSlice(self.allocator, "&quot;"),
                '\'' => try result.appendSlice(self.allocator, "&#x27;"),
                else => try result.append(self.allocator, c),
            }
        }

        return result.toOwnedSlice(self.allocator);
    }

    fn escapeJson(self: *Renderer, value: []const u8) ![]const u8 {
        return json_utils.escapeJsonContent(self.allocator, value);
    }
};

test "render simple template" {
    const allocator = std.testing.allocator;
    var p = parser.Parser.init(allocator);
    const tokens = try p.parse("Hello, {{name}}!");
    defer {
        for (tokens) |token| {
            switch (token) {
                .text => |text| allocator.free(text),
                .variable => |v| {
                    allocator.free(v.name);
                    if (v.default) |d| allocator.free(d);
                    allocator.free(v.filters);
                },
            }
        }
        allocator.free(tokens);
    }

    var renderer_instance = Renderer.init(allocator, .{});
    const result = try renderer_instance.render(tokens, .{ .name = "World" });
    defer allocator.free(result);

    try std.testing.expectEqualStrings("Hello, World!", result);
}

test "render with default value" {
    const allocator = std.testing.allocator;
    var p = parser.Parser.init(allocator);
    const tokens = try p.parse("Hello, {{name|Guest}}!");
    defer {
        for (tokens) |token| {
            switch (token) {
                .text => |text| allocator.free(text),
                .variable => |v| {
                    allocator.free(v.name);
                    if (v.default) |d| allocator.free(d);
                    allocator.free(v.filters);
                },
            }
        }
        allocator.free(tokens);
    }

    var renderer_instance = Renderer.init(allocator, .{});
    const result = try renderer_instance.render(tokens, .{});
    defer allocator.free(result);

    try std.testing.expectEqualStrings("Hello, Guest!", result);
}

test "render strict mode missing variable" {
    const allocator = std.testing.allocator;
    var p = parser.Parser.init(allocator);
    const tokens = try p.parse("Hello, {{name}}!");
    defer {
        for (tokens) |token| {
            switch (token) {
                .text => |text| allocator.free(text),
                .variable => |v| {
                    allocator.free(v.name);
                    if (v.default) |d| allocator.free(d);
                    allocator.free(v.filters);
                },
            }
        }
        allocator.free(tokens);
    }

    var renderer_instance = Renderer.init(allocator, .{ .strict = true });
    const result = renderer_instance.render(tokens, .{});
    try std.testing.expectError(RenderError.MissingVariable, result);
}
