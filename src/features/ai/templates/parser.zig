//! Template parser for extracting variables and text segments.
//!
//! Supports Mustache-style {{variable}} syntax with optional default values
//! using {{variable|default}} format.

const std = @import("std");

pub const ParseError = error{
    UnterminatedVariable,
    EmptyVariableName,
    InvalidSyntax,
    OutOfMemory,
};

/// Token types produced by parsing.
pub const Token = union(enum) {
    /// Literal text segment.
    text: []const u8,
    /// Variable placeholder with optional default.
    variable: Variable,

    pub const Variable = struct {
        name: []const u8,
        default: ?[]const u8,
        filters: []const Filter,
    };

    pub const Filter = enum {
        upper,
        lower,
        trim,
        escape_html,
        escape_json,
    };
};

pub const Parser = struct {
    allocator: std.mem.Allocator,

    pub fn init(allocator: std.mem.Allocator) Parser {
        return .{ .allocator = allocator };
    }

    /// Parse a template string into tokens.
    pub fn parse(self: *Parser, source: []const u8) ParseError![]Token {
        var tokens = std.ArrayListUnmanaged(Token).empty;
        errdefer {
            for (tokens.items) |token| {
                self.freeToken(token);
            }
            tokens.deinit(self.allocator);
        }

        var pos: usize = 0;

        while (pos < source.len) {
            // Look for variable start
            if (pos + 1 < source.len and source[pos] == '{' and source[pos + 1] == '{') {
                // Find variable end
                const var_start = pos + 2;
                const var_end = std.mem.indexOf(u8, source[var_start..], "}}") orelse {
                    return ParseError.UnterminatedVariable;
                };

                const var_content = source[var_start .. var_start + var_end];
                if (var_content.len == 0) {
                    return ParseError.EmptyVariableName;
                }

                // Parse variable with potential default value
                const variable = try self.parseVariable(var_content);
                tokens.append(self.allocator, .{ .variable = variable }) catch return ParseError.OutOfMemory;

                pos = var_start + var_end + 2;
            } else {
                // Accumulate text until next variable or end
                const text_start = pos;
                while (pos < source.len) {
                    if (pos + 1 < source.len and source[pos] == '{' and source[pos + 1] == '{') {
                        break;
                    }
                    pos += 1;
                }

                if (pos > text_start) {
                    const text = self.allocator.dupe(u8, source[text_start..pos]) catch return ParseError.OutOfMemory;
                    tokens.append(self.allocator, .{ .text = text }) catch {
                        self.allocator.free(text);
                        return ParseError.OutOfMemory;
                    };
                }
            }
        }

        return tokens.toOwnedSlice(self.allocator) catch return ParseError.OutOfMemory;
    }

    fn parseVariable(self: *Parser, content: []const u8) ParseError!Token.Variable {
        // Trim whitespace
        const trimmed = std.mem.trim(u8, content, " \t\n\r");
        if (trimmed.len == 0) {
            return ParseError.EmptyVariableName;
        }

        // Check for default value (separated by |)
        var name: []const u8 = undefined;
        var default: ?[]const u8 = null;
        var filters = std.ArrayListUnmanaged(Token.Filter).empty;
        errdefer filters.deinit(self.allocator);

        // Split by | for default and filters
        var parts = std.mem.splitScalar(u8, trimmed, '|');

        // First part is the name
        name = std.mem.trim(u8, parts.first(), " \t");
        if (name.len == 0) {
            return ParseError.EmptyVariableName;
        }

        // Check for subsequent parts (first is default, rest are filters)
        var has_default = false;
        while (parts.next()) |part| {
            const trimmed_part = std.mem.trim(u8, part, " \t");
            if (!has_default) {
                // First part after name is the default value
                if (trimmed_part.len > 0) {
                    default = self.allocator.dupe(u8, trimmed_part) catch return ParseError.OutOfMemory;
                }
                has_default = true;
            } else {
                // Subsequent parts are filters
                if (std.mem.eql(u8, trimmed_part, "upper")) {
                    filters.append(self.allocator, .upper) catch return ParseError.OutOfMemory;
                } else if (std.mem.eql(u8, trimmed_part, "lower")) {
                    filters.append(self.allocator, .lower) catch return ParseError.OutOfMemory;
                } else if (std.mem.eql(u8, trimmed_part, "trim")) {
                    filters.append(self.allocator, .trim) catch return ParseError.OutOfMemory;
                } else if (std.mem.eql(u8, trimmed_part, "escape_html")) {
                    filters.append(self.allocator, .escape_html) catch return ParseError.OutOfMemory;
                } else if (std.mem.eql(u8, trimmed_part, "escape_json")) {
                    filters.append(self.allocator, .escape_json) catch return ParseError.OutOfMemory;
                }
            }
        }

        const name_copy = self.allocator.dupe(u8, name) catch return ParseError.OutOfMemory;
        errdefer self.allocator.free(name_copy);

        return .{
            .name = name_copy,
            .default = default,
            .filters = filters.toOwnedSlice(self.allocator) catch return ParseError.OutOfMemory,
        };
    }

    /// Extract all variable names from tokens.
    pub fn extractVariables(self: *Parser, tokens: []const Token) ![]const []const u8 {
        var vars = std.ArrayListUnmanaged([]const u8).empty;
        errdefer {
            for (vars.items) |v| {
                self.allocator.free(v);
            }
            vars.deinit(self.allocator);
        }

        var seen = std.StringHashMapUnmanaged(void){};
        defer seen.deinit(self.allocator);

        for (tokens) |token| {
            if (token == .variable) {
                if (!seen.contains(token.variable.name)) {
                    try seen.put(self.allocator, token.variable.name, {});
                    const name_copy = try self.allocator.dupe(u8, token.variable.name);
                    try vars.append(self.allocator, name_copy);
                }
            }
        }

        return vars.toOwnedSlice(self.allocator);
    }

    fn freeToken(self: *Parser, token: Token) void {
        switch (token) {
            .text => |text| self.allocator.free(text),
            .variable => |v| {
                self.allocator.free(v.name);
                if (v.default) |d| self.allocator.free(d);
                self.allocator.free(v.filters);
            },
        }
    }
};

test "parse simple variable" {
    const allocator = std.testing.allocator;
    var parser_instance = Parser.init(allocator);
    const tokens = try parser_instance.parse("Hello, {{name}}!");
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

    try std.testing.expectEqual(@as(usize, 3), tokens.len);
    try std.testing.expectEqualStrings("Hello, ", tokens[0].text);
    try std.testing.expectEqualStrings("name", tokens[1].variable.name);
    try std.testing.expectEqualStrings("!", tokens[2].text);
}

test "parse variable with default" {
    const allocator = std.testing.allocator;
    var parser_instance = Parser.init(allocator);
    const tokens = try parser_instance.parse("{{name|World}}");
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

    try std.testing.expectEqual(@as(usize, 1), tokens.len);
    try std.testing.expectEqualStrings("name", tokens[0].variable.name);
    try std.testing.expectEqualStrings("World", tokens[0].variable.default.?);
}

test "parse unterminated variable" {
    const allocator = std.testing.allocator;
    var parser_instance = Parser.init(allocator);
    const result = parser_instance.parse("Hello, {{name");
    try std.testing.expectError(ParseError.UnterminatedVariable, result);
}
