//! Prompt template system for structured prompt generation.
//!
//! Provides template parsing, rendering, and a library of built-in templates
//! for common LLM interaction patterns like chat, RAG, and tool use.

const std = @import("std");
const parser = @import("parser.zig");
const renderer = @import("renderer.zig");
const library = @import("library.zig");

pub const Parser = parser.Parser;
pub const ParseError = parser.ParseError;
pub const Token = parser.Token;

pub const Renderer = renderer.Renderer;
pub const RenderError = renderer.RenderError;
pub const RenderOptions = renderer.RenderOptions;

pub const BuiltinTemplates = library.BuiltinTemplates;
pub const getBuiltinTemplate = library.getBuiltinTemplate;

/// A parsed template ready for rendering.
pub const Template = struct {
    allocator: std.mem.Allocator,
    name: []const u8,
    tokens: []const Token,
    source: []const u8,
    variables: []const []const u8,

    pub fn init(allocator: std.mem.Allocator, name: []const u8, source: []const u8) !Template {
        var p = Parser.init(allocator);
        const tokens = try p.parse(source);
        const variables = try p.extractVariables(tokens);

        return .{
            .allocator = allocator,
            .name = try allocator.dupe(u8, name),
            .tokens = tokens,
            .source = try allocator.dupe(u8, source),
            .variables = variables,
        };
    }

    pub fn deinit(self: *Template) void {
        for (self.variables) |v| {
            self.allocator.free(v);
        }
        self.allocator.free(self.variables);
        for (self.tokens) |token| {
            if (token == .variable) {
                self.allocator.free(token.variable.name);
                if (token.variable.default) |d| {
                    self.allocator.free(d);
                }
            }
        }
        self.allocator.free(self.tokens);
        self.allocator.free(self.source);
        self.allocator.free(self.name);
        self.* = undefined;
    }

    /// Render the template with the given values.
    pub fn render(self: *const Template, values: anytype) ![]u8 {
        var r = Renderer.init(self.allocator, .{});
        return r.render(self.tokens, values);
    }

    /// Render the template with options.
    pub fn renderWithOptions(self: *const Template, values: anytype, options: RenderOptions) ![]u8 {
        var r = Renderer.init(self.allocator, options);
        return r.render(self.tokens, values);
    }

    /// Check if all required variables are provided.
    pub fn validateValues(self: *const Template, values: anytype) bool {
        const ValuesType = @TypeOf(values);
        const fields = std.meta.fields(ValuesType);

        for (self.variables) |variable| {
            var found = false;
            inline for (fields) |field| {
                if (std.mem.eql(u8, field.name, variable)) {
                    found = true;
                    break;
                }
            }
            if (!found) {
                // Check if there's a default value in tokens
                for (self.tokens) |token| {
                    if (token == .variable and
                        std.mem.eql(u8, token.variable.name, variable) and
                        token.variable.default != null)
                    {
                        found = true;
                        break;
                    }
                }
            }
            if (!found) return false;
        }
        return true;
    }

    /// Get required variable names.
    pub fn getVariables(self: *const Template) []const []const u8 {
        return self.variables;
    }
};

/// Template registry for managing named templates.
pub const TemplateRegistry = struct {
    allocator: std.mem.Allocator,
    templates: std.StringHashMapUnmanaged(Template),

    pub fn init(allocator: std.mem.Allocator) TemplateRegistry {
        return .{
            .allocator = allocator,
            .templates = std.StringHashMapUnmanaged(Template){},
        };
    }

    pub fn deinit(self: *TemplateRegistry) void {
        var iter = self.templates.iterator();
        while (iter.next()) |entry| {
            self.allocator.free(entry.key_ptr.*);
            var template = entry.value_ptr.*;
            template.deinit();
        }
        self.templates.deinit(self.allocator);
    }

    /// Register a template with a name.
    pub fn register(self: *TemplateRegistry, name: []const u8, source: []const u8) !void {
        var template = try Template.init(self.allocator, name, source);
        errdefer template.deinit();

        const name_copy = try self.allocator.dupe(u8, name);
        errdefer self.allocator.free(name_copy);

        try self.templates.put(self.allocator, name_copy, template);
    }

    /// Register a built-in template by its enum value.
    pub fn registerBuiltin(self: *TemplateRegistry, builtin: BuiltinTemplates) !void {
        const info = library.getBuiltinTemplate(builtin);
        try self.register(info.name, info.source);
    }

    /// Get a template by name.
    pub fn get(self: *const TemplateRegistry, name: []const u8) ?*const Template {
        if (self.templates.getPtr(name)) |ptr| {
            return ptr;
        }
        return null;
    }

    /// Render a template by name with values.
    pub fn renderTemplate(self: *const TemplateRegistry, name: []const u8, values: anytype) ![]u8 {
        const template = self.get(name) orelse return error.TemplateNotFound;
        return template.render(values);
    }

    /// List all registered template names.
    pub fn listTemplates(self: *const TemplateRegistry, allocator: std.mem.Allocator) ![][]const u8 {
        var names = std.ArrayListUnmanaged([]const u8){};
        errdefer names.deinit(allocator);

        var iter = self.templates.iterator();
        while (iter.next()) |entry| {
            try names.append(allocator, try allocator.dupe(u8, entry.key_ptr.*));
        }

        return names.toOwnedSlice(allocator);
    }
};

/// Convenience function to parse and render a template in one step.
pub fn renderTemplate(allocator: std.mem.Allocator, source: []const u8, values: anytype) ![]u8 {
    var template = try Template.init(allocator, "inline", source);
    defer template.deinit();
    return template.render(values);
}

/// Format a chat message with role and content.
pub fn formatChatMessage(allocator: std.mem.Allocator, role: []const u8, content: []const u8) ![]u8 {
    return std.fmt.allocPrint(allocator, "<|{s}|>\n{s}\n<|end|>", .{ role, content });
}

/// Format multiple chat messages.
pub fn formatChatHistory(allocator: std.mem.Allocator, messages: []const ChatMessage) ![]u8 {
    var result = std.ArrayListUnmanaged(u8){};
    errdefer result.deinit(allocator);

    for (messages) |msg| {
        const formatted = try formatChatMessage(allocator, msg.role, msg.content);
        defer allocator.free(formatted);
        try result.appendSlice(allocator, formatted);
        try result.append(allocator, '\n');
    }

    return result.toOwnedSlice(allocator);
}

pub const ChatMessage = struct {
    role: []const u8,
    content: []const u8,
};

test "template basic rendering" {
    const allocator = std.testing.allocator;
    var template = try Template.init(allocator, "test", "Hello, {{name}}!");
    defer template.deinit();

    const result = try template.render(.{ .name = "World" });
    defer allocator.free(result);

    try std.testing.expectEqualStrings("Hello, World!", result);
}

test "template registry" {
    const allocator = std.testing.allocator;
    var registry = TemplateRegistry.init(allocator);
    defer registry.deinit();

    try registry.register("greeting", "Hello, {{name}}!");
    const template = registry.get("greeting");
    try std.testing.expect(template != null);
}

test "template with default value" {
    const allocator = std.testing.allocator;
    var template = try Template.init(allocator, "test", "Hello, {{name|World}}!");
    defer template.deinit();

    const result = try template.render(.{});
    defer allocator.free(result);

    try std.testing.expectEqualStrings("Hello, World!", result);
}

test {
    _ = parser;
    _ = renderer;
    _ = library;
}
