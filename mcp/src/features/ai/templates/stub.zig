//! Templates stub — disabled at compile time.

const std = @import("std");
const types = @import("types.zig");

pub const ParseError = types.ParseError;
pub const RenderError = types.RenderError;
pub const Token = types.Token;

pub const Parser = struct {
    allocator: std.mem.Allocator,
    pub fn init(allocator: std.mem.Allocator) Parser {
        return .{ .allocator = allocator };
    }
    pub fn parse(_: *Parser, _: []const u8) ![]Token {
        return error.FeatureDisabled;
    }
    pub fn extractVariables(_: *Parser, _: []const Token) ![]const []const u8 {
        return error.FeatureDisabled;
    }
};

pub const RenderOptions = types.RenderOptions;

pub const Renderer = struct {
    allocator: std.mem.Allocator,
    options: RenderOptions,
    pub fn init(allocator: std.mem.Allocator, options: RenderOptions) Renderer {
        return .{ .allocator = allocator, .options = options };
    }
    pub fn render(_: *Renderer, _: []const Token, _: anytype) ![]u8 {
        return error.FeatureDisabled;
    }
    pub fn renderWithMap(_: *Renderer, _: []const Token, _: std.StringHashMapUnmanaged([]const u8)) ![]u8 {
        return error.FeatureDisabled;
    }
};

pub const BuiltinTemplates = types.BuiltinTemplates;

const TemplateInfo = struct { name: []const u8, description: []const u8, source: []const u8, variables: []const []const u8 };

pub fn getBuiltinTemplate(_: BuiltinTemplates) TemplateInfo {
    return .{ .name = "", .description = "Templates feature is disabled", .source = "", .variables = &.{} };
}

pub const Template = struct {
    allocator: std.mem.Allocator,
    name: []const u8,
    tokens: []const Token,
    source: []const u8,
    variables: []const []const u8,
    pub fn init(_: std.mem.Allocator, _: []const u8, _: []const u8) !Template {
        return error.FeatureDisabled;
    }
    pub fn deinit(_: *Template) void {}
    pub fn render(_: *const Template, _: anytype) ![]u8 {
        return error.FeatureDisabled;
    }
    pub fn renderWithOptions(_: *const Template, _: anytype, _: RenderOptions) ![]u8 {
        return error.FeatureDisabled;
    }
    pub fn validateValues(_: *const Template, _: anytype) bool {
        return false;
    }
    pub fn getVariables(_: *const Template) []const []const u8 {
        return &.{};
    }
};

pub const TemplateRegistry = struct {
    allocator: std.mem.Allocator,
    templates: std.StringHashMapUnmanaged(Template),
    pub fn init(allocator: std.mem.Allocator) TemplateRegistry {
        return .{ .allocator = allocator, .templates = std.StringHashMapUnmanaged(Template).empty };
    }
    pub fn deinit(_: *TemplateRegistry) void {}
    pub fn register(_: *TemplateRegistry, _: []const u8, _: []const u8) !void {
        return error.FeatureDisabled;
    }
    pub fn registerBuiltin(_: *TemplateRegistry, _: BuiltinTemplates) !void {
        return error.FeatureDisabled;
    }
    pub fn get(_: *const TemplateRegistry, _: []const u8) ?*const Template {
        return null;
    }
    pub fn renderTemplate(_: *const TemplateRegistry, _: []const u8, _: anytype) ![]u8 {
        return error.FeatureDisabled;
    }
    pub fn listTemplates(_: *const TemplateRegistry, _: std.mem.Allocator) ![][]const u8 {
        return error.FeatureDisabled;
    }
};

pub fn renderTemplate(_: std.mem.Allocator, _: []const u8, _: anytype) ![]u8 {
    return error.FeatureDisabled;
}
pub fn formatChatMessage(_: std.mem.Allocator, _: []const u8, _: []const u8) ![]u8 {
    return error.FeatureDisabled;
}
pub fn formatChatHistory(_: std.mem.Allocator, _: []const ChatMessage) ![]u8 {
    return error.FeatureDisabled;
}

pub const ChatMessage = types.ChatMessage;

test {
    std.testing.refAllDecls(@This());
}
