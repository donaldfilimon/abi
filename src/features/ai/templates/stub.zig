//! Templates stub — disabled at compile time.

const std = @import("std");

pub const ParseError = error{TemplatesDisabled};
pub const RenderError = error{TemplatesDisabled};

pub const Token = union(enum) {
    text: []const u8,
    variable: Variable,
    pub const Variable = struct { name: []const u8, default: ?[]const u8, filters: []const Filter };
    pub const Filter = enum { upper, lower, trim, escape_html, escape_json };
};

pub const Parser = struct {
    allocator: std.mem.Allocator,
    pub fn init(allocator: std.mem.Allocator) Parser {
        return .{ .allocator = allocator };
    }
    pub fn parse(_: *Parser, _: []const u8) ParseError![]Token {
        return ParseError.TemplatesDisabled;
    }
    pub fn extractVariables(_: *Parser, _: []const Token) ![]const []const u8 {
        return ParseError.TemplatesDisabled;
    }
};

pub const RenderOptions = struct { strict: bool = false, missing_placeholder: []const u8 = "", auto_escape_html: bool = false };

pub const Renderer = struct {
    allocator: std.mem.Allocator,
    options: RenderOptions,
    pub fn init(allocator: std.mem.Allocator, options: RenderOptions) Renderer {
        return .{ .allocator = allocator, .options = options };
    }
    pub fn render(_: *Renderer, _: []const Token, _: anytype) RenderError![]u8 {
        return RenderError.TemplatesDisabled;
    }
    pub fn renderWithMap(_: *Renderer, _: []const Token, _: std.StringHashMapUnmanaged([]const u8)) RenderError![]u8 {
        return RenderError.TemplatesDisabled;
    }
};

pub const BuiltinTemplates = enum {
    system_message,
    chat_completion,
    rag_context,
    tool_prompt,
    code_generation,
    code_review,
    summarization,
    question_answer,
    translation,
    json_extraction,
    classification,
    conversation,
};

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
        return error.TemplatesDisabled;
    }
    pub fn deinit(_: *Template) void {}
    pub fn render(_: *const Template, _: anytype) ![]u8 {
        return error.TemplatesDisabled;
    }
    pub fn renderWithOptions(_: *const Template, _: anytype, _: RenderOptions) ![]u8 {
        return error.TemplatesDisabled;
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
        return .{ .allocator = allocator, .templates = std.StringHashMapUnmanaged(Template){} };
    }
    pub fn deinit(_: *TemplateRegistry) void {}
    pub fn register(_: *TemplateRegistry, _: []const u8, _: []const u8) !void {
        return error.TemplatesDisabled;
    }
    pub fn registerBuiltin(_: *TemplateRegistry, _: BuiltinTemplates) !void {
        return error.TemplatesDisabled;
    }
    pub fn get(_: *const TemplateRegistry, _: []const u8) ?*const Template {
        return null;
    }
    pub fn renderTemplate(_: *const TemplateRegistry, _: []const u8, _: anytype) ![]u8 {
        return error.TemplatesDisabled;
    }
    pub fn listTemplates(_: *const TemplateRegistry, _: std.mem.Allocator) ![][]const u8 {
        return error.TemplatesDisabled;
    }
};

pub fn renderTemplate(_: std.mem.Allocator, _: []const u8, _: anytype) ![]u8 {
    return error.TemplatesDisabled;
}
pub fn formatChatMessage(_: std.mem.Allocator, _: []const u8, _: []const u8) ![]u8 {
    return error.TemplatesDisabled;
}
pub fn formatChatHistory(_: std.mem.Allocator, _: []const ChatMessage) ![]u8 {
    return error.TemplatesDisabled;
}

pub const ChatMessage = struct { role: []const u8, content: []const u8 };
