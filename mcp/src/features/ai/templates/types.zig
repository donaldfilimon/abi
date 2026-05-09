//! Shared types for the templates module.
//!
//! Used by mod.zig (via parser.zig, renderer.zig, library.zig)
//! and stub.zig to prevent type drift between enabled and disabled paths.
//!
//! Source of truth: parser.zig and renderer.zig definitions.

const std = @import("std");

/// Errors produced during template parsing.
pub const ParseError = error{
    UnterminatedVariable,
    EmptyVariableName,
    InvalidSyntax,
    OutOfMemory,
};

/// Errors produced during template rendering.
pub const RenderError = error{
    MissingVariable,
    InvalidValueType,
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

/// Options controlling template rendering behavior.
pub const RenderOptions = struct {
    /// Error on missing variables (otherwise use empty string).
    strict: bool = false,
    /// String to use for missing variables when not strict.
    missing_placeholder: []const u8 = "",
    /// Automatically escape HTML entities.
    auto_escape_html: bool = false,
};

/// Available built-in templates.
pub const BuiltinTemplates = enum {
    /// Basic system message template.
    system_message,
    /// Chat completion with system, context, and user message.
    chat_completion,
    /// RAG context injection template.
    rag_context,
    /// Tool/function calling prompt.
    tool_prompt,
    /// Code generation prompt.
    code_generation,
    /// Code review prompt.
    code_review,
    /// Text summarization prompt.
    summarization,
    /// Question answering prompt.
    question_answer,
    /// Translation prompt.
    translation,
    /// JSON extraction prompt.
    json_extraction,
    /// Classification prompt.
    classification,
    /// Conversation continuation.
    conversation,
};

/// A chat message with role and content.
pub const ChatMessage = struct {
    role: []const u8,
    content: []const u8,
};

test {
    std.testing.refAllDecls(@This());
}
