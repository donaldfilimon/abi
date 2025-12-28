//! Shared types and utilities for connector modules.
//!
//! Provides common data structures used across different AI service connectors
//! such as OpenAI, Ollama, and HuggingFace.

const std = @import("std");

/// Chat message structure used in conversation-based APIs.
/// Compatible with OpenAI, Ollama, and similar chat completions formats.
pub const ChatMessage = struct {
    /// The role of the message sender (e.g., "system", "user", "assistant")
    role: []const u8,
    /// The content of the message
    content: []const u8,
};

/// Message role constants for common API formats
pub const Role = struct {
    pub const SYSTEM = "system";
    pub const USER = "user";
    pub const ASSISTANT = "assistant";
};

test "ChatMessage can be created" {
    const msg = ChatMessage{
        .role = Role.USER,
        .content = "Hello, world!",
    };

    try std.testing.expectEqualStrings("user", msg.role);
    try std.testing.expectEqualStrings("Hello, world!", msg.content);
}
