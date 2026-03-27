//! Shared types for the prompts module.
//!
//! Canonical definitions used by both mod.zig and stub.zig to prevent type drift.

const std = @import("std");

/// Available profile types
pub const ProfileType = enum {
    /// General-purpose helpful assistant
    assistant,
    /// Code-focused programming assistant
    coder,
    /// Creative writing assistant
    writer,
    /// Data analysis and research assistant
    analyst,
    /// Friendly conversational companion
    companion,
    /// Technical documentation helper
    docs,
    /// Code review specialist
    reviewer,
    /// Minimal/direct response mode
    minimal,
    /// Abbey - opinionated, emotionally intelligent AI
    abbey,
    /// Ralph - Iterative, tireless worker for complex tasks
    ralph,
    /// Aviva - direct expert for concise, factual output
    aviva,
    /// Abi - adaptive moderator and router
    abi,
    /// Ava - locally-trained assistant based on gpt-oss
    ava,
};

/// Profile definition with complete system instructions
pub const Profile = struct {
    /// Short identifier
    name: []const u8 = "disabled",
    /// Human-readable description
    description: []const u8 = "",
    /// Full system prompt with instructions
    system_prompt: []const u8 = "",
    /// Suggested temperature (0.0-2.0)
    suggested_temperature: f32 = 0.7,
    /// Whether to include examples in responses
    include_examples: bool = false,
};

/// Message role in conversation
pub const Role = enum {
    system,
    user,
    assistant,
    tool,

    pub fn toString(self: Role) []const u8 {
        return switch (self) {
            .system => "system",
            .user => "user",
            .assistant => "assistant",
            .tool => "tool",
        };
    }

    pub fn toPrefix(self: Role) []const u8 {
        return switch (self) {
            .system => "System: ",
            .user => "User: ",
            .assistant => "Assistant: ",
            .tool => "Tool: ",
        };
    }
};

/// A single message in the conversation
pub const Message = struct {
    role: Role,
    content: []const u8,
    /// Whether content is owned by builder (needs freeing)
    owned: bool = false,
};

/// Output format for built prompts
pub const PromptFormat = enum {
    /// Plain text with role prefixes (User: / Assistant:)
    text,
    /// JSON array of message objects
    json,
    /// ChatML format (<|system|>, <|user|>, etc.)
    chatml,
    /// Llama2/Mistral instruction format
    llama,
    /// Raw messages without formatting
    raw,
};

test {
    std.testing.refAllDecls(@This());
}
