//! Prompt Builder with Export Support
//!
//! Constructs prompts in various formats with full export capability
//! for debugging and inspection via --show-prompt flags.

const std = @import("std");
const personas = @import("personas.zig");

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

/// Prompt builder with history and export support
pub const PromptBuilder = struct {
    allocator: std.mem.Allocator,
    persona: personas.Persona,
    messages: std.ArrayListUnmanaged(Message),
    include_system: bool = true,

    const Self = @This();

    /// Initialize with a persona type
    pub fn init(allocator: std.mem.Allocator, persona_type: personas.PersonaType) Self {
        return .{
            .allocator = allocator,
            .persona = personas.getPersona(persona_type),
            .messages = .{},
        };
    }

    /// Initialize with a custom persona
    pub fn initCustom(allocator: std.mem.Allocator, persona: personas.Persona) Self {
        return .{
            .allocator = allocator,
            .persona = persona,
            .messages = .{},
        };
    }

    pub fn deinit(self: *Self) void {
        for (self.messages.items) |msg| {
            if (msg.owned) {
                self.allocator.free(msg.content);
            }
        }
        self.messages.deinit(self.allocator);
    }

    /// Add a system message (overrides persona system prompt)
    pub fn addSystemMessage(self: *Self, content: []const u8) !void {
        try self.messages.append(self.allocator, .{
            .role = .system,
            .content = content,
            .owned = false,
        });
    }

    /// Add a user message
    pub fn addUserMessage(self: *Self, content: []const u8) !void {
        try self.messages.append(self.allocator, .{
            .role = .user,
            .content = content,
            .owned = false,
        });
    }

    /// Add a user message (owned copy)
    pub fn addUserMessageOwned(self: *Self, content: []const u8) !void {
        const copy = try self.allocator.dupe(u8, content);
        try self.messages.append(self.allocator, .{
            .role = .user,
            .content = copy,
            .owned = true,
        });
    }

    /// Add an assistant message
    pub fn addAssistantMessage(self: *Self, content: []const u8) !void {
        try self.messages.append(self.allocator, .{
            .role = .assistant,
            .content = content,
            .owned = false,
        });
    }

    /// Add an assistant message (owned copy)
    pub fn addAssistantMessageOwned(self: *Self, content: []const u8) !void {
        const copy = try self.allocator.dupe(u8, content);
        try self.messages.append(self.allocator, .{
            .role = .assistant,
            .content = copy,
            .owned = true,
        });
    }

    /// Add a tool response message
    pub fn addToolMessage(self: *Self, content: []const u8) !void {
        try self.messages.append(self.allocator, .{
            .role = .tool,
            .content = content,
            .owned = false,
        });
    }

    /// Add a generic message
    pub fn addMessage(self: *Self, role: Role, content: []const u8) !void {
        try self.messages.append(self.allocator, .{
            .role = role,
            .content = content,
            .owned = false,
        });
    }

    /// Clear all messages (keeps persona)
    pub fn clear(self: *Self) void {
        for (self.messages.items) |msg| {
            if (msg.owned) {
                self.allocator.free(msg.content);
            }
        }
        self.messages.clearRetainingCapacity();
    }

    /// Build the prompt in the specified format
    pub fn build(self: *Self, format: PromptFormat) ![]u8 {
        return switch (format) {
            .text => try self.buildText(),
            .json => try self.buildJson(),
            .chatml => try self.buildChatML(),
            .llama => try self.buildLlama(),
            .raw => try self.buildRaw(),
        };
    }

    /// Export prompt for debugging (human-readable format)
    pub fn exportDebug(self: *Self) ![]u8 {
        var result = std.ArrayListUnmanaged(u8).empty;
        errdefer result.deinit(self.allocator);

        // Header
        try result.appendSlice(self.allocator, "=== PROMPT EXPORT ===\n");
        try result.appendSlice(self.allocator, "Persona: ");
        try result.appendSlice(self.allocator, self.persona.name);
        try result.appendSlice(self.allocator, "\n");
        try result.appendSlice(self.allocator, "Temperature: ");
        var temp_buf: [8]u8 = undefined;
        const temp_str = std.fmt.bufPrint(&temp_buf, "{d:.1}", .{self.persona.suggested_temperature}) catch "?";
        try result.appendSlice(self.allocator, temp_str);
        try result.appendSlice(self.allocator, "\n\n");

        // System prompt
        if (self.include_system) {
            try result.appendSlice(self.allocator, "--- SYSTEM ---\n");
            try result.appendSlice(self.allocator, self.persona.system_prompt);
            try result.appendSlice(self.allocator, "\n\n");
        }

        // Messages
        try result.appendSlice(self.allocator, "--- MESSAGES ---\n");
        for (self.messages.items) |msg| {
            try result.appendSlice(self.allocator, "[");
            try result.appendSlice(self.allocator, msg.role.toString());
            try result.appendSlice(self.allocator, "]\n");
            try result.appendSlice(self.allocator, msg.content);
            try result.appendSlice(self.allocator, "\n\n");
        }

        try result.appendSlice(self.allocator, "=== END EXPORT ===\n");

        return result.toOwnedSlice(self.allocator);
    }

    /// Get message count
    pub fn messageCount(self: *const Self) usize {
        return self.messages.items.len;
    }

    /// Get all messages (including implicit system message)
    pub fn getAllMessages(self: *Self) ![]Message {
        var all = std.ArrayListUnmanaged(Message).empty;
        errdefer all.deinit(self.allocator);

        // Add system prompt first if enabled
        if (self.include_system) {
            try all.append(self.allocator, .{
                .role = .system,
                .content = self.persona.system_prompt,
                .owned = false,
            });
        }

        // Add all other messages
        for (self.messages.items) |msg| {
            try all.append(self.allocator, msg);
        }

        return all.toOwnedSlice(self.allocator);
    }

    // ========================================================================
    // Format Builders
    // ========================================================================

    fn buildText(self: *Self) ![]u8 {
        var result = std.ArrayListUnmanaged(u8).empty;
        errdefer result.deinit(self.allocator);

        // System prompt
        if (self.include_system) {
            try result.appendSlice(self.allocator, "System: ");
            try result.appendSlice(self.allocator, self.persona.system_prompt);
            try result.appendSlice(self.allocator, "\n\n");
        }

        // Messages with role prefixes
        for (self.messages.items) |msg| {
            try result.appendSlice(self.allocator, msg.role.toPrefix());
            try result.appendSlice(self.allocator, msg.content);
            try result.appendSlice(self.allocator, "\n");
        }

        // End with Assistant prompt
        try result.appendSlice(self.allocator, "Assistant:");

        return result.toOwnedSlice(self.allocator);
    }

    fn buildJson(self: *Self) ![]u8 {
        var result = std.ArrayListUnmanaged(u8).empty;
        errdefer result.deinit(self.allocator);

        try result.appendSlice(self.allocator, "[");

        var first = true;

        // System prompt
        if (self.include_system) {
            try self.appendJsonMessage(&result, .system, self.persona.system_prompt);
            first = false;
        }

        // Messages
        for (self.messages.items) |msg| {
            if (!first) {
                try result.appendSlice(self.allocator, ",");
            }
            try self.appendJsonMessage(&result, msg.role, msg.content);
            first = false;
        }

        try result.appendSlice(self.allocator, "]");

        return result.toOwnedSlice(self.allocator);
    }

    fn appendJsonMessage(self: *Self, result: *std.ArrayListUnmanaged(u8), role: Role, content: []const u8) !void {
        try result.appendSlice(self.allocator, "{\"role\":\"");
        try result.appendSlice(self.allocator, role.toString());
        try result.appendSlice(self.allocator, "\",\"content\":\"");
        // Escape content
        for (content) |c| {
            switch (c) {
                '"' => try result.appendSlice(self.allocator, "\\\""),
                '\\' => try result.appendSlice(self.allocator, "\\\\"),
                '\n' => try result.appendSlice(self.allocator, "\\n"),
                '\r' => try result.appendSlice(self.allocator, "\\r"),
                '\t' => try result.appendSlice(self.allocator, "\\t"),
                else => try result.append(self.allocator, c),
            }
        }
        try result.appendSlice(self.allocator, "\"}");
    }

    fn buildChatML(self: *Self) ![]u8 {
        var result = std.ArrayListUnmanaged(u8).empty;
        errdefer result.deinit(self.allocator);

        // System
        if (self.include_system) {
            try result.appendSlice(self.allocator, "<|system|>\n");
            try result.appendSlice(self.allocator, self.persona.system_prompt);
            try result.appendSlice(self.allocator, "\n<|end|>\n");
        }

        // Messages
        for (self.messages.items) |msg| {
            try result.appendSlice(self.allocator, "<|");
            try result.appendSlice(self.allocator, msg.role.toString());
            try result.appendSlice(self.allocator, "|>\n");
            try result.appendSlice(self.allocator, msg.content);
            try result.appendSlice(self.allocator, "\n<|end|>\n");
        }

        // Assistant turn
        try result.appendSlice(self.allocator, "<|assistant|>\n");

        return result.toOwnedSlice(self.allocator);
    }

    fn buildLlama(self: *Self) ![]u8 {
        var result = std.ArrayListUnmanaged(u8).empty;
        errdefer result.deinit(self.allocator);

        // Llama2/Mistral format
        if (self.include_system) {
            try result.appendSlice(self.allocator, "[INST] <<SYS>>\n");
            try result.appendSlice(self.allocator, self.persona.system_prompt);
            try result.appendSlice(self.allocator, "\n<</SYS>>\n\n");
        } else {
            try result.appendSlice(self.allocator, "[INST] ");
        }

        // Combine user messages for instruction
        for (self.messages.items) |msg| {
            if (msg.role == .user) {
                try result.appendSlice(self.allocator, msg.content);
                try result.appendSlice(self.allocator, " [/INST]");
            } else if (msg.role == .assistant) {
                try result.appendSlice(self.allocator, " ");
                try result.appendSlice(self.allocator, msg.content);
                try result.appendSlice(self.allocator, " [INST] ");
            }
        }

        return result.toOwnedSlice(self.allocator);
    }

    fn buildRaw(self: *Self) ![]u8 {
        var result = std.ArrayListUnmanaged(u8).empty;
        errdefer result.deinit(self.allocator);

        for (self.messages.items) |msg| {
            try result.appendSlice(self.allocator, msg.content);
            try result.appendSlice(self.allocator, "\n");
        }

        return result.toOwnedSlice(self.allocator);
    }
};

test "prompt builder text format" {
    const allocator = std.testing.allocator;

    var builder = PromptBuilder.init(allocator, .minimal);
    defer builder.deinit();

    try builder.addUserMessage("Hello");
    const prompt = try builder.build(.text);
    defer allocator.free(prompt);

    try std.testing.expect(std.mem.indexOf(u8, prompt, "User: Hello") != null);
    try std.testing.expect(std.mem.indexOf(u8, prompt, "Assistant:") != null);
}

test "prompt builder json format" {
    const allocator = std.testing.allocator;

    var builder = PromptBuilder.init(allocator, .minimal);
    defer builder.deinit();

    try builder.addUserMessage("Hi");
    const prompt = try builder.build(.json);
    defer allocator.free(prompt);

    try std.testing.expect(std.mem.indexOf(u8, prompt, "\"role\":\"user\"") != null);
    try std.testing.expect(std.mem.indexOf(u8, prompt, "\"content\":\"Hi\"") != null);
}

test "prompt builder export" {
    const allocator = std.testing.allocator;

    var builder = PromptBuilder.init(allocator, .coder);
    defer builder.deinit();

    try builder.addUserMessage("Write a function");
    const exported = try builder.exportDebug();
    defer allocator.free(exported);

    try std.testing.expect(std.mem.indexOf(u8, exported, "=== PROMPT EXPORT ===") != null);
    try std.testing.expect(std.mem.indexOf(u8, exported, "Persona: coder") != null);
}
