const std = @import("std");
const core = @import("../core/mod.zig");

/// Re-export commonly used types for convenience
pub const Allocator = core.Allocator;
pub const ArrayList = core.ArrayList;

/// Basic AI types for minimal functionality
pub const PersonaType = enum {
    adaptive,
    casual,
    technical,

    pub fn getDescription(self: PersonaType) []const u8 {
        return switch (self) {
            .adaptive => "Adaptive AI that adjusts to user needs",
            .casual => "Casual, friendly AI for everyday conversation",
            .technical => "Technical AI focused on detailed analysis",
        };
    }

    pub fn getSystemPrompt(self: PersonaType) []const u8 {
        return switch (self) {
            .adaptive => "You are an adaptive AI assistant.",
            .casual => "You are a friendly, casual AI assistant.",
            .technical => "Technical, precise AI assistant.",
        };
    }
};

/// AI backend types
pub const Backend = enum {
    local,
    openai,
    anthropic,
    google,
    azure,

    pub fn getDescription(self: Backend) []const u8 {
        return switch (self) {
            .local => "Local AI model",
            .openai => "OpenAI API",
            .anthropic => "Anthropic Claude API",
            .google => "Google AI API",
            .azure => "Microsoft Azure OpenAI",
        };
    }
};

/// Basic message structure
pub const Message = struct {
    role: Role,
    content: []const u8,

    pub const Role = enum {
        system,
        user,
        assistant,
        function,
    };

    pub fn dupe(self: Message, allocator: core.Allocator) !Message {
        return .{
            .role = self.role,
            .content = try allocator.dupe(u8, self.content),
        };
    }

    pub fn deinit(self: Message, allocator: core.Allocator) void {
        allocator.free(self.content);
    }
};

/// Basic context for conversation
pub const Context = struct {
    messages: core.ArrayList(Message),
    persona: PersonaType,
    allocator: core.Allocator,

    pub fn init(allocator: core.Allocator, persona: PersonaType) !*Context {
        var messages = core.ArrayList(Message){};
        try messages.ensureTotalCapacity(allocator, 10);

        const context = try allocator.create(Context);
        context.* = Context{
            .messages = messages,
            .persona = persona,
            .allocator = allocator,
        };
        return context;
    }

    pub fn deinit(self: *Context) void {
        for (self.messages.items) |*msg| {
            msg.deinit(self.allocator);
        }
        self.messages.deinit(self.allocator);
        self.allocator.destroy(self);
    }

    pub fn addMessage(self: *Context, msg: Message) !void {
        const duped = try msg.dupe(self.allocator);
        try self.messages.append(self.allocator, duped);
    }

    pub fn clearHistory(self: *Context) void {
        for (self.messages.items) |*msg| {
            msg.deinit(self.allocator);
        }
        self.messages.clearRetainingCapacity();
    }

    pub fn getSystemPrompt(self: *const Context) []const u8 {
        return self.persona.getSystemPrompt();
    }
};

/// Basic AI agent
pub const Agent = struct {
    allocator: core.Allocator,
    context: *Context,

    pub fn init(allocator: core.Allocator, persona: PersonaType) !Agent {
        return Agent{
            .allocator = allocator,
            .context = try Context.init(allocator, persona),
        };
    }

    pub fn deinit(self: *Agent) void {
        self.context.deinit();
    }

    pub fn setPersona(self: *Agent, persona: PersonaType) void {
        self.context.persona = persona;
    }

    pub fn clearHistory(self: *Agent) void {
        self.context.clearHistory();
    }

    pub fn generate(self: *Agent, prompt: []const u8, options: GenerationOptions) !GenerationResult {
        _ = options; // Options not yet implemented but kept for future use
        // Add user message to context
        const user_message = Message{
            .role = .user,
            .content = try self.allocator.dupe(u8, prompt),
        };
        try self.context.messages.append(self.allocator, user_message);

        // Generate persona-aware response
        const persona_desc = self.context.persona.getSystemPrompt();
        const response_content = try std.fmt.allocPrint(self.allocator, "{s} {s}", .{ persona_desc, "I understand your message and would help you with that." });

        // Add assistant response to context
        const assistant_message = Message{
            .role = .assistant,
            .content = try self.allocator.dupe(u8, response_content),
        };
        try self.context.messages.append(self.allocator, assistant_message);

        // Calculate basic usage stats
        const usage = GenerationResult.Usage{
            .prompt_tokens = @intCast(prompt.len / 4), // Rough estimate
            .completion_tokens = @intCast(response_content.len / 4),
            .total_tokens = @intCast((prompt.len + response_content.len) / 4),
        };

        return GenerationResult{
            .content = response_content,
            .usage = usage,
        };
    }
};

/// Estimate the number of tokens in text (simplified implementation)
pub fn estimateTokens(text: []const u8) !usize {
    // Simple estimation: roughly 4 characters per token
    return text.len / 4;
}

/// Model configuration for AI operations
pub const ModelConfig = struct {
    model_id: []const u8,
    backend: Backend,
    capabilities: Capabilities,

    pub const Capabilities = struct {
        streaming: bool = false,
        context_window: usize = 4096,
    };
};

/// Generation options for AI operations
pub const GenerationOptions = struct {
    stream_callback: ?*const fn (chunk: []const u8) void = null,
};

/// Generation result from AI operations
pub const GenerationResult = struct {
    content: []const u8,
    usage: ?Usage = null,

    pub const Usage = struct {
        prompt_tokens: usize,
        completion_tokens: usize,
        total_tokens: usize,
    };
};

test "AI agent basic functionality" {
    const allocator = std.testing.allocator;

    var agent = try Agent.init(allocator, .adaptive);
    defer agent.deinit();

    try std.testing.expectEqual(PersonaType.adaptive, agent.context.persona);

    agent.setPersona(.technical);
    try std.testing.expectEqual(PersonaType.technical, agent.context.persona);
}

test "AI context management" {
    const allocator = std.testing.allocator;

    var ctx = try Context.init(allocator, .adaptive);
    defer ctx.deinit();

    try ctx.addMessage(Message{
        .role = .user,
        .content = "Hello, AI!",
    });

    try std.testing.expectEqual(@as(usize, 1), ctx.messages.items.len);

    ctx.clearHistory();
    try std.testing.expectEqual(@as(usize, 0), ctx.messages.items.len);
}
