//! AI Agent Module
//!
//! Provides intelligent AI agents with configurable personas and conversation management.
//! Supports multiple backend integrations and maintains conversation context.

const std = @import("std");
const core = @import("core/mod.zig");

/// Re-export core types for convenience
pub const Allocator = core.Allocator;
pub const ArrayList = core.ArrayList;

/// Agent-specific error types with standardized naming
pub const AgentError = error{
    InvalidQuery,
    ApiKeyMissing,
    PersonaNotFound,
    ContextWindowExceeded,
    ModelNotAvailable,
    RateLimitExceeded,
    InvalidConfiguration,
    ResourceExhausted,
    OperationTimeout,
} || core.Error;

pub const PersonaType = enum {
    empathetic,
    direct,
    adaptive,
    creative,
    technical,
    solver,
    educator,
    counselor,

    /// Get a description string for the persona
    pub fn getDescription(self: PersonaType) []const u8 {
        return switch (self) {
            .empathetic => "empathetic and understanding",
            .direct => "direct and to the point",
            .adaptive => "adaptive to user needs",
            .creative => "creative and imaginative",
            .technical => "technical and precise",
            .solver => "problem-solving focused",
            .educator => "educational and explanatory",
            .counselor => "supportive and guiding",
        };
    }
};

pub const MessageRole = enum {
    user,
    assistant,
    system,
};

pub const Message = struct {
    role: MessageRole,
    content: []const u8,

    pub fn deinit(self: Message, allocator: std.mem.Allocator) void {
        allocator.free(self.content);
    }
};

/// AI Agent configuration
pub const AgentConfig = struct {
    default_persona: PersonaType = .adaptive,
    max_context_length: usize = 4096,
    enable_history: bool = true,
    temperature: f32 = 0.7,
};

/// Intelligent AI Agent with persona management
pub const Agent = struct {
    allocator: std.mem.Allocator,
    config: AgentConfig,
    current_persona: ?PersonaType = null,
    conversation_history: std.ArrayListUnmanaged(Message) = .{},

    /// Initialize a new AI agent
    pub fn init(allocator: std.mem.Allocator, config: AgentConfig) !*Agent {
        const self = try allocator.create(Agent);
        errdefer allocator.destroy(self);

        self.* = Agent{
            .allocator = allocator,
            .config = config,
            .current_persona = config.default_persona,
        };

        // Initialize conversation history if enabled
        if (config.enable_history) {
            self.conversation_history = std.ArrayListUnmanaged(Message){};
        }

        return self;
    }

    /// Clean up agent resources
    pub fn deinit(self: *Agent) void {
        // Clean up conversation history
        if (self.config.enable_history) {
            for (self.conversation_history.items) |*message| {
                message.deinit(self.allocator);
            }
            self.conversation_history.deinit(self.allocator);
        }

        self.allocator.destroy(self);
    }

    /// Set the agent's persona
    pub fn setPersona(self: *Agent, persona: PersonaType) void {
        self.current_persona = persona;
    }

    /// Get the current persona
    pub fn getPersona(self: *const Agent) ?PersonaType {
        return self.current_persona;
    }

    /// Start the agent (placeholder implementation)
    pub fn start(self: *Agent) !void {
        const logger = core.logging.ai_logger;
        const persona_desc = if (self.current_persona) |p| p.getDescription() else "no persona";
        logger.info("AI Agent started with {s} persona", .{persona_desc});
    }

    /// Add a message to conversation history
    pub fn addMessage(self: *Agent, role: MessageRole, content: []const u8) !void {
        if (!self.config.enable_history) return;

        const content_copy = try self.allocator.dupe(u8, content);
        errdefer self.allocator.free(content_copy);

        const message = Message{
            .role = role,
            .content = content_copy,
        };

        try self.conversation_history.append(self.allocator, message);

        // Trim history if it exceeds context length
        self.trimHistory();
    }

    /// Get conversation history
    pub fn getHistory(self: *const Agent) []const Message {
        if (!self.config.enable_history) return &.{};
        return self.conversation_history.items;
    }

    /// Clear conversation history
    pub fn clearHistory(self: *Agent) void {
        if (!self.config.enable_history) return;

        for (self.conversation_history.items) |*message| {
            message.deinit(self.allocator);
        }
        self.conversation_history.clearRetainingCapacity();
    }

    /// Trim conversation history to stay within context limits
    fn trimHistory(self: *Agent) void {
        if (!self.config.enable_history) return;

        var total_length: usize = 0;
        var trim_index: usize = 0;

        // Find where to trim to stay under context limit
        for (self.conversation_history.items, 0..) |message, i| {
            const new_length = total_length + message.content.len;
            if (new_length > self.config.max_context_length) {
                trim_index = i;
                break;
            }
            total_length = new_length;
        }

        // Remove older messages if needed
        if (trim_index > 0) {
            // Clean up messages being removed
            for (self.conversation_history.items[0..trim_index]) |*message| {
                message.deinit(self.allocator);
            }

            // Shift remaining messages
            const remaining = self.conversation_history.items[trim_index..];
            std.mem.copyForwards(Message, self.conversation_history.items[0..remaining.len], remaining);
            self.conversation_history.items.len = remaining.len;
        }
    }
};
