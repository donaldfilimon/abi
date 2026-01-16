//! Abbey Discord Bot Integration
//!
//! Integrates Abbey AI with Discord for conversational interactions:
//! - Message handling with context awareness
//! - Per-channel/user session management
//! - Emotional response adaptation
//! - Slash command support
//!
//! Usage:
//!   var bot = try AbbeyDiscordBot.init(allocator, config);
//!   defer bot.deinit();
//!   try bot.start(); // Starts listening for messages

const std = @import("std");
const engine = @import("engine.zig");
const core_types = @import("core/types.zig");
const core_config = @import("core/config.zig");
const discord = @import("../../connectors/discord/mod.zig");
const emotions = @import("emotions.zig");

// ============================================================================
// Discord Bot Types
// ============================================================================

pub const DiscordBotError = error{
    ClientCreationFailed,
    SessionNotFound,
    MessageSendFailed,
    InvalidConfiguration,
    BotNotStarted,
    BotAlreadyRunning,
} || std.mem.Allocator.Error;

/// Configuration for Abbey Discord bot
pub const DiscordBotConfig = struct {
    /// Abbey engine configuration
    abbey: core_config.AbbeyConfig = .{},
    /// Discord bot token (overrides env var)
    bot_token: ?[]const u8 = null,
    /// Whether to respond to all messages or only mentions
    respond_to_all: bool = false,
    /// Command prefix for text commands (e.g., "!abbey")
    command_prefix: ?[]const u8 = null,
    /// Maximum message length before truncating
    max_message_length: usize = 2000,
    /// Whether to show typing indicator while processing
    show_typing: bool = true,
    /// Whether to add emotional reactions based on context
    add_emotional_reactions: bool = true,
    /// Default channel IDs to listen on (empty = all)
    allowed_channels: []const []const u8 = &.{},
};

// ============================================================================
// Session Manager
// ============================================================================

/// Manages per-user/channel conversation sessions
pub const SessionManager = struct {
    allocator: std.mem.Allocator,
    sessions: std.StringHashMapUnmanaged(Session),

    pub const Session = struct {
        user_id: []const u8,
        channel_id: []const u8,
        engine_session_id: ?core_types.SessionId,
        last_interaction: i64,
        message_count: usize,
        emotional_trend: emotions.EmotionType,
    };

    const Self = @This();

    pub fn init(allocator: std.mem.Allocator) Self {
        return Self{
            .allocator = allocator,
            .sessions = .{},
        };
    }

    pub fn deinit(self: *Self) void {
        var it = self.sessions.iterator();
        while (it.next()) |entry| {
            self.allocator.free(entry.key_ptr.*);
        }
        self.sessions.deinit(self.allocator);
    }

    /// Get or create a session for a user in a channel
    pub fn getOrCreateSession(self: *Self, user_id: []const u8, channel_id: []const u8) !*Session {
        const key = try self.makeKey(user_id, channel_id);

        const gop = try self.sessions.getOrPut(self.allocator, key);
        if (!gop.found_existing) {
            gop.value_ptr.* = Session{
                .user_id = user_id,
                .channel_id = channel_id,
                .engine_session_id = null,
                .last_interaction = core_types.getTimestampSec(),
                .message_count = 0,
                .emotional_trend = .neutral,
            };
        } else {
            // Key was already present, free the duplicate
            self.allocator.free(key);
        }
        return gop.value_ptr;
    }

    fn makeKey(self: *Self, user_id: []const u8, channel_id: []const u8) ![]u8 {
        return std.fmt.allocPrint(self.allocator, "{s}:{s}", .{ user_id, channel_id });
    }

    /// Clean up old sessions (older than max_age_seconds)
    pub fn cleanup(self: *Self, max_age_seconds: i64) void {
        const now = core_types.getTimestampSec();
        var to_remove = std.ArrayListUnmanaged([]const u8){};
        defer to_remove.deinit(self.allocator);

        var it = self.sessions.iterator();
        while (it.next()) |entry| {
            if (now - entry.value_ptr.last_interaction > max_age_seconds) {
                to_remove.append(self.allocator, entry.key_ptr.*) catch continue;
            }
        }

        for (to_remove.items) |key| {
            _ = self.sessions.remove(key);
            self.allocator.free(key);
        }
    }
};

// ============================================================================
// Abbey Discord Bot
// ============================================================================

/// Discord bot powered by Abbey AI
pub const AbbeyDiscordBot = struct {
    allocator: std.mem.Allocator,
    config: DiscordBotConfig,
    abbey_engine: engine.AbbeyEngine,
    session_manager: SessionManager,
    running: bool = false,
    discord_client: ?DiscordClientWrapper = null,

    const Self = @This();

    /// Discord client wrapper for API calls
    const DiscordClientWrapper = struct {
        allocator: std.mem.Allocator,

        pub fn init(allocator: std.mem.Allocator) !DiscordClientWrapper {
            return .{ .allocator = allocator };
        }

        pub fn deinit(self: *DiscordClientWrapper) void {
            _ = self;
        }

        /// Send a message to a channel
        pub fn sendMessage(self: *DiscordClientWrapper, channel_id: []const u8, content: []const u8) !void {
            var client = discord.createClient(self.allocator) catch return error.MessageSendFailed;
            defer client.deinit();

            _ = client.createMessage(channel_id, content) catch return error.MessageSendFailed;
        }

        /// Add a reaction to a message
        pub fn addReaction(self: *DiscordClientWrapper, channel_id: []const u8, message_id: []const u8, emoji: []const u8) !void {
            var client = discord.createClient(self.allocator) catch return;
            defer client.deinit();

            client.createReaction(channel_id, message_id, emoji) catch {};
        }

        /// Show typing indicator
        pub fn triggerTyping(self: *DiscordClientWrapper, channel_id: []const u8) void {
            var client = discord.createClient(self.allocator) catch return;
            defer client.deinit();

            client.triggerTypingIndicator(channel_id) catch {};
        }
    };

    /// Initialize the Abbey Discord bot
    pub fn init(allocator: std.mem.Allocator, config: DiscordBotConfig) !Self {
        var abbey_eng = try engine.AbbeyEngine.init(allocator, config.abbey);
        errdefer abbey_eng.deinit();

        return Self{
            .allocator = allocator,
            .config = config,
            .abbey_engine = abbey_eng,
            .session_manager = SessionManager.init(allocator),
        };
    }

    /// Clean up resources
    pub fn deinit(self: *Self) void {
        if (self.discord_client) |*client| {
            client.deinit();
        }
        self.session_manager.deinit();
        self.abbey_engine.deinit();
    }

    /// Process an incoming Discord message
    pub fn handleMessage(
        self: *Self,
        message: discord.Message,
    ) !?MessageResponse {
        // Skip bot messages
        if (message.author.bot) return null;

        // Check if we should respond
        if (!self.shouldRespond(&message)) return null;

        // Get or create session
        var session = try self.session_manager.getOrCreateSession(
            message.author.id,
            message.channel_id,
        );
        session.last_interaction = core_types.getTimestampSec();
        session.message_count += 1;

        // Start Abbey conversation if needed
        if (!self.abbey_engine.conversation_active) {
            try self.abbey_engine.startConversation(message.author.id);
        }

        // Show typing indicator
        if (self.config.show_typing) {
            if (self.discord_client) |*client| {
                client.triggerTyping(message.channel_id);
            }
        }

        // Process with Abbey
        var response = try self.abbey_engine.process(message.content);
        defer response.deinit(self.allocator);

        // Update session emotional trend
        session.emotional_trend = response.emotional_context.detected;

        // Build response
        return MessageResponse{
            .content = try self.formatResponse(&response),
            .channel_id = message.channel_id,
            .reply_to = message.id,
            .emotional_reaction = if (self.config.add_emotional_reactions)
                getEmotionalEmoji(response.emotional_context.detected)
            else
                null,
        };
    }

    /// Determine if bot should respond to a message
    fn shouldRespond(self: *Self, message: *const discord.Message) bool {
        // Check if message is in allowed channels
        if (self.config.allowed_channels.len > 0) {
            var allowed = false;
            for (self.config.allowed_channels) |ch| {
                if (std.mem.eql(u8, ch, message.channel_id)) {
                    allowed = true;
                    break;
                }
            }
            if (!allowed) return false;
        }

        // Check for command prefix
        if (self.config.command_prefix) |prefix| {
            if (std.mem.startsWith(u8, message.content, prefix)) {
                return true;
            }
        }

        // Check for mentions (would need to check mentions array in real impl)
        // For now, respond to all if configured
        return self.config.respond_to_all;
    }

    /// Format Abbey response for Discord
    fn formatResponse(self: *Self, response: *const engine.Response) ![]u8 {
        var output = std.ArrayListUnmanaged(u8){};
        errdefer output.deinit(self.allocator);

        // Add content
        const content_len = @min(response.content.len, self.config.max_message_length - 50);
        try output.appendSlice(self.allocator, response.content[0..content_len]);

        // Add truncation indicator if needed
        if (response.content.len > self.config.max_message_length - 50) {
            try output.appendSlice(self.allocator, "...");
        }

        // Add confidence indicator for low confidence responses
        if (response.confidence.level == .low or response.confidence.level == .uncertain) {
            try output.appendSlice(self.allocator, "\n\n_Note: I'm not fully confident about this response._");
        }

        return output.toOwnedSlice(self.allocator);
    }

    /// Get emoji based on emotional context
    fn getEmotionalEmoji(emotion: core_types.EmotionType) ?[]const u8 {
        return switch (emotion) {
            .happy, .excited => "\xF0\x9F\x98\x8A", // smiling face
            .curious => "\xF0\x9F\xA4\x94", // thinking face
            .concerned, .worried => "\xF0\x9F\x98\x9F", // worried face
            .empathetic => "\xE2\x9D\xA4\xEF\xB8\x8F", // heart
            .frustrated => "\xF0\x9F\x98\xA4", // huffing face
            .neutral => null,
            .sad => "\xF0\x9F\x98\x94", // pensive face
            .confused => "\xF0\x9F\x98\x95", // confused face
            .focused => "\xF0\x9F\x8E\xAF", // target
            .pleased => "\xF0\x9F\x98\x8C", // relieved face
            else => null,
        };
    }

    /// Send a response to Discord
    pub fn sendResponse(self: *Self, response: MessageResponse) !void {
        const client = self.discord_client orelse return DiscordBotError.BotNotStarted;
        _ = client;

        // Would use discord client to send message
        // For now, this is a stub showing the pattern
        _ = response;
    }

    /// Get bot statistics
    pub fn getStats(self: *Self) BotStats {
        const engine_stats = self.abbey_engine.getStats();
        return BotStats{
            .total_messages_processed = engine_stats.total_queries,
            .active_sessions = self.session_manager.sessions.count(),
            .current_emotion = engine_stats.current_emotion,
            .avg_response_time_ms = engine_stats.avg_response_time_ms,
            .relationship_score = engine_stats.relationship_score,
        };
    }

    /// Reset all sessions
    pub fn resetSessions(self: *Self) void {
        self.session_manager.deinit();
        self.session_manager = SessionManager.init(self.allocator);
    }
};

/// Response to send to Discord
pub const MessageResponse = struct {
    content: []const u8,
    channel_id: []const u8,
    reply_to: ?[]const u8 = null,
    emotional_reaction: ?[]const u8 = null,
};

/// Bot statistics
pub const BotStats = struct {
    total_messages_processed: usize,
    active_sessions: usize,
    current_emotion: core_types.EmotionType,
    avg_response_time_ms: f32,
    relationship_score: f32,
};

// ============================================================================
// Command Handling
// ============================================================================

/// Discord slash command definitions for Abbey
pub const AbbeyCommands = struct {
    /// Chat with Abbey
    pub const chat_command = discord.ApplicationCommand{
        .name = "abbey",
        .description = "Chat with Abbey AI",
        .type = .chat_input,
        .options = &[_]discord.ApplicationCommandOption{
            .{
                .name = "message",
                .description = "Your message to Abbey",
                .type = .string,
                .required = true,
            },
        },
    };

    /// Get Abbey's current emotional state
    pub const mood_command = discord.ApplicationCommand{
        .name = "abbey-mood",
        .description = "See Abbey's current emotional state",
        .type = .chat_input,
        .options = &[_]discord.ApplicationCommandOption{},
    };

    /// Get conversation statistics
    pub const stats_command = discord.ApplicationCommand{
        .name = "abbey-stats",
        .description = "View conversation statistics",
        .type = .chat_input,
        .options = &[_]discord.ApplicationCommandOption{},
    };

    /// Clear conversation context
    pub const clear_command = discord.ApplicationCommand{
        .name = "abbey-clear",
        .description = "Clear conversation context with Abbey",
        .type = .chat_input,
        .options = &[_]discord.ApplicationCommandOption{},
    };
};

/// Handle a slash command interaction
pub fn handleSlashCommand(
    bot: *AbbeyDiscordBot,
    interaction: discord.Interaction,
) !discord.InteractionResponse {
    const data = interaction.data orelse {
        return buildErrorResponse("Invalid interaction data");
    };

    const command_name = data.name orelse {
        return buildErrorResponse("Missing command name");
    };

    if (std.mem.eql(u8, command_name, "abbey")) {
        return handleChatCommand(bot, &data);
    } else if (std.mem.eql(u8, command_name, "abbey-mood")) {
        return handleMoodCommand(bot);
    } else if (std.mem.eql(u8, command_name, "abbey-stats")) {
        return handleStatsCommand(bot);
    } else if (std.mem.eql(u8, command_name, "abbey-clear")) {
        return handleClearCommand(bot);
    }

    return buildErrorResponse("Unknown command");
}

fn handleChatCommand(bot: *AbbeyDiscordBot, data: *const discord.InteractionData) !discord.InteractionResponse {
    // Extract message from options
    const options = data.options orelse {
        return buildErrorResponse("Missing options");
    };

    var message_content: ?[]const u8 = null;
    for (options) |opt| {
        if (opt.name) |name| {
            if (std.mem.eql(u8, name, "message")) {
                message_content = opt.value;
                break;
            }
        }
    }

    const user_message = message_content orelse {
        return buildErrorResponse("Missing message");
    };

    // Process with Abbey
    var response = try bot.abbey_engine.process(user_message);
    defer response.deinit(bot.allocator);

    return discord.InteractionResponse{
        .type = .channel_message_with_source,
        .data = .{
            .content = response.content,
        },
    };
}

fn handleMoodCommand(bot: *AbbeyDiscordBot) discord.InteractionResponse {
    const emotional = bot.abbey_engine.getEmotionalState();
    var buf: [256]u8 = undefined;
    const content = std.fmt.bufPrint(&buf, "Current mood: {t} (confidence: {d:.0}%)\nValence: {d:.2}, Arousal: {d:.2}", .{
        emotional.detected,
        emotional.confidence * 100,
        emotional.valence,
        emotional.arousal,
    }) catch "Unable to get mood";

    return discord.InteractionResponse{
        .type = .channel_message_with_source,
        .data = .{
            .content = content,
        },
    };
}

fn handleStatsCommand(bot: *AbbeyDiscordBot) discord.InteractionResponse {
    const stats = bot.getStats();
    var buf: [512]u8 = undefined;
    const content = std.fmt.bufPrint(&buf, "**Abbey Stats**\nMessages: {d}\nSessions: {d}\nAvg Response: {d:.1}ms\nRelationship: {d:.0}%", .{
        stats.total_messages_processed,
        stats.active_sessions,
        stats.avg_response_time_ms,
        stats.relationship_score * 100,
    }) catch "Unable to get stats";

    return discord.InteractionResponse{
        .type = .channel_message_with_source,
        .data = .{
            .content = content,
        },
    };
}

fn handleClearCommand(bot: *AbbeyDiscordBot) discord.InteractionResponse {
    bot.abbey_engine.clearConversation();
    return discord.InteractionResponse{
        .type = .channel_message_with_source,
        .data = .{
            .content = "Conversation context cleared! Let's start fresh.",
        },
    };
}

fn buildErrorResponse(message: []const u8) discord.InteractionResponse {
    return discord.InteractionResponse{
        .type = .channel_message_with_source,
        .data = .{
            .content = message,
            .flags = discord.MessageFlags.ephemeral,
        },
    };
}

// ============================================================================
// Tests
// ============================================================================

test "session manager" {
    const allocator = std.testing.allocator;

    var manager = SessionManager.init(allocator);
    defer manager.deinit();

    const session = try manager.getOrCreateSession("user123", "channel456");
    try std.testing.expectEqualStrings("user123", session.user_id);
    try std.testing.expectEqual(@as(usize, 0), session.message_count);

    // Getting same session should return existing
    const session2 = try manager.getOrCreateSession("user123", "channel456");
    session2.message_count = 5;
    try std.testing.expectEqual(@as(usize, 5), session.message_count);
}

test "abbey discord bot initialization" {
    const allocator = std.testing.allocator;

    var bot = try AbbeyDiscordBot.init(allocator, .{});
    defer bot.deinit();

    const stats = bot.getStats();
    try std.testing.expectEqual(@as(usize, 0), stats.total_messages_processed);
}

test "emotional emoji mapping" {
    const emoji = AbbeyDiscordBot.getEmotionalEmoji(.happy);
    try std.testing.expect(emoji != null);

    const neutral_emoji = AbbeyDiscordBot.getEmotionalEmoji(.neutral);
    try std.testing.expect(neutral_emoji == null);
}
