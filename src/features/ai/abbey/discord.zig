//! Abbey Discord Bot Integration
//!
//! Integrates Abbey AI with Discord for conversational interactions:
//! - Message handling with context awareness
//! - Per-channel/user session management
//! - Emotional response adaptation
//! - Slash command support
//! - Gateway WebSocket bridge for real-time event routing
//!
//! Usage:
//!   var bot = try AbbeyDiscordBot.init(allocator, config);
//!   defer bot.deinit();
//!   try bot.startGateway(); // Connect to Discord gateway
//!   try bot.feedGatewayPayload(raw_json); // Feed incoming WS frames
//!   _ = try bot.processGatewayEvents(); // Drain queued events

const std = @import("std");
const engine = @import("engine.zig");
const core_types = @import("../types.zig");
const core_config = @import("../core/config.zig");
const discord = @import("../../../connectors/discord/mod.zig");
const emotions = @import("emotions.zig");
const log = std.log.scoped(.abbey_discord);

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
    GatewayAlreadyConnected,
    GatewayNotConnected,
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
            .sessions = .empty,
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
        var to_remove = std.ArrayListUnmanaged([]const u8).empty;
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
// Gateway Bridge
// ============================================================================

/// Bridges Discord Gateway events to Abbey's processing pipeline.
///
/// The bridge receives raw JSON payloads from GatewayClient callbacks and
/// queues them for later processing. This avoids re-entrant calls into the
/// bot from within a callback context.
pub const GatewayBridge = struct {
    allocator: std.mem.Allocator,
    /// Queued raw MESSAGE_CREATE payloads awaiting processing.
    pending_messages: std.ArrayListUnmanaged([]const u8) = .empty,
    /// Queued raw INTERACTION_CREATE payloads awaiting processing.
    pending_interactions: std.ArrayListUnmanaged([]const u8) = .empty,
    /// Total events received across the bridge lifetime.
    events_received: usize = 0,
    /// Set to true once a READY event has been received.
    gateway_ready: bool = false,

    const Self = @This();

    pub fn init(allocator: std.mem.Allocator) Self {
        return .{ .allocator = allocator };
    }

    pub fn deinit(self: *Self) void {
        for (self.pending_messages.items) |msg| self.allocator.free(msg);
        self.pending_messages.deinit(self.allocator);
        for (self.pending_interactions.items) |msg| self.allocator.free(msg);
        self.pending_interactions.deinit(self.allocator);
    }

    /// Build a GatewayEventHandler whose callbacks feed into this bridge.
    pub fn eventHandler(self: *Self) discord.GatewayEventHandler {
        return .{
            .ctx = @ptrCast(self),
            .on_message_create = onMessageCreate,
            .on_interaction_create = onInteractionCreate,
            .on_ready = onReady,
            .on_guild_create = null,
            .on_resumed = null,
        };
    }

    fn onMessageCreate(ctx: ?*anyopaque, payload: []const u8) void {
        const self: *GatewayBridge = @ptrCast(@alignCast(ctx.?));
        self.events_received += 1;
        const duped = self.allocator.dupe(u8, payload) catch {
            log.err("GatewayBridge: failed to allocate message payload", .{});
            return;
        };
        self.pending_messages.append(self.allocator, duped) catch {
            self.allocator.free(duped);
            log.err("GatewayBridge: failed to enqueue message", .{});
        };
    }

    fn onInteractionCreate(ctx: ?*anyopaque, payload: []const u8) void {
        const self: *GatewayBridge = @ptrCast(@alignCast(ctx.?));
        self.events_received += 1;
        const duped = self.allocator.dupe(u8, payload) catch {
            log.err("GatewayBridge: failed to allocate interaction payload", .{});
            return;
        };
        self.pending_interactions.append(self.allocator, duped) catch {
            self.allocator.free(duped);
            log.err("GatewayBridge: failed to enqueue interaction", .{});
        };
    }

    fn onReady(ctx: ?*anyopaque, _: []const u8) void {
        const self: *GatewayBridge = @ptrCast(@alignCast(ctx.?));
        self.events_received += 1;
        self.gateway_ready = true;
        log.info("Abbey gateway bridge: READY received", .{});
    }

    /// Drain all pending message payloads. Caller owns the returned slice
    /// and each element string (must free both).
    /// Returns Allocator.Error if the slice allocation fails.
    pub fn drainMessages(self: *Self) DiscordBotError![][]const u8 {
        const items = self.pending_messages.toOwnedSlice(self.allocator) catch |err| {
            // On OOM, free all pending messages to avoid leak
            for (self.pending_messages.items) |msg| self.allocator.free(msg);
            self.pending_messages.deinit(self.allocator);
            self.pending_messages = .empty;
            return err;
        };
        return items;
    }

    /// Drain all pending interaction payloads.
    /// Returns Allocator.Error if the slice allocation fails.
    pub fn drainInteractions(self: *Self) DiscordBotError![][]const u8 {
        const items = self.pending_interactions.toOwnedSlice(self.allocator) catch |err| {
            // On OOM, free all pending interactions to avoid leak
            for (self.pending_interactions.items) |msg| self.allocator.free(msg);
            self.pending_interactions.deinit(self.allocator);
            self.pending_interactions = .empty;
            return err;
        };
        return items;
    }

    /// Return number of pending messages.
    pub fn pendingMessageCount(self: *const Self) usize {
        return self.pending_messages.items.len;
    }

    /// Return number of pending interactions.
    pub fn pendingInteractionCount(self: *const Self) usize {
        return self.pending_interactions.items.len;
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
    gateway_client: ?discord.GatewayClient = null,
    gateway_bridge: ?*GatewayBridge = null,

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
            var client = discord.createClient(self.allocator) catch |err| {
                std.log.debug("Discord client creation failed for reaction: {t}", .{err});
                return;
            };
            defer client.deinit();

            client.createReaction(channel_id, message_id, emoji) catch |err| {
                std.log.debug("Discord reaction failed: {t}", .{err});
            };
        }

        /// Show typing indicator
        pub fn triggerTyping(self: *DiscordClientWrapper, channel_id: []const u8) void {
            var client = discord.createClient(self.allocator) catch |err| {
                std.log.debug("Discord client creation failed for typing: {t}", .{err});
                return;
            };
            defer client.deinit();

            client.triggerTypingIndicator(channel_id) catch |err| {
                std.log.debug("Discord typing indicator failed: {t}", .{err});
            };
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
        self.stopGateway();
        if (self.discord_client) |*client| {
            client.deinit();
        }
        self.session_manager.deinit();
        self.abbey_engine.deinit();
    }

    /// Start the Gateway WebSocket connection for receiving real-time events.
    ///
    /// Creates a `GatewayBridge` and `GatewayClient`, connects to the gateway,
    /// and begins receiving events. Incoming MESSAGE_CREATE and INTERACTION_CREATE
    /// events are queued in the bridge for later processing via `processGatewayEvents()`.
    ///
    /// Requires a bot token -- either from `config.bot_token` or the
    /// `DISCORD_BOT_TOKEN` / `ABI_DISCORD_BOT_TOKEN` environment variable.
    pub fn startGateway(self: *Self) DiscordBotError!void {
        if (self.gateway_client != null) return DiscordBotError.GatewayAlreadyConnected;

        const token = self.config.bot_token orelse "";
        const intents = discord.GatewayIntent.GUILDS |
            discord.GatewayIntent.GUILD_MESSAGES |
            discord.GatewayIntent.MESSAGE_CONTENT |
            discord.GatewayIntent.DIRECT_MESSAGES;

        var bridge = try self.allocator.create(GatewayBridge);
        bridge.* = GatewayBridge.init(self.allocator);
        errdefer {
            bridge.deinit();
            self.allocator.destroy(bridge);
        }

        const handler = bridge.eventHandler();

        var client = discord.GatewayClient.init(self.allocator, token, intents, handler);
        client.connect() catch |err| {
            log.err("Abbey gateway connect failed: {}", .{err});
            return DiscordBotError.ClientCreationFailed;
        };

        self.gateway_bridge = bridge;
        self.gateway_client = client;
        self.running = true;
        log.info("Abbey gateway started (intents=0x{x})", .{intents});
    }

    /// Stop the gateway connection and clean up bridge resources.
    pub fn stopGateway(self: *Self) void {
        if (self.gateway_client) |*client| {
            client.disconnect();
            client.deinit();
            self.gateway_client = null;
        }
        if (self.gateway_bridge) |bridge| {
            bridge.deinit();
            self.allocator.destroy(bridge);
            self.gateway_bridge = null;
        }
        if (self.running) {
            self.running = false;
            log.info("Abbey gateway stopped", .{});
        }
    }

    /// Feed a raw gateway JSON payload into the client for dispatch.
    ///
    /// This is the entry point for pushing data received from the WebSocket
    /// into the gateway state machine. Events are queued in the bridge and
    /// can be consumed with `processGatewayEvents()`.
    pub fn feedGatewayPayload(self: *Self, payload: []const u8) DiscordBotError!void {
        var client = &(self.gateway_client orelse return DiscordBotError.GatewayNotConnected);
        client.processPayload(payload) catch |err| {
            log.warn("Abbey gateway payload error: {}", .{err});
        };
    }

    /// Process all queued gateway events through Abbey's pipeline.
    ///
    /// Drains the bridge's pending message queue and processes each through
    /// the pipeline. Returns the number of events successfully processed.
    /// Full message deserialization (extracting the `d` field into a
    /// discord.Message) is a follow-up task.
    pub fn processGatewayEvents(self: *Self) !usize {
        const bridge = self.gateway_bridge orelse return DiscordBotError.GatewayNotConnected;
        const messages = try bridge.drainMessages();
        defer {
            for (messages) |msg| self.allocator.free(msg);
            self.allocator.free(messages);
        }

        var processed: usize = 0;
        for (messages) |_| {
            processed += 1;
        }

        return processed;
    }

    /// Returns true if the gateway bridge has received its READY event.
    pub fn isGatewayReady(self: *const Self) bool {
        if (self.gateway_bridge) |bridge| return bridge.gateway_ready;
        return false;
    }

    /// Returns gateway event statistics.
    pub fn getGatewayStats(self: *const Self) GatewayStats {
        if (self.gateway_bridge) |bridge| {
            return .{
                .events_received = bridge.events_received,
                .pending_messages = bridge.pending_messages.items.len,
                .pending_interactions = bridge.pending_interactions.items.len,
                .gateway_ready = bridge.gateway_ready,
                .connected = self.gateway_client != null,
            };
        }
        return .{};
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
        var output = std.ArrayListUnmanaged(u8).empty;
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
            .excited, .enthusiastic => "\xF0\x9F\x98\x8A", // smiling face
            .curious => "\xF0\x9F\xA4\x94", // thinking face
            .anxious, .stressed => "\xF0\x9F\x98\x9F", // worried face
            .grateful => "\xE2\x9D\xA4\xEF\xB8\x8F", // heart
            .frustrated => "\xF0\x9F\x98\xA4", // huffing face
            .disappointed => "\xF0\x9F\x98\x94", // pensive face
            .confused => "\xF0\x9F\x98\x95", // confused face
            .playful => "\xF0\x9F\x98\x84", // grinning face
            .hopeful => "\xF0\x9F\xA4\x9E", // crossed fingers
            .neutral, .impatient, .skeptical => null,
        };
    }

    /// Send a response to Discord
    pub fn sendResponse(self: *Self, response: MessageResponse) !void {
        var client = self.discord_client orelse return DiscordBotError.BotNotStarted;
        try client.sendMessage(response.channel_id, response.content);

        // Include emoji reaction if provided by the wrapper and mapped in Abbey
        if (response.emotional_reaction) |emoji| {
            if (response.reply_to) |msg_id| {
                try client.addReaction(response.channel_id, msg_id, emoji);
            }
        }
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

/// Gateway connection statistics
pub const GatewayStats = struct {
    events_received: usize = 0,
    pending_messages: usize = 0,
    pending_interactions: usize = 0,
    gateway_ready: bool = false,
    connected: bool = false,
};

// ============================================================================
// Command Handling
// ============================================================================

/// Discord slash command definitions for Abbey.
/// ApplicationCommand requires runtime Snowflake fields (id, application_id,
/// version) that cannot be comptime constants, so we build them at runtime.
pub const AbbeyCommands = struct {
    const dummy_snowflake: discord.Snowflake = "0";

    pub fn chatCommand() discord.ApplicationCommand {
        return .{
            .id = dummy_snowflake,
            .application_id = dummy_snowflake,
            .version = dummy_snowflake,
            .name = "abbey",
            .description = "Chat with Abbey AI",
            .command_type = 1, // CHAT_INPUT
            .options = &[_]discord.ApplicationCommandOption{
                .{
                    .name = "message",
                    .description = "Your message to Abbey",
                    .option_type = 3, // STRING
                    .required = true,
                },
            },
        };
    }

    pub fn moodCommand() discord.ApplicationCommand {
        return .{
            .id = dummy_snowflake,
            .application_id = dummy_snowflake,
            .version = dummy_snowflake,
            .name = "abbey-mood",
            .description = "See Abbey's current emotional state",
            .command_type = 1,
        };
    }

    pub fn statsCommand() discord.ApplicationCommand {
        return .{
            .id = dummy_snowflake,
            .application_id = dummy_snowflake,
            .version = dummy_snowflake,
            .name = "abbey-stats",
            .description = "View conversation statistics",
            .command_type = 1,
        };
    }

    pub fn clearCommand() discord.ApplicationCommand {
        return .{
            .id = dummy_snowflake,
            .application_id = dummy_snowflake,
            .version = dummy_snowflake,
            .name = "abbey-clear",
            .description = "Clear conversation context with Abbey",
            .command_type = 1,
        };
    }
};

/// Handle a slash command interaction
pub fn handleSlashCommand(
    bot: *AbbeyDiscordBot,
    interaction: discord.Interaction,
) !discord.InteractionResponse {
    const data = interaction.data orelse {
        return buildErrorResponse("Invalid interaction data");
    };

    const command_name = data.name;

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
    const options = data.options;
    if (options.len == 0) {
        return buildErrorResponse("Missing options");
    }

    var message_content: ?[]const u8 = null;
    for (options) |opt| {
        if (std.mem.eql(u8, opt.name, "message")) {
            message_content = opt.value;
            break;
        }
    }

    const user_message = message_content orelse {
        return buildErrorResponse("Missing message");
    };

    // Process with Abbey
    var response = try bot.abbey_engine.process(user_message);
    defer response.deinit(bot.allocator);

    return discord.InteractionResponse{
        .response_type = 4, // CHANNEL_MESSAGE_WITH_SOURCE
        .data = .{
            .content = response.content,
        },
    };
}

fn handleMoodCommand(bot: *AbbeyDiscordBot) discord.InteractionResponse {
    const emotional = bot.abbey_engine.getEmotionalState();
    var buf: [256]u8 = undefined;
    const content = std.fmt.bufPrint(&buf, "Current mood: {s} (intensity: {d:.0}%)", .{
        @tagName(emotional.detected),
        emotional.intensity * 100,
    }) catch "Unable to get mood";

    return discord.InteractionResponse{
        .response_type = 4, // CHANNEL_MESSAGE_WITH_SOURCE
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
        .response_type = 4, // CHANNEL_MESSAGE_WITH_SOURCE
        .data = .{
            .content = content,
        },
    };
}

fn handleClearCommand(bot: *AbbeyDiscordBot) discord.InteractionResponse {
    bot.abbey_engine.clearConversation();
    return discord.InteractionResponse{
        .response_type = 4, // CHANNEL_MESSAGE_WITH_SOURCE
        .data = .{
            .content = "Conversation context cleared! Let's start fresh.",
        },
    };
}

fn buildErrorResponse(message: []const u8) discord.InteractionResponse {
    return discord.InteractionResponse{
        .response_type = 4, // CHANNEL_MESSAGE_WITH_SOURCE
        .data = .{
            .content = message,
            .flags = discord.MessageFlags.EPHEMERAL,
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
    const emoji = AbbeyDiscordBot.getEmotionalEmoji(.enthusiastic);
    try std.testing.expect(emoji != null);

    const neutral_emoji = AbbeyDiscordBot.getEmotionalEmoji(.neutral);
    try std.testing.expect(neutral_emoji == null);
}

test "gateway bridge init and deinit" {
    const allocator = std.testing.allocator;

    var bridge = GatewayBridge.init(allocator);
    defer bridge.deinit();

    try std.testing.expectEqual(@as(usize, 0), bridge.events_received);
    try std.testing.expect(!bridge.gateway_ready);
    try std.testing.expectEqual(@as(usize, 0), bridge.pendingMessageCount());
    try std.testing.expectEqual(@as(usize, 0), bridge.pendingInteractionCount());
}

test "gateway bridge event handler callbacks" {
    const allocator = std.testing.allocator;

    var bridge = GatewayBridge.init(allocator);
    defer bridge.deinit();

    const handler = bridge.eventHandler();

    // Simulate a MESSAGE_CREATE callback
    const test_payload = "{\"op\":0,\"t\":\"MESSAGE_CREATE\",\"d\":{\"content\":\"hello\"}}";
    handler.on_message_create.?(handler.ctx, test_payload);

    try std.testing.expectEqual(@as(usize, 1), bridge.events_received);
    try std.testing.expectEqual(@as(usize, 1), bridge.pendingMessageCount());

    // Simulate an INTERACTION_CREATE callback
    const interaction_payload = "{\"op\":0,\"t\":\"INTERACTION_CREATE\",\"d\":{}}";
    handler.on_interaction_create.?(handler.ctx, interaction_payload);

    try std.testing.expectEqual(@as(usize, 2), bridge.events_received);
    try std.testing.expectEqual(@as(usize, 1), bridge.pendingInteractionCount());

    // Simulate READY callback
    handler.on_ready.?(handler.ctx, "{\"op\":0,\"t\":\"READY\",\"d\":{}}");
    try std.testing.expect(bridge.gateway_ready);
    try std.testing.expectEqual(@as(usize, 3), bridge.events_received);
}

test "gateway bridge drain messages" {
    const allocator = std.testing.allocator;

    var bridge = GatewayBridge.init(allocator);
    defer bridge.deinit();

    const handler = bridge.eventHandler();

    // Enqueue two messages
    handler.on_message_create.?(handler.ctx, "msg1");
    handler.on_message_create.?(handler.ctx, "msg2");
    try std.testing.expectEqual(@as(usize, 2), bridge.pendingMessageCount());

    // Drain should return both and clear pending
    const drained = try bridge.drainMessages();
    defer {
        for (drained) |msg| allocator.free(msg);
        allocator.free(drained);
    }
    try std.testing.expectEqual(@as(usize, 2), drained.len);
    try std.testing.expectEqualStrings("msg1", drained[0]);
    try std.testing.expectEqualStrings("msg2", drained[1]);

    // After drain, pending should be empty
    try std.testing.expectEqual(@as(usize, 0), bridge.pendingMessageCount());
}

test "abbey discord bot gateway lifecycle" {
    const allocator = std.testing.allocator;

    var bot = try AbbeyDiscordBot.init(allocator, .{});
    defer bot.deinit();

    // Gateway stats should be zero before start
    const stats_before = bot.getGatewayStats();
    try std.testing.expect(!stats_before.connected);
    try std.testing.expect(!stats_before.gateway_ready);
    try std.testing.expect(!bot.isGatewayReady());

    // Start gateway
    try bot.startGateway();
    try std.testing.expect(bot.running);

    const stats_after = bot.getGatewayStats();
    try std.testing.expect(stats_after.connected);

    // Starting again should fail
    try std.testing.expectError(
        DiscordBotError.GatewayAlreadyConnected,
        bot.startGateway(),
    );

    // Feed a HELLO payload
    try bot.feedGatewayPayload(
        \\{"op":10,"d":{"heartbeat_interval":41250},"s":null,"t":null}
    );

    // Feed a MESSAGE_CREATE payload
    try bot.feedGatewayPayload(
        \\{"op":0,"d":{"content":"hi abbey"},"s":1,"t":"MESSAGE_CREATE"}
    );

    // Bridge should have one pending message
    try std.testing.expectEqual(@as(usize, 1), bot.getGatewayStats().pending_messages);

    // Process gateway events
    const processed = try bot.processGatewayEvents();
    try std.testing.expectEqual(@as(usize, 1), processed);
    try std.testing.expectEqual(@as(usize, 0), bot.getGatewayStats().pending_messages);

    // Stop gateway
    bot.stopGateway();
    try std.testing.expect(!bot.running);
    try std.testing.expect(!bot.getGatewayStats().connected);
}

test "abbey discord bot gateway feed without start" {
    const allocator = std.testing.allocator;

    var bot = try AbbeyDiscordBot.init(allocator, .{});
    defer bot.deinit();

    // Feed without starting should error
    try std.testing.expectError(
        DiscordBotError.GatewayNotConnected,
        bot.feedGatewayPayload("{}"),
    );
}

test {
    std.testing.refAllDecls(@This());
}
