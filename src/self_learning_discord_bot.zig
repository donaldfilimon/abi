//! Self-Learning Discord Bot with AI Agent Integration and Persistent Learning
//!
//! This bot combines:
//! - Real-time Discord WebSocket connection (gateway)
//! - AI agent system with multiple personas
//! - WDBX database for learning from interactions
//! - Context-aware response generation
//! - Rate limiting and error handling

const std = @import("std");
const agent = @import("agent.zig");
const database = @import("mlai/wdbx/db.zig");
const discord_gateway = @import("discord/gateway.zig");
const discord_api = @import("discord/api.zig");

/// Configuration for the self-learning Discord bot
pub const BotConfig = struct {
    /// Discord bot token
    discord_token: []const u8,
    /// OpenAI API key for AI responses
    openai_api_key: ?[]const u8 = null,
    /// Default persona for the bot
    default_persona: agent.PersonaType = .AdaptiveModerator,
    /// Database configuration
    db_config: database.Config = .{ .shard_count = 5 },
    /// Maximum response length
    max_response_length: usize = 2000,
    /// Learning threshold (similarity score to store interactions)
    learning_threshold: f32 = 0.7,
    /// Context search limit
    context_limit: usize = 3,
    /// Enable debug logging
    debug: bool = false,
};

/// Learning context for improving responses
pub const LearningContext = struct {
    similar_interactions: []database.Entry,
    confidence_score: f32,
    should_learn: bool,

    pub fn deinit(self: LearningContext, allocator: std.mem.Allocator) void {
        allocator.free(self.similar_interactions);
    }
};

/// Self-learning Discord bot with AI integration
pub const SelfLearningBot = struct {
    allocator: std.mem.Allocator,
    config: BotConfig,
    discord_bot: *discord_gateway.DiscordBot,
    ai_agent: *agent.Agent,
    learning_db: *database.Database,
    is_running: bool = false,
    message_count: u64 = 0,
    learned_interactions: u64 = 0,

    /// Initialize the self-learning Discord bot
    pub fn init(allocator: std.mem.Allocator, config: BotConfig) !*SelfLearningBot {
        // Initialize AI agent
        var agent_config = agent.AgentConfig{
            .default_persona = config.default_persona,
            .max_tokens = @min(config.max_response_length / 4, 1000), // Estimate tokens
            .temperature = 0.7,
            .model = "gpt-3.5-turbo",
        };
        if (config.openai_api_key) |key| {
            agent_config.api_key = key;
        }

        const ai_agent = try agent.Agent.init(allocator, agent_config);
        errdefer ai_agent.deinit();

        // Initialize learning database
        const learning_db = try allocator.create(database.Database);
        learning_db.* = try database.Database.init(allocator, config.db_config);
        errdefer {
            learning_db.deinit();
            allocator.destroy(learning_db);
        }

        // Initialize Discord bot
        const discord_bot = try discord_gateway.DiscordBot.init(
            allocator,
            config.discord_token,
            ai_agent,
            learning_db,
        );
        errdefer discord_bot.deinit();

        // Create self-learning bot
        const self_learning_bot = try allocator.create(SelfLearningBot);
        self_learning_bot.* = .{
            .allocator = allocator,
            .config = config,
            .discord_bot = discord_bot,
            .ai_agent = ai_agent,
            .learning_db = learning_db,
        };

        // Set message handler
        discord_bot.setMessageHandler(handleMessage);

        return self_learning_bot;
    }

    /// Clean up resources
    pub fn deinit(self: *SelfLearningBot) void {
        self.discord_bot.deinit();
        self.ai_agent.deinit();
        self.learning_db.deinit();
        self.allocator.destroy(self.learning_db);
        self.allocator.destroy(self);
    }

    /// Start the bot and connect to Discord
    pub fn start(self: *SelfLearningBot) !void {
        self.is_running = true;

        std.debug.print("🚀 Starting Self-Learning Discord Bot...\n", .{});
        std.debug.print("🧠 Default Persona: {s}\n", .{@tagName(self.config.default_persona)});
        std.debug.print("💾 Database Shards: {}\n", .{self.config.db_config.shard_count});

        // Connect to Discord and start event loop
        try self.discord_bot.connect();
    }

    /// Stop the bot gracefully
    pub fn stop(self: *SelfLearningBot) void {
        std.debug.print("🛑 Stopping Self-Learning Discord Bot...\n", .{});
        std.debug.print("📊 Total Messages Processed: {}\n", .{self.message_count});
        std.debug.print("🧠 Interactions Learned: {}\n", .{self.learned_interactions});

        self.is_running = false;
        self.discord_bot.disconnect();
    }

    /// Get bot statistics
    pub fn getStats(self: *SelfLearningBot) BotStats {
        return BotStats{
            .messages_processed = self.message_count,
            .interactions_learned = self.learned_interactions,
            .current_persona = self.ai_agent.current_persona,
            .is_running = self.is_running,
        };
    }

    /// Switch the bot's persona
    pub fn switchPersona(self: *SelfLearningBot, persona: agent.PersonaType) void {
        self.ai_agent.switchPersona(persona);
        if (self.config.debug) {
            std.debug.print("🎭 Switched to persona: {s}\n", .{@tagName(persona)});
        }
    }

    /// Search for similar past interactions for learning context
    fn findSimilarInteractions(self: *SelfLearningBot, query: []const u8, limit: usize) !LearningContext {
        var similar_interactions = std.ArrayList(database.Entry).init(self.allocator);
        defer similar_interactions.deinit();

        // Simple similarity search based on common words
        // In production, this would use embeddings/vector similarity
        const query_words = try self.extractKeywords(query);
        defer self.allocator.free(query_words);

        var best_matches = std.ArrayList(struct { entry: database.Entry, score: f32 }).init(self.allocator);
        defer best_matches.deinit();

        // This is a simplified similarity calculation
        // In production, use proper vector embeddings
        var total_confidence: f32 = 0.0;
        var search_count: usize = 0;

        // Search across all shards (simplified approach)
        for (0..self.learning_db.shards.len) |shard_idx| {
            for (self.learning_db.shards[shard_idx].entries.items) |entry| {
                const similarity = self.calculateSimilarity(query_words, entry.key);
                if (similarity > 0.3) { // Minimum similarity threshold
                    try best_matches.append(.{ .entry = entry, .score = similarity });
                    total_confidence += similarity;
                    search_count += 1;
                }
            }
        }

        // Sort by similarity score
        std.sort.insertion(@TypeOf(best_matches.items[0]), best_matches.items, {}, struct {
            fn lessThan(_: void, a: @TypeOf(best_matches.items[0]), b: @TypeOf(best_matches.items[0])) bool {
                return a.score > b.score;
            }
        }.lessThan);

        // Take top matches
        const final_limit = @min(limit, best_matches.items.len);
        var final_matches = try self.allocator.alloc(database.Entry, final_limit);
        for (0..final_limit) |i| {
            final_matches[i] = best_matches.items[i].entry;
        }

        const avg_confidence = if (search_count > 0) total_confidence / @as(f32, @floatFromInt(search_count)) else 0.0;

        return LearningContext{
            .similar_interactions = final_matches,
            .confidence_score = avg_confidence,
            .should_learn = avg_confidence < self.config.learning_threshold,
        };
    }

    /// Extract keywords from text for similarity matching
    fn extractKeywords(self: *SelfLearningBot, text: []const u8) ![][]const u8 {
        var keywords = std.ArrayList([]const u8).init(self.allocator);
        defer keywords.deinit();

        var word_iter = std.mem.split(u8, text, " ");
        while (word_iter.next()) |word| {
            if (word.len > 3) { // Only consider longer words
                const clean_word = std.mem.trim(u8, word, " \t\n\r.,!?");
                if (clean_word.len > 3) {
                    try keywords.append(try self.allocator.dupe(u8, clean_word));
                }
            }
        }

        return keywords.toOwnedSlice();
    }

    /// Calculate similarity between two sets of keywords
    fn calculateSimilarity(_: *SelfLearningBot, query_words: [][]const u8, text: []const u8) f32 {
        var matches: f32 = 0.0;

        for (query_words) |word| {
            // Simple case-sensitive substring search
            if (std.mem.indexOf(u8, text, word) != null) {
                matches += 1.0;
            }
        }

        return if (query_words.len > 0) matches / @as(f32, @floatFromInt(query_words.len)) else 0.0;
    }

    /// Store interaction for future learning
    fn storeInteraction(self: *SelfLearningBot, query: []const u8, response: []const u8, persona: agent.PersonaType) !void {
        try self.learning_db.storeInteraction(query, response, persona);
        self.learned_interactions += 1;

        if (self.config.debug) {
            std.debug.print("💾 Stored interaction (total: {})\n", .{self.learned_interactions});
        }
    }
};

/// Bot statistics structure
pub const BotStats = struct {
    messages_processed: u64,
    interactions_learned: u64,
    current_persona: agent.PersonaType,
    is_running: bool,
};

/// Message handler function for Discord gateway
fn handleMessage(discord_bot: *discord_gateway.DiscordBot, message: discord_gateway.IncomingMessage) !void {
    // Get the self-learning bot instance from the discord bot
    // This is a simplified approach - in production, use proper context passing
    const self = @fieldParentPtr(SelfLearningBot, "discord_bot", discord_bot);

    self.message_count += 1;

    if (self.config.debug) {
        std.debug.print("🔍 Processing message #{}: {s}\n", .{ self.message_count, message.content });
    }

    // Skip empty messages or commands
    if (message.content.len == 0 or message.content[0] == '!') {
        return;
    }

    // Find similar past interactions for context
    const learning_context = self.findSimilarInteractions(message.content, self.config.context_limit) catch |err| {
        std.debug.print("❌ Error finding similar interactions: {}\n", .{err});
        return;
    };
    defer learning_context.deinit(self.allocator);

    // Build context for AI agent
    var context_messages = std.ArrayList(agent.Message).init(self.allocator);
    defer {
        for (context_messages.items) |msg| {
            msg.deinit(self.allocator);
        }
        context_messages.deinit();
    }

    // Add similar interactions as context
    for (learning_context.similar_interactions) |interaction| {
        try context_messages.append(agent.Message{
            .role = .user,
            .content = try self.allocator.dupe(u8, interaction.key),
        });
        try context_messages.append(agent.Message{
            .role = .assistant,
            .content = try self.allocator.dupe(u8, interaction.value),
        });
    }

    // Add current message
    try context_messages.append(agent.Message{
        .role = .user,
        .content = try self.allocator.dupe(u8, message.content),
    });

    // Generate AI response
    const ai_response = self.ai_agent.generateResponse(context_messages.items) catch |err| {
        std.debug.print("❌ AI response error: {}\n", .{err});

        // Fallback response
        try discord_bot.sendMessage(message.channel_id, "I'm having trouble processing that right now. Please try again!");
        return;
    };
    defer self.allocator.free(ai_response);

    // Truncate response if too long
    var final_response = ai_response;
    if (ai_response.len > self.config.max_response_length) {
        final_response = ai_response[0..self.config.max_response_length];
    }

    // Send response to Discord
    discord_bot.sendMessage(message.channel_id, final_response) catch |err| {
        std.debug.print("❌ Discord send error: {}\n", .{err});
        return;
    };

    // Store interaction for learning if confidence is low enough
    if (learning_context.should_learn) {
        self.storeInteraction(message.content, final_response, self.ai_agent.current_persona) catch |err| {
            std.debug.print("❌ Learning storage error: {}\n", .{err});
        };
    }

    if (self.config.debug) {
        std.debug.print("✅ Response sent (confidence: {d:.2})\n", .{learning_context.confidence_score});
    }
}

/// Example usage and testing
pub fn main() !void {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();
    const allocator = gpa.allocator();

    // Read Discord token from environment
    const discord_token = std.process.getEnvVarOwned(allocator, "DISCORD_BOT_TOKEN") catch |err| switch (err) {
        error.EnvironmentVariableNotFound => {
            std.debug.print("❌ Please set DISCORD_BOT_TOKEN environment variable\n", .{});
            return;
        },
        else => return err,
    };
    defer allocator.free(discord_token);

    // Read OpenAI API key (optional)
    const openai_key = std.process.getEnvVarOwned(allocator, "OPENAI_API_KEY") catch null;
    defer if (openai_key) |key| allocator.free(key);

    // Configure the bot
    const config = BotConfig{
        .discord_token = discord_token,
        .openai_api_key = openai_key,
        .default_persona = .AdaptiveModerator,
        .debug = true,
        .max_response_length = 1800, // Leave room for Discord's limit
        .learning_threshold = 0.6,
        .context_limit = 5,
    };

    // Initialize and start the bot
    var bot = try SelfLearningBot.init(allocator, config);
    defer bot.deinit();

    std.debug.print("🤖 Self-Learning Discord Bot initialized!\n", .{});
    std.debug.print("🔑 Using persona: {s}\n", .{@tagName(config.default_persona)});

    // Start the bot (this will block until disconnected)
    try bot.start();
}

test "Self-learning bot initialization" {
    const allocator = std.testing.allocator;

    const config = BotConfig{
        .discord_token = "test_token",
        .default_persona = .EmpatheticAnalyst,
        .debug = false,
    };

    var bot = try SelfLearningBot.init(allocator, config);
    defer bot.deinit();

    const stats = bot.getStats();
    try std.testing.expectEqual(@as(u64, 0), stats.messages_processed);
    try std.testing.expectEqual(agent.PersonaType.EmpatheticAnalyst, stats.current_persona);
    try std.testing.expectEqual(false, stats.is_running);
}

test "Keyword extraction" {
    const allocator = std.testing.allocator;

    const config = BotConfig{
        .discord_token = "test_token",
    };

    var bot = try SelfLearningBot.init(allocator, config);
    defer bot.deinit();

    const keywords = try bot.extractKeywords("Hello there, how are you doing today?");
    defer {
        for (keywords) |keyword| {
            allocator.free(keyword);
        }
        allocator.free(keywords);
    }

    try std.testing.expect(keywords.len > 0);
    // Should extract words longer than 3 characters
    var found_hello = false;
    for (keywords) |keyword| {
        if (std.mem.eql(u8, keyword, "Hello")) {
            found_hello = true;
            break;
        }
    }
    try std.testing.expect(found_hello);
}
