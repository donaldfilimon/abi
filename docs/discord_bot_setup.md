# 🤖 Discord Bot Setup Guide

> **Build a self-learning AI bot for Discord with the Abi AI Framework**

[![Discord Bot](https://img.shields.io/badge/Discord-Bot-blue.svg)](docs/discord_bot_setup.md)
[![AI Framework](https://img.shields.io/badge/AI-Framework-brightgreen.svg)]()

This guide will walk you through setting up a self-learning Discord bot using the Abi AI Framework. Your bot will learn from conversations, provide intelligent responses, and integrate seamlessly with Discord's platform.

## 📋 **Table of Contents**

- [Prerequisites](#prerequisites)
- [Environment Setup](#environment-setup)
- [Bot Features](#bot-features)
- [Configuration](#configuration)
- [Learning System](#learning-system)
- [Discord Bot Setup](#discord-bot-setup)
- [Deployment](#deployment)
- [Monitoring](#monitoring)
- [Troubleshooting](#troubleshooting)
- [API Reference](#api-reference)
- [Contributing](#contributing)

---

## ✅ **Prerequisites**

- **Zig**: Version 0.15.1 or later
- **Discord Account**: For bot creation and testing
- **Discord Application**: Bot application registered on Discord Developer Portal
- **OpenAI API Key**: For advanced AI capabilities (optional)
- **Git**: For version control
- **Basic Knowledge**: Zig programming and Discord bot concepts

---

## 🛠️ **Environment Setup**

### **1. Install Dependencies**

```bash
# Clone the repository
git clone https://github.com/your-org/abi.git
cd abi

# Install Zig (if not already installed)
# Windows: Download from https://ziglang.org/download/
# macOS: brew install zig
# Linux: Download and add to PATH

# Verify installation
zig version
```

### **2. Build the Project**

```bash
# Build the project
zig build

# Verify build
./zig-out/bin/abi --version
```

### **3. Environment Variables**

Create a `.env` file in your project root:

```bash
# Discord Bot Configuration
DISCORD_TOKEN=your_discord_bot_token_here
DISCORD_CLIENT_ID=your_discord_client_id_here
DISCORD_GUILD_ID=your_discord_server_id_here

# OpenAI Configuration (Optional)
OPENAI_API_KEY=your_openai_api_key_here
OPENAI_MODEL=gpt-3.5-turbo

# Bot Configuration
BOT_PREFIX=!
BOT_LEARNING_RATE=0.1
BOT_MAX_MEMORY=1000
BOT_RESPONSE_TIMEOUT=30

# Database Configuration
DB_PATH=./discord_bot_data.wdbx
DB_BACKUP_INTERVAL=3600
```

---

## 🚀 **Bot Features**

### **Core Capabilities**

- **🎯 Intelligent Responses**: Context-aware AI-powered responses
- **🧠 Self-Learning**: Learns from conversations and improves over time
- **📚 Memory Management**: Maintains conversation history and user preferences
- **🔍 Context Understanding**: Remembers previous interactions and references
- **⚡ Fast Response**: Optimized for real-time Discord interactions
- **🛡️ Safety Features**: Content filtering and rate limiting

### **Advanced Features**

- **🎭 Multi-Personality**: Switch between different AI personalities
- **🌐 Multi-Language**: Support for multiple languages
- **📊 Analytics**: Track bot usage and learning progress
- **🔧 Custom Commands**: Extensible command system
- **📱 Mobile Optimized**: Responsive design for mobile Discord users

---

## ⚙️ **Configuration**

### **1. Bot Configuration File**

Create `config/bot_config.zig`:

```zig
pub const BotConfig = struct {
    // Discord Settings
    pub const discord = struct {
        pub const token = "your_discord_bot_token";
        pub const client_id = "your_discord_client_id";
        pub const guild_id = "your_discord_server_id";
        pub const intents = .{
            .guilds = true,
            .guild_messages = true,
            .guild_members = true,
            .direct_messages = true,
            .message_content = true,
        };
    };

    // AI Settings
    pub const ai = struct {
        pub const model = "gpt-3.5-turbo";
        pub const max_tokens = 150;
        pub const temperature = 0.7;
        pub const learning_rate = 0.1;
        pub const max_context_length = 2000;
    };

    // Memory Settings
    pub const memory = struct {
        pub const max_conversations = 1000;
        pub const max_messages_per_conversation = 50;
        pub const cleanup_interval = 3600; // seconds
        pub const persistence_enabled = true;
    };

    // Response Settings
    pub const response = struct {
        pub const default_timeout = 30; // seconds
        pub const max_response_length = 2000;
        pub const enable_typing_indicator = true;
        pub const rate_limit_per_user = 5; // messages per minute
    };
};
```

### **2. Learning Configuration**

```zig
pub const LearningConfig = struct {
    // Conversation Learning
    pub const conversation = struct {
        pub const min_messages_for_learning = 3;
        pub const context_window_size = 10;
        pub const sentiment_analysis_enabled = true;
        pub const topic_extraction_enabled = true;
    };

    // User Learning
    pub const user = struct {
        pub const personality_tracking = true;
        pub const preference_learning = true;
        pub const interaction_history_size = 100;
        pub const privacy_respect_enabled = true;
    };

    // Content Learning
    pub const content = struct {
        pub const language_detection = true;
        pub const cultural_awareness = true;
        pub const content_filtering = true;
        pub const safe_content_only = true;
    };
};
```

---

## 🧠 **Learning System**

### **1. Conversation Learning**

The bot learns from conversations by analyzing:

- **Message Patterns**: Common phrases and responses
- **Context Relationships**: How messages relate to previous ones
- **User Preferences**: Individual user communication styles
- **Sentiment Analysis**: Emotional context of conversations
- **Topic Evolution**: How conversations flow between topics

### **2. Memory Architecture**

```zig
const ConversationMemory = struct {
    conversation_id: []const u8,
    participants: []User,
    messages: []Message,
    context: ConversationContext,
    metadata: ConversationMetadata,
    
    const Message = struct {
        id: []const u8,
        author: User,
        content: []const u8,
        timestamp: i64,
        sentiment: f32,
        topics: []Topic,
        reactions: []Reaction,
    };
    
    const ConversationContext = struct {
        current_topic: Topic,
        mood: Mood,
        formality_level: FormalityLevel,
        language: Language,
        cultural_context: CulturalContext,
    };
    
    const Topic = struct {
        name: []const u8,
        confidence: f32,
        keywords: []const u8,
        related_topics: []Topic,
    };
    
    const Mood = enum {
        positive,
        neutral,
        negative,
        mixed,
    };
    
    const FormalityLevel = enum {
        casual,
        informal,
        formal,
        professional,
    };
};
```

### **3. Learning Algorithms**

#### **Pattern Recognition**
```zig
const PatternLearner = struct {
    patterns: std.AutoHashMap(Pattern, PatternStats),
    allocator: std.mem.Allocator,
    
    const Pattern = struct {
        trigger: []const u8,
        response: []const u8,
        context: []const u8,
        user_id: []const u8,
    };
    
    const PatternStats = struct {
        frequency: u32,
        success_rate: f32,
        last_used: i64,
        confidence: f32,
    };
    
    pub fn learnPattern(self: *@This(), message: Message, response: Message) !void {
        const pattern = Pattern{
            .trigger = message.content,
            .response = response.content,
            .context = self.extractContext(message),
            .user_id = message.author.id,
        };
        
        if (self.patterns.get(pattern)) |existing| {
            // Update existing pattern
            existing.frequency += 1;
            existing.success_rate = (existing.success_rate + 1.0) / 2.0;
            existing.last_used = std.time.milliTimestamp();
        } else {
            // Create new pattern
            try self.patterns.put(pattern, PatternStats{
                .frequency = 1,
                .success_rate = 1.0,
                .last_used = std.time.milliTimestamp(),
                .confidence = 0.5,
            });
        }
    }
    
    pub fn findBestResponse(self: *@This(), message: Message) ?[]const u8 {
        var best_pattern: ?Pattern = null;
        var best_score: f32 = 0.0;
        
        var iter = self.patterns.iterator();
        while (iter.next()) |entry| {
            const pattern = entry.key_ptr.*;
            const stats = entry.value_ptr.*;
            
            const score = self.calculatePatternScore(pattern, message, stats);
            if (score > best_score) {
                best_score = score;
                best_pattern = pattern;
            }
        }
        
        return if (best_pattern) |pattern| pattern.response else null;
    }
    
    fn calculatePatternScore(self: *@This(), pattern: Pattern, message: Message, stats: PatternStats) f32 {
        const content_similarity = self.calculateContentSimilarity(pattern.trigger, message.content);
        const context_similarity = self.calculateContextSimilarity(pattern.context, message);
        const user_similarity = if (std.mem.eql(u8, pattern.user_id, message.author.id)) 1.0 else 0.5;
        
        return (content_similarity * 0.4 + context_similarity * 0.3 + user_similarity * 0.3) * stats.confidence;
    }
};
```

#### **Sentiment Analysis**
```zig
const SentimentAnalyzer = struct {
    positive_words: std.StringHashMap(f32),
    negative_words: std.StringHashMap(f32),
    neutral_words: std.StringHashMap(f32),
    allocator: std.mem.Allocator,
    
    pub fn init(allocator: std.mem.Allocator) !@This() {
        var analyzer = @This(){
            .positive_words = std.StringHashMap(f32).init(allocator),
            .negative_words = std.StringHashMap(f32).init(allocator),
            .neutral_words = std.StringHashMap(f32).init(allocator),
            .allocator = allocator,
        };
        
        try analyzer.loadSentimentDictionary();
        return analyzer;
    }
    
    pub fn analyzeSentiment(self: *@This(), text: []const u8) SentimentResult {
        var positive_score: f32 = 0.0;
        var negative_score: f32 = 0.0;
        var neutral_score: f32 = 0.0;
        
        const words = self.tokenize(text);
        defer self.allocator.free(words);
        
        for (words) |word| {
            if (self.positive_words.get(word)) |score| {
                positive_score += score;
            } else if (self.negative_words.get(word)) |score| {
                negative_score += score;
            } else {
                neutral_score += 0.1;
            }
        }
        
        const total_score = positive_score + negative_score + neutral_score;
        if (total_score == 0) {
            return SentimentResult{ .sentiment = .neutral, .confidence = 0.0, .scores = .{ 0.0, 0.0, 0.0 } };
        }
        
        const normalized_positive = positive_score / total_score;
        const normalized_negative = negative_score / total_score;
        const normalized_neutral = neutral_score / total_score;
        
        const sentiment = if (normalized_positive > normalized_negative and normalized_positive > normalized_neutral)
            .positive
        else if (normalized_negative > normalized_positive and normalized_negative > normalized_neutral)
            .negative
        else
            .neutral;
        
        const confidence = @max(normalized_positive, @max(normalized_negative, normalized_neutral));
        
        return SentimentResult{
            .sentiment = sentiment,
            .confidence = confidence,
            .scores = .{ normalized_positive, normalized_negative, normalized_neutral },
        };
    }
    
    const SentimentResult = struct {
        sentiment: Sentiment,
        confidence: f32,
        scores: [3]f32, // [positive, negative, neutral]
        
        const Sentiment = enum {
            positive,
            negative,
            neutral,
        };
    };
};
```

---

## 🤖 **Discord Bot Setup**

### **1. Create Discord Application**

1. **Go to [Discord Developer Portal](https://discord.com/developers/applications)**
2. **Click "New Application"**
3. **Name your bot (e.g., "Abi AI Bot")**
4. **Go to "Bot" section**
5. **Click "Add Bot"**
6. **Copy the bot token**

### **2. Bot Permissions**

Set these permissions for your bot:

```
General Permissions:
✅ Read Messages/View Channels
✅ Send Messages
✅ Use Slash Commands
✅ Add Reactions
✅ Embed Links
✅ Attach Files
✅ Read Message History
✅ Use External Emojis
✅ Add Reactions

Text Permissions:
✅ Send Messages
✅ Send Messages in Threads
✅ Use Slash Commands
✅ Manage Messages
✅ Embed Links
✅ Attach Files
✅ Read Message History
✅ Mention Everyone
✅ Use External Emojis
✅ Add Reactions
```

### **3. Invite Bot to Server**

Generate invite link with these scopes:
- `bot`
- `applications.commands`

### **4. Bot Code Structure**

```zig
const DiscordBot = struct {
    client: discord.Client,
    ai_engine: AIEngine,
    memory: ConversationMemory,
    config: BotConfig,
    allocator: std.mem.Allocator,
    
    pub fn init(allocator: std.mem.Allocator, config: BotConfig) !@This() {
        return @This(){
            .client = try discord.Client.init(config.discord.token),
            .ai_engine = try AIEngine.init(allocator, config.ai),
            .memory = try ConversationMemory.init(allocator, config.memory),
            .config = config,
            .allocator = allocator,
        };
    }
    
    pub fn start(self: *@This()) !void {
        // Set up event handlers
        try self.setupEventHandlers();
        
        // Start the bot
        try self.client.start();
    }
    
    fn setupEventHandlers(self: *@This()) !void {
        // Message handler
        self.client.on(.message_create, self.handleMessage);
        
        // Ready handler
        self.client.on(.ready, self.handleReady);
        
        // Interaction handler
        self.client.on(.interaction_create, self.handleInteraction);
    }
    
    fn handleMessage(self: *@This(), event: discord.Message) !void {
        // Ignore bot messages
        if (event.author.bot) return;
        
        // Process message for learning
        try self.processMessageForLearning(event);
        
        // Generate response if bot is mentioned
        if (self.isBotMentioned(event.content)) {
            try self.generateAndSendResponse(event);
        }
    }
    
    fn handleReady(self: *@This(), event: discord.Ready) !void {
        std.log.info("Bot is ready! Logged in as {}", .{event.user.username});
        
        // Register slash commands
        try self.registerSlashCommands();
    }
    
    fn handleInteraction(self: *@This(), event: discord.Interaction) !void {
        switch (event.data) {
            .application_command => |command| {
                try self.handleSlashCommand(event, command);
            },
            else => {},
        }
    }
};
```

### **5. Slash Commands**

```zig
const SlashCommands = struct {
    bot: *DiscordBot,
    
    pub fn init(bot: *DiscordBot) @This() {
        return @This(){ .bot = bot };
    }
    
    pub fn registerCommands(self: *@This()) !void {
        const commands = [_]discord.ApplicationCommand{
            .{
                .name = "chat",
                .description = "Chat with the AI bot",
                .options = &[_]discord.ApplicationCommandOption{
                    .{
                        .type = .string,
                        .name = "message",
                        .description = "Your message to the bot",
                        .required = true,
                    },
                },
            },
            .{
                .name = "learn",
                .description = "Teach the bot something new",
                .options = &[_]discord.ApplicationCommandOption{
                    .{
                        .type = .string,
                        .name = "question",
                        .description = "The question or statement",
                        .required = true,
                    },
                    .{
                        .type = .string,
                        .name = "answer",
                        .description = "The correct answer or response",
                        .required = true,
                    },
                },
            },
            .{
                .name = "personality",
                .description = "Change the bot's personality",
                .options = &[_]discord.ApplicationCommandOption{
                    .{
                        .type = .string,
                        .name = "style",
                        .description = "Personality style (friendly, professional, creative, etc.)",
                        .required = true,
                    },
                },
            },
            .{
                .name = "stats",
                .description = "View bot learning statistics",
            },
            .{
                .name = "forget",
                .description = "Make the bot forget recent conversations",
                .options = &[_]discord.ApplicationCommandOption{
                    .{
                        .type = .integer,
                        .name = "hours",
                        .description = "Number of hours to go back",
                        .required = false,
                    },
                },
            },
        };
        
        for (commands) |command| {
            try self.bot.client.createGlobalApplicationCommand(command);
        }
    }
    
    pub fn handleSlashCommand(self: *@This(), interaction: discord.Interaction, command: discord.ApplicationCommandData) !void {
        const command_name = command.name;
        
        if (std.mem.eql(u8, command_name, "chat")) {
            try self.handleChatCommand(interaction, command);
        } else if (std.mem.eql(u8, command_name, "learn")) {
            try self.handleLearnCommand(interaction, command);
        } else if (std.mem.eql(u8, command_name, "personality")) {
            try self.handlePersonalityCommand(interaction, command);
        } else if (std.mem.eql(u8, command_name, "stats")) {
            try self.handleStatsCommand(interaction, command);
        } else if (std.mem.eql(u8, command_name, "forget")) {
            try self.handleForgetCommand(interaction, command);
        }
    }
    
    fn handleChatCommand(self: *@This(), interaction: discord.Interaction, command: discord.ApplicationCommandData) !void {
        const message = command.options[0].value.string;
        
        // Defer response for long processing
        try interaction.defer();
        
        // Generate AI response
        const response = try self.bot.ai_engine.generateResponse(message, interaction.user.id);
        
        // Send response
        try interaction.followUp(.{
            .content = response,
            .flags = .{ .ephemeral = false },
        });
        
        // Learn from this interaction
        try self.bot.memory.recordInteraction(interaction.user.id, message, response);
    }
    
    fn handleLearnCommand(self: *@This(), interaction: discord.Interaction, command: discord.ApplicationCommandData) !void {
        const question = command.options[0].value.string;
        const answer = command.options[1].value.string;
        
        // Teach the bot
        try self.bot.ai_engine.learn(question, answer, interaction.user.id);
        
        try interaction.createResponse(.{
            .content = "Thanks! I've learned something new.",
            .flags = .{ .ephemeral = true },
        });
    }
};
```

---

## 🚀 **Deployment**

### **1. Local Development**

```bash
# Run bot locally
zig build run -- discord_bot

# Enable debug logging
RUST_LOG=debug zig build run -- discord_bot
```

### **2. Production Deployment**

#### **Docker Deployment**
```dockerfile
FROM zig:latest as builder

WORKDIR /app
COPY . .
RUN zig build -Drelease-small

FROM alpine:latest
RUN apk add --no-cache ca-certificates
WORKDIR /root/
COPY --from=builder /app/zig-out/bin/discord_bot .
CMD ["./discord_bot"]
```

#### **Systemd Service (Linux)**
```ini
[Unit]
Description=Abi AI Discord Bot
After=network.target

[Service]
Type=simple
User=discord-bot
WorkingDirectory=/opt/discord-bot
ExecStart=/opt/discord-bot/discord_bot
Restart=always
RestartSec=10
Environment=DISCORD_TOKEN=your_token_here

[Install]
WantedBy=multi-user.target
```

#### **Windows Service**
```powershell
# Install as Windows service
sc create "AbiDiscordBot" binPath="C:\path\to\discord_bot.exe" start=auto
sc description "AbiDiscordBot" "Abi AI Discord Bot Service"
sc start "AbiDiscordBot"
```

### **3. Environment-Specific Configs**

#### **Development**
```bash
# .env.development
NODE_ENV=development
LOG_LEVEL=debug
ENABLE_DEBUG_COMMANDS=true
TEST_MODE=true
```

#### **Staging**
```bash
# .env.staging
NODE_ENV=staging
LOG_LEVEL=info
ENABLE_DEBUG_COMMANDS=false
TEST_MODE=false
```

#### **Production**
```bash
# .env.production
NODE_ENV=production
LOG_LEVEL=warn
ENABLE_DEBUG_COMMANDS=false
TEST_MODE=false
ENABLE_METRICS=true
```

---

## 📊 **Monitoring**

### **1. Health Checks**

```zig
const HealthMonitor = struct {
    bot: *DiscordBot,
    last_heartbeat: i64,
    health_status: HealthStatus,
    
    const HealthStatus = enum {
        healthy,
        degraded,
        unhealthy,
    };
    
    pub fn checkHealth(self: *@This()) !HealthReport {
        const now = std.time.milliTimestamp();
        self.last_heartbeat = now;
        
        // Check Discord connection
        const discord_healthy = self.bot.client.isConnected();
        
        // Check AI engine
        const ai_healthy = self.bot.ai_engine.isHealthy();
        
        // Check memory system
        const memory_healthy = self.bot.memory.isHealthy();
        
        // Determine overall health
        const health = if (discord_healthy and ai_healthy and memory_healthy)
            .healthy
        else if (discord_healthy or ai_healthy or memory_healthy)
            .degraded
        else
            .unhealthy;
        
        self.health_status = health;
        
        return HealthReport{
            .status = health,
            .timestamp = now,
            .discord_connection = discord_healthy,
            .ai_engine = ai_healthy,
            .memory_system = memory_healthy,
            .uptime = self.bot.getUptime(),
            .message_count = self.bot.getTotalMessages(),
            .user_count = self.bot.getUniqueUsers(),
        };
    }
    
    const HealthReport = struct {
        status: HealthStatus,
        timestamp: i64,
        discord_connection: bool,
        ai_engine: bool,
        memory_system: bool,
        uptime: u64,
        message_count: u64,
        user_count: u64,
    };
};
```

### **2. Metrics Collection**

```zig
const MetricsCollector = struct {
    message_count: std.atomic.Atomic(u64),
    response_time: std.atomic.Atomic(u64),
    error_count: std.atomic.Atomic(u64),
    user_interactions: std.atomic.Atomic(u64),
    learning_events: std.atomic.Atomic(u64),
    
    pub fn recordMessage(self: *@This()) void {
        _ = self.message_count.fetchAdd(1, .Monotonic);
    }
    
    pub fn recordResponseTime(self: *@This(), time_ns: u64) void {
        _ = self.response_time.store(time_ns, .Monotonic);
    }
    
    pub fn recordError(self: *@This()) void {
        _ = self.error_count.fetchAdd(1, .Monotonic);
    }
    
    pub fn recordUserInteraction(self: *@This()) void {
        _ = self.user_interactions.fetchAdd(1, .Monotonic);
    }
    
    pub fn recordLearningEvent(self: *@This()) void {
        _ = self.learning_events.fetchAdd(1, .Monotonic);
    }
    
    pub fn getMetrics(self: *@This()) Metrics {
        return Metrics{
            .total_messages = self.message_count.load(.Monotonic),
            .avg_response_time_ns = self.response_time.load(.Monotonic),
            .total_errors = self.error_count.load(.Monotonic),
            .total_user_interactions = self.user_interactions.load(.Monotonic),
            .total_learning_events = self.learning_events.load(.Monotonic),
        };
    }
    
    const Metrics = struct {
        total_messages: u64,
        avg_response_time_ns: u64,
        total_errors: u64,
        total_user_interactions: u64,
        total_learning_events: u64,
    };
};
```

### **3. Logging**

```zig
const BotLogger = struct {
    allocator: std.mem.Allocator,
    log_file: ?std.fs.File,
    log_level: LogLevel,
    
    const LogLevel = enum {
        debug,
        info,
        warn,
        error,
        fatal,
    };
    
    pub fn init(allocator: std.mem.Allocator, log_file_path: ?[]const u8, level: LogLevel) !@This() {
        var logger = @This(){
            .allocator = allocator,
            .log_file = null,
            .log_level = level,
        };
        
        if (log_file_path) |path| {
            logger.log_file = try std.fs.cwd().createFile(path, .{});
        }
        
        return logger;
    }
    
    pub fn log(self: *@This(), level: LogLevel, comptime format: []const u8, args: anytype) !void {
        if (@enumToInt(level) < @enumToInt(self.log_level)) return;
        
        const timestamp = std.time.milliTimestamp();
        const level_str = switch (level) {
            .debug => "DEBUG",
            .info => "INFO",
            .warn => "WARN",
            .error => "ERROR",
            .fatal => "FATAL",
        };
        
        const message = try std.fmt.allocPrint(
            self.allocator,
            "[{d}] [{}] " ++ format ++ "\n",
            .{ timestamp, level_str } ++ args
        );
        defer self.allocator.free(message);
        
        // Console output
        try std.io.getStdOut().writeAll(message);
        
        // File output
        if (self.log_file) |file| {
            try file.writeAll(message);
            try file.flush();
        }
        
        // Fatal errors should exit
        if (level == .fatal) {
            std.process.exit(1);
        }
    }
};
```

---

## 🔧 **Troubleshooting**

### **Common Issues & Solutions**

#### **1. Bot Not Responding**
```zig
// Check bot permissions
pub fn checkBotPermissions(self: *@This(), guild_id: discord.Snowflake) !void {
    const guild = try self.client.getGuild(guild_id);
    const member = try guild.getMember(self.client.user.id);
    
    if (!member.hasPermission(.send_messages)) {
        std.log.err("Bot lacks SEND_MESSAGES permission", .{});
        return error.InsufficientPermissions;
    }
    
    if (!member.hasPermission(.use_slash_commands)) {
        std.log.err("Bot lacks USE_SLASH_COMMANDS permission", .{});
        return error.InsufficientPermissions;
    }
}
```

#### **2. Rate Limiting Issues**
```zig
// Implement rate limiting
const RateLimiter = struct {
    user_limits: std.AutoHashMap(discord.Snowflake, UserLimit),
    allocator: std.mem.Allocator,
    
    const UserLimit = struct {
        message_count: u32,
        last_reset: i64,
        max_messages: u32 = 5,
        reset_interval: i64 = 60 * 1000, // 1 minute
    };
    
    pub fn canSendMessage(self: *@This(), user_id: discord.Snowflake) bool {
        const now = std.time.milliTimestamp();
        
        if (self.user_limits.get(user_id)) |limit| {
            if (now - limit.last_reset > limit.reset_interval) {
                // Reset counter
                limit.message_count = 0;
                limit.last_reset = now;
            }
            
            if (limit.message_count >= limit.max_messages) {
                return false;
            }
            
            limit.message_count += 1;
            return true;
        } else {
            // First message from user
            try self.user_limits.put(user_id, UserLimit{
                .message_count = 1,
                .last_reset = now,
            });
            return true;
        }
    }
};
```

#### **3. Memory Issues**
```zig
// Memory cleanup
pub fn cleanupMemory(self: *@This()) !void {
    const now = std.time.milliTimestamp();
    const cleanup_threshold = now - (self.config.memory.cleanup_interval * 1000);
    
    // Clean up old conversations
    try self.memory.cleanupOldConversations(cleanup_threshold);
    
    // Clean up old patterns
    try self.ai_engine.cleanupOldPatterns(cleanup_threshold);
    
    // Force garbage collection if needed
    if (self.getMemoryUsage() > self.config.memory.max_memory) {
        try self.forceMemoryCleanup();
    }
}
```

---

## 📚 **API Reference**

### **Core Functions**

#### **Message Handling**
```zig
pub fn handleMessage(self: *@This(), message: discord.Message) !void
pub fn generateResponse(self: *@This(), message: discord.Message) ![]const u8
pub fn sendResponse(self: *@This(), channel_id: discord.Snowflake, content: []const u8) !void
```

#### **Learning Functions**
```zig
pub fn learn(self: *@This(), input: []const u8, output: []const u8, user_id: discord.Snowflake) !void
pub fn forget(self: *@This(), user_id: discord.Snowflake, hours_back: ?u32) !void
pub fn getStats(self: *@This()) LearningStats
```

#### **Memory Functions**
```zig
pub fn recordConversation(self: *@This(), conversation: Conversation) !void
pub fn getConversationHistory(self: *@This(), user_id: discord.Snowflake) ![]Conversation
pub fn cleanupOldData(self: *@This(), threshold: i64) !void
```

### **Configuration Options**

- **`discord.token`**: Bot authentication token
- **`ai.model`**: AI model to use for responses
- **`memory.max_conversations`**: Maximum conversations to store
- **`response.timeout`**: Response generation timeout
- **`learning.rate`**: Learning rate for pattern recognition

---

## 🤝 **Contributing**

### **How to Contribute**

1. **Fork the repository**
2. **Create a feature branch**
3. **Make your changes**
4. **Add tests**
5. **Submit a pull request**

### **Areas for Contribution**

- **New AI Models**: Integrate additional AI services
- **Language Support**: Add support for more languages
- **Advanced Learning**: Implement more sophisticated learning algorithms
- **Analytics**: Enhanced bot usage analytics
- **Documentation**: Improve guides and examples

### **Development Setup**

```bash
# Clone your fork
git clone https://github.com/your-username/abi.git
cd abi

# Add upstream remote
git remote add upstream https://github.com/original-org/abi.git

# Create feature branch
git checkout -b feature/discord-bot-improvements

# Make changes and test
zig build test
zig build run -- discord_bot

# Commit and push
git add .
git commit -m "Add new Discord bot features"
git push origin feature/discord-bot-improvements
```

---

## 🔗 **Additional Resources**

- **[Discord Developer Portal](https://discord.com/developers)** - Official Discord API documentation
- **[Abi AI Framework](README.md)** - Main framework documentation
- **[Bot Examples](examples/)** - Additional bot examples
- **[Community Discord](https://discord.gg/your-server)** - Join our community

---

**🤖 Ready to build an intelligent Discord bot? The Abi AI Framework provides everything you need for a self-learning, context-aware bot that gets smarter over time!**

**🚀 Start with the examples above and create a bot that truly understands and learns from your Discord community.** 