# Self-Learning Discord Bot Setup Guide

The Abi framework includes a complete self-learning Discord bot that combines real-time Discord integration, AI-powered responses, and persistent learning capabilities.

## 🚀 Quick Start

### Prerequisites

1. **Zig Installation**: Install Zig using the provided script [[memory:1151497]]
   ```bash
   bash scripts/install_zig.sh
   ```

2. **Discord Bot Token**: Create a Discord application and bot at [Discord Developer Portal](https://discord.com/developers/applications)

3. **OpenAI API Key** (Optional): Get your API key from [OpenAI Platform](https://platform.openai.com/api-keys)

### Environment Setup

1. Set required environment variables:
   ```bash
   export DISCORD_BOT_TOKEN="your_discord_bot_token_here"
   export OPENAI_API_KEY="your_openai_api_key_here"  # Optional
   ```

2. Build and run the bot:
   ```bash
   zig build run-discord-bot-demo
   ```

## 🤖 Features

### Core Capabilities

- **Real-time Discord Integration**: WebSocket-based connection with automatic reconnection
- **AI-Powered Responses**: Multiple personas with different response styles
- **Persistent Learning**: WDBX database stores interactions for continuous improvement
- **Context-Aware**: Uses past interactions to provide better responses
- **Rate Limiting**: Built-in protection against Discord API limits
- **Graceful Error Handling**: Robust error recovery and logging

### Available Personas

| Persona | Description | Use Cases |
|---------|-------------|-----------|
| `EmpatheticAnalyst` | Supportive and analytical | Customer support, counseling |
| `DirectExpert` | Clear and professional | Technical documentation, FAQ |
| `AdaptiveModerator` | Context-aware and balanced | General purpose, community management |
| `CreativeWriter` | Imaginative and expressive | Creative projects, storytelling |
| `TechnicalAdvisor` | Technical and detailed | Programming help, troubleshooting |
| `ProblemSolver` | Systematic problem-solving | Project planning, debugging |

## 🛠️ Configuration

### Environment Configurations

The bot supports different environments with optimized settings:

```bash
# Development (default)
zig build run-discord-bot-demo --env development

# Production
zig build run-discord-bot-demo --env production

# Testing
zig build run-discord-bot-demo --env testing
```

### Configuration Options

```bash
# Set specific persona
zig build run-discord-bot-demo --persona CreativeWriter

# Limit messages for testing
zig build run-discord-bot-demo --max-messages 100

# Production deployment
zig build run-discord-bot-demo --env production --persona AdaptiveModerator
```

### Advanced Configuration

Create a custom configuration by modifying the `BotConfig` in your code:

```zig
const config = BotConfig{
    .discord_token = discord_token,
    .openai_api_key = openai_key,
    .default_persona = .AdaptiveModerator,
    .debug = false,
    .max_response_length = 2000,
    .learning_threshold = 0.7,  // Higher = learn less frequently
    .context_limit = 3,         // Number of past interactions to consider
    .db_config = .{ .shard_count = 5 },  // Database performance tuning
};
```

## 📊 Learning System

### How Learning Works

1. **Message Processing**: Each incoming message is analyzed for keywords and context
2. **Similarity Search**: The bot searches past interactions for similar queries
3. **Context Building**: Relevant past conversations are used to inform responses
4. **Response Generation**: AI agent generates contextually aware responses
5. **Learning Storage**: New interactions are stored if confidence is below threshold

### Learning Parameters

- **Learning Threshold** (0.0-1.0): Controls how aggressively the bot learns
  - Lower values = Learn from more interactions
  - Higher values = Only learn from novel interactions

- **Context Limit**: Number of similar past interactions to consider
  - Higher values = More context but slower response
  - Lower values = Faster but less informed responses

### Database Structure

The bot uses a sharded WDBX database:

```
Database
├── Shard 0 (Prime: 31)
├── Shard 1 (Prime: 37)
├── Shard 2 (Prime: 43)
└── ...
```

Each entry contains:
- **Key**: User's message/query
- **Value**: Bot's response
- **Persona**: Which persona generated the response
- **Version**: Timestamp for chronological ordering

## 🔧 Discord Bot Setup

### 1. Create Discord Application

1. Go to [Discord Developer Portal](https://discord.com/developers/applications)
2. Click "New Application"
3. Give your bot a name
4. Go to "Bot" section
5. Click "Add Bot"
6. Copy the bot token

### 2. Configure Bot Permissions

Required permissions for the bot:
- Read Messages
- Send Messages  
- Read Message History
- Use Slash Commands (optional)

Permission integer: `2048` (basic) or `8192` (with slash commands)

### 3. Invite Bot to Server

Use this URL format:
```
https://discord.com/api/oauth2/authorize?client_id=YOUR_CLIENT_ID&permissions=2048&scope=bot
```

Replace `YOUR_CLIENT_ID` with your application's client ID.

### 4. Bot Intents

The bot requires these intents:
- `GUILD_MESSAGES` (1 << 9)
- `MESSAGE_CONTENT` (1 << 15)

These are automatically configured in the gateway connection.

## 🚀 Deployment

### Local Development

```bash
# Development mode with debug logging
export DISCORD_BOT_TOKEN="your_token"
export OPENAI_API_KEY="your_key"
zig build run-discord-bot-demo --env development
```

### Production Deployment

```bash
# Production mode with optimized settings
export DISCORD_BOT_TOKEN="your_token"
export OPENAI_API_KEY="your_key"
zig build run-discord-bot-demo --env production
```

### Docker Deployment (Optional)

Create a `Dockerfile`:

```dockerfile
FROM alpine:latest

# Install Zig
RUN wget https://ziglang.org/download/0.12.0/zig-linux-x86_64-0.12.0.tar.xz
RUN tar -xf zig-linux-x86_64-0.12.0.tar.xz
RUN mv zig-linux-x86_64-0.12.0 /usr/local/zig
ENV PATH="/usr/local/zig:$PATH"

# Copy source code
COPY . /app
WORKDIR /app

# Build the bot
RUN zig build

# Set environment variables
ENV DISCORD_BOT_TOKEN=""
ENV OPENAI_API_KEY=""

# Run the bot
CMD ["zig", "build", "run-discord-bot", "--", "--env", "production"]
```

### Systemd Service (Linux)

Create `/etc/systemd/system/discord-bot.service`:

```ini
[Unit]
Description=Self-Learning Discord Bot
After=network.target

[Service]
Type=simple
User=discord-bot
WorkingDirectory=/opt/discord-bot
ExecStart=/usr/local/bin/zig build run-discord-bot -- --env production
Restart=always
RestartSec=10
Environment=DISCORD_BOT_TOKEN=your_token_here
Environment=OPENAI_API_KEY=your_key_here

[Install]
WantedBy=multi-user.target
```

Enable and start:
```bash
sudo systemctl enable discord-bot
sudo systemctl start discord-bot
```

## 📈 Monitoring and Analytics

### Built-in Statistics

The bot provides real-time statistics:

```
📊 === Bot Statistics ===
⏱️  Uptime: 2h 15m 30s
💬 Messages Processed: 1,247
🧠 Interactions Learned: 89
🎭 Current Persona: AdaptiveModerator
🔄 Status: Running
📈 Learning Rate: 7.1%
========================
```

### Health Monitoring

Monitor these key metrics:

- **Message Processing Rate**: Messages per minute
- **Learning Rate**: Percentage of interactions stored
- **Error Rate**: Failed responses per hour
- **Response Time**: Average AI response generation time
- **Memory Usage**: Database size and memory consumption

### Logging

Enable debug logging for troubleshooting:

```bash
zig build run-discord-bot-demo --env development  # Enables debug mode
```

Log levels:
- **INFO**: Basic operation info
- **DEBUG**: Detailed processing information
- **ERROR**: Error conditions and recovery
- **WARN**: Non-fatal issues

## 🔧 Troubleshooting

### Common Issues

#### "Invalid Token" Error
- Verify your Discord bot token is correct
- Check token hasn't been regenerated in Discord Developer Portal
- Ensure token is properly set in environment variable

#### "Missing Permissions" Error
- Bot needs "Send Messages" and "Read Messages" permissions
- Check bot role hierarchy in Discord server
- Verify bot is invited with correct permissions

#### "Rate Limited" Messages
- Bot automatically handles rate limits
- If persistent, reduce message frequency
- Check for other bots using same token

#### AI Response Errors
- Verify OpenAI API key is valid and has credits
- Check internet connectivity
- Bot falls back to simple responses if AI fails

#### Memory Issues
- Monitor database size growth
- Consider increasing `learning_threshold` to learn less
- Restart bot periodically in production

### Debug Commands

```bash
# Test basic functionality
zig build test

# Run with maximum debug output
zig build run-discord-bot-demo --env development

# Test specific persona
zig build run-discord-bot-demo --persona TechnicalAdvisor --max-messages 10
```

### Performance Tuning

#### Database Optimization
- Increase `shard_count` for better performance with many interactions
- Higher shard counts reduce collision probability
- Monitor memory usage with different shard configurations

#### Response Optimization
- Lower `context_limit` for faster responses
- Higher `learning_threshold` reduces database writes
- Shorter `max_response_length` improves Discord API performance

#### Memory Management
- Bot automatically manages memory for responses
- Database entries persist across restarts
- Consider periodic database cleanup for long-running deployments

## 🧪 Testing

### Unit Tests

```bash
# Run all tests
zig build test

# Test specific components
zig build test -- --filter "discord"
zig build test -- --filter "learning"
```

### Integration Testing

```bash
# Test bot with limited messages
zig build run-discord-bot-demo --env testing --max-messages 50

# Test specific persona
zig build run-discord-bot-demo --persona EmpatheticAnalyst --max-messages 20
```

### Load Testing

Create test scenarios to validate performance:

1. **High Message Volume**: Test with rapid message succession
2. **Learning Capacity**: Test database performance with many stored interactions
3. **Persona Switching**: Test dynamic persona changes during conversation
4. **Error Recovery**: Test behavior when Discord API is unavailable

## 📚 API Reference

### Key Classes

#### `SelfLearningBot`
Main bot class with initialization, learning, and Discord integration.

```zig
pub const SelfLearningBot = struct {
    // Initialize with configuration
    pub fn init(allocator: std.mem.Allocator, config: BotConfig) !*SelfLearningBot

    // Start bot connection
    pub fn start(self: *SelfLearningBot) !void

    // Switch AI persona
    pub fn switchPersona(self: *SelfLearningBot, persona: agent.PersonaType) void

    // Get runtime statistics
    pub fn getStats(self: *SelfLearningBot) BotStats
};
```

#### `BotConfig`
Configuration structure for customizing bot behavior.

```zig
pub const BotConfig = struct {
    discord_token: []const u8,
    openai_api_key: ?[]const u8 = null,
    default_persona: agent.PersonaType = .AdaptiveModerator,
    db_config: database.Config = .{ .shard_count = 5 },
    max_response_length: usize = 2000,
    learning_threshold: f32 = 0.7,
    context_limit: usize = 3,
    debug: bool = false,
};
```

### Database API

The WDBX database provides persistent learning storage:

```zig
// Store new interaction
try bot.learning_db.storeInteraction(query, response, persona);

// Retrieve similar interactions
const entry = bot.learning_db.retrieve(query);
```

## 🤝 Contributing

### Development Setup

1. Fork the repository
2. Install Zig using the provided script
3. Set up test Discord bot for development
4. Run tests: `zig build test`
5. Make changes and test thoroughly
6. Submit pull request

### Code Style

Follow Zig conventions:
- Use snake_case for functions and variables
- Use PascalCase for types and structs
- Include comprehensive tests for new features
- Document public APIs with doc comments

### Testing Guidelines

- Write unit tests for all new functionality
- Test error conditions and edge cases
- Verify Discord API integration with test bot
- Performance test with realistic message volumes

## 📝 License

This Discord bot implementation is part of the Abi AI framework and follows the project's licensing terms.

---

## 🎯 Next Steps

After setting up your Discord bot:

1. **Customize Personas**: Create custom personas for your specific use case
2. **Integration**: Integrate with other Abi framework components
3. **Monitoring**: Set up production monitoring and alerting
4. **Scaling**: Consider horizontal scaling for high-traffic servers
5. **Features**: Add custom commands and advanced Discord features

For more information, see the main [Abi framework documentation](../README.md). 