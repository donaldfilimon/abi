//! Discord Integration Module - Discord Bot and API Integration
//!
//! This module provides comprehensive Discord integration capabilities:
//! - Discord API client with rate limiting
//! - Real-time WebSocket gateway connection
//! - Bot command processing and event handling
//! - Message formatting and embed support
//! - Voice channel integration
//! - Rich presence and activity management
//! - Discord slash commands support

const std = @import("std");
const core = @import("../core/mod.zig");

/// Re-export commonly used types
pub const Allocator = core.Allocator;

/// Core Discord components
pub const api = @import("api.zig");
pub const gateway = @import("gateway.zig");
pub const types = @import("types.zig");

/// Re-export main types for convenience
pub const DiscordConfig = api.DiscordConfig;
pub const DiscordMessage = api.DiscordMessage;
pub const DiscordBot = gateway.DiscordBot;
pub const MessageHandler = gateway.MessageHandler;

/// Discord integration configuration
pub const DiscordIntegrationConfig = struct {
    /// Bot token (required)
    token: []const u8,

    /// Bot application ID
    application_id: ?[]const u8 = null,

    /// Guild ID for guild-specific commands
    guild_id: ?[]const u8 = null,

    /// Command prefix for text commands
    command_prefix: []const u8 = "!",

    /// Enable slash commands
    enable_slash_commands: bool = true,

    /// Enable message reactions
    enable_reactions: bool = true,

    /// Enable voice channel support
    enable_voice: bool = false,

    /// Enable rich presence
    enable_rich_presence: bool = true,

    /// Enable message caching
    enable_caching: bool = true,

    /// Cache size limit
    cache_size: usize = 10000,

    /// Enable debug logging
    debug_mode: bool = false,
};

/// Bot status and activity
pub const BotStatus = enum {
    online,
    idle,
    dnd,
    invisible,
};

/// Activity types for rich presence
pub const ActivityType = enum {
    playing,
    streaming,
    listening,
    watching,
    competing,
};

/// Bot activity configuration
pub const Activity = struct {
    /// Activity type
    type: ActivityType,

    /// Activity name
    name: []const u8,

    /// Activity details (optional)
    details: ?[]const u8 = null,

    /// Activity state (optional)
    state: ?[]const u8 = null,

    /// Streaming URL (for streaming activity)
    url: ?[]const u8 = null,

    /// Activity timestamps
    timestamps: ?ActivityTimestamps = null,

    /// Activity assets (images)
    assets: ?ActivityAssets = null,

    /// Activity party information
    party: ?ActivityParty = null,

    /// Activity secrets (for joining/spectating)
    secrets: ?ActivitySecrets = null,
};

/// Activity timestamps
pub const ActivityTimestamps = struct {
    start: ?i64 = null,
    end: ?i64 = null,
};

/// Activity assets (images)
pub const ActivityAssets = struct {
    large_image: ?[]const u8 = null,
    large_text: ?[]const u8 = null,
    small_image: ?[]const u8 = null,
    small_text: ?[]const u8 = null,
};

/// Activity party information
pub const ActivityParty = struct {
    id: ?[]const u8 = null,
    size: ?struct { current: u32, max: u32 } = null,
};

/// Activity secrets for joining/spectating
pub const ActivitySecrets = struct {
    join: ?[]const u8 = null,
    spectate: ?[]const u8 = null,
    match: ?[]const u8 = null,
};

/// Command handler function type
pub const CommandHandler = *const fn (
    bot: *DiscordBot,
    message: gateway.IncomingMessage,
    args: []const []const u8,
) anyerror!void;

/// Slash command handler function type
pub const SlashCommandHandler = *const fn (
    bot: *DiscordBot,
    interaction: *anyopaque, // Discord interaction object
) anyerror!void;

/// Command registry for managing bot commands
pub const CommandRegistry = struct {
    allocator: Allocator,
    text_commands: std.StringHashMap(CommandHandler),
    slash_commands: std.StringHashMap(SlashCommandHandler),

    /// Initialize a new command registry
    pub fn init(allocator: Allocator) CommandRegistry {
        return .{
            .allocator = allocator,
            .text_commands = std.StringHashMap(CommandHandler).init(allocator),
            .slash_commands = std.StringHashMap(SlashCommandHandler).init(allocator),
        };
    }

    /// Deinitialize the registry
    pub fn deinit(self: *CommandRegistry) void {
        self.text_commands.deinit();
        self.slash_commands.deinit();
    }

    /// Register a text command
    pub fn registerTextCommand(
        self: *CommandRegistry,
        name: []const u8,
        handler: CommandHandler,
    ) !void {
        const name_copy = try self.allocator.dupe(u8, name);
        try self.text_commands.put(name_copy, handler);
    }

    /// Register a slash command
    pub fn registerSlashCommand(
        self: *CommandRegistry,
        name: []const u8,
        handler: SlashCommandHandler,
    ) !void {
        const name_copy = try self.allocator.dupe(u8, name);
        try self.slash_commands.put(name_copy, handler);
    }

    /// Get a text command handler
    pub fn getTextCommand(self: *const CommandRegistry, name: []const u8) ?CommandHandler {
        return self.text_commands.get(name);
    }

    /// Get a slash command handler
    pub fn getSlashCommand(self: *const CommandRegistry, name: []const u8) ?SlashCommandHandler {
        return self.slash_commands.get(name);
    }
};

/// Message builder for creating rich Discord messages
pub const MessageBuilder = struct {
    content: ?[]const u8 = null,
    embeds: std.ArrayList(api.Embed),
    components: std.ArrayList(Component),
    tts: bool = false,
    flags: u32 = 0,

    /// Initialize a new message builder
    pub fn init(allocator: Allocator) MessageBuilder {
        return .{
            .embeds = std.ArrayList(api.Embed).init(allocator),
            .components = std.ArrayList(Component).init(allocator),
        };
    }

    /// Deinitialize the message builder
    pub fn deinit(self: *MessageBuilder) void {
        self.embeds.deinit();
        self.components.deinit();
        if (self.content) |content| {
            self.embeds.allocator.free(content);
        }
    }

    /// Set message content
    pub fn setContent(self: *MessageBuilder, content: []const u8) !void {
        if (self.content) |old_content| {
            self.embeds.allocator.free(old_content);
        }
        self.content = try self.embeds.allocator.dupe(u8, content);
    }

    /// Add an embed
    pub fn addEmbed(self: *MessageBuilder, embed: api.Embed) !void {
        try self.embeds.append(embed);
    }

    /// Build the final Discord message
    pub fn build(self: *MessageBuilder) api.DiscordMessage {
        return .{
            .content = self.content,
            .embeds = if (self.embeds.items.len > 0) self.embeds.items else null,
            .tts = self.tts,
            .flags = self.flags,
            .allowed_mentions = null, // TODO: Implement allowed mentions
        };
    }
};

/// UI component types
pub const ComponentType = enum {
    action_row,
    button,
    select_menu,
    text_input,
};

/// Generic UI component
pub const Component = struct {
    type: ComponentType,
    data: union(ComponentType) {
        action_row: []Component,
        button: Button,
        select_menu: SelectMenu,
        text_input: TextInput,
    },
};

/// Button component
pub const Button = struct {
    style: ButtonStyle,
    label: ?[]const u8 = null,
    custom_id: ?[]const u8 = null,
    url: ?[]const u8 = null,
    disabled: bool = false,
    emoji: ?Emoji = null,
};

/// Button styles
pub const ButtonStyle = enum {
    primary,
    secondary,
    success,
    danger,
    link,
};

/// Select menu component
pub const SelectMenu = struct {
    custom_id: []const u8,
    options: []SelectOption,
    placeholder: ?[]const u8 = null,
    min_values: u32 = 1,
    max_values: u32 = 1,
    disabled: bool = false,
};

/// Select option
pub const SelectOption = struct {
    label: []const u8,
    value: []const u8,
    description: ?[]const u8 = null,
    emoji: ?Emoji = null,
    default: bool = false,
};

/// Text input component
pub const TextInput = struct {
    custom_id: []const u8,
    style: TextInputStyle,
    label: []const u8,
    min_length: ?u32 = null,
    max_length: ?u32 = null,
    required: bool = true,
    value: ?[]const u8 = null,
    placeholder: ?[]const u8 = null,
};

/// Text input styles
pub const TextInputStyle = enum {
    short,
    paragraph,
};

/// Emoji structure
pub const Emoji = struct {
    id: ?[]const u8 = null,
    name: ?[]const u8 = null,
    animated: bool = false,
};

/// Initialize Discord integration
pub fn init(config: DiscordIntegrationConfig) !*DiscordBot {
    _ = config;
    // TODO: Implement Discord integration initialization
    return undefined;
}

/// Create a command registry
pub fn createCommandRegistry(allocator: Allocator) CommandRegistry {
    return CommandRegistry.init(allocator);
}

/// Create a message builder
pub fn createMessageBuilder(allocator: Allocator) MessageBuilder {
    return MessageBuilder.init(allocator);
}
