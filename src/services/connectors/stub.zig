//! Stub for Connectors module when disabled.
//!
//! Mirrors the full API of mod.zig, returning error.ConnectorsDisabled for all operations.

const std = @import("std");

/// Shared connector types (available even when connectors are disabled).
pub const shared = @import("shared.zig");

/// Connectors module errors.
pub const Error = error{
    ConnectorsDisabled,
    MissingApiKey,
    MissingApiToken,
    MissingBotToken,
    ApiRequestFailed,
    InvalidResponse,
    RateLimitExceeded,
    OutOfMemory,
};

var initialized: bool = false;

pub fn init(_: std.mem.Allocator) !void {
    return Error.ConnectorsDisabled;
}

pub fn deinit() void {
    initialized = false;
}

pub fn isEnabled() bool {
    return false;
}

pub fn isInitialized() bool {
    return initialized;
}

pub fn getEnvOwned(_: std.mem.Allocator, _: []const u8) !?[]u8 {
    return Error.ConnectorsDisabled;
}

pub fn getFirstEnvOwned(_: std.mem.Allocator, _: []const []const u8) !?[]u8 {
    return Error.ConnectorsDisabled;
}

/// Auth header stub.
pub const AuthHeader = struct {
    value: []u8,

    pub fn header(_: *const AuthHeader) std.http.Header {
        return .{ .name = "authorization", .value = "" };
    }

    pub fn deinit(self: *AuthHeader, allocator: std.mem.Allocator) void {
        allocator.free(self.value);
        self.* = undefined;
    }
};

pub fn buildBearerHeader(_: std.mem.Allocator, _: []const u8) !AuthHeader {
    return Error.ConnectorsDisabled;
}

// ============================================================================
// OpenAI Connector Stub
// ============================================================================

pub const openai = @import("stubs/openai.zig");

// ============================================================================
// HuggingFace Connector Stub
// ============================================================================

pub const huggingface = @import("stubs/huggingface.zig");

// ============================================================================
// Ollama Connector Stub
// ============================================================================

pub const ollama = @import("stubs/ollama.zig");

// ============================================================================
// Anthropic Connector Stub
// ============================================================================

pub const anthropic = @import("stubs/anthropic.zig");

// ============================================================================
// Mistral Connector Stub
// ============================================================================

pub const mistral = @import("stubs/mistral.zig");

// ============================================================================
// Cohere Connector Stub
// ============================================================================

pub const cohere = @import("stubs/cohere.zig");

// ============================================================================
// LM Studio Connector Stub
// ============================================================================

pub const lm_studio = @import("stubs/lm_studio.zig");

// ============================================================================
// vLLM Connector Stub
// ============================================================================

pub const vllm = @import("stubs/vllm.zig");

// ============================================================================
// MLX Connector Stub
// ============================================================================

pub const mlx = @import("stubs/mlx.zig");

// ============================================================================
// Local Scheduler Connector Stub
// ============================================================================

pub const local_scheduler = @import("stubs/local_scheduler.zig");

// ============================================================================
// Discord Connector Stub
// ============================================================================

pub const discord = struct {
    pub const DiscordError = error{
        MissingBotToken,
        ApiRequestFailed,
        InvalidResponse,
        RateLimitExceeded,
        Unauthorized,
        Forbidden,
        NotFound,
        InvalidGatewayUrl,
        GatewayConnectionFailed,
        InvalidInteractionSignature,
    };

    pub const Config = struct {
        bot_token: []u8,
        client_id: ?[]u8 = null,
        client_secret: ?[]u8 = null,
        public_key: ?[]u8 = null,

        pub fn deinit(self: *Config, allocator: std.mem.Allocator) void {
            shared.secureFree(allocator, self.bot_token);
            if (self.client_id) |id| allocator.free(id);
            shared.secureFreeOptional(allocator, self.client_secret);
            shared.secureFreeOptional(allocator, self.public_key);
            self.* = undefined;
        }
    };

    pub const Snowflake = []const u8;

    pub const User = struct {
        id: Snowflake,
        username: []const u8,
        discriminator: []const u8,
        avatar: ?[]const u8 = null,
        bot: bool = false,
    };

    pub const Guild = struct {
        id: Snowflake,
        name: []const u8,
        icon: ?[]const u8 = null,
        owner_id: Snowflake,
    };

    pub const Channel = struct {
        id: Snowflake,
        type: u8,
        guild_id: ?Snowflake = null,
        name: ?[]const u8 = null,
    };

    pub const Message = struct {
        id: Snowflake,
        channel_id: Snowflake,
        author: User,
        content: []const u8,
        timestamp: []const u8,
    };

    pub const GatewayIntent = struct {
        pub const GUILDS: u32 = 1 << 0;
        pub const GUILD_MEMBERS: u32 = 1 << 1;
        pub const GUILD_MESSAGES: u32 = 1 << 9;
        pub const MESSAGE_CONTENT: u32 = 1 << 15;
    };

    pub const Permission = struct {
        pub const SEND_MESSAGES: u64 = 1 << 11;
        pub const VIEW_CHANNEL: u64 = 1 << 10;
        pub const ADMINISTRATOR: u64 = 1 << 3;
        pub const MANAGE_GUILD: u64 = 1 << 5;
    };

    pub fn hasPermission(perms: u64, check: u64) bool {
        return (perms & check) != 0;
    }

    pub const Client = struct {
        allocator: std.mem.Allocator,

        pub fn init(_: std.mem.Allocator, _: Config) !Client {
            return Error.ConnectorsDisabled;
        }

        pub fn deinit(_: *Client) void {}
    };

    pub fn loadFromEnv(_: std.mem.Allocator) !Config {
        return Error.ConnectorsDisabled;
    }

    pub fn createClient(_: std.mem.Allocator) !Client {
        return Error.ConnectorsDisabled;
    }

    pub fn isAvailable() bool {
        return false;
    }
};

// ============================================================================
// Loader Functions (stub implementations)
// ============================================================================

pub fn loadOpenAI(_: std.mem.Allocator) !openai.Config {
    return Error.ConnectorsDisabled;
}

pub fn tryLoadOpenAI(_: std.mem.Allocator) !?openai.Config {
    return null;
}

pub fn loadHuggingFace(_: std.mem.Allocator) !huggingface.Config {
    return Error.ConnectorsDisabled;
}

pub fn tryLoadHuggingFace(_: std.mem.Allocator) !?huggingface.Config {
    return null;
}

pub fn loadOllama(_: std.mem.Allocator) !ollama.Config {
    return Error.ConnectorsDisabled;
}

pub fn tryLoadOllama(_: std.mem.Allocator) !?ollama.Config {
    return null;
}

pub fn loadLocalScheduler(_: std.mem.Allocator) !local_scheduler.Config {
    return Error.ConnectorsDisabled;
}

pub fn loadDiscord(_: std.mem.Allocator) !discord.Config {
    return Error.ConnectorsDisabled;
}

pub fn tryLoadDiscord(_: std.mem.Allocator) !?discord.Config {
    return null;
}

pub fn loadAnthropic(_: std.mem.Allocator) !anthropic.Config {
    return Error.ConnectorsDisabled;
}

pub fn tryLoadAnthropic(_: std.mem.Allocator) !?anthropic.Config {
    return null;
}

pub fn loadMistral(_: std.mem.Allocator) !mistral.Config {
    return Error.ConnectorsDisabled;
}

pub fn tryLoadMistral(_: std.mem.Allocator) !?mistral.Config {
    return null;
}

pub fn loadCohere(_: std.mem.Allocator) !cohere.Config {
    return Error.ConnectorsDisabled;
}

pub fn tryLoadCohere(_: std.mem.Allocator) !?cohere.Config {
    return null;
}

pub fn loadLMStudio(_: std.mem.Allocator) !lm_studio.Config {
    return Error.ConnectorsDisabled;
}

pub fn tryLoadLMStudio(_: std.mem.Allocator) !?lm_studio.Config {
    return null;
}

pub fn loadVLLM(_: std.mem.Allocator) !vllm.Config {
    return Error.ConnectorsDisabled;
}

pub fn tryLoadVLLM(_: std.mem.Allocator) !?vllm.Config {
    return null;
}

pub fn loadMLX(_: std.mem.Allocator) !mlx.Config {
    return Error.ConnectorsDisabled;
}

pub fn tryLoadMLX(_: std.mem.Allocator) !?mlx.Config {
    return null;
}

// ============================================================================
// Tests
// ============================================================================

test "connectors stub returns disabled" {
    try std.testing.expectEqual(false, isEnabled());
    try std.testing.expectEqual(false, isInitialized());
}

test "connectors stub init returns error" {
    try std.testing.expectError(Error.ConnectorsDisabled, init(std.testing.allocator));
}

test "connectors stub loaders return disabled or null" {
    try std.testing.expectError(Error.ConnectorsDisabled, loadOpenAI(std.testing.allocator));
    try std.testing.expectError(Error.ConnectorsDisabled, loadAnthropic(std.testing.allocator));
    try std.testing.expectError(Error.ConnectorsDisabled, loadLMStudio(std.testing.allocator));
    try std.testing.expectError(Error.ConnectorsDisabled, loadVLLM(std.testing.allocator));
    try std.testing.expectError(Error.ConnectorsDisabled, loadMLX(std.testing.allocator));

    const openai_opt = try tryLoadOpenAI(std.testing.allocator);
    try std.testing.expectEqual(@as(?openai.Config, null), openai_opt);

    const anthropic_opt = try tryLoadAnthropic(std.testing.allocator);
    try std.testing.expectEqual(@as(?anthropic.Config, null), anthropic_opt);

    const lm_studio_opt = try tryLoadLMStudio(std.testing.allocator);
    try std.testing.expectEqual(@as(?lm_studio.Config, null), lm_studio_opt);

    const vllm_opt = try tryLoadVLLM(std.testing.allocator);
    try std.testing.expectEqual(@as(?vllm.Config, null), vllm_opt);

    const mlx_opt = try tryLoadMLX(std.testing.allocator);
    try std.testing.expectEqual(@as(?mlx.Config, null), mlx_opt);
}

test "connectors stub isAvailable returns false" {
    try std.testing.expect(!openai.isAvailable());
    try std.testing.expect(!huggingface.isAvailable());
    try std.testing.expect(!ollama.isAvailable());
    try std.testing.expect(!anthropic.isAvailable());
    try std.testing.expect(!mistral.isAvailable());
    try std.testing.expect(!cohere.isAvailable());
    try std.testing.expect(!lm_studio.isAvailable());
    try std.testing.expect(!vllm.isAvailable());
    try std.testing.expect(!mlx.isAvailable());
}
