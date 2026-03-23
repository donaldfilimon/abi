//! Stub for Connectors module when disabled.
//!
//! Mirrors the full API of mod.zig, returning error.FeatureDisabled for all operations.

const std = @import("std");

/// Shared connector types (available even when connectors are disabled).
pub const shared = @import("shared.zig");

/// Connectors module errors.
pub const Error = error{
    FeatureDisabled,
    MissingApiKey,
    MissingApiToken,
    MissingBotToken,
    ApiRequestFailed,
    InvalidResponse,
    RateLimitExceeded,
    OutOfMemory,
};

var initialized = std.atomic.Value(bool).init(false);

pub fn init(_: std.mem.Allocator) !void {
    return Error.FeatureDisabled;
}

pub fn deinit() void {
    initialized.store(false, .release);
}

pub fn isEnabled() bool {
    return false;
}

pub fn isInitialized() bool {
    return initialized.load(.acquire);
}

pub fn getEnvOwned(_: std.mem.Allocator, _: []const u8) !?[]u8 {
    return Error.FeatureDisabled;
}

pub fn getFirstEnvOwned(_: std.mem.Allocator, _: []const []const u8) !?[]u8 {
    return Error.FeatureDisabled;
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
    return Error.FeatureDisabled;
}

// ============================================================================
// OpenAI Connector Stub
// ============================================================================

pub const openai = @import("stubs/openai.zig");
pub const codex = @import("stubs/codex.zig");
pub const opencode = @import("stubs/opencode.zig");
pub const claude = @import("stubs/claude.zig");
pub const gemini = @import("stubs/gemini.zig");

// ============================================================================
// HuggingFace Connector Stub
// ============================================================================

pub const huggingface = @import("stubs/huggingface.zig");

// ============================================================================
// Ollama Connector Stub
// ============================================================================

pub const ollama = @import("stubs/ollama.zig");
pub const ollama_passthrough = @import("stubs/ollama_passthrough.zig");

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
// llama.cpp Connector Stub
// ============================================================================

pub const llama_cpp = @import("stubs/llama_cpp.zig");

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
            return Error.FeatureDisabled;
        }

        pub fn deinit(_: *Client) void {}
    };

    pub fn loadFromEnv(_: std.mem.Allocator) !Config {
        return Error.FeatureDisabled;
    }

    pub fn createClient(_: std.mem.Allocator) !Client {
        return Error.FeatureDisabled;
    }

    pub fn isAvailable() bool {
        return false;
    }
};

// ============================================================================
// Loader Functions (stub implementations)
// ============================================================================

pub fn loadOpenAI(_: std.mem.Allocator) !openai.Config {
    return Error.FeatureDisabled;
}

pub fn tryLoadOpenAI(_: std.mem.Allocator) !?openai.Config {
    return null;
}

pub fn loadCodex(_: std.mem.Allocator) !codex.Config {
    return Error.FeatureDisabled;
}

pub fn tryLoadCodex(_: std.mem.Allocator) !?codex.Config {
    return null;
}

pub fn loadOpenCode(_: std.mem.Allocator) !opencode.Config {
    return Error.FeatureDisabled;
}

pub fn tryLoadOpenCode(_: std.mem.Allocator) !?opencode.Config {
    return null;
}

pub fn loadClaude(_: std.mem.Allocator) !claude.Config {
    return Error.FeatureDisabled;
}

pub fn tryLoadClaude(_: std.mem.Allocator) !?claude.Config {
    return null;
}

pub fn loadGemini(_: std.mem.Allocator) !gemini.Config {
    return Error.FeatureDisabled;
}

pub fn tryLoadGemini(_: std.mem.Allocator) !?gemini.Config {
    return null;
}

pub fn loadHuggingFace(_: std.mem.Allocator) !huggingface.Config {
    return Error.FeatureDisabled;
}

pub fn tryLoadHuggingFace(_: std.mem.Allocator) !?huggingface.Config {
    return null;
}

pub fn loadOllama(_: std.mem.Allocator) !ollama.Config {
    return Error.FeatureDisabled;
}

pub fn tryLoadOllama(_: std.mem.Allocator) !?ollama.Config {
    return null;
}

pub fn loadOllamaPassthrough(_: std.mem.Allocator) !ollama_passthrough.Config {
    return Error.FeatureDisabled;
}

pub fn tryLoadOllamaPassthrough(_: std.mem.Allocator) !?ollama_passthrough.Config {
    return null;
}

pub fn loadLocalScheduler(_: std.mem.Allocator) !local_scheduler.Config {
    return Error.FeatureDisabled;
}

pub fn loadDiscord(_: std.mem.Allocator) !discord.Config {
    return Error.FeatureDisabled;
}

pub fn tryLoadDiscord(_: std.mem.Allocator) !?discord.Config {
    return null;
}

pub fn loadAnthropic(_: std.mem.Allocator) !anthropic.Config {
    return Error.FeatureDisabled;
}

pub fn tryLoadAnthropic(_: std.mem.Allocator) !?anthropic.Config {
    return null;
}

pub fn loadMistral(_: std.mem.Allocator) !mistral.Config {
    return Error.FeatureDisabled;
}

pub fn tryLoadMistral(_: std.mem.Allocator) !?mistral.Config {
    return null;
}

pub fn loadCohere(_: std.mem.Allocator) !cohere.Config {
    return Error.FeatureDisabled;
}

pub fn tryLoadCohere(_: std.mem.Allocator) !?cohere.Config {
    return null;
}

pub fn loadLMStudio(_: std.mem.Allocator) !lm_studio.Config {
    return Error.FeatureDisabled;
}

pub fn tryLoadLMStudio(_: std.mem.Allocator) !?lm_studio.Config {
    return null;
}

pub fn loadVLLM(_: std.mem.Allocator) !vllm.Config {
    return Error.FeatureDisabled;
}

pub fn tryLoadVLLM(_: std.mem.Allocator) !?vllm.Config {
    return null;
}

pub fn loadMLX(_: std.mem.Allocator) !mlx.Config {
    return Error.FeatureDisabled;
}

pub fn tryLoadMLX(_: std.mem.Allocator) !?mlx.Config {
    return null;
}

pub fn loadLlamaCpp(_: std.mem.Allocator) !llama_cpp.Config {
    return Error.FeatureDisabled;
}

pub fn tryLoadLlamaCpp(_: std.mem.Allocator) !?llama_cpp.Config {
    return null;
}

pub const ProviderInfo = struct {
    name: []const u8,
    display_name: []const u8,
    env_key: []const u8,
    base_url: []const u8,
    is_alias: bool,
};

pub const ProviderRegistry = struct {
    pub const providers: [16]ProviderInfo = .{
        .{ .name = "openai", .display_name = "OpenAI", .env_key = "OPENAI_API_KEY", .base_url = "https://api.openai.com/v1", .is_alias = false },
        .{ .name = "anthropic", .display_name = "Anthropic", .env_key = "ANTHROPIC_API_KEY", .base_url = "https://api.anthropic.com/v1", .is_alias = false },
        .{ .name = "claude", .display_name = "Claude", .env_key = "ANTHROPIC_API_KEY", .base_url = "https://api.anthropic.com/v1", .is_alias = true },
        .{ .name = "codex", .display_name = "Codex", .env_key = "OPENAI_API_KEY", .base_url = "https://api.openai.com/v1", .is_alias = true },
        .{ .name = "opencode", .display_name = "OpenCode", .env_key = "ABI_OPENCODE_API_KEY", .base_url = "https://api.openai.com/v1", .is_alias = true },
        .{ .name = "gemini", .display_name = "Google Gemini", .env_key = "GEMINI_API_KEY", .base_url = "https://generativelanguage.googleapis.com/v1beta", .is_alias = false },
        .{ .name = "huggingface", .display_name = "HuggingFace", .env_key = "HF_API_TOKEN", .base_url = "https://api-inference.huggingface.co", .is_alias = false },
        .{ .name = "ollama", .display_name = "Ollama", .env_key = "OLLAMA_HOST", .base_url = "http://127.0.0.1:11434", .is_alias = false },
        .{ .name = "ollama_passthrough", .display_name = "Ollama Passthrough", .env_key = "OLLAMA_PASSTHROUGH_URL", .base_url = "http://127.0.0.1:11434", .is_alias = false },
        .{ .name = "mistral", .display_name = "Mistral AI", .env_key = "MISTRAL_API_KEY", .base_url = "https://api.mistral.ai/v1", .is_alias = false },
        .{ .name = "cohere", .display_name = "Cohere", .env_key = "COHERE_API_KEY", .base_url = "https://api.cohere.ai/v1", .is_alias = false },
        .{ .name = "lm_studio", .display_name = "LM Studio", .env_key = "ABI_LM_STUDIO_HOST", .base_url = "http://localhost:1234", .is_alias = false },
        .{ .name = "vllm", .display_name = "vLLM", .env_key = "VLLM_HOST", .base_url = "http://localhost:8000", .is_alias = false },
        .{ .name = "mlx", .display_name = "MLX", .env_key = "MLX_HOST", .base_url = "http://localhost:8080", .is_alias = false },
        .{ .name = "llama_cpp", .display_name = "llama.cpp", .env_key = "LLAMA_CPP_HOST", .base_url = "http://localhost:8080", .is_alias = false },
        .{ .name = "discord", .display_name = "Discord", .env_key = "DISCORD_BOT_TOKEN", .base_url = "https://discord.com/api/v10", .is_alias = false },
    };

    pub fn listAll() []const ProviderInfo {
        return &providers;
    }

    pub fn listAvailable() []const ProviderInfo {
        return &.{};
    }

    pub fn getByName(name: []const u8) ?ProviderInfo {
        for (providers) |p| {
            if (std.mem.eql(u8, p.name, name)) return p;
        }
        return null;
    }
};

// ============================================================================
// Tests
// ============================================================================

test "connectors stub returns disabled" {
    try std.testing.expectEqual(false, isEnabled());
    try std.testing.expectEqual(false, isInitialized());
}

test "connectors stub init returns error" {
    try std.testing.expectError(Error.FeatureDisabled, init(std.testing.allocator));
}

test "connectors stub loaders return disabled or null" {
    try std.testing.expectError(Error.FeatureDisabled, loadOpenAI(std.testing.allocator));
    try std.testing.expectError(Error.FeatureDisabled, loadAnthropic(std.testing.allocator));
    try std.testing.expectError(Error.FeatureDisabled, loadLMStudio(std.testing.allocator));
    try std.testing.expectError(Error.FeatureDisabled, loadVLLM(std.testing.allocator));
    try std.testing.expectError(Error.FeatureDisabled, loadMLX(std.testing.allocator));

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
    try std.testing.expect(!codex.isAvailable());
    try std.testing.expect(!opencode.isAvailable());
    try std.testing.expect(!claude.isAvailable());
    try std.testing.expect(!gemini.isAvailable());
    try std.testing.expect(!huggingface.isAvailable());
    try std.testing.expect(!ollama.isAvailable());
    try std.testing.expect(!ollama_passthrough.isAvailable());
    try std.testing.expect(!anthropic.isAvailable());
    try std.testing.expect(!mistral.isAvailable());
    try std.testing.expect(!cohere.isAvailable());
    try std.testing.expect(!lm_studio.isAvailable());
    try std.testing.expect(!vllm.isAvailable());
    try std.testing.expect(!mlx.isAvailable());
}

test "stub ProviderRegistry.listAll returns 16 providers" {
    const all = ProviderRegistry.listAll();
    try std.testing.expectEqual(@as(usize, 16), all.len);
}

test "stub ProviderRegistry.listAvailable returns empty" {
    const available = ProviderRegistry.listAvailable();
    try std.testing.expectEqual(@as(usize, 0), available.len);
}

test "stub ProviderRegistry.getByName finds openai" {
    const info = ProviderRegistry.getByName("openai");
    try std.testing.expect(info != null);
    try std.testing.expectEqualStrings("OpenAI", info.?.display_name);
}

test "stub ProviderRegistry.getByName returns null for nonexistent" {
    const info = ProviderRegistry.getByName("nonexistent");
    try std.testing.expect(info == null);
}

test {
    std.testing.refAllDecls(@This());
}
