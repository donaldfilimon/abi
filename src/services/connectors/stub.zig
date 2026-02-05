//! Stub for Connectors module when disabled.
//!
//! Mirrors the full API of mod.zig, returning error.ConnectorsDisabled for all operations.

const std = @import("std");

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

pub const openai = struct {
    pub const OpenAIError = error{
        MissingApiKey,
        ApiRequestFailed,
        InvalidResponse,
        RateLimitExceeded,
    };

    pub const Config = struct {
        api_key: []u8,
        base_url: []u8,
        model: []const u8 = "gpt-4",
        timeout_ms: u32 = 60_000,

        pub fn deinit(self: *Config, allocator: std.mem.Allocator) void {
            allocator.free(self.api_key);
            allocator.free(self.base_url);
            self.* = undefined;
        }
    };

    pub const Message = struct {
        role: []const u8,
        content: []const u8,
    };

    pub const ChatCompletionRequest = struct {
        model: []const u8,
        messages: []Message,
        temperature: f32 = 0.7,
        max_tokens: ?u32 = null,
        stream: bool = false,
    };

    pub const ChatCompletionResponse = struct {
        id: []const u8,
        object: []const u8,
        created: u64,
        model: []const u8,
        choices: []Choice,
        usage: Usage,
    };

    pub const Choice = struct {
        index: u32,
        message: Message,
        finish_reason: []const u8,
    };

    pub const Usage = struct {
        prompt_tokens: u32,
        completion_tokens: u32,
        total_tokens: u32,
    };

    pub const StreamingChunk = struct {
        id: []const u8,
        object: []const u8,
        created: u64,
        model: []const u8,
        choices: []StreamingChoice,
        delta: ?StreamingDelta = null,
    };

    pub const StreamingChoice = struct {
        index: u32,
        delta: ?StreamingDelta,
        finish_reason: ?[]const u8 = null,
    };

    pub const StreamingDelta = struct {
        role: ?[]const u8 = null,
        content: ?[]const u8 = null,
    };

    pub const Client = struct {
        allocator: std.mem.Allocator,

        pub fn init(_: std.mem.Allocator, _: Config) !Client {
            return Error.ConnectorsDisabled;
        }

        pub fn deinit(_: *Client) void {}

        pub fn chatCompletion(_: *Client, _: ChatCompletionRequest) !ChatCompletionResponse {
            return Error.ConnectorsDisabled;
        }

        pub fn chat(_: *Client, _: []Message) !ChatCompletionResponse {
            return Error.ConnectorsDisabled;
        }

        pub fn chatSimple(_: *Client, _: []const u8) !ChatCompletionResponse {
            return Error.ConnectorsDisabled;
        }
    };

    pub fn loadFromEnv(_: std.mem.Allocator) !Config {
        return Error.ConnectorsDisabled;
    }

    pub fn createClient(_: std.mem.Allocator) !Client {
        return Error.ConnectorsDisabled;
    }
};

// ============================================================================
// HuggingFace Connector Stub
// ============================================================================

pub const huggingface = struct {
    pub const HuggingFaceError = error{
        MissingApiToken,
        ModelLoading,
        ApiRequestFailed,
        InvalidResponse,
    };

    pub const Config = struct {
        api_token: []u8,
        base_url: []u8,
        model: []const u8 = "gpt2",
        timeout_ms: u32 = 60_000,

        pub fn deinit(self: *Config, allocator: std.mem.Allocator) void {
            allocator.free(self.api_token);
            allocator.free(self.base_url);
            self.* = undefined;
        }
    };

    pub const InferenceRequest = struct {
        inputs: ?[]const u8 = null,
        parameters: ?Parameters = null,
    };

    pub const Parameters = struct {
        top_k: ?u32 = null,
        top_p: ?f32 = null,
        temperature: ?f32 = null,
        max_new_tokens: ?u32 = null,
        return_full_text: ?bool = null,
    };

    pub const InferenceResponse = struct {
        generated_text: []const u8,
    };

    pub const TextGenerationRequest = struct {
        inputs: []const u8,
        parameters: ?Parameters = null,
    };

    pub const Client = struct {
        allocator: std.mem.Allocator,

        pub fn init(_: std.mem.Allocator, _: Config) !Client {
            return Error.ConnectorsDisabled;
        }

        pub fn deinit(_: *Client) void {}

        pub fn inference(_: *Client, _: InferenceRequest) !InferenceResponse {
            return Error.ConnectorsDisabled;
        }

        pub fn generateText(_: *Client, _: []const u8, _: ?Parameters) !InferenceResponse {
            return Error.ConnectorsDisabled;
        }

        pub fn generateTextSimple(_: *Client, _: []const u8) !InferenceResponse {
            return Error.ConnectorsDisabled;
        }
    };

    pub fn loadFromEnv(_: std.mem.Allocator) !Config {
        return Error.ConnectorsDisabled;
    }

    pub fn createClient(_: std.mem.Allocator) !Client {
        return Error.ConnectorsDisabled;
    }
};

// ============================================================================
// Ollama Connector Stub
// ============================================================================

pub const ollama = struct {
    pub const Config = struct {
        host: []u8,
        model: []const u8 = "llama2",
        model_owned: bool = false,
        timeout_ms: u32 = 120_000,

        pub fn deinit(self: *Config, allocator: std.mem.Allocator) void {
            allocator.free(self.host);
            if (self.model_owned) {
                allocator.free(@constCast(self.model));
            }
            self.* = undefined;
        }
    };

    pub const Message = struct {
        role: []const u8,
        content: []const u8,
    };

    pub const GenerateRequest = struct {
        model: []const u8,
        prompt: []const u8,
        stream: bool = false,
        options: ?Options = null,
    };

    pub const ChatRequest = struct {
        model: []const u8,
        messages: []Message,
        stream: bool = false,
    };

    pub const Options = struct {
        temperature: f32 = 0.7,
        num_predict: u32 = 128,
        top_p: f32 = 0.9,
        top_k: u32 = 40,
    };

    pub const GenerateResponse = struct {
        model: []const u8,
        response: []const u8,
        done: bool,
        context: ?[]u64 = null,

        pub fn deinit(_: *GenerateResponse, _: std.mem.Allocator) void {}
    };

    pub const ChatResponse = struct {
        model: []const u8,
        message: Message,
        done: bool,

        pub fn deinit(_: *ChatResponse, _: std.mem.Allocator) void {}
    };

    pub const Client = struct {
        allocator: std.mem.Allocator,

        pub fn init(_: std.mem.Allocator, _: Config) !Client {
            return Error.ConnectorsDisabled;
        }

        pub fn deinit(_: *Client) void {}

        pub fn generate(_: *Client, _: GenerateRequest) !GenerateResponse {
            return Error.ConnectorsDisabled;
        }

        pub fn generateSimple(_: *Client, _: []const u8) !GenerateResponse {
            return Error.ConnectorsDisabled;
        }

        pub fn chat(_: *Client, _: ChatRequest) !ChatResponse {
            return Error.ConnectorsDisabled;
        }

        pub fn chatSimple(_: *Client, _: []const u8) !ChatResponse {
            return Error.ConnectorsDisabled;
        }
    };

    pub fn loadFromEnv(_: std.mem.Allocator) !Config {
        return Error.ConnectorsDisabled;
    }

    pub fn createClient(_: std.mem.Allocator) !Client {
        return Error.ConnectorsDisabled;
    }
};

// ============================================================================
// Anthropic Connector Stub
// ============================================================================

pub const anthropic = struct {
    pub const AnthropicError = error{
        MissingApiKey,
        ApiRequestFailed,
        InvalidResponse,
        RateLimitExceeded,
        ContentFiltered,
    };

    pub const Config = struct {
        api_key: []u8,
        base_url: []u8,
        model: []const u8 = "claude-3-5-sonnet-20241022",
        max_tokens: u32 = 4096,
        timeout_ms: u32 = 120_000,

        pub fn deinit(self: *Config, allocator: std.mem.Allocator) void {
            allocator.free(self.api_key);
            allocator.free(self.base_url);
            self.* = undefined;
        }
    };

    pub const Message = struct {
        role: []const u8,
        content: []const u8,
    };

    pub const MessagesRequest = struct {
        model: []const u8,
        messages: []const Message,
        max_tokens: u32 = 4096,
        temperature: f32 = 0.7,
        system: ?[]const u8 = null,
        stream: bool = false,
    };

    pub const ContentBlock = struct {
        type: []const u8,
        text: []const u8,
    };

    pub const MessagesResponse = struct {
        id: []const u8,
        type: []const u8,
        role: []const u8,
        content: []ContentBlock,
        model: []const u8,
        stop_reason: ?[]const u8,
        usage: Usage,
    };

    pub const Usage = struct {
        input_tokens: u32,
        output_tokens: u32,
    };

    pub const EmbeddingRequest = struct {
        model: []const u8 = "voyage-3",
        input: []const []const u8,
        input_type: []const u8 = "document",
    };

    pub const EmbeddingResponse = struct {
        object: []const u8,
        data: []EmbeddingData,
        model: []const u8,
        usage: EmbeddingUsage,
    };

    pub const EmbeddingData = struct {
        object: []const u8,
        index: u32,
        embedding: []f32,
    };

    pub const EmbeddingUsage = struct {
        total_tokens: u32,
    };

    pub const Client = struct {
        allocator: std.mem.Allocator,

        pub fn init(_: std.mem.Allocator, _: Config) !Client {
            return Error.ConnectorsDisabled;
        }

        pub fn deinit(_: *Client) void {}

        pub fn messages(_: *Client, _: MessagesRequest) !MessagesResponse {
            return Error.ConnectorsDisabled;
        }

        pub fn chat(_: *Client, _: []const Message) !MessagesResponse {
            return Error.ConnectorsDisabled;
        }

        pub fn chatSimple(_: *Client, _: []const u8) !MessagesResponse {
            return Error.ConnectorsDisabled;
        }

        pub fn chatWithSystem(_: *Client, _: []const u8, _: []const Message) !MessagesResponse {
            return Error.ConnectorsDisabled;
        }

        pub fn getResponseText(_: *Client, _: MessagesResponse) ![]u8 {
            return Error.ConnectorsDisabled;
        }
    };

    pub fn loadFromEnv(_: std.mem.Allocator) !Config {
        return Error.ConnectorsDisabled;
    }

    pub fn createClient(_: std.mem.Allocator) !Client {
        return Error.ConnectorsDisabled;
    }
};

// ============================================================================
// Mistral Connector Stub
// ============================================================================

pub const mistral = struct {
    pub const MistralError = error{
        MissingApiKey,
        ApiRequestFailed,
        InvalidResponse,
        RateLimitExceeded,
    };

    pub const Config = struct {
        api_key: []u8,
        base_url: []u8,
        model: []const u8 = "mistral-large-latest",
        timeout_ms: u32 = 60_000,

        pub fn deinit(self: *Config, allocator: std.mem.Allocator) void {
            allocator.free(self.api_key);
            allocator.free(self.base_url);
            self.* = undefined;
        }
    };

    pub const Message = struct {
        role: []const u8,
        content: []const u8,
    };

    pub const ChatCompletionRequest = struct {
        model: []const u8,
        messages: []const Message,
        temperature: f32 = 0.7,
        max_tokens: ?u32 = null,
        top_p: f32 = 1.0,
        stream: bool = false,
        safe_prompt: bool = false,
    };

    pub const ChatCompletionResponse = struct {
        id: []const u8,
        object: []const u8,
        created: u64,
        model: []const u8,
        choices: []Choice,
        usage: Usage,
    };

    pub const Choice = struct {
        index: u32,
        message: Message,
        finish_reason: []const u8,
    };

    pub const Usage = struct {
        prompt_tokens: u32,
        completion_tokens: u32,
        total_tokens: u32,
    };

    pub const EmbeddingRequest = struct {
        model: []const u8 = "mistral-embed",
        input: []const []const u8,
        encoding_format: []const u8 = "float",
    };

    pub const EmbeddingResponse = struct {
        id: []const u8,
        object: []const u8,
        model: []const u8,
        data: []EmbeddingData,
        usage: EmbeddingUsage,
    };

    pub const EmbeddingData = struct {
        object: []const u8,
        index: u32,
        embedding: []f32,
    };

    pub const EmbeddingUsage = struct {
        prompt_tokens: u32,
        total_tokens: u32,
    };

    pub const Client = struct {
        allocator: std.mem.Allocator,

        pub fn init(_: std.mem.Allocator, _: Config) !Client {
            return Error.ConnectorsDisabled;
        }

        pub fn deinit(_: *Client) void {}

        pub fn chatCompletion(_: *Client, _: ChatCompletionRequest) !ChatCompletionResponse {
            return Error.ConnectorsDisabled;
        }

        pub fn chat(_: *Client, _: []const Message) !ChatCompletionResponse {
            return Error.ConnectorsDisabled;
        }

        pub fn chatSimple(_: *Client, _: []const u8) !ChatCompletionResponse {
            return Error.ConnectorsDisabled;
        }

        pub fn embeddings(_: *Client, _: EmbeddingRequest) !EmbeddingResponse {
            return Error.ConnectorsDisabled;
        }

        pub fn embed(_: *Client, _: []const []const u8) !EmbeddingResponse {
            return Error.ConnectorsDisabled;
        }

        pub fn embedSingle(_: *Client, _: []const u8) ![]f32 {
            return Error.ConnectorsDisabled;
        }
    };

    pub fn loadFromEnv(_: std.mem.Allocator) !Config {
        return Error.ConnectorsDisabled;
    }

    pub fn createClient(_: std.mem.Allocator) !Client {
        return Error.ConnectorsDisabled;
    }
};

// ============================================================================
// Cohere Connector Stub
// ============================================================================

pub const cohere = struct {
    pub const CohereError = error{
        MissingApiKey,
        ApiRequestFailed,
        InvalidResponse,
        RateLimitExceeded,
    };

    pub const Config = struct {
        api_key: []u8,
        base_url: []u8,
        model: []const u8 = "command-r-plus",
        timeout_ms: u32 = 60_000,

        pub fn deinit(self: *Config, allocator: std.mem.Allocator) void {
            allocator.free(self.api_key);
            allocator.free(self.base_url);
            self.* = undefined;
        }
    };

    pub const ChatRole = enum {
        user,
        assistant,
        system,
        tool,

        pub fn toString(self: ChatRole) []const u8 {
            return switch (self) {
                .user => "USER",
                .assistant => "CHATBOT",
                .system => "SYSTEM",
                .tool => "TOOL",
            };
        }
    };

    pub const ChatMessage = struct {
        role: ChatRole,
        message: []const u8,
    };

    pub const ChatRequest = struct {
        model: []const u8,
        message: []const u8,
        chat_history: []const ChatMessage = &.{},
        preamble: ?[]const u8 = null,
        temperature: f32 = 0.3,
        max_tokens: ?u32 = null,
        stream: bool = false,
    };

    pub const ChatResponse = struct {
        response_id: []const u8,
        text: []const u8,
        generation_id: []const u8,
        finish_reason: []const u8,
        meta: ChatMeta,
    };

    pub const ChatMeta = struct {
        api_version: ApiVersion,
        billed_units: BilledUnits,
        tokens: TokenUsage,
    };

    pub const ApiVersion = struct {
        version: []const u8,
    };

    pub const BilledUnits = struct {
        input_tokens: u32,
        output_tokens: u32,
    };

    pub const TokenUsage = struct {
        input_tokens: u32,
        output_tokens: u32,
    };

    pub const EmbedRequest = struct {
        model: []const u8 = "embed-english-v3.0",
        texts: []const []const u8,
        input_type: []const u8 = "search_document",
        truncate: []const u8 = "END",
    };

    pub const EmbedResponse = struct {
        id: []const u8,
        embeddings: [][]f32,
        texts: [][]const u8,
        meta: EmbedMeta,
    };

    pub const EmbedMeta = struct {
        api_version: ApiVersion,
        billed_units: EmbedBilledUnits,
    };

    pub const EmbedBilledUnits = struct {
        input_tokens: u32,
    };

    pub const RerankRequest = struct {
        model: []const u8 = "rerank-english-v3.0",
        query: []const u8,
        documents: []const []const u8,
        top_n: ?u32 = null,
        return_documents: bool = false,
    };

    pub const RerankResponse = struct {
        id: []const u8,
        results: []RerankResult,
        meta: RerankMeta,
    };

    pub const RerankResult = struct {
        index: u32,
        relevance_score: f32,
        document: ?[]const u8 = null,
    };

    pub const RerankMeta = struct {
        api_version: ApiVersion,
        billed_units: RerankBilledUnits,
    };

    pub const RerankBilledUnits = struct {
        search_units: u32,
    };

    pub const Client = struct {
        allocator: std.mem.Allocator,

        pub fn init(_: std.mem.Allocator, _: Config) !Client {
            return Error.ConnectorsDisabled;
        }

        pub fn deinit(_: *Client) void {}

        pub fn chat(_: *Client, _: ChatRequest) !ChatResponse {
            return Error.ConnectorsDisabled;
        }

        pub fn chatSimple(_: *Client, _: []const u8) !ChatResponse {
            return Error.ConnectorsDisabled;
        }

        pub fn chatWithHistory(_: *Client, _: []const u8, _: []const ChatMessage) !ChatResponse {
            return Error.ConnectorsDisabled;
        }

        pub fn embed(_: *Client, _: EmbedRequest) !EmbedResponse {
            return Error.ConnectorsDisabled;
        }

        pub fn embedTexts(_: *Client, _: []const []const u8) !EmbedResponse {
            return Error.ConnectorsDisabled;
        }

        pub fn embedSingle(_: *Client, _: []const u8) ![]f32 {
            return Error.ConnectorsDisabled;
        }

        pub fn rerank(_: *Client, _: RerankRequest) !RerankResponse {
            return Error.ConnectorsDisabled;
        }

        pub fn rerankDocuments(_: *Client, _: []const u8, _: []const []const u8, _: ?u32) !RerankResponse {
            return Error.ConnectorsDisabled;
        }
    };

    pub fn loadFromEnv(_: std.mem.Allocator) !Config {
        return Error.ConnectorsDisabled;
    }

    pub fn createClient(_: std.mem.Allocator) !Client {
        return Error.ConnectorsDisabled;
    }
};

// ============================================================================
// Local Scheduler Connector Stub
// ============================================================================

pub const local_scheduler = struct {
    pub const Config = struct {
        url: []u8,

        pub fn deinit(self: *Config, allocator: std.mem.Allocator) void {
            allocator.free(self.url);
            self.* = undefined;
        }

        pub fn endpoint(_: *const Config, _: std.mem.Allocator, _: []const u8) ![]u8 {
            return Error.ConnectorsDisabled;
        }

        pub fn healthUrl(_: *const Config, _: std.mem.Allocator) ![]u8 {
            return Error.ConnectorsDisabled;
        }

        pub fn submitUrl(_: *const Config, _: std.mem.Allocator) ![]u8 {
            return Error.ConnectorsDisabled;
        }
    };

    pub fn loadFromEnv(_: std.mem.Allocator) !Config {
        return Error.ConnectorsDisabled;
    }
};

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
            allocator.free(self.bot_token);
            if (self.client_id) |id| allocator.free(id);
            if (self.client_secret) |secret| allocator.free(secret);
            if (self.public_key) |key| allocator.free(key);
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

    const openai_opt = try tryLoadOpenAI(std.testing.allocator);
    try std.testing.expectEqual(@as(?openai.Config, null), openai_opt);

    const anthropic_opt = try tryLoadAnthropic(std.testing.allocator);
    try std.testing.expectEqual(@as(?anthropic.Config, null), anthropic_opt);
}
