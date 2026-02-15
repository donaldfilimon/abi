const std = @import("std");
const shared = @import("../shared.zig");
const contract = @import("contract.zig");

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
    model_owned: bool = false,
    timeout_ms: u32 = 60_000,

    pub fn deinit(self: *Config, allocator: std.mem.Allocator) void {
        shared.deinitConfig(allocator, self.api_key, self.base_url);
        if (self.model_owned) allocator.free(@constCast(self.model));
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
        return contract.disabled(Client);
    }

    pub fn deinit(_: *Client) void {}

    pub fn chat(_: *Client, _: ChatRequest) !ChatResponse {
        return contract.disabled(ChatResponse);
    }

    pub fn chatSimple(_: *Client, _: []const u8) !ChatResponse {
        return contract.disabled(ChatResponse);
    }

    pub fn chatWithHistory(_: *Client, _: []const u8, _: []const ChatMessage) !ChatResponse {
        return contract.disabled(ChatResponse);
    }

    pub fn chatStreaming(_: *Client, _: ChatRequest) !void {
        return contract.disabled(void);
    }

    pub fn embed(_: *Client, _: EmbedRequest) !EmbedResponse {
        return contract.disabled(EmbedResponse);
    }

    pub fn embedTexts(_: *Client, _: []const []const u8) !EmbedResponse {
        return contract.disabled(EmbedResponse);
    }

    pub fn embedSingle(_: *Client, _: []const u8) ![]f32 {
        return contract.disabled([]f32);
    }

    pub fn rerank(_: *Client, _: RerankRequest) !RerankResponse {
        return contract.disabled(RerankResponse);
    }

    pub fn rerankDocuments(_: *Client, _: []const u8, _: []const []const u8, _: ?u32) !RerankResponse {
        return contract.disabled(RerankResponse);
    }
};

pub fn loadFromEnv(_: std.mem.Allocator) !Config {
    return contract.disabled(Config);
}

pub fn createClient(_: std.mem.Allocator) !Client {
    return contract.disabled(Client);
}

pub fn isAvailable() bool {
    return contract.unavailable();
}
