const std = @import("std");
const anthropic = @import("anthropic.zig");

pub const ClaudeError = anthropic.AnthropicError;
pub const Config = anthropic.Config;
pub const Message = anthropic.Message;
pub const MessagesRequest = anthropic.MessagesRequest;
pub const ContentBlock = anthropic.ContentBlock;
pub const MessagesResponse = anthropic.MessagesResponse;
pub const Usage = anthropic.Usage;
pub const EmbeddingRequest = anthropic.EmbeddingRequest;
pub const EmbeddingResponse = anthropic.EmbeddingResponse;
pub const EmbeddingData = anthropic.EmbeddingData;
pub const EmbeddingUsage = anthropic.EmbeddingUsage;
pub const Client = anthropic.Client;

pub fn loadFromEnv(_: std.mem.Allocator) !Config {
    return error.ConnectorsDisabled;
}

pub fn createClient(_: std.mem.Allocator) !Client {
    return error.ConnectorsDisabled;
}

pub fn isAvailable() bool {
    return false;
}
