const std = @import("std");
const openai = @import("openai.zig");

pub const OpenCodeError = openai.OpenAIError;
pub const Config = openai.Config;
pub const Message = openai.Message;
pub const ChatCompletionRequest = openai.ChatCompletionRequest;
pub const ChatCompletionResponse = openai.ChatCompletionResponse;
pub const StreamingChunk = openai.StreamingChunk;
pub const StreamingChoice = openai.StreamingChoice;
pub const StreamingDelta = openai.StreamingDelta;
pub const Client = openai.Client;

pub fn loadFromEnv(_: std.mem.Allocator) !Config {
    return error.ConnectorsDisabled;
}

pub fn createClient(_: std.mem.Allocator) !Client {
    return error.ConnectorsDisabled;
}

pub fn isAvailable() bool {
    return false;
}
