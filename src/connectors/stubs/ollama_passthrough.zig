const std = @import("std");
const vllm = @import("vllm.zig");

pub const OllamaPassthroughError = vllm.VLLMError;
pub const Config = vllm.Config;
pub const Message = vllm.Message;
pub const ChatCompletionRequest = vllm.ChatCompletionRequest;
pub const ChatCompletionResponse = vllm.ChatCompletionResponse;
pub const Choice = vllm.Choice;
pub const Usage = vllm.Usage;
pub const Client = vllm.Client;

pub fn loadFromEnv(_: std.mem.Allocator) !Config {
    return error.ConnectorsDisabled;
}

pub fn createClient(_: std.mem.Allocator) !Client {
    return error.ConnectorsDisabled;
}

pub fn isAvailable() bool {
    return false;
}

test {
    std.testing.refAllDecls(@This());
}
