const std = @import("std");
const http_backends = @import("http_backends.zig");
const providers = @import("providers.zig");

pub fn generateResponse(
    agent: anytype,
    input: []const u8,
    allocator: std.mem.Allocator,
) ![]u8 {
    return switch (agent.config.backend) {
        .echo => generateEchoResponse(input, allocator),
        .openai => http_backends.generateOpenAIResponse(agent, allocator),
        .anthropic, .gemini, .codex, .llama_cpp => providers.generateProviderRouterResponse(agent, input, allocator),
        .ollama => http_backends.generateOllamaResponse(agent, allocator),
        .huggingface => http_backends.generateHuggingFaceResponse(agent, allocator),
        .local => generateLocalResponse(agent, allocator),
        .provider_router => providers.generateProviderRouterResponse(agent, input, allocator),
    };
}

fn generateEchoResponse(input: []const u8, allocator: std.mem.Allocator) ![]u8 {
    return std.fmt.allocPrint(allocator, "Echo: {s}", .{input});
}

fn generateLocalResponse(agent: anytype, allocator: std.mem.Allocator) ![]u8 {
    return std.fmt.allocPrint(allocator,
        \\[Local Model Response]
        \\Agent: {s}
        \\Temperature: {d:.2}
        \\
        \\Local transformer inference is available when AI features are enabled.
    , .{
        agent.config.name,
        agent.config.temperature,
    });
}
