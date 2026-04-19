const std = @import("std");

pub const openai = @import("openai.zig");
pub const codex = @import("codex.zig");
pub const opencode = @import("opencode.zig");
pub const claude = @import("claude.zig");
pub const gemini = @import("gemini.zig");
pub const huggingface = @import("huggingface.zig");
pub const ollama = @import("ollama.zig");
pub const ollama_passthrough = @import("ollama_passthrough.zig");
pub const local_scheduler = @import("local_scheduler.zig");
pub const discord = @import("discord/mod.zig");
pub const anthropic = @import("anthropic.zig");
pub const mistral = @import("mistral.zig");
pub const cohere = @import("cohere.zig");
pub const lm_studio = @import("lm_studio.zig");
pub const vllm = @import("vllm.zig");
pub const mlx = @import("mlx.zig");
pub const llama_cpp = @import("llama_cpp.zig");

fn LoaderPayload(comptime load_fn: anytype) type {
    const return_type = @typeInfo(@TypeOf(load_fn)).@"fn".return_type orelse
        @compileError("connector loader must have a return type");
    return switch (@typeInfo(return_type)) {
        .error_union => |info| info.payload,
        else => @compileError("connector loader must return an error union"),
    };
}

fn LoaderErrorSet(comptime load_fn: anytype) type {
    const return_type = @typeInfo(@TypeOf(load_fn)).@"fn".return_type orelse
        @compileError("connector loader must have a return type");
    return switch (@typeInfo(return_type)) {
        .error_union => |info| info.error_set,
        else => @compileError("connector loader must return an error union"),
    };
}

fn tryLoadWithMissing(
    allocator: std.mem.Allocator,
    comptime load_fn: anytype,
    comptime missing_err: LoaderErrorSet(load_fn),
) !?LoaderPayload(load_fn) {
    return load_fn(allocator) catch |err| {
        if (err == missing_err) {
            std.log.warn("connector: missing required env var for {s}, provider will be unavailable", .{@typeName(LoaderPayload(load_fn))});
            return null;
        }
        return err;
    };
}

fn tryLoadNullable(allocator: std.mem.Allocator, comptime load_fn: anytype) !?LoaderPayload(load_fn) {
    return load_fn(allocator) catch |err| {
        std.log.warn("connector: failed to load config for {s}: {s}, provider will be unavailable", .{ @typeName(LoaderPayload(load_fn)), @errorName(err) });
        return null;
    };
}

pub fn loadOpenAI(allocator: std.mem.Allocator) !openai.Config {
    return openai.loadFromEnv(allocator);
}

pub fn tryLoadOpenAI(allocator: std.mem.Allocator) !?openai.Config {
    return tryLoadWithMissing(allocator, openai.loadFromEnv, openai.OpenAIError.MissingApiKey);
}

pub fn loadCodex(allocator: std.mem.Allocator) !codex.Config {
    return codex.loadFromEnv(allocator);
}

pub fn tryLoadCodex(allocator: std.mem.Allocator) !?codex.Config {
    return tryLoadWithMissing(allocator, codex.loadFromEnv, codex.CodexError.MissingApiKey);
}

pub fn loadOpenCode(allocator: std.mem.Allocator) !opencode.Config {
    return opencode.loadFromEnv(allocator);
}

pub fn tryLoadOpenCode(allocator: std.mem.Allocator) !?opencode.Config {
    return tryLoadWithMissing(allocator, opencode.loadFromEnv, opencode.OpenCodeError.MissingApiKey);
}

pub fn loadClaude(allocator: std.mem.Allocator) !claude.Config {
    return claude.loadFromEnv(allocator);
}

pub fn tryLoadClaude(allocator: std.mem.Allocator) !?claude.Config {
    return tryLoadWithMissing(allocator, claude.loadFromEnv, claude.ClaudeError.MissingApiKey);
}

pub fn loadGemini(allocator: std.mem.Allocator) !gemini.Config {
    return gemini.loadFromEnv(allocator);
}

pub fn tryLoadGemini(allocator: std.mem.Allocator) !?gemini.Config {
    return tryLoadWithMissing(allocator, gemini.loadFromEnv, gemini.GeminiError.MissingApiKey);
}

pub fn loadHuggingFace(allocator: std.mem.Allocator) !huggingface.Config {
    return huggingface.loadFromEnv(allocator);
}

pub fn tryLoadHuggingFace(allocator: std.mem.Allocator) !?huggingface.Config {
    return tryLoadWithMissing(allocator, huggingface.loadFromEnv, huggingface.HuggingFaceError.MissingApiToken);
}

pub fn loadOllama(allocator: std.mem.Allocator) !ollama.Config {
    return ollama.loadFromEnv(allocator);
}

pub fn tryLoadOllama(allocator: std.mem.Allocator) !?ollama.Config {
    return tryLoadNullable(allocator, ollama.loadFromEnv);
}

pub fn loadOllamaPassthrough(allocator: std.mem.Allocator) !ollama_passthrough.Config {
    return ollama_passthrough.loadFromEnv(allocator);
}

pub fn tryLoadOllamaPassthrough(allocator: std.mem.Allocator) !?ollama_passthrough.Config {
    return tryLoadNullable(allocator, ollama_passthrough.loadFromEnv);
}

pub fn loadLocalScheduler(allocator: std.mem.Allocator) !local_scheduler.Config {
    return local_scheduler.loadFromEnv(allocator);
}

pub fn loadDiscord(allocator: std.mem.Allocator) !discord.Config {
    return discord.loadFromEnv(allocator);
}

pub fn tryLoadDiscord(allocator: std.mem.Allocator) !?discord.Config {
    return tryLoadWithMissing(allocator, discord.loadFromEnv, discord.DiscordError.MissingBotToken);
}

pub fn loadAnthropic(allocator: std.mem.Allocator) !anthropic.Config {
    return anthropic.loadFromEnv(allocator);
}

pub fn tryLoadAnthropic(allocator: std.mem.Allocator) !?anthropic.Config {
    return tryLoadWithMissing(allocator, anthropic.loadFromEnv, anthropic.AnthropicError.MissingApiKey);
}

pub fn loadMistral(allocator: std.mem.Allocator) !mistral.Config {
    return mistral.loadFromEnv(allocator);
}

pub fn tryLoadMistral(allocator: std.mem.Allocator) !?mistral.Config {
    return tryLoadWithMissing(allocator, mistral.loadFromEnv, mistral.MistralError.MissingApiKey);
}

pub fn loadCohere(allocator: std.mem.Allocator) !cohere.Config {
    return cohere.loadFromEnv(allocator);
}

pub fn tryLoadCohere(allocator: std.mem.Allocator) !?cohere.Config {
    return tryLoadWithMissing(allocator, cohere.loadFromEnv, cohere.CohereError.MissingApiKey);
}

pub fn loadLMStudio(allocator: std.mem.Allocator) !lm_studio.Config {
    return lm_studio.loadFromEnv(allocator);
}

pub fn tryLoadLMStudio(allocator: std.mem.Allocator) !?lm_studio.Config {
    return tryLoadNullable(allocator, lm_studio.loadFromEnv);
}

pub fn loadVLLM(allocator: std.mem.Allocator) !vllm.Config {
    return vllm.loadFromEnv(allocator);
}

pub fn tryLoadVLLM(allocator: std.mem.Allocator) !?vllm.Config {
    return tryLoadNullable(allocator, vllm.loadFromEnv);
}

pub fn loadMLX(allocator: std.mem.Allocator) !mlx.Config {
    return mlx.loadFromEnv(allocator);
}

pub fn tryLoadMLX(allocator: std.mem.Allocator) !?mlx.Config {
    return tryLoadNullable(allocator, mlx.loadFromEnv);
}

pub fn loadLlamaCpp(allocator: std.mem.Allocator) !llama_cpp.Config {
    return llama_cpp.loadFromEnv(allocator);
}

pub fn tryLoadLlamaCpp(allocator: std.mem.Allocator) !?llama_cpp.Config {
    return tryLoadNullable(allocator, llama_cpp.loadFromEnv);
}

fn mockMissingLoader(_: std.mem.Allocator) error{MissingConfig}!u8 {
    return error.MissingConfig;
}

fn mockOtherLoader(_: std.mem.Allocator) error{ MissingConfig, InvalidConfig }!u8 {
    return error.InvalidConfig;
}

fn mockSuccessLoader(_: std.mem.Allocator) error{MissingConfig}!u8 {
    return 7;
}

test "tryLoadWithMissing returns null for configured missing error" {
    const result = try tryLoadWithMissing(std.testing.allocator, mockMissingLoader, error.MissingConfig);
    try std.testing.expect(result == null);
}

test "tryLoadWithMissing preserves non-missing errors" {
    try std.testing.expectError(
        error.InvalidConfig,
        tryLoadWithMissing(std.testing.allocator, mockOtherLoader, error.MissingConfig),
    );
}

test "tryLoadNullable returns null for any error" {
    const result = try tryLoadNullable(std.testing.allocator, mockOtherLoader);
    try std.testing.expect(result == null);
}

test "tryLoadWithMissing preserves successful payloads" {
    const result = try tryLoadWithMissing(std.testing.allocator, mockSuccessLoader, error.MissingConfig);
    try std.testing.expectEqual(@as(u8, 7), result.?);
}

test {
    std.testing.refAllDecls(@This());
}
