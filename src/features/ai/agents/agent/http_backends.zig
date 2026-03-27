const std = @import("std");
const foundation = @import("../../../../foundation/mod.zig");
const connectors = @import("../../../../connectors/mod.zig");
const types = @import("../types.zig");
const payloads = @import("payloads.zig");

const http = foundation.utils.async_http;
const retry = foundation.utils.http_retry;
const time = foundation.utils;
const platform_time = foundation.time;
const escape_json = foundation.utils.json.escapeJsonContent;

pub fn generateOpenAIResponse(agent: anytype, allocator: std.mem.Allocator) ![]u8 {
    const api_key = try connectors.getFirstEnvOwned(allocator, &.{
        "ABI_OPENAI_API_KEY",
        "OPENAI_API_KEY",
    }) orelse return types.AgentError.ApiKeyMissing;
    defer allocator.free(api_key);

    const base_url = try connectors.getEnvOwned(allocator, "ABI_OPENAI_BASE_URL") orelse
        try allocator.dupe(u8, "https://api.openai.com/v1");
    defer allocator.free(base_url);

    const endpoint = try std.fmt.allocPrint(allocator, "{s}/chat/completions", .{base_url});
    defer allocator.free(endpoint);

    const messages_json = try payloads.allocJsonMessages(allocator, agent.history.items);
    defer allocator.free(messages_json);

    const request_body = try std.fmt.allocPrint(allocator,
        \\{{"model":"{s}","messages":{s},"temperature":{d:.2},"max_tokens":{d}}}
    , .{
        agent.config.model,
        messages_json,
        agent.config.temperature,
        agent.config.max_tokens,
    });
    defer allocator.free(request_body);

    var response = try fetchWithRetry(agent, allocator, endpoint, request_body, api_key);
    defer response.deinit();

    if (response.status_code == 429) {
        return types.AgentError.RateLimitExceeded;
    }

    if (!response.isSuccess()) {
        std.log.err(
            "OpenAI API returned status {d}: {s}",
            .{ response.status_code, response.body },
        );
        return types.AgentError.HttpRequestFailed;
    }

    const parsed = std.json.parseFromSlice(
        std.json.Value,
        allocator,
        response.body,
        .{},
    ) catch |err| {
        std.log.err("Failed to parse OpenAI response: {}", .{err});
        return types.AgentError.InvalidApiResponse;
    };
    defer parsed.deinit();

    const choices = parsed.value.object.get("choices") orelse return types.AgentError.InvalidApiResponse;
    if (choices.array.items.len == 0) return types.AgentError.InvalidApiResponse;

    const first_choice = choices.array.items[0];
    const message = first_choice.object.get("message") orelse return types.AgentError.InvalidApiResponse;
    const content = message.object.get("content") orelse return types.AgentError.InvalidApiResponse;

    if (parsed.value.object.get("usage")) |usage| {
        if (usage.object.get("total_tokens")) |total| {
            agent.total_tokens_used += @as(u64, @intCast(total.integer));
        }
    }

    return try allocator.dupe(u8, content.string);
}

pub fn generateOllamaResponse(agent: anytype, allocator: std.mem.Allocator) ![]u8 {
    const ollama_host = try connectors.getFirstEnvOwned(allocator, &.{
        "ABI_OLLAMA_HOST",
        "OLLAMA_HOST",
    }) orelse try allocator.dupe(u8, "http://127.0.0.1:11434");
    defer allocator.free(ollama_host);

    const model_name = try connectors.getEnvOwned(allocator, "ABI_OLLAMA_MODEL") orelse
        try allocator.dupe(u8, agent.config.model);
    defer allocator.free(model_name);

    const endpoint = try std.fmt.allocPrint(allocator, "{s}/api/chat", .{ollama_host});
    defer allocator.free(endpoint);

    const messages_json = try payloads.allocJsonMessages(allocator, agent.history.items);
    defer allocator.free(messages_json);

    const request_body = try std.fmt.allocPrint(allocator,
        \\{{"model":"{s}","messages":{s},"stream":false,"options":{{"temperature":{d:.2},"num_predict":{d}}}}}
    , .{
        model_name,
        messages_json,
        agent.config.temperature,
        agent.config.max_tokens,
    });
    defer allocator.free(request_body);

    var client = try http.AsyncHttpClient.init(allocator);
    defer client.deinit();

    var request = try http.HttpRequest.init(allocator, .post, endpoint);
    defer request.deinit();

    try request.setHeader("Content-Type", "application/json");
    try request.setBody(request_body);

    var response = client.fetch(&request) catch |err| {
        std.log.err(
            "Ollama API request failed: {}. Is Ollama running on {s}?",
            .{ err, ollama_host },
        );
        return types.AgentError.HttpRequestFailed;
    };
    defer response.deinit();

    if (!response.isSuccess()) {
        std.log.err(
            "Ollama API returned status {d}: {s}",
            .{ response.status_code, response.body },
        );
        return types.AgentError.HttpRequestFailed;
    }

    const parsed = std.json.parseFromSlice(
        std.json.Value,
        allocator,
        response.body,
        .{},
    ) catch |err| {
        std.log.err("Failed to parse Ollama response: {}", .{err});
        return types.AgentError.InvalidApiResponse;
    };
    defer parsed.deinit();

    const message = parsed.value.object.get("message") orelse return types.AgentError.InvalidApiResponse;
    const content = message.object.get("content") orelse return types.AgentError.InvalidApiResponse;

    if (parsed.value.object.get("prompt_eval_count")) |prompt_tokens| {
        if (parsed.value.object.get("eval_count")) |completion_tokens| {
            agent.total_tokens_used += @as(u64, @intCast(prompt_tokens.integer));
            agent.total_tokens_used += @as(u64, @intCast(completion_tokens.integer));
        }
    }

    return try allocator.dupe(u8, content.string);
}

pub fn generateHuggingFaceResponse(agent: anytype, allocator: std.mem.Allocator) ![]u8 {
    const api_token = try connectors.getFirstEnvOwned(allocator, &.{
        "ABI_HF_API_TOKEN",
        "HF_API_TOKEN",
        "HUGGING_FACE_HUB_TOKEN",
    }) orelse return types.AgentError.ApiKeyMissing;
    defer allocator.free(api_token);

    const base_url = try connectors.getEnvOwned(allocator, "ABI_HF_BASE_URL") orelse
        try allocator.dupe(u8, "https://api-inference.huggingface.co");
    defer allocator.free(base_url);

    const endpoint = try std.fmt.allocPrint(allocator, "{s}/models/{s}", .{
        base_url,
        agent.config.model,
    });
    defer allocator.free(endpoint);

    const prompt_text = try payloads.allocTranscript(allocator, agent.history.items);
    defer allocator.free(prompt_text);

    const escaped_prompt = try escape_json(allocator, prompt_text);
    defer allocator.free(escaped_prompt);

    const request_body = try std.fmt.allocPrint(allocator,
        \\{{"inputs":"{s}","parameters":{{"temperature":{d:.2},"max_new_tokens":{d},"return_full_text":false}}}}
    , .{
        escaped_prompt,
        agent.config.temperature,
        agent.config.max_tokens,
    });
    defer allocator.free(request_body);

    var client = try http.AsyncHttpClient.init(allocator);
    defer client.deinit();

    var request = try http.HttpRequest.init(allocator, .post, endpoint);
    defer request.deinit();

    try request.setBearerToken(api_token);
    try request.setHeader("Content-Type", "application/json");
    try request.setBody(request_body);

    var response = client.fetch(&request) catch |err| {
        std.log.err("HuggingFace API request failed: {}", .{err});
        return types.AgentError.HttpRequestFailed;
    };
    defer response.deinit();

    if (response.status_code == 429) {
        return types.AgentError.RateLimitExceeded;
    }

    if (!response.isSuccess()) {
        std.log.err(
            "HuggingFace API returned status {d}: {s}",
            .{ response.status_code, response.body },
        );
        return types.AgentError.HttpRequestFailed;
    }

    const parsed = std.json.parseFromSlice(
        std.json.Value,
        allocator,
        response.body,
        .{},
    ) catch |err| {
        std.log.err("Failed to parse HuggingFace response: {}", .{err});
        return types.AgentError.InvalidApiResponse;
    };
    defer parsed.deinit();

    if (parsed.value != .array) return types.AgentError.InvalidApiResponse;
    if (parsed.value.array.items.len == 0) return types.AgentError.InvalidApiResponse;

    const first_result = parsed.value.array.items[0];
    const generated_text = first_result.object.get("generated_text") orelse return types.AgentError.InvalidApiResponse;

    return try allocator.dupe(u8, generated_text.string);
}

fn fetchWithRetry(
    agent: anytype,
    allocator: std.mem.Allocator,
    endpoint: []const u8,
    request_body: []const u8,
    api_key: []const u8,
) !http.HttpResponse {
    var client = try http.AsyncHttpClient.init(allocator);
    defer client.deinit();

    var request = try http.HttpRequest.init(allocator, .post, endpoint);
    defer request.deinit();

    try request.setBearerToken(api_key);
    try request.setHeader("Content-Type", "application/json");
    try request.setBody(request_body);

    var attempt: u32 = 0;
    var backoff_ms = agent.config.retry_config.initial_delay_ms;
    const seed = platform_time.getSeed();
    var prng = std.Random.DefaultPrng.init(seed);
    const random = prng.random();
    const max_attempts = agent.config.retry_config.max_retries;

    return while (attempt <= max_attempts) : (attempt += 1) {
        var response = client.fetch(&request) catch |err| {
            if (attempt >= max_attempts) {
                std.log.err(
                    "OpenAI API request failed after {d} attempts: {}",
                    .{ attempt + 1, err },
                );
                return types.AgentError.HttpRequestFailed;
            }

            try sleepWithBackoff(agent, random, &backoff_ms);
            continue;
        };

        const is_retryable = retry.isStatusRetryable(response.status_code);
        if (is_retryable and attempt < max_attempts) {
            response.deinit();
            try sleepWithBackoff(agent, random, &backoff_ms);
            continue;
        }

        break response;
    } else {
        std.log.err(
            "OpenAI API request failed after {d} attempts",
            .{max_attempts + 1},
        );
        return types.AgentError.HttpRequestFailed;
    };
}

fn sleepWithBackoff(
    agent: anytype,
    random: std.Random,
    backoff_ms: *u64,
) !void {
    const sleep_ms = if (agent.config.retry_config.jitter) blk: {
        const jitter_min = backoff_ms.* / 2;
        const jitter_range = backoff_ms.* - jitter_min;
        const jitter = random.intRangeAtMost(u64, 0, jitter_range);
        break :blk jitter_min + jitter;
    } else backoff_ms.*;

    if (sleep_ms > 0) {
        time.sleepMs(sleep_ms);
    }

    const multiplied = @as(f64, @floatFromInt(backoff_ms.*)) *
        agent.config.retry_config.multiplier;
    backoff_ms.* = @min(
        @as(u64, @intFromFloat(multiplied)),
        agent.config.retry_config.max_delay_ms,
    );
}

test {
    std.testing.refAllDecls(@This());
}
