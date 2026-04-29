//! OpenAI-Compatible Endpoint Handlers
//!
//! Handles OpenAI-compatible chat completions (streaming and non-streaming)
//! and the custom ABI streaming endpoint.

const std = @import("std");
const time = @import("../../../../foundation/mod.zig").time;
const shared_utils = @import("../../../../foundation/mod.zig").utils;
const mod = @import("../mod.zig");
const sse = @import("../sse.zig");
const backends = @import("../backends/mod.zig");
const formats = @import("../formats/mod.zig");
const request_types = @import("../request_types.zig");
const handlers = @import("handlers.zig");
const routing = @import("routing.zig");
const server_types = @import("types.zig");

const ConnectionContext = handlers.ConnectionContext;
const parseAbiStreamRequest = request_types.parseAbiStreamRequest;

/// Handle OpenAI-compatible chat completions.
pub fn handleOpenAIChatCompletions(server: anytype, request: *std.http.Server.Request, conn_ctx: *ConnectionContext) !void {
    if (request.head.method != .POST) {
        return routing.respondJson(
            request,
            "{\"error\":{\"message\":\"method not allowed\",\"type\":\"invalid_request_error\"}}",
            .method_not_allowed,
        );
    }

    const body = try routing.readRequestBody(server.allocator, request);
    defer server.allocator.free(body);

    const chat_request = try formats.openai.parseRequest(server.allocator, body);
    defer chat_request.deinit(server.allocator);

    if (chat_request.stream) {
        return streamOpenAIResponse(server, conn_ctx, chat_request);
    } else {
        return nonStreamingOpenAIResponse(server, request, chat_request);
    }
}

/// Stream OpenAI-format response with true SSE streaming.
fn streamOpenAIResponse(
    server: anytype,
    conn_ctx: *ConnectionContext,
    chat_request: formats.openai.ChatCompletionRequest,
) !void {
    const current = server.active_streams.fetchAdd(1, .seq_cst);
    if (current >= server.config.max_concurrent_streams) {
        _ = server.active_streams.fetchSub(1, .seq_cst);
        const error_response =
            "HTTP/1.1 429 Too Many Requests\r\n" ++
            "Content-Type: application/json\r\n" ++
            "Content-Length: 71\r\n" ++
            "\r\n" ++
            "{\"error\":{\"message\":\"too many concurrent streams\",\"type\":\"rate_limit_error\"}}";
        try conn_ctx.write(error_response);
        try conn_ctx.flush();
        return;
    }
    defer _ = server.active_streams.fetchSub(1, .seq_cst);

    const backend = server.getAvailableBackend(server.config.default_backend) catch |err| {
        if (err == error.CircuitBreakerOpen) {
            const error_response =
                "HTTP/1.1 503 Service Unavailable\r\n" ++
                "Content-Type: application/json\r\n" ++
                "Retry-After: 30\r\n" ++
                "Content-Length: 78\r\n" ++
                "\r\n" ++
                "{\"error\":{\"message\":\"backend temporarily unavailable\",\"type\":\"service_unavailable_error\"}}";
            try conn_ctx.write(error_response);
            try conn_ctx.flush();
            return;
        }
        return err;
    };
    const backend_type = backend.backend_type;
    const metrics = server.getMetrics();
    const prompt = try chat_request.formatPrompt(server.allocator);
    defer server.allocator.free(prompt);

    try conn_ctx.writeSseHeaders();

    if (metrics) |m| {
        m.recordStreamStart(backend_type);
    }
    var stream_iter = backend.streamTokens(prompt, chat_request.toGenerationConfig()) catch |err| {
        if (metrics) |m| {
            m.recordStreamFailure(backend_type);
        }
        server.backend_router.recordFailure(backend_type);
        try conn_ctx.write("event: error\ndata: {\"type\":\"stream_error\",\"message\":\"failed to start stream\"}\n\n");
        try conn_ctx.flush();
        return err;
    };
    defer stream_iter.deinit();

    var stream_ok = false;
    errdefer if (!stream_ok) {
        if (metrics) |m| {
            m.recordStreamFailure(backend_type);
        }
        server.backend_router.recordFailure(backend_type);
    };

    var heartbeat_timer = time.Timer.start() catch null;
    const hb_interval_ns: u64 = server.config.heartbeat_interval_ms * 1_000_000;

    const stream_start_ms = shared_utils.unixMs();
    var last_token_ms = stream_start_ms;
    var token_index: u32 = 0;
    while (true) {
        const maybe_token = stream_iter.next() catch |err| return err;
        if (maybe_token == null) break;
        const token = maybe_token.?;
        if (server.config.heartbeat_interval_ms > 0) {
            if (heartbeat_timer) |*timer| {
                if (timer.read() >= hb_interval_ns) {
                    try conn_ctx.write(": heartbeat\n\n");
                    try conn_ctx.flush();
                    timer.reset();
                }
            }
        }

        if (metrics) |m| {
            const now_ms = shared_utils.unixMs();
            const latency_ms: u64 = if (now_ms >= last_token_ms)
                @intCast(now_ms - last_token_ms)
            else
                0;
            m.recordTokenLatency(backend_type, latency_ms);
            last_token_ms = now_ms;
        }

        const chunk_json = try formats.openai.formatStreamChunk(
            server.allocator,
            token.text,
            chat_request.model,
            token_index,
            token.is_end,
        );
        defer server.allocator.free(chunk_json);

        try conn_ctx.write("data: ");
        try conn_ctx.write(chunk_json);
        try conn_ctx.write("\n\n");
        try conn_ctx.flush();

        token_index += 1;
        if (token.is_end) break;
    }

    try conn_ctx.write("data: [DONE]\n\n");
    try conn_ctx.flush();

    stream_ok = true;
    if (metrics) |m| {
        const end_ms = shared_utils.unixMs();
        const duration_ms: u64 = if (end_ms >= stream_start_ms)
            @intCast(end_ms - stream_start_ms)
        else
            0;
        m.recordStreamComplete(backend_type, duration_ms);
    }
    server.backend_router.recordSuccess(backend_type);
}

/// Non-streaming OpenAI response.
fn nonStreamingOpenAIResponse(
    server: anytype,
    request: *std.http.Server.Request,
    chat_request: formats.openai.ChatCompletionRequest,
) !void {
    const backend = try server.backend_router.getBackend(server.config.default_backend);
    const prompt = try chat_request.formatPrompt(server.allocator);
    defer server.allocator.free(prompt);

    const response_text = try backend.generate(prompt, chat_request.toGenerationConfig());
    defer server.allocator.free(response_text);

    const response_body = try formats.openai.formatResponse(
        server.allocator,
        response_text,
        chat_request.model,
    );
    defer server.allocator.free(response_body);

    return routing.respondJson(request, response_body, .ok);
}

/// Handle custom ABI streaming request with true SSE streaming.
pub fn handleStreamRequest(server: anytype, request: *std.http.Server.Request, conn_ctx: *ConnectionContext) !void {
    if (request.head.method != .POST) {
        const error_response =
            "HTTP/1.1 405 Method Not Allowed\r\n" ++
            "Content-Type: application/json\r\n" ++
            "Content-Length: 29\r\n" ++
            "\r\n" ++
            "{\"error\":\"method not allowed\"}";
        try conn_ctx.write(error_response);
        try conn_ctx.flush();
        return;
    }

    const body = try routing.readRequestBody(server.allocator, request);
    defer server.allocator.free(body);

    const stream_request = try parseAbiStreamRequest(server.allocator, body);
    defer stream_request.deinit(server.allocator);

    const current = server.active_streams.fetchAdd(1, .seq_cst);
    if (current >= server.config.max_concurrent_streams) {
        _ = server.active_streams.fetchSub(1, .seq_cst);
        const error_response =
            "HTTP/1.1 429 Too Many Requests\r\n" ++
            "Content-Type: application/json\r\n" ++
            "Content-Length: 35\r\n" ++
            "\r\n" ++
            "{\"error\":\"too many concurrent streams\"}";
        try conn_ctx.write(error_response);
        try conn_ctx.flush();
        return;
    }
    defer _ = server.active_streams.fetchSub(1, .seq_cst);

    try conn_ctx.writeSseHeaders();

    var sse_encoder = sse.SseEncoder.init(server.allocator, .{
        .include_id = true,
        .event_prefix = "abi.",
    });
    defer sse_encoder.deinit();

    const backend_type = stream_request.backend orelse server.config.default_backend;
    const metrics = server.getMetrics();
    const backend = try server.backend_router.getBackend(backend_type);

    if (metrics) |m| {
        m.recordStreamStart(backend_type);
    }

    var stream_iter = backend.streamTokens(stream_request.prompt, stream_request.config) catch |err| {
        if (metrics) |m| {
            m.recordStreamFailure(backend_type);
        }
        server.backend_router.recordFailure(backend_type);
        return err;
    };
    defer stream_iter.deinit();

    var stream_ok = false;
    errdefer if (!stream_ok) {
        if (metrics) |m| {
            m.recordStreamFailure(backend_type);
        }
        server.backend_router.recordFailure(backend_type);
    };

    var heartbeat_timer = time.Timer.start() catch null;
    const hb_interval_ns: u64 = server.config.heartbeat_interval_ms * 1_000_000;

    const stream_start_ms = shared_utils.unixMs();
    var last_token_ms = stream_start_ms;

    const start_event = mod.StreamEvent.startEvent();
    const start_sse = try sse_encoder.encode(start_event);
    defer server.allocator.free(start_sse);
    try conn_ctx.write(start_sse);
    try conn_ctx.flush();

    while (true) {
        const maybe_token = stream_iter.next() catch |err| return err;
        if (maybe_token == null) break;
        const token = maybe_token.?;
        if (server.config.heartbeat_interval_ms > 0) {
            if (heartbeat_timer) |*timer| {
                if (timer.read() >= hb_interval_ns) {
                    const heartbeat_data = try sse_encoder.encodeHeartbeat();
                    defer server.allocator.free(heartbeat_data);
                    try conn_ctx.write(heartbeat_data);
                    try conn_ctx.flush();
                    timer.reset();
                }
            }
        }

        if (metrics) |m| {
            const now_ms = shared_utils.unixMs();
            const latency_ms: u64 = if (now_ms >= last_token_ms)
                @intCast(now_ms - last_token_ms)
            else
                0;
            m.recordTokenLatency(backend_type, latency_ms);
            last_token_ms = now_ms;
        }

        const token_event = mod.StreamEvent{
            .event_type = .token,
            .token = .{
                .id = @intCast(token.id orelse 0),
                .text = token.text,
                .is_end = token.is_end,
            },
        };

        const sse_data = try sse_encoder.encode(token_event);
        defer server.allocator.free(sse_data);

        try conn_ctx.write(sse_data);
        try conn_ctx.flush();

        if (token.is_end) break;
    }

    const end_event = mod.StreamEvent.endEvent();
    const end_sse = try sse_encoder.encode(end_event);
    defer server.allocator.free(end_sse);
    try conn_ctx.write(end_sse);
    try conn_ctx.flush();

    stream_ok = true;
    if (metrics) |m| {
        const end_ms = shared_utils.unixMs();
        const duration_ms: u64 = if (end_ms >= stream_start_ms)
            @intCast(end_ms - stream_start_ms)
        else
            0;
        m.recordStreamComplete(backend_type, duration_ms);
    }
    server.backend_router.recordSuccess(backend_type);
}

test {
    std.testing.refAllDecls(@This());
}
