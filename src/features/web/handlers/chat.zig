//! Persona Chat Handler
//!
//! Handles HTTP requests for the Multi-Persona AI Assistant.
//! Provides request parsing, response formatting, and error handling.
//!
//! API Endpoints:
//! - POST /api/v1/chat - Auto-routing to best persona
//! - POST /api/v1/chat/abbey - Force Abbey persona
//! - POST /api/v1/chat/aviva - Force Aviva persona
//! - GET /api/v1/personas - List available personas
//! - GET /api/v1/personas/metrics - Get persona metrics
//!
//! Rate Limiting:
//! - 100 requests per minute per user (token bucket algorithm)
//! - Auto-ban after 10 consecutive violations (1 hour ban)

const std = @import("std");
const rate_limit = @import("../../../services/shared/security/rate_limit.zig");

/// Helper to serialize a value to JSON using Zig 0.16 API
fn jsonStringifyAlloc(allocator: std.mem.Allocator, value: anytype, options: std.json.Stringify.Options) ![]u8 {
    var out: std.Io.Writer.Allocating = .init(allocator);
    errdefer out.deinit();
    try std.json.Stringify.value(value, options, &out.writer);
    return out.toOwnedSlice();
}
const personas = @import("../../ai/personas/mod.zig");
const types = @import("../../ai/personas/types.zig");

/// Chat request from client.
pub const ChatRequest = struct {
    /// The user's message content.
    content: []const u8,
    /// User identifier for conversation tracking.
    user_id: ?[]const u8 = null,
    /// Session/conversation ID for context.
    session_id: ?[]const u8 = null,
    /// Preferred persona (null for auto-routing).
    persona: ?[]const u8 = null,
    /// Additional context or system instructions.
    context: ?[]const u8 = null,
    /// Maximum response tokens.
    max_tokens: ?u32 = null,
    /// Temperature for generation.
    temperature: ?f32 = null,

    pub fn deinit(self: *ChatRequest, allocator: std.mem.Allocator) void {
        allocator.free(self.content);
        if (self.user_id) |value| allocator.free(value);
        if (self.session_id) |value| allocator.free(value);
        if (self.persona) |value| allocator.free(value);
        if (self.context) |value| allocator.free(value);
    }

    pub fn dupe(allocator: std.mem.Allocator, other: ChatRequest) !ChatRequest {
        return .{
            .content = try allocator.dupe(u8, other.content),
            .user_id = try dupeOptional(allocator, other.user_id),
            .session_id = try dupeOptional(allocator, other.session_id),
            .persona = try dupeOptional(allocator, other.persona),
            .context = try dupeOptional(allocator, other.context),
            .max_tokens = other.max_tokens,
            .temperature = other.temperature,
        };
    }
};

/// Chat response to client.
pub const ChatResponse = struct {
    /// The generated response content.
    content: []const u8,
    /// Which persona generated the response.
    persona: []const u8,
    /// Confidence score (0.0 - 1.0).
    confidence: f32,
    /// Generation time in milliseconds.
    latency_ms: u64,
    /// Code blocks extracted from response.
    code_blocks: ?[]const CodeBlockJson = null,
    /// Source references if any.
    references: ?[]const SourceJson = null,
    /// Request ID for tracking.
    request_id: ?[]const u8 = null,
};

/// Chat handler result with HTTP status.
pub const ChatResult = struct {
    status: u16,
    body: []const u8,
};

/// Code block in JSON format.
pub const CodeBlockJson = struct {
    language: []const u8,
    code: []const u8,
};

/// Source reference in JSON format.
pub const SourceJson = struct {
    title: []const u8,
    url: ?[]const u8 = null,
    confidence: f32,
};

/// Error response format.
pub const ErrorResponse = struct {
    @"error": ErrorDetail,
};

const ErrorDetail = struct {
    code: []const u8,
    message: []const u8,
    request_id: ?[]const u8 = null,
};

/// Persona info for listing.
pub const PersonaInfo = struct {
    name: []const u8,
    type_name: []const u8,
    description: []const u8,
    available: bool,
};

/// Metrics response.
pub const MetricsResponse = struct {
    personas: []const PersonaMetricsJson,
    total_requests: u64,
    overall_success_rate: f32,
};

const PersonaMetricsJson = struct {
    name: []const u8,
    total_requests: u64,
    success_rate: f32,
    error_count: u64,
    latency_p50_ms: ?f64 = null,
    latency_p99_ms: ?f64 = null,
};

/// Rate limiting errors.
pub const RateLimitError = error{
    /// Request rate limit exceeded.
    RateLimited,
};

fn dupeOptional(allocator: std.mem.Allocator, value: ?[]const u8) !?[]const u8 {
    if (value) |v| {
        return try allocator.dupe(u8, v);
    }
    return null;
}

/// Chat handler that wraps the persona system.
pub const ChatHandler = struct {
    allocator: std.mem.Allocator,
    orchestrator: ?*personas.MultiPersonaSystem = null,
    rate_limiter: rate_limit.RateLimiter,

    const Self = @This();

    /// Rate limit configuration for chat endpoint.
    /// - 100 requests per minute per user
    /// - Token bucket algorithm (allows bursts)
    /// - Auto-ban after 10 consecutive violations for 1 hour
    pub const rate_limit_config: rate_limit.RateLimitConfig = .{
        .requests = 100,
        .window_seconds = 60,
        .burst = 20,
        .algorithm = .token_bucket,
        .scope = .user,
        .ban_duration = 3600, // 1 hour ban
        .ban_threshold = 10, // 10 consecutive violations triggers ban
    };

    /// Initialize the chat handler with rate limiting.
    pub fn init(allocator: std.mem.Allocator) Self {
        return .{
            .allocator = allocator,
            .orchestrator = null,
            .rate_limiter = rate_limit.RateLimiter.init(allocator, rate_limit_config),
        };
    }

    /// Deinitialize the chat handler.
    pub fn deinit(self: *Self) void {
        self.rate_limiter.deinit();
    }

    /// Set the orchestrator reference.
    pub fn setOrchestrator(self: *Self, orchestrator: *personas.MultiPersonaSystem) void {
        self.orchestrator = orchestrator;
    }

    /// Handle a chat request (auto-routing).
    pub fn handleChat(self: *Self, request_json: []const u8) ![]const u8 {
        return self.handleChatWithPersona(request_json, null);
    }

    /// Handle a chat request with optional forced persona.
    /// Returns `error.RateLimited` if the user has exceeded the rate limit.
    pub fn handleChatWithPersona(self: *Self, request_json: []const u8, forced_persona: ?types.PersonaType) ![]const u8 {
        const result = try self.handleChatWithPersonaResult(request_json, forced_persona);
        if (result.status == HttpStatus.too_many_requests) {
            if (std.mem.indexOf(u8, result.body, "temporarily banned") != null) {
                return result.body;
            }
            self.allocator.free(result.body);
            return RateLimitError.RateLimited;
        }
        return result.body;
    }

    /// Handle a chat request and return status + body for HTTP routing.
    pub fn handleChatWithPersonaResult(
        self: *Self,
        request_json: []const u8,
        forced_persona: ?types.PersonaType,
    ) !ChatResult {
        // Parse request
        var request = self.parseRequest(request_json) catch |err| {
            const err_name = try std.fmt.allocPrint(self.allocator, "{t}", .{err});
            defer self.allocator.free(err_name);
            return .{
                .status = HttpStatus.bad_request,
                .body = try self.formatError("PARSE_ERROR", err_name, null),
            };
        };
        defer request.deinit(self.allocator);

        // Check rate limit before processing
        // Use user_id if available, fall back to session_id or "anonymous"
        const rate_limit_key = request.user_id orelse request.session_id orelse "anonymous";
        const rate_status = self.rate_limiter.check(rate_limit_key);

        if (!rate_status.allowed) {
            const message = if (rate_status.banned)
                "Too many requests. You have been temporarily banned due to excessive violations."
            else
                "Too many requests. Please retry later.";
            return .{
                .status = HttpStatus.too_many_requests,
                .body = try self.formatError("RATE_LIMITED", message, null),
            };
        }

        // Get orchestrator
        const orch = self.orchestrator orelse {
            return .{
                .status = HttpStatus.service_unavailable,
                .body = try self.formatError("SERVICE_UNAVAILABLE", "Persona service not initialized", null),
            };
        };

        // Build persona request
        var persona_request = types.PersonaRequest{
            .content = request.content,
            .user_id = request.user_id,
            .session_id = request.session_id,
            .system_instruction = request.context,
            .max_tokens = request.max_tokens,
            .temperature = request.temperature,
        };

        // Force persona if specified
        if (forced_persona) |p| {
            if (!self.isPersonaAvailable(p)) {
                return .{
                    .status = HttpStatus.not_found,
                    .body = try self.formatError("PERSONA_UNAVAILABLE", "Requested persona is not registered", null),
                };
            }
            persona_request.preferred_persona = p;
        } else if (request.persona) |persona_name| {
            const parsed = parsePersonaType(persona_name) orelse {
                return .{
                    .status = HttpStatus.bad_request,
                    .body = try self.formatError("INVALID_PERSONA", "Unknown persona", null),
                };
            };
            if (!self.isPersonaAvailable(parsed)) {
                return .{
                    .status = HttpStatus.not_found,
                    .body = try self.formatError("PERSONA_UNAVAILABLE", "Requested persona is not registered", null),
                };
            }
            persona_request.preferred_persona = parsed;
        }

        // Process request
        const response = orch.process(persona_request) catch |err| {
            const err_name = try std.fmt.allocPrint(self.allocator, "{t}", .{err});
            defer self.allocator.free(err_name);
            return .{
                .status = HttpStatus.internal_server_error,
                .body = try self.formatError("PROCESSING_ERROR", err_name, null),
            };
        };
        defer @constCast(&response).deinit(self.allocator);

        // Format response
        return .{
            .status = HttpStatus.ok,
            .body = try self.formatResponse(response),
        };
    }

    /// Handle Abbey-specific request.
    pub fn handleAbbeyChat(self: *Self, request_json: []const u8) ![]const u8 {
        return self.handleChatWithPersona(request_json, .abbey);
    }

    /// Handle Aviva-specific request.
    pub fn handleAvivaChat(self: *Self, request_json: []const u8) ![]const u8 {
        return self.handleChatWithPersona(request_json, .aviva);
    }

    /// List available personas.
    pub fn listPersonas(self: *Self) ![]const u8 {
        var persona_list = std.ArrayListUnmanaged(PersonaInfo).empty;
        defer persona_list.deinit(self.allocator);

        const persona_types = types.allPersonaTypes();
        for (persona_types) |pt| {
            try persona_list.append(self.allocator, .{
                .name = @tagName(pt),
                .type_name = @tagName(pt),
                .description = describePersonaType(pt),
                .available = self.isPersonaAvailable(pt),
            });
        }

        return jsonStringifyAlloc(self.allocator, .{
            .personas = persona_list.items,
        }, .{});
    }

    /// Get persona metrics.
    pub fn getMetrics(self: *Self) ![]const u8 {
        var persona_metrics_list = std.ArrayListUnmanaged(PersonaMetricsJson).empty;
        defer persona_metrics_list.deinit(self.allocator);

        const persona_types = types.allPersonaTypes();
        var total_requests: u64 = 0;
        var total_successes: f64 = 0.0;
        const metrics_manager = if (self.orchestrator) |orch| orch.getMetrics() else null;

        for (persona_types) |pt| {
            var total: u64 = 0;
            var success_rate: f32 = 1.0;
            var error_count: u64 = 0;
            var latency_p50: ?f64 = null;
            var latency_p99: ?f64 = null;

            if (metrics_manager) |m| {
                if (m.getStats(pt)) |stats| {
                    total = stats.total_requests;
                    success_rate = stats.success_rate;
                    error_count = stats.error_count;
                    if (stats.latency) |lat| {
                        latency_p50 = lat.p50;
                        latency_p99 = lat.p99;
                    }
                }
            }

            total_requests += total;
            total_successes += @as(f64, @floatFromInt(total)) * @as(f64, success_rate);

            try persona_metrics_list.append(self.allocator, .{
                .name = @tagName(pt),
                .total_requests = total,
                .success_rate = success_rate,
                .error_count = error_count,
                .latency_p50_ms = latency_p50,
                .latency_p99_ms = latency_p99,
            });
        }

        const overall_success_rate: f32 = if (total_requests > 0)
            @floatCast(total_successes / @as(f64, @floatFromInt(total_requests)))
        else
            1.0;

        return jsonStringifyAlloc(self.allocator, MetricsResponse{
            .personas = persona_metrics_list.items,
            .total_requests = total_requests,
            .overall_success_rate = overall_success_rate,
        }, .{});
    }

    /// Parse a chat request from JSON.
    fn parseRequest(self: *Self, json: []const u8) !ChatRequest {
        const parsed = try std.json.parseFromSlice(ChatRequest, self.allocator, json, .{
            .ignore_unknown_fields = true,
        });
        defer parsed.deinit();
        return ChatRequest.dupe(self.allocator, parsed.value);
    }

    /// Format a persona response as JSON.
    fn formatResponse(self: *Self, response: types.PersonaResponse) ![]const u8 {
        var code_blocks: ?[]CodeBlockJson = null;
        if (response.code_blocks) |blocks| {
            const cb_list = try self.allocator.alloc(CodeBlockJson, blocks.len);
            for (blocks, 0..) |block, i| {
                cb_list[i] = .{
                    .language = block.language,
                    .code = block.code,
                };
            }
            code_blocks = cb_list;
        }
        defer if (code_blocks) |cb| self.allocator.free(cb);

        var references: ?[]SourceJson = null;
        if (response.references) |refs| {
            const ref_list = try self.allocator.alloc(SourceJson, refs.len);
            for (refs, 0..) |ref, i| {
                ref_list[i] = .{
                    .title = ref.title,
                    .url = ref.url,
                    .confidence = ref.confidence,
                };
            }
            references = ref_list;
        }
        defer if (references) |r| self.allocator.free(r);

        const chat_response = ChatResponse{
            .content = response.content,
            .persona = @tagName(response.persona),
            .confidence = response.confidence,
            .latency_ms = response.generation_time_ms,
            .code_blocks = code_blocks,
            .references = references,
        };

        return jsonStringifyAlloc(self.allocator, chat_response, .{});
    }

    /// Format an error response as JSON.
    pub fn formatError(self: *Self, code: []const u8, message: []const u8, request_id: ?[]const u8) ![]const u8 {
        const error_response = ErrorResponse{
            .@"error" = .{
                .code = code,
                .message = message,
                .request_id = request_id,
            },
        };
        return jsonStringifyAlloc(self.allocator, error_response, .{});
    }

    /// Check if a persona is available.
    fn isPersonaAvailable(self: *const Self, persona_type: types.PersonaType) bool {
        if (self.orchestrator) |orch| {
            return orch.ctx.getPersona(persona_type) != null;
        }
        return false;
    }

    /// Get rate limit status for a user (useful for returning rate limit headers).
    /// Returns the current rate limit status without consuming a request.
    pub fn getRateLimitStatus(self: *Self, user_id: ?[]const u8) rate_limit.RateLimitStatus {
        const key = user_id orelse "anonymous";
        // Note: check() does consume a request, so this should be called after handling
        // For header purposes, we can use the limiter's bucket info
        if (self.rate_limiter.getBucketInfo(key)) |info| {
            const current: u32 = @intCast(@min(info.total_requests, rate_limit_config.requests));
            const remaining: u32 = if (current >= rate_limit_config.requests) 0 else rate_limit_config.requests - current;
            return .{
                .allowed = !info.is_banned,
                .remaining = remaining,
                .limit = rate_limit_config.requests,
                .reset_at = 0,
                .current = current,
                .banned = info.is_banned,
                .ban_expires_at = info.ban_expires_at,
            };
        }
        // No bucket yet means no requests made
        return .{
            .allowed = true,
            .remaining = rate_limit_config.requests,
            .limit = rate_limit_config.requests,
            .reset_at = 0,
            .current = 0,
        };
    }

    /// Get rate limiter statistics.
    pub fn getRateLimiterStats(self: *Self) rate_limit.RateLimiter.RateLimiterStats {
        return self.rate_limiter.getStats();
    }

    /// Reset rate limit for a specific user (admin operation).
    pub fn resetRateLimit(self: *Self, user_id: []const u8) void {
        self.rate_limiter.reset(user_id);
    }

    /// Unban a specific user (admin operation).
    pub fn unbanUser(self: *Self, user_id: []const u8) bool {
        return self.rate_limiter.unban(user_id);
    }
};

fn describePersonaType(persona_type: types.PersonaType) []const u8 {
    return switch (persona_type) {
        .assistant => "General-purpose helpful assistant",
        .coder => "Code-focused programming specialist",
        .writer => "Creative writing and content generation specialist",
        .analyst => "Data analysis and research specialist",
        .companion => "Friendly conversational companion",
        .docs => "Technical documentation specialist",
        .reviewer => "Code and logic reviewer",
        .minimal => "Minimal, direct response model",
        .abbey => "Empathetic polymath - supportive, thorough responses",
        .aviva => "Direct expert - concise, factual responses",
        .abi => "Router and content moderator",
        .ralph => "Iterative agent loop specialist",
    };
}

/// Parse persona type from string.
pub fn parsePersonaType(name: []const u8) ?types.PersonaType {
    for (types.allPersonaTypes()) |persona_type| {
        if (std.mem.eql(u8, name, @tagName(persona_type))) {
            return persona_type;
        }
    }
    return null;
}

/// HTTP status codes.
pub const HttpStatus = struct {
    pub const ok = 200;
    pub const created = 201;
    pub const bad_request = 400;
    pub const unauthorized = 401;
    pub const not_found = 404;
    pub const method_not_allowed = 405;
    pub const too_many_requests = 429;
    pub const internal_server_error = 500;
    pub const service_unavailable = 503;
};

// Tests

test "parse persona type" {
    try std.testing.expectEqual(types.PersonaType.abbey, parsePersonaType("abbey").?);
    try std.testing.expectEqual(types.PersonaType.aviva, parsePersonaType("aviva").?);
    try std.testing.expectEqual(types.PersonaType.abi, parsePersonaType("abi").?);
    try std.testing.expect(parsePersonaType("unknown") == null);
}

test "chat handler initialization" {
    var handler = ChatHandler.init(std.testing.allocator);
    defer handler.deinit();
    try std.testing.expect(handler.orchestrator == null);
}

test "error response format" {
    var handler = ChatHandler.init(std.testing.allocator);
    defer handler.deinit();
    const error_json = try handler.formatError("TEST_ERROR", "Test message", null);
    defer std.testing.allocator.free(error_json);

    try std.testing.expect(std.mem.indexOf(u8, error_json, "TEST_ERROR") != null);
    try std.testing.expect(std.mem.indexOf(u8, error_json, "Test message") != null);
}

test "rate limiter configuration" {
    // Verify rate limit config matches requirements
    try std.testing.expectEqual(@as(u32, 100), ChatHandler.rate_limit_config.requests);
    try std.testing.expectEqual(@as(u32, 60), ChatHandler.rate_limit_config.window_seconds);
    try std.testing.expectEqual(rate_limit.Algorithm.token_bucket, ChatHandler.rate_limit_config.algorithm);
    try std.testing.expectEqual(rate_limit.Scope.user, ChatHandler.rate_limit_config.scope);
    try std.testing.expectEqual(@as(u32, 3600), ChatHandler.rate_limit_config.ban_duration);
    try std.testing.expectEqual(@as(u32, 10), ChatHandler.rate_limit_config.ban_threshold);
}

test "rate limiting allows requests under limit" {
    var handler = ChatHandler.init(std.testing.allocator);
    defer handler.deinit();

    // Initial status should show full capacity
    const status = handler.getRateLimitStatus("test_user");
    try std.testing.expect(status.allowed);
    try std.testing.expectEqual(@as(u32, 100), status.limit);
}

test "rate limiter stats tracking" {
    var handler = ChatHandler.init(std.testing.allocator);
    defer handler.deinit();

    // Initial stats should be zero
    const stats = handler.getRateLimiterStats();
    try std.testing.expectEqual(@as(u64, 0), stats.total_requests);
    try std.testing.expectEqual(@as(u64, 0), stats.blocked_requests);
}

test "http status includes rate limit code" {
    try std.testing.expectEqual(@as(u16, 429), HttpStatus.too_many_requests);
}
