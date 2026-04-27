//! Profile Chat Handler
//!
//! Handles HTTP requests for the Multi-Profile AI Assistant.
//! Provides request parsing, response formatting, and error handling.
//!
//! API Endpoints:
//! - POST /api/v1/chat - Auto-routing to best profile
//! - POST /api/v1/chat/abbey - Force Abbey profile
//! - POST /api/v1/chat/aviva - Force Aviva profile
//! - GET /api/v1/profiles - List available profiles
//! - GET /api/v1/profiles/metrics - Get profile metrics
//!
//! Rate Limiting:
//! - 100 requests per minute per user (token bucket algorithm)
//! - Auto-ban after 10 consecutive violations (1 hour ban)

const std = @import("std");
const rate_limit = @import("../../foundation/mod.zig").security.rate_limit;

/// Helper to serialize a value to JSON using Zig 0.16 API
fn jsonStringifyAlloc(allocator: std.mem.Allocator, value: anytype, options: std.json.Stringify.Options) ![]u8 {
    var out: std.Io.Writer.Allocating = .init(allocator);
    errdefer out.deinit();
    try std.json.Stringify.value(value, options, &out.writer);
    return out.toOwnedSlice();
}
const build_options = @import("build_options");
const ai_mod = if (build_options.feat_ai) @import("../ai/mod.zig") else @import("../ai/stub.zig");
const profiles = ai_mod.profiles;
const types = ai_mod.types;

/// Chat request from client.
pub const ChatRequest = struct {
    /// The user's message content.
    content: []const u8,
    /// User identifier for conversation tracking.
    user_id: ?[]const u8 = null,
    /// Session/conversation ID for context.
    session_id: ?[]const u8 = null,
    /// Preferred profile (null for auto-routing).
    profile: ?[]const u8 = null,
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
        if (self.profile) |value| allocator.free(value);
        if (self.context) |value| allocator.free(value);
    }

    pub fn dupe(allocator: std.mem.Allocator, other: ChatRequest) !ChatRequest {
        return .{
            .content = try allocator.dupe(u8, other.content),
            .user_id = try dupeOptional(allocator, other.user_id),
            .session_id = try dupeOptional(allocator, other.session_id),
            .profile = try dupeOptional(allocator, other.profile),
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
    /// Which profile generated the response.
    profile: []const u8,
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

/// Profile info for listing.
pub const ProfileInfo = struct {
    name: []const u8,
    type_name: []const u8,
    description: []const u8,
    available: bool,
};

/// Metrics response.
pub const MetricsResponse = struct {
    profiles: []const ProfileMetricsJson,
    total_requests: u64,
    overall_success_rate: f32,
};

const ProfileMetricsJson = struct {
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

/// Chat handler that wraps the profile system.
pub const ChatHandler = struct {
    allocator: std.mem.Allocator,
    orchestrator: ?*profiles.MultiProfileSystem = null,
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
    pub fn setOrchestrator(self: *Self, orchestrator: *profiles.MultiProfileSystem) void {
        self.orchestrator = orchestrator;
    }

    /// Handle a chat request (auto-routing).
    pub fn handleChat(self: *Self, request_json: []const u8) ![]const u8 {
        return self.handleChatWithProfile(request_json, null);
    }

    /// Handle a chat request with optional forced profile.
    /// Returns `error.RateLimited` if the user has exceeded the rate limit.
    pub fn handleChatWithProfile(self: *Self, request_json: []const u8, forced_profile: ?types.ProfileType) ![]const u8 {
        const result = try self.handleChatWithProfileResult(request_json, forced_profile);
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
    pub fn handleChatWithProfileResult(
        self: *Self,
        request_json: []const u8,
        forced_profile: ?types.ProfileType,
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
                .body = try self.formatError("SERVICE_UNAVAILABLE", "Profile service not initialized", null),
            };
        };

        // Build profile request
        var profile_request = types.ProfileRequest{
            .content = request.content,
            .user_id = request.user_id,
            .session_id = request.session_id,
            .system_instruction = request.context,
            .max_tokens = request.max_tokens,
            .temperature = request.temperature,
        };

        // Force profile if specified
        if (forced_profile) |p| {
            if (!self.isProfileAvailable(p)) {
                return .{
                    .status = HttpStatus.not_found,
                    .body = try self.formatError("PROFILE_UNAVAILABLE", "Requested profile is not registered", null),
                };
            }
            profile_request.preferred_profile = p;
        } else if (request.profile) |profile_name| {
            const parsed = parseProfileType(profile_name) orelse {
                return .{
                    .status = HttpStatus.bad_request,
                    .body = try self.formatError("INVALID_PROFILE", "Unknown profile", null),
                };
            };
            if (!self.isProfileAvailable(parsed)) {
                return .{
                    .status = HttpStatus.not_found,
                    .body = try self.formatError("PROFILE_UNAVAILABLE", "Requested profile is not registered", null),
                };
            }
            profile_request.preferred_profile = parsed;
        }

        // Process request
        const response = orch.process(profile_request) catch |err| {
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
        return self.handleChatWithProfile(request_json, .abbey);
    }

    /// Handle Aviva-specific request.
    pub fn handleAvivaChat(self: *Self, request_json: []const u8) ![]const u8 {
        return self.handleChatWithProfile(request_json, .aviva);
    }

    /// List available profiles.
    pub fn listProfiles(self: *Self) ![]const u8 {
        var profile_list = std.ArrayListUnmanaged(ProfileInfo).empty;
        defer profile_list.deinit(self.allocator);

        const profile_types = types.allProfileTypes();
        for (profile_types) |pt| {
            try profile_list.append(self.allocator, .{
                .name = @tagName(pt),
                .type_name = @tagName(pt),
                .description = describeProfileType(pt),
                .available = self.isProfileAvailable(pt),
            });
        }

        return jsonStringifyAlloc(self.allocator, .{
            .profiles = profile_list.items,
        }, .{});
    }

    /// Get profile metrics.
    pub fn getMetrics(self: *Self) ![]const u8 {
        var profile_metrics_list = std.ArrayListUnmanaged(ProfileMetricsJson).empty;
        defer profile_metrics_list.deinit(self.allocator);

        const profile_types = types.allProfileTypes();

        for (profile_types) |pt| {
            try profile_metrics_list.append(self.allocator, .{
                .name = @tagName(pt),
                .total_requests = 0,
                .success_rate = 1.0,
                .error_count = 0,
                .latency_p50_ms = null,
                .latency_p99_ms = null,
            });
        }

        return jsonStringifyAlloc(self.allocator, MetricsResponse{
            .profiles = profile_metrics_list.items,
            .total_requests = 0,
            .overall_success_rate = 1.0,
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

    /// Format a profile response as JSON.
    fn formatResponse(self: *Self, response: types.ProfileResponse) ![]const u8 {
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
            .profile = @tagName(response.profile),
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

    /// Check if a profile is available.
    fn isProfileAvailable(self: *const Self, profile_type: types.ProfileType) bool {
        if (self.orchestrator) |orch| {
            return orch.ctx.getProfile(profile_type) != null;
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

fn describeProfileType(profile_type: types.ProfileType) []const u8 {
    return switch (profile_type) {
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
        .ava => "Locally-trained versatile assistant based on gpt-oss",
    };
}

/// Parse profile type from string.
pub fn parseProfileType(name: []const u8) ?types.ProfileType {
    for (types.allProfileTypes()) |profile_type| {
        if (std.mem.eql(u8, name, @tagName(profile_type))) {
            return profile_type;
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

test "parse profile type" {
    try std.testing.expectEqual(types.ProfileType.abbey, parseProfileType("abbey").?);
    try std.testing.expectEqual(types.ProfileType.aviva, parseProfileType("aviva").?);
    try std.testing.expectEqual(types.ProfileType.abi, parseProfileType("abi").?);
    try std.testing.expect(parseProfileType("unknown") == null);
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

test {
    std.testing.refAllDecls(@This());
}
