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

const std = @import("std");
const personas = @import("../../ai/personas/mod.zig");
const types = @import("../../ai/personas/types.zig");
const metrics = @import("../../ai/personas/metrics.zig");

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

/// Chat handler that wraps the persona orchestrator.
pub const ChatHandler = struct {
    allocator: std.mem.Allocator,
    orchestrator: ?*personas.PersonaOrchestrator = null,

    const Self = @This();

    /// Initialize the chat handler.
    pub fn init(allocator: std.mem.Allocator) Self {
        return .{
            .allocator = allocator,
            .orchestrator = null,
        };
    }

    /// Set the orchestrator reference.
    pub fn setOrchestrator(self: *Self, orchestrator: *personas.PersonaOrchestrator) void {
        self.orchestrator = orchestrator;
    }

    /// Handle a chat request (auto-routing).
    pub fn handleChat(self: *Self, request_json: []const u8) ![]const u8 {
        return self.handleChatWithPersona(request_json, null);
    }

    /// Handle a chat request with optional forced persona.
    pub fn handleChatWithPersona(self: *Self, request_json: []const u8, forced_persona: ?types.PersonaType) ![]const u8 {
        // Parse request
        const request = self.parseRequest(request_json) catch |err| {
            return self.formatError("PARSE_ERROR", @errorName(err), null);
        };

        // Get orchestrator
        const orch = self.orchestrator orelse {
            return self.formatError("SERVICE_UNAVAILABLE", "Persona service not initialized", null);
        };

        // Build persona request
        var persona_request = types.PersonaRequest{
            .content = request.content,
            .user_id = request.user_id,
            .session_id = request.session_id,
            .context = if (request.context) |ctx| .{ .system_instruction = ctx } else null,
        };

        // Force persona if specified
        if (forced_persona) |p| {
            persona_request.preferred_persona = p;
        } else if (request.persona) |persona_name| {
            persona_request.preferred_persona = parsePersonaType(persona_name);
        }

        // Process request
        const response = orch.process(persona_request) catch |err| {
            return self.formatError("PROCESSING_ERROR", @errorName(err), null);
        };

        // Format response
        return self.formatResponse(response);
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
        var persona_list = std.ArrayList(PersonaInfo).init(self.allocator);
        defer persona_list.deinit();

        // Add known personas
        const persona_types = [_]types.PersonaType{ .abi, .abbey, .aviva };
        const descriptions = [_][]const u8{
            "Router and content moderator",
            "Empathetic polymath - supportive, thorough responses",
            "Direct expert - concise, factual responses",
        };

        for (persona_types, 0..) |pt, i| {
            try persona_list.append(.{
                .name = @tagName(pt),
                .type_name = @tagName(pt),
                .description = descriptions[i],
                .available = self.isPersonaAvailable(pt),
            });
        }

        return std.json.stringifyAlloc(self.allocator, .{
            .personas = persona_list.items,
        }, .{});
    }

    /// Get persona metrics.
    pub fn getMetrics(self: *Self) ![]const u8 {
        var persona_metrics_list = std.ArrayList(PersonaMetricsJson).init(self.allocator);
        defer persona_metrics_list.deinit();

        var total_requests: u64 = 0;
        var total_successes: u64 = 0;

        // Get metrics for each persona if orchestrator has metrics
        if (self.orchestrator) |orch| {
            if (orch.metrics) |m| {
                const persona_types = [_]types.PersonaType{ .abi, .abbey, .aviva };
                for (persona_types) |pt| {
                    if (m.getStats(pt)) |stats| {
                        total_requests += stats.total_requests;
                        total_successes += @intFromFloat(@as(f32, @floatFromInt(stats.total_requests)) * stats.success_rate);

                        var latency_p50: ?f64 = null;
                        var latency_p99: ?f64 = null;
                        if (stats.latency) |lat| {
                            latency_p50 = lat.p50;
                            latency_p99 = lat.p99;
                        }

                        try persona_metrics_list.append(.{
                            .name = @tagName(pt),
                            .total_requests = stats.total_requests,
                            .success_rate = stats.success_rate,
                            .error_count = stats.error_count,
                            .latency_p50_ms = latency_p50,
                            .latency_p99_ms = latency_p99,
                        });
                    }
                }
            }
        }

        const overall_success_rate: f32 = if (total_requests > 0)
            @as(f32, @floatFromInt(total_successes)) / @as(f32, @floatFromInt(total_requests))
        else
            1.0;

        return std.json.stringifyAlloc(self.allocator, MetricsResponse{
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
        return parsed.value;
    }

    /// Format a persona response as JSON.
    fn formatResponse(self: *Self, response: types.PersonaResponse) ![]const u8 {
        var code_blocks: ?[]CodeBlockJson = null;
        if (response.code_blocks) |blocks| {
            var cb_list = try self.allocator.alloc(CodeBlockJson, blocks.len);
            for (blocks, 0..) |block, i| {
                cb_list[i] = .{
                    .language = block.language,
                    .code = block.code,
                };
            }
            code_blocks = cb_list;
        }

        var references: ?[]SourceJson = null;
        if (response.references) |refs| {
            var ref_list = try self.allocator.alloc(SourceJson, refs.len);
            for (refs, 0..) |ref, i| {
                ref_list[i] = .{
                    .title = ref.title,
                    .url = ref.url,
                    .confidence = ref.confidence,
                };
            }
            references = ref_list;
        }

        const chat_response = ChatResponse{
            .content = response.content,
            .persona = @tagName(response.persona),
            .confidence = response.confidence,
            .latency_ms = response.generation_time_ms,
            .code_blocks = code_blocks,
            .references = references,
        };

        return std.json.stringifyAlloc(self.allocator, chat_response, .{});
    }

    /// Format an error response as JSON.
    fn formatError(self: *Self, code: []const u8, message: []const u8, request_id: ?[]const u8) ![]const u8 {
        const error_response = ErrorResponse{
            .@"error" = .{
                .code = code,
                .message = message,
                .request_id = request_id,
            },
        };
        return std.json.stringifyAlloc(self.allocator, error_response, .{});
    }

    /// Check if a persona is available.
    fn isPersonaAvailable(self: *const Self, persona_type: types.PersonaType) bool {
        if (self.orchestrator) |orch| {
            return orch.registry.get(persona_type) != null;
        }
        return false;
    }
};

/// Parse persona type from string.
pub fn parsePersonaType(name: []const u8) ?types.PersonaType {
    if (std.mem.eql(u8, name, "abbey")) return .abbey;
    if (std.mem.eql(u8, name, "aviva")) return .aviva;
    if (std.mem.eql(u8, name, "abi")) return .abi;
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
    pub const internal_server_error = 500;
    pub const service_unavailable = 503;
};

// Tests

test "parse persona type" {
    try std.testing.expectEqual(types.PersonaType.abbey, parsePersonaType("abbey").?);
    try std.testing.expectEqual(types.PersonaType.aviva, parsePersonaType("aviva").?);
    try std.testing.expect(parsePersonaType("unknown") == null);
}

test "chat handler initialization" {
    var handler = ChatHandler.init(std.testing.allocator);
    try std.testing.expect(handler.orchestrator == null);
}

test "error response format" {
    var handler = ChatHandler.init(std.testing.allocator);
    const error_json = try handler.formatError("TEST_ERROR", "Test message", null);
    defer std.testing.allocator.free(error_json);

    try std.testing.expect(std.mem.indexOf(u8, error_json, "TEST_ERROR") != null);
    try std.testing.expect(std.mem.indexOf(u8, error_json, "Test message") != null);
}
