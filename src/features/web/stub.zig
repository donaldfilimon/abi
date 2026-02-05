//! Web Stub Module
//!
//! This module provides API-compatible no-op implementations for all public
//! web/HTTP functions when the web feature is disabled at compile time.
//! All functions return `error.WebDisabled` or empty/default values as
//! appropriate. Type definitions are provided to maintain API compatibility.
//!
//! The web module encompasses:
//! - HTTP client for external API requests
//! - Chat handlers for AI persona interactions
//! - Route definitions and request routing
//! - JSON parsing and response formatting
//! - Weather and utility clients
//!
//! To enable the real implementation, build with `-Denable-web=true`.

const std = @import("std");
const config_module = @import("../../core/config/mod.zig");
const persona_types = @import("../ai/personas/types.zig");

// ============================================================================
// Local Stubs Imports
// ============================================================================

pub const types = @import("stubs/types.zig");
pub const client = @import("stubs/client.zig");

// ============================================================================
// Handlers and Routes Stubs (match mod.zig structure)
// ============================================================================

/// Stub handlers namespace for API parity.
pub const handlers = struct {
    pub const chat = struct {
        pub const ChatHandler = StubChatHandler;
        pub const ChatRequest = StubChatRequest;
        pub const ChatResponse = StubChatResponse;
        pub const ChatResult = StubChatResult;
    };
};

/// Stub routes namespace for API parity.
pub const routes = struct {
    pub const personas = struct {
        pub const Router = StubRouter;
        pub const Route = StubRoute;
        pub const RouteContext = StubRouteContext;
    };
};

// Re-export handler types at top level (like mod.zig)
pub const ChatHandler = StubChatHandler;
pub const ChatRequest = StubChatRequest;
pub const ChatResponse = StubChatResponse;
pub const ChatResult = StubChatResult;

// Re-export route types at top level (like mod.zig)
pub const PersonaRouter = StubRouter;
pub const Route = StubRoute;
pub const RouteContext = StubRouteContext;

// ============================================================================
// Stub Type Definitions
// ============================================================================

/// Stub chat request - matches mod.zig ChatRequest.
pub const StubChatRequest = struct {
    content: []const u8,
    user_id: ?[]const u8 = null,
    session_id: ?[]const u8 = null,
    persona: ?[]const u8 = null,
    context: ?[]const u8 = null,
    max_tokens: ?u32 = null,
    temperature: ?f32 = null,

    pub fn deinit(_: *StubChatRequest, _: std.mem.Allocator) void {}

    pub fn dupe(_: std.mem.Allocator, other: StubChatRequest) !StubChatRequest {
        return other;
    }
};

/// Stub chat response - matches mod.zig ChatResponse.
pub const StubChatResponse = struct {
    content: []const u8,
    persona: []const u8,
    confidence: f32,
    latency_ms: u64,
    code_blocks: ?[]const StubCodeBlock = null,
    references: ?[]const StubSource = null,
    request_id: ?[]const u8 = null,
};

pub const StubChatResult = struct {
    status: u16,
    body: []const u8,
};

const StubCodeBlock = struct {
    language: []const u8,
    code: []const u8,
};

const StubSource = struct {
    title: []const u8,
    url: ?[]const u8 = null,
    confidence: f32,
};

/// Stub chat handler - returns WebDisabled for all operations.
pub const StubChatHandler = struct {
    allocator: std.mem.Allocator,

    pub fn init(allocator: std.mem.Allocator) StubChatHandler {
        return .{ .allocator = allocator };
    }

    pub fn handleChat(self: *StubChatHandler, request_json: []const u8) ![]const u8 {
        _ = self;
        _ = request_json;
        return error.WebDisabled;
    }

    pub fn handleAbbeyChat(self: *StubChatHandler, request_json: []const u8) ![]const u8 {
        _ = self;
        _ = request_json;
        return error.WebDisabled;
    }

    pub fn handleAvivaChat(self: *StubChatHandler, request_json: []const u8) ![]const u8 {
        _ = self;
        _ = request_json;
        return error.WebDisabled;
    }

    pub fn handleChatWithPersonaResult(
        self: *StubChatHandler,
        request_json: []const u8,
        forced_persona: ?persona_types.PersonaType,
    ) !StubChatResult {
        _ = self;
        _ = request_json;
        _ = forced_persona;
        return error.WebDisabled;
    }

    pub fn listPersonas(self: *StubChatHandler) ![]const u8 {
        _ = self;
        return error.WebDisabled;
    }

    pub fn getMetrics(self: *StubChatHandler) ![]const u8 {
        _ = self;
        return error.WebDisabled;
    }

    pub fn formatError(self: *StubChatHandler, code: []const u8, message: []const u8, request_id: ?[]const u8) ![]const u8 {
        _ = self;
        _ = code;
        _ = message;
        _ = request_id;
        return error.WebDisabled;
    }
};

/// Stub HTTP method enum.
pub const Method = enum {
    GET,
    POST,
    PUT,
    DELETE,
    PATCH,
    OPTIONS,
    HEAD,
};

/// Stub route definition.
pub const StubRoute = struct {
    path: []const u8,
    method: Method,
    description: []const u8,
    requires_auth: bool = false,
};

/// Stub route context.
pub const StubRouteContext = struct {
    allocator: std.mem.Allocator,
    body: []const u8 = "",
    response_status: u16 = 200,
    response_content_type: []const u8 = "application/json",

    pub fn init(allocator: std.mem.Allocator, handler: *StubChatHandler) StubRouteContext {
        _ = handler;
        return .{ .allocator = allocator };
    }

    pub fn deinit(self: *StubRouteContext) void {
        _ = self;
    }

    pub fn write(self: *StubRouteContext, data: []const u8) !void {
        _ = self;
        _ = data;
    }

    pub fn setStatus(self: *StubRouteContext, status: u16) void {
        self.response_status = status;
    }

    pub fn setContentType(self: *StubRouteContext, content_type: []const u8) void {
        self.response_content_type = content_type;
    }
};

/// Stub router.
pub const StubRouter = struct {
    allocator: std.mem.Allocator,

    pub fn init(allocator: std.mem.Allocator, handler: *StubChatHandler) StubRouter {
        _ = handler;
        return .{ .allocator = allocator };
    }

    pub fn match(self: *const StubRouter, path: []const u8, method: Method) ?StubRoute {
        _ = self;
        _ = path;
        _ = method;
        return null;
    }

    pub fn getRouteDefinitions(self: *const StubRouter) []const StubRoute {
        _ = self;
        return &.{};
    }
};

// ============================================================================
// Re-exports
// ============================================================================

pub const WebError = types.WebError;
pub const Response = types.Response;
pub const RequestOptions = types.RequestOptions;
pub const WeatherConfig = types.WeatherConfig;
pub const JsonValue = types.JsonValue;
pub const ParsedJson = types.ParsedJson;

pub const HttpClient = client.HttpClient;
pub const WeatherClient = client.WeatherClient;

/// Web Context stub for Framework integration.
pub const Context = struct {
    allocator: std.mem.Allocator,
    config: config_module.WebConfig,
    http_client: ?HttpClient = null,

    pub fn init(allocator: std.mem.Allocator, cfg: config_module.WebConfig) !*Context {
        _ = allocator;
        _ = cfg;
        return error.WebDisabled;
    }

    pub fn deinit(self: *Context) void {
        _ = self;
    }

    pub fn get(self: *Context, url: []const u8) !Response {
        _ = self;
        _ = url;
        return error.WebDisabled;
    }

    pub fn getWithOptions(self: *Context, url: []const u8, options: RequestOptions) !Response {
        _ = self;
        _ = url;
        _ = options;
        return error.WebDisabled;
    }

    pub fn postJson(self: *Context, url: []const u8, body: []const u8) !Response {
        _ = self;
        _ = url;
        _ = body;
        return error.WebDisabled;
    }

    pub fn freeResponse(self: *Context, response: Response) void {
        _ = self;
        _ = response;
    }

    pub fn parseJsonValue(self: *Context, response: Response) !ParsedJson {
        _ = self;
        _ = response;
        return error.WebDisabled;
    }
};

// HTTP utilities stub
pub const http = struct {
    pub fn isSuccess(status: u16) bool {
        return status >= 200 and status < 300;
    }

    pub fn isRedirect(status: u16) bool {
        return status >= 300 and status < 400;
    }

    pub fn isClientError(status: u16) bool {
        return status >= 400 and status < 500;
    }

    pub fn isServerError(status: u16) bool {
        return status >= 500 and status < 600;
    }
};

// Module lifecycle
var initialized: bool = false;

pub fn init(_: std.mem.Allocator) !void {
    return error.WebDisabled;
}

pub fn deinit() void {
    initialized = false;
}

pub fn isEnabled() bool {
    return false;
}

pub fn isInitialized() bool {
    return initialized;
}

// Convenience functions
pub fn get(allocator: std.mem.Allocator, url: []const u8) !Response {
    _ = allocator;
    _ = url;
    return error.WebDisabled;
}

pub fn getWithOptions(
    allocator: std.mem.Allocator,
    url: []const u8,
    options: RequestOptions,
) !Response {
    _ = allocator;
    _ = url;
    _ = options;
    return error.WebDisabled;
}

pub fn postJson(allocator: std.mem.Allocator, url: []const u8, body: []const u8) !Response {
    _ = allocator;
    _ = url;
    _ = body;
    return error.WebDisabled;
}

pub fn freeResponse(allocator: std.mem.Allocator, response: Response) void {
    _ = allocator;
    _ = response;
}

pub fn parseJsonValue(allocator: std.mem.Allocator, response: Response) !ParsedJson {
    _ = allocator;
    _ = response;
    return error.WebDisabled;
}

pub fn isSuccessStatus(status: u16) bool {
    return http.isSuccess(status);
}
