//! Web stub — disabled at compile time.

const std = @import("std");
const config_module = @import("../../core/config/mod.zig");

const persona_types = struct {
    pub const PersonaType = enum { assistant, coder, writer, analyst, companion, docs, reviewer, minimal, abbey, aviva, abi, ralph };
};

// --- Local Stubs Imports ---

pub const types = @import("stubs/types.zig");
pub const client = @import("stubs/client.zig");

// --- Handlers and Routes ---

pub const handlers = struct {
    pub const chat = struct {
        pub const ChatHandler = StubChatHandler;
        pub const ChatRequest = StubChatRequest;
        pub const ChatResponse = StubChatResponse;
        pub const ChatResult = StubChatResult;
    };
};

pub const routes = struct {
    pub const personas = struct {
        pub const Router = StubRouter;
        pub const Route = StubRoute;
        pub const RouteContext = StubRouteContext;
    };
};

pub const ChatHandler = StubChatHandler;
pub const ChatRequest = StubChatRequest;
pub const ChatResponse = StubChatResponse;
pub const ChatResult = StubChatResult;
pub const PersonaRouter = StubRouter;
pub const Route = StubRoute;
pub const RouteContext = StubRouteContext;

// --- Stub Type Definitions ---

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

pub const StubChatResponse = struct {
    content: []const u8,
    persona: []const u8,
    confidence: f32,
    latency_ms: u64,
    code_blocks: ?[]const StubCodeBlock = null,
    references: ?[]const StubSource = null,
    request_id: ?[]const u8 = null,
};

pub const StubChatResult = struct { status: u16, body: []const u8 };
const StubCodeBlock = struct { language: []const u8, code: []const u8 };
const StubSource = struct { title: []const u8, url: ?[]const u8 = null, confidence: f32 };

pub const StubChatHandler = struct {
    allocator: std.mem.Allocator,
    pub fn init(allocator: std.mem.Allocator) StubChatHandler {
        return .{ .allocator = allocator };
    }
    pub fn handleChat(_: *StubChatHandler, _: []const u8) ![]const u8 {
        return error.WebDisabled;
    }
    pub fn handleAbbeyChat(_: *StubChatHandler, _: []const u8) ![]const u8 {
        return error.WebDisabled;
    }
    pub fn handleAvivaChat(_: *StubChatHandler, _: []const u8) ![]const u8 {
        return error.WebDisabled;
    }
    pub fn handleChatWithPersonaResult(_: *StubChatHandler, _: []const u8, _: ?persona_types.PersonaType) !StubChatResult {
        return error.WebDisabled;
    }
    pub fn listPersonas(_: *StubChatHandler) ![]const u8 {
        return error.WebDisabled;
    }
    pub fn getMetrics(_: *StubChatHandler) ![]const u8 {
        return error.WebDisabled;
    }
    pub fn formatError(_: *StubChatHandler, _: []const u8, _: []const u8, _: ?[]const u8) ![]const u8 {
        return error.WebDisabled;
    }
};

pub const Method = enum { GET, POST, PUT, DELETE, PATCH, OPTIONS, HEAD };

pub const StubRoute = struct {
    path: []const u8,
    method: Method,
    description: []const u8,
    requires_auth: bool = false,
};

pub const StubRouteContext = struct {
    allocator: std.mem.Allocator,
    body: []const u8 = "",
    response_status: u16 = 200,
    response_content_type: []const u8 = "application/json",
    pub fn init(allocator: std.mem.Allocator, _: *StubChatHandler) StubRouteContext {
        return .{ .allocator = allocator };
    }
    pub fn deinit(_: *StubRouteContext) void {}
    pub fn write(_: *StubRouteContext, _: []const u8) !void {}
    pub fn setStatus(self: *StubRouteContext, status: u16) void {
        self.response_status = status;
    }
    pub fn setContentType(self: *StubRouteContext, content_type: []const u8) void {
        self.response_content_type = content_type;
    }
};

pub const StubRouter = struct {
    allocator: std.mem.Allocator,
    pub fn init(allocator: std.mem.Allocator, _: *StubChatHandler) StubRouter {
        return .{ .allocator = allocator };
    }
    pub fn match(_: *const StubRouter, _: []const u8, _: Method) ?StubRoute {
        return null;
    }
    pub fn getRouteDefinitions(_: *const StubRouter) []const StubRoute {
        return &.{};
    }
};

// --- Server and Middleware Stubs ---

pub const server = struct {
    pub const Server = StubServer;
    pub const ServerConfig = struct {
        host: []const u8 = "127.0.0.1",
        port: u16 = 8080,
        max_connections: usize = 1024,
        read_timeout_ms: u32 = 30_000,
        write_timeout_ms: u32 = 30_000,
        keep_alive: bool = true,
        keep_alive_timeout_ms: u32 = 5_000,
        worker_threads: u32 = 4,
    };
    pub const ServerState = enum { stopped, starting, running, stopping };
    pub const ServerStats = struct {};
    pub const ServerError = error{WebDisabled};
};

const StubServer = struct {
    pub fn init(_: std.mem.Allocator, _: anytype) StubServer {
        return .{};
    }
    pub fn deinit(_: *StubServer) void {}
};

pub const middleware = struct {
    pub const observability = struct {
        pub const BUCKET_COUNT: usize = 8;
        pub const bucket_bounds_us: [BUCKET_COUNT]u64 = .{
            100,
            500,
            1_000,
            5_000,
            50_000,
            200_000,
            1_000_000,
            std.math.maxInt(u64),
        };
        pub const RequestMetrics = StubRequestMetrics;
        pub const MetricsSnapshot = StubMetricsSnapshot;
        pub const MetricsMiddleware = StubMetricsMiddleware;
    };

    pub const MetricsMiddleware = StubMetricsMiddleware;
    pub const RequestMetrics = StubRequestMetrics;
    pub const MetricsSnapshot = StubMetricsSnapshot;
};

const stub_bucket_count = 8;

const StubRequestMetrics = struct {
    start_ns: i128,
};

const StubMetricsSnapshot = struct {
    total_requests: u64,
    total_errors: u64,
    active_requests: u64,
    request_durations_us: [stub_bucket_count]u64,
    status_counts: [6]u64,
};

const StubMetricsMiddleware = struct {
    total_requests: u64 = 0,
    total_errors: u64 = 0,
    active_requests: u64 = 0,
    request_durations_us: [stub_bucket_count]u64 = .{0} ** stub_bucket_count,
    status_counts: [6]u64 = .{0} ** 6,

    pub fn init() StubMetricsMiddleware {
        return .{};
    }
    pub fn processRequest(_: *StubMetricsMiddleware) StubRequestMetrics {
        return .{ .start_ns = 0 };
    }
    pub fn recordResponse(_: *StubMetricsMiddleware, _: StubRequestMetrics, _: u16) void {}
    pub fn getSnapshot(_: *const StubMetricsMiddleware) StubMetricsSnapshot {
        return .{
            .total_requests = 0,
            .total_errors = 0,
            .active_requests = 0,
            .request_durations_us = .{0} ** stub_bucket_count,
            .status_counts = .{0} ** 6,
        };
    }
    pub fn formatPrometheus(_: *const StubMetricsMiddleware, _: std.mem.Allocator) ![]u8 {
        return error.WebDisabled;
    }
    pub fn reset(_: *StubMetricsMiddleware) void {}
};

// --- Re-exports ---

pub const WebError = types.WebError;
pub const Response = types.Response;
pub const RequestOptions = types.RequestOptions;
pub const WeatherConfig = types.WeatherConfig;
pub const JsonValue = types.JsonValue;
pub const ParsedJson = types.ParsedJson;
pub const HttpClient = client.HttpClient;
pub const WeatherClient = client.WeatherClient;

// --- Context ---

pub const Context = struct {
    allocator: std.mem.Allocator,
    config: config_module.WebConfig,
    http_client: ?HttpClient = null,
    pub fn init(_: std.mem.Allocator, _: config_module.WebConfig) !*Context {
        return error.WebDisabled;
    }
    pub fn deinit(_: *Context) void {}
    pub fn get(_: *Context, _: []const u8) !Response {
        return error.WebDisabled;
    }
    pub fn getWithOptions(_: *Context, _: []const u8, _: RequestOptions) !Response {
        return error.WebDisabled;
    }
    pub fn postJson(_: *Context, _: []const u8, _: []const u8) !Response {
        return error.WebDisabled;
    }
    pub fn freeResponse(_: *Context, _: Response) void {}
    pub fn parseJsonValue(_: *Context, _: Response) !ParsedJson {
        return error.WebDisabled;
    }
};

// --- HTTP Utilities ---

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

// --- Module Lifecycle ---

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

// --- Convenience Functions ---

pub fn get(_: std.mem.Allocator, _: []const u8) !Response {
    return error.WebDisabled;
}
pub fn getWithOptions(_: std.mem.Allocator, _: []const u8, _: RequestOptions) !Response {
    return error.WebDisabled;
}
pub fn postJson(_: std.mem.Allocator, _: []const u8, _: []const u8) !Response {
    return error.WebDisabled;
}
pub fn freeResponse(_: std.mem.Allocator, _: Response) void {}
pub fn parseJsonValue(_: std.mem.Allocator, _: Response) !ParsedJson {
    return error.WebDisabled;
}
pub fn isSuccessStatus(status: u16) bool {
    return http.isSuccess(status);
}
