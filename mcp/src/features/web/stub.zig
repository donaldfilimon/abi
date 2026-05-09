//! Web stub -- disabled at compile time.

const std = @import("std");
const stub_helpers = @import("../core/stub_helpers.zig");
const config_module = @import("../core/config/mod.zig");

const stub_types = @import("stubs/types.zig");
const client = @import("stubs/client.zig");

// --- Shared types (from types.zig) ---
pub const types = @import("types.zig");
pub const WebError = types.WebError;
pub const Response = types.Response;
pub const RequestOptions = types.RequestOptions;
pub const WeatherConfig = types.WeatherConfig;
pub const JsonValue = types.JsonValue;
pub const ParsedJson = types.ParsedJson;

// --- Client re-exports ---
pub const HttpClient = client.HttpClient;
pub const WeatherClient = client.WeatherClient;

// --- Chat handler re-exports ---
pub const ChatHandler = stub_types.ChatHandler;
pub const ChatRequest = stub_types.ChatRequest;
pub const ChatResponse = stub_types.ChatResponse;
pub const ChatResult = stub_types.ChatResult;
pub const ProfileRouter = stub_types.Router;
pub const Route = stub_types.Route;
pub const RouteContext = stub_types.RouteContext;

// --- Handlers and Routes namespaces ---
pub const handlers = struct {
    pub const chat = struct {
        pub const ChatHandler = stub_types.ChatHandler;
        pub const ChatRequest = stub_types.ChatRequest;
        pub const ChatResponse = stub_types.ChatResponse;
        pub const ChatResult = stub_types.ChatResult;
    };
};

pub const routes = struct {
    pub const profiles = struct {
        pub const Router = stub_types.Router;
        pub const Route = stub_types.Route;
        pub const RouteContext = stub_types.RouteContext;
    };
};

// --- Server and Middleware ---
pub const server = struct {
    pub const Server = stub_types.Server;
    pub const ServerConfig = stub_types.ServerConfig;
    pub const ServerState = stub_types.ServerState;
    pub const ServerStats = stub_types.ServerStats;
    pub const ServerError = stub_types.ServerError;
};

pub const middleware = struct {
    pub const observability = struct {
        pub const BUCKET_COUNT = stub_types.BUCKET_COUNT;
        pub const bucket_bounds_us = stub_types.bucket_bounds_us;
        pub const RequestMetrics = stub_types.RequestMetrics;
        pub const MetricsSnapshot = stub_types.MetricsSnapshot;
        pub const MetricsMiddleware = stub_types.MetricsMiddleware;
    };
    pub const MetricsMiddleware = stub_types.MetricsMiddleware;
    pub const RequestMetrics = stub_types.RequestMetrics;
    pub const MetricsSnapshot = stub_types.MetricsSnapshot;
};

// --- Context ---
pub const Context = struct {
    allocator: std.mem.Allocator,
    config: config_module.WebConfig,
    http_client: ?HttpClient = null,
    pub fn init(_: std.mem.Allocator, _: config_module.WebConfig) !*Context {
        return error.FeatureDisabled;
    }
    pub fn deinit(_: *Context) void {}
    pub fn get(_: *Context, _: []const u8) !Response {
        return error.FeatureDisabled;
    }
    pub fn getWithOptions(_: *Context, _: []const u8, _: RequestOptions) !Response {
        return error.FeatureDisabled;
    }
    pub fn postJson(_: *Context, _: []const u8, _: []const u8) !Response {
        return error.FeatureDisabled;
    }
    pub fn freeResponse(_: *Context, _: Response) void {}
    pub fn parseJsonValue(_: *Context, _: Response) !ParsedJson {
        return error.FeatureDisabled;
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
const _stub = stub_helpers.StubFeatureNoConfig(error{FeatureDisabled});
pub const init = _stub.init;
pub const deinit = _stub.deinit;
pub const isEnabled = _stub.isEnabled;
pub const isInitialized = _stub.isInitialized;

// --- Convenience Functions ---
pub fn get(_: std.mem.Allocator, _: []const u8) !Response {
    return error.FeatureDisabled;
}
pub fn getWithOptions(_: std.mem.Allocator, _: []const u8, _: RequestOptions) !Response {
    return error.FeatureDisabled;
}
pub fn postJson(_: std.mem.Allocator, _: []const u8, _: []const u8) !Response {
    return error.FeatureDisabled;
}
pub fn freeResponse(_: std.mem.Allocator, _: Response) void {}
pub fn parseJsonValue(_: std.mem.Allocator, _: Response) !ParsedJson {
    return error.FeatureDisabled;
}
pub fn isSuccessStatus(status: u16) bool {
    return http.isSuccess(status);
}

test {
    std.testing.refAllDecls(@This());
}
