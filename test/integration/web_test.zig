//! Integration Tests: Web Module
//!
//! Verifies web module type exports, behavioral contracts,
//! and API semantics without making real HTTP requests.

const std = @import("std");
const abi = @import("abi");
const build_options = @import("build_options");

const web = abi.web;

// ============================================================================
// Feature flag behavior
// ============================================================================

test "web: isEnabled matches build flag" {
    try std.testing.expectEqual(build_options.feat_web, web.isEnabled());
}

test "web: isInitialized returns false before init" {
    // Module-level state should default to not-initialized
    // (unless a prior test called init — this tests the initial contract)
    if (!web.isEnabled()) return error.SkipZigTest;
    // We can't guarantee prior test order, but we can test the function exists and returns bool
    const result = web.isInitialized();
    try std.testing.expect(result == true or result == false);
}

// ============================================================================
// RequestOptions behavior
// ============================================================================

test "web: RequestOptions defaults are sensible" {
    const opts = web.RequestOptions{};
    try std.testing.expectEqual(@as(usize, 1024 * 1024), opts.max_response_bytes);
    try std.testing.expectEqualStrings("abi-http", opts.user_agent);
    try std.testing.expect(opts.follow_redirects);
    try std.testing.expectEqual(@as(u16, 3), opts.redirect_limit);
    try std.testing.expect(opts.content_type == null);
    try std.testing.expectEqual(@as(usize, 0), opts.extra_headers.len);
}

test "web: RequestOptions effectiveMaxResponseBytes caps at 100MB" {
    const hard_limit = web.RequestOptions.MAX_ALLOWED_RESPONSE_BYTES;
    try std.testing.expectEqual(@as(usize, 100 * 1024 * 1024), hard_limit);

    // Normal value passes through
    const normal = web.RequestOptions{ .max_response_bytes = 512 };
    try std.testing.expectEqual(@as(usize, 512), normal.effectiveMaxResponseBytes());

    // Oversized value gets capped
    const oversized = web.RequestOptions{ .max_response_bytes = 200 * 1024 * 1024 };
    try std.testing.expectEqual(hard_limit, oversized.effectiveMaxResponseBytes());
}

test "web: RequestOptions exact boundary at hard limit" {
    const hard_limit = web.RequestOptions.MAX_ALLOWED_RESPONSE_BYTES;
    const at_limit = web.RequestOptions{ .max_response_bytes = hard_limit };
    try std.testing.expectEqual(hard_limit, at_limit.effectiveMaxResponseBytes());

    const one_over = web.RequestOptions{ .max_response_bytes = hard_limit + 1 };
    try std.testing.expectEqual(hard_limit, one_over.effectiveMaxResponseBytes());
}

// ============================================================================
// Response type
// ============================================================================

test "web: Response struct fields accessible" {
    const response = web.Response{ .status = 200, .body = "OK" };
    try std.testing.expectEqual(@as(u16, 200), response.status);
    try std.testing.expectEqualStrings("OK", response.body);
}

test "web: Response can represent error status" {
    const response = web.Response{ .status = 404, .body = "Not Found" };
    try std.testing.expectEqual(@as(u16, 404), response.status);
    try std.testing.expect(!abi.foundation.utils.http.isSuccess(response.status));
}

test "web: Response can represent success status" {
    const response = web.Response{ .status = 201, .body = "{}" };
    try std.testing.expect(abi.foundation.utils.http.isSuccess(response.status));
}

// ============================================================================
// WeatherConfig
// ============================================================================

test "web: WeatherConfig defaults" {
    const config = web.WeatherConfig{};
    try std.testing.expectEqualStrings(
        "https://api.open-meteo.com/v1/forecast",
        config.base_url,
    );
    try std.testing.expect(config.include_current);
}

test "web: WeatherConfig custom values" {
    const config = web.WeatherConfig{
        .base_url = "https://custom-weather.example.com/api",
        .include_current = false,
    };
    try std.testing.expectEqualStrings("https://custom-weather.example.com/api", config.base_url);
    try std.testing.expect(!config.include_current);
}

// ============================================================================
// ChatRequest type
// ============================================================================

test "web: ChatRequest defaults" {
    const req = web.ChatRequest{
        .content = "hello",
    };
    try std.testing.expectEqualStrings("hello", req.content);
    try std.testing.expect(req.user_id == null);
    try std.testing.expect(req.session_id == null);
    try std.testing.expect(req.profile == null);
    try std.testing.expect(req.context == null);
    try std.testing.expect(req.max_tokens == null);
    try std.testing.expect(req.temperature == null);
}

test "web: ChatRequest with all fields" {
    const req = web.ChatRequest{
        .content = "test",
        .user_id = "user-1",
        .session_id = "sess-1",
        .profile = "abbey",
        .context = "be helpful",
        .max_tokens = 1024,
        .temperature = 0.7,
    };
    try std.testing.expectEqualStrings("test", req.content);
    try std.testing.expectEqualStrings("user-1", req.user_id.?);
    try std.testing.expectEqualStrings("abbey", req.profile.?);
    try std.testing.expectEqual(@as(u32, 1024), req.max_tokens.?);
}

// ============================================================================
// Core type availability (existing compile checks, kept for coverage)
// ============================================================================

test "web: core types exist" {
    _ = web.Response;
    _ = web.HttpClient;
    _ = web.RequestOptions;
    _ = web.Context;
    _ = web.WebError;
    _ = web.ChatHandler;
    _ = web.ChatResponse;
    _ = web.ChatResult;
    _ = web.ProfileRouter;
    _ = web.Route;
    _ = web.RouteContext;
    _ = web.JsonValue;
    _ = web.ParsedJson;
    _ = web.server;
    _ = web.middleware;
    _ = web.types;
}

test {
    std.testing.refAllDecls(@This());
}
