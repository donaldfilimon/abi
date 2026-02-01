//! Type definitions for the web module stub.
//!
//! These types mirror the real implementations to maintain API compatibility
//! when the web feature is disabled.

const std = @import("std");

/// Web module error type.
pub const WebError = error{
    /// The web feature is disabled in the build configuration.
    WebDisabled,
};

/// HTTP response structure.
pub const Response = struct {
    /// HTTP status code.
    status: u16,
    /// Response body content.
    body: []const u8,
};

/// HTTP request configuration options.
pub const RequestOptions = struct {
    /// Maximum bytes to read from response body.
    max_response_bytes: usize = 1024 * 1024,
    /// User-Agent header value.
    user_agent: []const u8 = "abi-http",
    /// Whether to follow HTTP redirects.
    follow_redirects: bool = true,
    /// Maximum number of redirects to follow.
    redirect_limit: u16 = 3,
    /// Content-Type header for requests with a body.
    content_type: ?[]const u8 = null,
    /// Additional headers to include in the request.
    extra_headers: []const std.http.Header = &.{},

    /// Hard upper limit for response size (100MB).
    pub const MAX_ALLOWED_RESPONSE_BYTES: usize = 100 * 1024 * 1024;

    /// Returns the effective max response bytes, capped at the hard limit.
    pub fn effectiveMaxResponseBytes(self: RequestOptions) usize {
        return @min(self.max_response_bytes, MAX_ALLOWED_RESPONSE_BYTES);
    }
};

/// Weather client configuration.
pub const WeatherConfig = struct {
    /// Base URL for the weather API.
    base_url: []const u8 = "https://api.open-meteo.com/v1/forecast",
    /// Whether to include current weather conditions.
    include_current: bool = true,
};

/// JSON value type for response parsing.
pub const JsonValue = std.json.Value;
/// Parsed JSON with ownership.
pub const ParsedJson = std.json.Parsed(JsonValue);
