const std = @import("std");

pub const WebError = error{
    WebDisabled,
};

pub const Response = struct {
    status: u16,
    body: []const u8,
};

pub const RequestOptions = struct {
    max_response_bytes: usize = 1024 * 1024,
    user_agent: []const u8 = "abi-http",
    follow_redirects: bool = true,
    redirect_limit: u16 = 3,
    content_type: ?[]const u8 = null,
    extra_headers: []const std.http.Header = &.{},
};

pub const WeatherConfig = struct {
    base_url: []const u8 = "https://api.open-meteo.com/v1/forecast",
    include_current: bool = true,
};

pub const JsonValue = std.json.Value;
pub const ParsedJson = std.json.Parsed(JsonValue);
