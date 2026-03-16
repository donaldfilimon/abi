//! Weather API Client
//!
//! Provides integration with the Open-Meteo weather API for retrieving
//! weather forecasts based on geographic coordinates.
//!
//! ## Supported Location Formats
//!
//! - Coordinates: `"40.7128,-74.0060"` (latitude, longitude)
//! - Direct URL: `"https://api.open-meteo.com/v1/forecast?latitude=..."`
//!
//! ## Example
//!
//! ```zig
//! const weather = @import("abi").web.weather;
//!
//! var client = try weather.WeatherClient.init(allocator, .{});
//! defer client.deinit();
//!
//! // Get weather for New York City
//! const response = try client.forecast("40.7128,-74.0060");
//! defer client.freeResponse(response);
//!
//! // Parse the JSON response
//! var parsed = try std.json.parseFromSlice(std.json.Value, allocator, response.body, .{});
//! defer parsed.deinit();
//! ```
//!
//! ## Open-Meteo API
//!
//! This client uses the Open-Meteo API (https://open-meteo.com/), which provides
//! free weather data without requiring an API key. For production use, consider
//! their terms of service and attribution requirements.

const std = @import("std");

const client = @import("client.zig");

/// Weather-specific errors.
pub const WeatherError = error{
    /// The provided location string could not be parsed as valid coordinates.
    /// Expected format: "latitude,longitude" (e.g., "40.7128,-74.0060").
    /// Latitude must be between -90 and 90, longitude between -180 and 180.
    InvalidLocation,
};

/// Configuration for the weather client.
pub const WeatherConfig = struct {
    /// Base URL for the weather API.
    /// Default is the Open-Meteo forecast endpoint.
    base_url: []const u8 = "https://api.open-meteo.com/v1/forecast",

    /// Whether to include current weather conditions in the response.
    /// When true, the `current_weather` field is populated in the API response.
    include_current: bool = true,
};

/// Client for fetching weather forecasts.
///
/// Wraps the HTTP client with weather-specific functionality including
/// coordinate parsing and URL building.
pub const WeatherClient = struct {
    allocator: std.mem.Allocator,
    http_client: client.HttpClient,
    config: WeatherConfig,

    /// Initialize a new weather client.
    ///
    /// Creates an underlying HTTP client for making API requests.
    pub fn init(allocator: std.mem.Allocator, config: WeatherConfig) !WeatherClient {
        const http_client = try client.HttpClient.init(allocator);
        return .{
            .allocator = allocator,
            .http_client = http_client,
            .config = config,
        };
    }

    /// Deinitialize the client and release resources.
    pub fn deinit(self: *WeatherClient) void {
        self.http_client.deinit();
        self.* = undefined;
    }

    /// Fetch a weather forecast for the given location.
    ///
    /// The location can be either:
    /// - Coordinates in "lat,lon" format (e.g., "40.7128,-74.0060")
    /// - A direct URL to a weather API endpoint
    ///
    /// Returns the raw HTTP response with JSON body from the weather API.
    /// Caller must free the response using `freeResponse()`.
    ///
    /// Returns `WeatherError.InvalidLocation` if coordinates cannot be parsed.
    pub fn forecast(self: *WeatherClient, location: []const u8) !client.Response {
        if (std.mem.startsWith(u8, location, "http://") or
            std.mem.startsWith(u8, location, "https://"))
        {
            return self.http_client.get(location);
        }

        const coords = try parseCoordinates(location);
        const url = try buildUrl(self.allocator, coords, self.config);
        defer self.allocator.free(url);
        return self.http_client.get(url);
    }

    /// Free a forecast response.
    ///
    /// Equivalent to freeing the response body through the HTTP client.
    pub fn freeResponse(self: *WeatherClient, response: client.Response) void {
        self.http_client.freeResponse(response);
    }
};

/// Geographic coordinates for weather queries.
const Coordinates = struct {
    /// Latitude in decimal degrees (-90 to 90).
    lat: f64,
    /// Longitude in decimal degrees (-180 to 180).
    lon: f64,
};

/// Parse a location string into validated coordinates.
///
/// Accepts "latitude,longitude" format with optional whitespace.
/// Validates that coordinates are within valid geographic bounds.
fn parseCoordinates(location: []const u8) WeatherError!Coordinates {
    var it = std.mem.splitScalar(u8, location, ',');
    const lat_text = it.next() orelse return WeatherError.InvalidLocation;
    const lon_text = it.next() orelse return WeatherError.InvalidLocation;
    if (it.next() != null) return WeatherError.InvalidLocation;

    const lat = std.fmt.parseFloat(f64, std.mem.trim(u8, lat_text, " \t")) catch
        return WeatherError.InvalidLocation;
    const lon = std.fmt.parseFloat(f64, std.mem.trim(u8, lon_text, " \t")) catch
        return WeatherError.InvalidLocation;
    if (lat < -90 or lat > 90 or lon < -180 or lon > 180) {
        return WeatherError.InvalidLocation;
    }
    return .{ .lat = lat, .lon = lon };
}

fn buildUrl(
    allocator: std.mem.Allocator,
    coords: Coordinates,
    config: WeatherConfig,
) ![]u8 {
    const current = if (config.include_current) "true" else "false";
    return std.fmt.allocPrint(
        allocator,
        "{s}?latitude={d:.4}&longitude={d:.4}&current_weather={s}",
        .{ config.base_url, coords.lat, coords.lon, current },
    );
}

test "weather coordinates parse and validate range" {
    const coords = try parseCoordinates(" 40.0, -73.5 ");
    try std.testing.expect(std.math.approxEqAbs(f64, coords.lat, 40.0, 0.0001));
    try std.testing.expect(std.math.approxEqAbs(f64, coords.lon, -73.5, 0.0001));

    try std.testing.expectError(WeatherError.InvalidLocation, parseCoordinates("91,0"));
    try std.testing.expectError(WeatherError.InvalidLocation, parseCoordinates("0,181"));
    try std.testing.expectError(WeatherError.InvalidLocation, parseCoordinates("0"));
}

test "weather url respects current toggle" {
    const allocator = std.testing.allocator;
    const coords = try parseCoordinates("37.7749,-122.4194");
    const url = try buildUrl(allocator, coords, .{ .include_current = false });
    defer allocator.free(url);
    try std.testing.expect(std.mem.indexOf(u8, url, "current_weather=false") != null);
}

test {
    std.testing.refAllDecls(@This());
}
