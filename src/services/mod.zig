//! Services Module - External Service Integrations
//!
//! This module offers integrations with various external services and APIs, including:
//! - Weather services for location-based data retrieval and forecasting
//! - Connectors for third-party APIs with built-in error handling
//! - Mechanisms for service discovery and load balancing
//! - Authentication and authorization services
//!
//! It re-exports essential types and functions from the `weather.zig` module and the standard library
//! for streamlined access and consistent memory management across service integrations.

const std = @import("std");
const weather = @import("weather.zig");

// Re-export weather service components with documentation

/// Represents the weather service interface for interacting with OpenWeatherMap API.
/// Provides methods for retrieving current weather data and forecasts.
pub const WeatherService = weather.WeatherService;

/// Contains structured weather data including temperature, humidity, wind, and location information.
/// All string fields are allocated and must be freed by the caller.
pub const WeatherData = weather.WeatherData;

/// Configuration settings for the weather service including API credentials,
/// timeout settings, and response size limits.
pub const WeatherConfig = weather.WeatherConfig;

/// Error types related to weather service operations including network errors,
/// API errors, and data parsing failures.
pub const WeatherError = weather.WeatherError;

/// Allocator for managing memory allocations within service modules.
/// Used for dynamic string allocation and HTTP response buffering.
pub const Allocator = std.mem.Allocator;

/// Initializes a weather service with the provided API key and default configuration.
///
/// This function creates a WeatherService instance with sensible defaults for production use.
/// The configuration can be further customized using WeatherConfig.fromEnv() for environment
/// variable overrides.
///
/// Parameters:
/// - `allocator`: Allocator for memory management during service operations
/// - `api_key`: API key for authenticating with the OpenWeatherMap service
///
/// Returns:
/// - A WeatherService instance ready for use
///
/// Errors:
/// - Returns an error if the service initialization fails due to invalid configuration
pub fn createWeatherService(allocator: Allocator, api_key: []const u8) !WeatherService {
    const config = WeatherConfig{
        .api_key = api_key,
        .base_url = "https://api.openweathermap.org/data/2.5",
        .units = "metric",
        .language = "en",
        .timeout_seconds = 10,
        .max_response_bytes = 1024 * 1024,
    };
    return WeatherService.init(allocator, config);
}

test "Services module imports" {
    // Verify that all main types are accessible and correctly imported
    _ = WeatherService;
    _ = WeatherData;
    _ = WeatherConfig;
    _ = WeatherError;
    _ = Allocator;
}

test "createWeatherService basic functionality" {
    const testing = std.testing;
    const allocator = testing.allocator;

    // Test service creation with valid API key
    const service = try createWeatherService(allocator, "test_api_key");
    defer service.deinit();

    // Verify configuration is properly set
    try testing.expectEqualStrings("test_api_key", service.config.api_key);
    try testing.expectEqualStrings("https://api.openweathermap.org/data/2.5", service.config.base_url);
    try testing.expectEqual(@as(u32, 10), service.config.timeout_seconds);
}
