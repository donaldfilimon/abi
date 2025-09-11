//! Services Module - External service integrations
//!
//! This module provides integrations with external services and APIs:
//! - Weather services for location-based data
//! - Third-party API connectors
//! - Service discovery and load balancing
//! - Authentication and authorization services

const std = @import("std");

// Re-export weather service
pub const WeatherService = @import("weather.zig").WeatherService;
pub const WeatherData = @import("weather.zig").WeatherData;
pub const WeatherConfig = @import("weather.zig").WeatherConfig;
pub const WeatherError = @import("weather.zig").WeatherError;

// Re-export commonly used types
pub const Allocator = std.mem.Allocator;

/// Initialize a weather service with default configuration
pub fn createWeatherService(allocator: std.mem.Allocator, api_key: []const u8) !*WeatherService {
    const config = WeatherConfig{
        .api_key = api_key,
        .base_url = "https://api.openweathermap.org/data/2.5",
        .timeout_ms = 10000,
    };
    return WeatherService.init(allocator, config);
}

test "Services module imports" {
    // Test that all main types are accessible
    _ = WeatherService;
    _ = WeatherData;
    _ = WeatherConfig;
}
