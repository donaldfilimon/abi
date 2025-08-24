//! Weather module tests
//!
//! Tests for the weather module functionality

const std = @import("std");
const abi = @import("abi");

test "weather utilities" {
    // Test temperature conversions
    try std.testing.expectEqual(@as(f32, 0.0), abi.weather.WeatherUtils.kelvinToCelsius(273.15));
    try std.testing.expectEqual(@as(f32, 32.0), abi.weather.WeatherUtils.celsiusToFahrenheit(0.0));
    try std.testing.expectEqual(@as(f32, 0.0), abi.weather.WeatherUtils.fahrenheitToCelsius(32.0));

    // Test wind direction
    try std.testing.expectEqualStrings("N", abi.weather.WeatherUtils.getWindDirection(0));
    try std.testing.expectEqualStrings("E", abi.weather.WeatherUtils.getWindDirection(90));
    try std.testing.expectEqualStrings("S", abi.weather.WeatherUtils.getWindDirection(180));
    try std.testing.expectEqualStrings("W", abi.weather.WeatherUtils.getWindDirection(270));
    try std.testing.expectEqualStrings("NE", abi.weather.WeatherUtils.getWindDirection(45));
    try std.testing.expectEqualStrings("SE", abi.weather.WeatherUtils.getWindDirection(135));
    try std.testing.expectEqualStrings("SW", abi.weather.WeatherUtils.getWindDirection(225));
    try std.testing.expectEqualStrings("NW", abi.weather.WeatherUtils.getWindDirection(315));

    // Test weather emojis
    try std.testing.expectEqualStrings("☀️", abi.weather.WeatherUtils.getWeatherEmoji("01d"));
    try std.testing.expectEqualStrings("⛅", abi.weather.WeatherUtils.getWeatherEmoji("02d"));
    try std.testing.expectEqualStrings("☁️", abi.weather.WeatherUtils.getWeatherEmoji("03d"));
    try std.testing.expectEqualStrings("🌧️", abi.weather.WeatherUtils.getWeatherEmoji("09d"));
    try std.testing.expectEqualStrings("🌦️", abi.weather.WeatherUtils.getWeatherEmoji("10d"));
    try std.testing.expectEqualStrings("⛈️", abi.weather.WeatherUtils.getWeatherEmoji("11d"));
    try std.testing.expectEqualStrings("🌨️", abi.weather.WeatherUtils.getWeatherEmoji("13d"));
    try std.testing.expectEqualStrings("🌫️", abi.weather.WeatherUtils.getWeatherEmoji("50d"));
}

test "weather config" {
    const config = abi.weather.WeatherConfig{
        .api_key = "test_key",
        .units = "metric",
        .language = "en",
    };

    try std.testing.expectEqualStrings("test_key", config.api_key);
    try std.testing.expectEqualStrings("metric", config.units);
    try std.testing.expectEqualStrings("en", config.language);
    try std.testing.expectEqual(@as(u32, 10), config.timeout_seconds);
}

test "weather data structure" {
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    const allocator = arena.allocator();

    var weather_data = abi.weather.WeatherData{
        .temperature = 20.5,
        .feels_like = 19.2,
        .humidity = 65,
        .pressure = 1013,
        .description = try allocator.dupe(u8, "scattered clouds"),
        .icon = try allocator.dupe(u8, "03d"),
        .wind_speed = 3.2,
        .wind_direction = 180,
        .visibility = 10000,
        .sunrise = 1640995200,
        .sunset = 1641027600,
        .city = try allocator.dupe(u8, "London"),
        .country = try allocator.dupe(u8, "GB"),
        .timestamp = 1641009600,
    };

    try std.testing.expectEqual(@as(f32, 20.5), weather_data.temperature);
    try std.testing.expectEqual(@as(f32, 19.2), weather_data.feels_like);
    try std.testing.expectEqual(@as(u8, 65), weather_data.humidity);
    try std.testing.expectEqual(@as(u16, 1013), weather_data.pressure);
    try std.testing.expectEqualStrings("scattered clouds", weather_data.description);
    try std.testing.expectEqualStrings("03d", weather_data.icon);
    try std.testing.expectEqual(@as(f32, 3.2), weather_data.wind_speed);
    try std.testing.expectEqual(@as(u16, 180), weather_data.wind_direction);
    try std.testing.expectEqual(@as(u32, 10000), weather_data.visibility);
    try std.testing.expectEqual(@as(u64, 1640995200), weather_data.sunrise);
    try std.testing.expectEqual(@as(u64, 1641027600), weather_data.sunset);
    try std.testing.expectEqualStrings("London", weather_data.city);
    try std.testing.expectEqualStrings("GB", weather_data.country);
    try std.testing.expectEqual(@as(u64, 1641009600), weather_data.timestamp);

    // Test deinit
    weather_data.deinit(allocator);
}

test "weather service initialization" {
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    const allocator = arena.allocator();

    const config = abi.weather.WeatherConfig{
        .api_key = "test_key",
        .units = "metric",
        .language = "en",
    };

    var service = try abi.weather.WeatherService.init(allocator, config);
    defer service.deinit();

    try std.testing.expectEqualStrings("test_key", service.config.api_key);
    try std.testing.expectEqualStrings("metric", service.config.units);
    try std.testing.expectEqualStrings("en", service.config.language);
}

test "weather json formatting" {
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    const allocator = arena.allocator();

    var weather_data = abi.weather.WeatherData{
        .temperature = 20.5,
        .feels_like = 19.2,
        .humidity = 65,
        .pressure = 1013,
        .description = try allocator.dupe(u8, "scattered clouds"),
        .icon = try allocator.dupe(u8, "03d"),
        .wind_speed = 3.2,
        .wind_direction = 180,
        .visibility = 10000,
        .sunrise = 1640995200,
        .sunset = 1641027600,
        .city = try allocator.dupe(u8, "London"),
        .country = try allocator.dupe(u8, "GB"),
        .timestamp = 1641009600,
    };
    defer weather_data.deinit(allocator);

    const json = try abi.weather.WeatherUtils.formatWeatherJson(weather_data, allocator);
    defer allocator.free(json);

    // Verify JSON contains expected fields
    try std.testing.expect(std.mem.indexOf(u8, json, "temperature") != null);
    try std.testing.expect(std.mem.indexOf(u8, json, "humidity") != null);
    try std.testing.expect(std.mem.indexOf(u8, json, "London") != null);
}
