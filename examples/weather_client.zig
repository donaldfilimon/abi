//! Weather Client Example
//!
//! This example demonstrates how to use the weather module to fetch weather data
//! from the OpenWeatherMap API.

const std = @import("std");
const root = @import("../src/root.zig");

pub fn main() !void {
    // Initialize allocator
    var arena = std.heap.ArenaAllocator.init(std.heap.page_allocator);
    defer arena.deinit();
    const allocator = arena.allocator();

    // Check command line arguments
    const args = try std.process.argsAlloc(allocator);
    defer std.process.argsFree(allocator, args);

    if (args.len < 3) {
        std.debug.print("Usage: {s} <api_key> <city>\n", .{args[0]});
        std.debug.print("Example: {s} YOUR_API_KEY London\n", .{args[0]});
        return;
    }

    const api_key = args[1];
    const city = args[2];

    // Initialize weather service
    const weather_config = root.weather.WeatherConfig{
        .api_key = api_key,
        .units = "metric", // Use metric units (Celsius)
        .language = "en",
    };

    var weather_service = try root.weather.WeatherService.init(allocator, weather_config);
    defer weather_service.deinit();

    std.debug.print("Fetching weather data for {s}...\n", .{city});

    // Get current weather
    const weather_data = weather_service.getCurrentWeather(city) catch |err| {
        std.debug.print("Error fetching weather data: {}\n", .{err});
        return;
    };
    defer weather_data.deinit(allocator);

    // Display weather information
    displayWeatherInfo(weather_data);

    // Get 5-day forecast
    std.debug.print("\nFetching 5-day forecast...\n", .{});
    const forecast_data = weather_service.getForecast(city) catch |err| {
        std.debug.print("Error fetching forecast data: {}\n", .{err});
        return;
    };
    defer {
        for (forecast_data) |*data| {
            data.deinit(allocator);
        }
        allocator.free(forecast_data);
    }

    displayForecast(forecast_data);
}

fn displayWeatherInfo(weather: root.weather.WeatherData) void {
    const emoji = root.weather.WeatherUtils.getWeatherEmoji(weather.icon);
    const wind_dir = root.weather.WeatherUtils.getWindDirection(weather.wind_direction);

    std.debug.print("\n=== Current Weather ===\n", .{});
    std.debug.print("📍 Location: {s}, {s}\n", .{ weather.city, weather.country });
    std.debug.print("🌡️  Temperature: {d:.1}°C (feels like {d:.1}°C)\n", .{ weather.temperature, weather.feels_like });
    std.debug.print("{s} Condition: {s}\n", .{ emoji, weather.description });
    std.debug.print("💧 Humidity: {}%\n", .{weather.humidity});
    std.debug.print("🌪️  Wind: {d:.1} m/s {s}\n", .{ weather.wind_speed, wind_dir });
    std.debug.print("👁️  Visibility: {} m\n", .{weather.visibility});
    std.debug.print("📊 Pressure: {} hPa\n", .{weather.pressure});

    // Convert timestamps to readable format
    const sunrise_time = std.time.epoch.Epoch{ .secs = weather.sunrise };
    const sunset_time = std.time.epoch.Epoch{ .secs = weather.sunset };
    const sunrise_day = sunrise_time.getDaySeconds();
    const sunset_day = sunset_time.getDaySeconds();

    const sunrise_hour = sunrise_day / 3600;
    const sunrise_min = (sunrise_day % 3600) / 60;
    const sunset_hour = sunset_day / 3600;
    const sunset_min = (sunset_day % 3600) / 60;

    std.debug.print("🌅 Sunrise: {:0>2}:{:0>2}\n", .{ sunrise_hour, sunrise_min });
    std.debug.print("🌇 Sunset: {:0>2}:{:0>2}\n", .{ sunset_hour, sunset_min });
}

fn displayForecast(forecast: []root.weather.WeatherData) void {
    std.debug.print("\n=== 5-Day Forecast ===\n", .{});

    for (forecast, 0..) |data, i| {
        if (i >= 40) break; // Limit to first 40 entries (5 days * 8 entries per day)

        const emoji = root.weather.WeatherUtils.getWeatherEmoji(data.icon);
        const wind_dir = root.weather.WeatherUtils.getWindDirection(data.wind_direction);

        // Convert timestamp to readable format
        const timestamp = std.time.epoch.Epoch{ .secs = data.timestamp };
        const day_seconds = timestamp.getDaySeconds();
        const hour = day_seconds / 3600;
        const minute = (day_seconds % 3600) / 60;

        std.debug.print("{:0>2}:{:0>2} | {s} {d:.1}°C | {s} | 💧{}% | 🌪️{d:.1}m/s {s}\n", .{
            hour,
            minute,
            emoji,
            data.temperature,
            data.description,
            data.humidity,
            data.wind_speed,
            wind_dir,
        });
    }
}

test "weather utilities" {
    try std.testing.expectEqual(@as(f32, 0.0), root.weather.WeatherUtils.kelvinToCelsius(273.15));
    try std.testing.expectEqual(@as(f32, 32.0), root.weather.WeatherUtils.celsiusToFahrenheit(0.0));
    try std.testing.expectEqual(@as(f32, 0.0), root.weather.WeatherUtils.fahrenheitToCelsius(32.0));
    try std.testing.expectEqualStrings("N", root.weather.WeatherUtils.getWindDirection(0));
    try std.testing.expectEqualStrings("E", root.weather.WeatherUtils.getWindDirection(90));
    try std.testing.expectEqualStrings("S", root.weather.WeatherUtils.getWindDirection(180));
    try std.testing.expectEqualStrings("W", root.weather.WeatherUtils.getWindDirection(270));
}
