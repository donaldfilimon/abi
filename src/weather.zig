//! Weather Integration Module - Weather Data and Forecasting Services
//!
//! This module provides comprehensive weather data integration including:
//! - Real-time weather data from multiple providers
//! - Weather forecasting with configurable time ranges
//! - Location-based weather queries (city, coordinates)
//! - Weather data caching and optimization
//! - Multiple units system support (metric, imperial)
//! - Weather alerts and severe weather warnings
//! - Historical weather data retrieval
//! - Weather data visualization helpers

const std = @import("std");
const core = @import("core/mod.zig");
const root = @import("root");

/// Re-export commonly used types
pub const Allocator = core.Allocator;

/// Weather-specific error types
pub const WeatherError = error{
    InvalidApiKey,
    CityNotFound,
    NetworkError,
    InvalidResponse,
    RateLimited,
    ServiceUnavailable,
    InvalidCoordinates,
    Timeout,
    ParseError,
} || core.Error;

pub const WeatherData = struct {
    temperature: f32,
    feels_like: f32,
    humidity: u8,
    pressure: u16,
    description: []const u8,
    icon: []const u8,
    wind_speed: f32,
    wind_direction: u16,
    visibility: u32,
    sunrise: u64,
    sunset: u64,
    city: []const u8,
    country: []const u8,
    timestamp: u64,

    fn deinit(self: *WeatherData, allocator: std.mem.Allocator) void {
        allocator.free(self.description);
        allocator.free(self.icon);
        allocator.free(self.city);
        allocator.free(self.country);
    }
};

pub const WeatherConfig = struct {
    api_key: []const u8,
    base_url: []const u8 = "https://api.openweathermap.org/data/2.5",
    units: []const u8 = "metric",
    language: []const u8 = "en",
    timeout_seconds: u32 = 10,
};

pub const WeatherService = struct {
    config: WeatherConfig,
    allocator: std.mem.Allocator,

    fn init(allocator: std.mem.Allocator, config: WeatherConfig) WeatherService {
        return .{ .config = config, .allocator = allocator };
    }

    fn getCurrentWeather(self: *WeatherService, city: []const u8) !WeatherData {
        const url = try std.fmt.allocPrint(self.allocator, "{s}/weather?q={s}&appid={s}&units={s}&lang={s}", .{ self.config.base_url, city, self.config.api_key, self.config.units, self.config.language });
        defer self.allocator.free(url);

        const json = try self.fetchJson(url);
        defer self.allocator.free(json);

        return self.parseWeatherResponse(json);
    }

    fn getCurrentWeatherByCoords(self: *WeatherService, lat: f32, lon: f32) !WeatherData {
        const url = try std.fmt.allocPrint(self.allocator, "{s}/weather?lat={d}&lon={d}&appid={s}&units={s}&lang={s}", .{ self.config.base_url, lat, lon, self.config.api_key, self.config.units, self.config.language });
        defer self.allocator.free(url);

        const json = try self.fetchJson(url);
        defer self.allocator.free(json);

        return self.parseWeatherResponse(json);
    }

    fn getForecast(self: *WeatherService, city: []const u8) ![]WeatherData {
        const url = try std.fmt.allocPrint(self.allocator, "{s}/forecast?q={s}&appid={s}&units={s}&lang={s}", .{ self.config.base_url, city, self.config.api_key, self.config.units, self.config.language });
        defer self.allocator.free(url);

        const json = try self.fetchJson(url);
        defer self.allocator.free(json);

        return self.parseForecastResponse(json);
    }

    fn fetchJson(self: *WeatherService, url: []const u8) ![]u8 {
        const http_client = @import("http_client.zig");

        // Configure HTTP client with retries and timeouts
        var client = http_client.HttpClient.init(self.allocator, .{
            .connect_timeout_ms = 10000,
            .read_timeout_ms = 15000,
            .max_retries = 3,
            .initial_backoff_ms = 1000,
            .max_backoff_ms = 5000,
            .user_agent = "WDBX-Weather-Client/1.0",
            .follow_redirects = true,
            .verify_ssl = true,
            .verbose = false,
        });

        const response = try client.get(url);
        defer response.deinit();

        if (response.status_code != 200) {
            std.debug.print("Weather API error: HTTP {d}\n", .{response.status_code});
            return error.WeatherApiError;
        }

        // Return a copy of the response body since response.deinit() will free it
        return try self.allocator.dupe(u8, response.body);
    }

    fn parseWeatherResponse(self: *WeatherService, json_str: []const u8) !WeatherData {
        var parser = std.json.Parser.init(self.allocator, false);
        defer parser.deinit();

        var tree = try parser.parse(json_str);
        defer tree.deinit();

        const root_obj = tree.root.object;
        const main = root_obj.get("main").?.object;
        const weather = root_obj.get("weather").?.array.items[0].object;
        const wind = root_obj.get("wind").?.object;
        const sys = root_obj.get("sys").?.object;

        return WeatherData{
            .temperature = @floatCast(main.get("temp").?.Float),
            .feels_like = @floatCast(main.get("feels_like").?.Float),
            .humidity = @intCast(main.get("humidity").?.Integer),
            .pressure = @intCast(main.get("pressure").?.Integer),
            .description = try self.allocator.dupe(u8, weather.get("description").?.String),
            .icon = try self.allocator.dupe(u8, weather.get("icon").?.String),
            .wind_speed = @floatCast(wind.get("speed").?.Float),
            .wind_direction = @intCast(wind.get("deg").?.Integer),
            .visibility = @intCast(root_obj.get("visibility").?.Integer),
            .sunrise = @intCast(sys.get("sunrise").?.Integer),
            .sunset = @intCast(sys.get("sunset").?.Integer),
            .city = try self.allocator.dupe(u8, root_obj.get("name").?.String),
            .country = try self.allocator.dupe(u8, sys.get("country").?.String),
            .timestamp = @intCast(root_obj.get("dt").?.Integer),
        };
    }

    fn parseForecastResponse(self: *WeatherService, json_str: []const u8) ![]WeatherData {
        var parser = std.json.Parser.init(self.allocator, false);
        defer parser.deinit();

        var tree = try parser.parse(json_str);
        defer tree.deinit();

        const list = tree.root.object.get("list").?.array;
        var forecast = try std.ArrayList(WeatherData).initCapacity(self.allocator, list.items.len);
        defer forecast.deinit();

        for (list.items) |item| {
            try forecast.append(try self.parseForecastItem(item));
        }

        return forecast.toOwnedSlice();
    }

    fn parseForecastItem(self: *WeatherService, item: std.json.Value) !WeatherData {
        const obj = item.object;
        const main = obj.get("main").?.object;
        const weather = obj.get("weather").?.array.items[0].object;
        const wind = obj.get("wind").?.object;

        return WeatherData{
            .temperature = @floatCast(main.get("temp").?.Float),
            .feels_like = @floatCast(main.get("feels_like").?.Float),
            .humidity = @intCast(main.get("humidity").?.Integer),
            .pressure = @intCast(main.get("pressure").?.Integer),
            .description = try self.allocator.dupe(u8, weather.get("description").?.String),
            .icon = try self.allocator.dupe(u8, weather.get("icon").?.String),
            .wind_speed = @floatCast(wind.get("speed").?.Float),
            .wind_direction = @intCast(wind.get("deg").?.Integer),
            .visibility = @intCast(obj.get("visibility").?.Integer),
            .sunrise = 0,
            .sunset = 0,
            .city = "",
            .country = "",
            .timestamp = @intCast(obj.get("dt").?.Integer),
        };
    }
};

pub const WeatherUtils = struct {
    fn kelvinToCelsius(kelvin: f32) f32 {
        return kelvin - 273.15;
    }

    fn celsiusToFahrenheit(celsius: f32) f32 {
        return celsius * 9.0 / 5.0 + 32.0;
    }

    fn fahrenheitToCelsius(fahrenheit: f32) f32 {
        return (fahrenheit - 32.0) * 5.0 / 9.0;
    }

    fn getWindDirection(degrees: u16) []const u8 {
        const dirs = [_][]const u8{ "N", "NNE", "NE", "ENE", "E", "ESE", "SE", "SSE", "S", "SSW", "SW", "WSW", "W", "WNW", "NW", "NNW" };
        return dirs[@as(usize, @intCast((degrees + 11.25) / 22.5)) % 16];
    }

    fn formatWeatherJson(weather: WeatherData, allocator: std.mem.Allocator) ![]u8 {
        return std.json.stringifyAlloc(allocator, weather, .{});
    }

    fn getWeatherEmoji(icon: []const u8) []const u8 {
        return switch (icon[0]) {
            '0', '1' => "🌤️",
            '2' => "⛅",
            '3', '4' => "☁️",
            '5' => "🌧️",
            '6' => "🌨️",
            '7', '9' => "🌫️",
            '8' => "⛈️",
            else => "🌡️",
        };
    }
};
