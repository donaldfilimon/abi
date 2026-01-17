const std = @import("std");

const client = @import("client.zig");

pub const WeatherError = error{
    InvalidLocation,
};

pub const WeatherConfig = struct {
    base_url: []const u8 = "https://api.open-meteo.com/v1/forecast",
    include_current: bool = true,
};

pub const WeatherClient = struct {
    allocator: std.mem.Allocator,
    http_client: client.HttpClient,
    config: WeatherConfig,

    pub fn init(allocator: std.mem.Allocator, config: WeatherConfig) !WeatherClient {
        const http_client = try client.HttpClient.init(allocator);
        return .{
            .allocator = allocator,
            .http_client = http_client,
            .config = config,
        };
    }

    pub fn deinit(self: *WeatherClient) void {
        self.http_client.deinit();
        self.* = undefined;
    }

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

    pub fn freeResponse(self: *WeatherClient, response: client.Response) void {
        self.http_client.freeResponse(response);
    }
};

const Coordinates = struct {
    lat: f64,
    lon: f64,
};

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
