const std = @import("std");

const client = @import("client.zig");
const weather = @import("weather.zig");

pub const Response = client.Response;
pub const HttpClient = client.HttpClient;
pub const RequestOptions = client.RequestOptions;
pub const WeatherClient = weather.WeatherClient;
pub const WeatherConfig = weather.WeatherConfig;
pub const http = @import("../../shared/utils/http/mod.zig");

pub fn init(_: std.mem.Allocator) !void {}

pub fn deinit() void {}
