const std = @import("std");
const weather = @import("weather");

test "parse current weather JSON" {
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    const allocator = arena.allocator();

    var svc = try weather.WeatherService.init(allocator, .{ .api_key = "test" });
    defer svc.deinit();

    const sample = "{\"main\":{\"temp\":20.5,\"feels_like\":19.0,\"humidity\":60,\"pressure\":1012},\"weather\":[{\"description\":\"clear sky\",\"icon\":\"01d\"}],\"wind\":{\"speed\":3.2,\"deg\":180},\"visibility\":10000,\"sys\":{\"sunrise\":1,\"sunset\":2,\"country\":\"GB\"},\"name\":\"London\",\"dt\":123}";
    const data = try svc.testParseWeatherResponse(sample);
    defer data.deinit(allocator);
    try std.testing.expectEqual(@as(f32, 20.5), data.temperature);
    try std.testing.expectEqualStrings("London", data.city);
}

test "parse forecast JSON" {
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    const allocator = arena.allocator();

    var svc = try weather.WeatherService.init(allocator, .{ .api_key = "test" });
    defer svc.deinit();

    const sample = "{\"list\":[{\"main\":{\"temp\":15.0,\"feels_like\":14.0,\"humidity\":50,\"pressure\":1015},\"weather\":[{\"description\":\"clouds\",\"icon\":\"03d\"}],\"wind\":{\"speed\":2.1,\"deg\":90},\"visibility\":9000,\"dt\":456}],\"city\":{\"name\":\"Test\"}}";
    const items = try svc.testParseForecastResponse(sample);
    defer allocator.free(items);
    try std.testing.expect(items.len >= 1);
}
