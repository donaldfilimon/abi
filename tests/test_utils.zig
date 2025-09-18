//! Comprehensive tests for the utils module

const std = @import("std");
const utils = @import("../src/utils.zig");

test "Config initialization" {
    const testing = std.testing;

    // Test default config
    const default_config = utils.Config{};
    try testing.expectEqualStrings("abi-ai", default_config.name);
    try testing.expectEqual(@as(u32, 1), default_config.version);
    try testing.expect(!default_config.debug_mode);

    // Test custom config
    const custom_config = utils.Config.init("test-app");
    try testing.expectEqualStrings("test-app", custom_config.name);
}

test "DefinitionType toString" {
    const testing = std.testing;

    try testing.expectEqualStrings("core", utils.DefinitionType.core.toString());
    try testing.expectEqualStrings("database", utils.DefinitionType.database.toString());
    try testing.expectEqualStrings("neural", utils.DefinitionType.neural.toString());
    try testing.expectEqualStrings("web", utils.DefinitionType.web.toString());
    try testing.expectEqualStrings("cli", utils.DefinitionType.cli.toString());
}

test "HttpStatus values and phrases" {
    const testing = std.testing;

    // Test all status codes and phrases
    try testing.expectEqual(@as(u16, 200), @intFromEnum(utils.HttpStatus.ok));
    try testing.expectEqualStrings("OK", utils.HttpStatus.ok.phrase());

    try testing.expectEqual(@as(u16, 404), @intFromEnum(utils.HttpStatus.not_found));
    try testing.expectEqualStrings("Not Found", utils.HttpStatus.not_found.phrase());

    try testing.expectEqual(@as(u16, 500), @intFromEnum(utils.HttpStatus.internal_server_error));
    try testing.expectEqualStrings("Internal Server Error", utils.HttpStatus.internal_server_error.phrase());

    try testing.expectEqual(@as(u16, 201), @intFromEnum(utils.HttpStatus.created));
    try testing.expectEqualStrings("Created", utils.HttpStatus.created.phrase());

    try testing.expectEqual(@as(u16, 429), @intFromEnum(utils.HttpStatus.too_many_requests));
    try testing.expectEqualStrings("Too Many Requests", utils.HttpStatus.too_many_requests.phrase());
}

test "HttpMethod comprehensive tests" {
    const testing = std.testing;

    // Test fromString with various cases
    try testing.expectEqual(utils.HttpMethod.GET, utils.HttpMethod.fromString("GET").?);
    try testing.expectEqual(utils.HttpMethod.POST, utils.HttpMethod.fromString("post").?);
    try testing.expectEqual(utils.HttpMethod.PUT, utils.HttpMethod.fromString("Put").?);
    try testing.expectEqual(utils.HttpMethod.DELETE, utils.HttpMethod.fromString("delete").?);
    try testing.expectEqual(utils.HttpMethod.PATCH, utils.HttpMethod.fromString("PATCH").?);
    try testing.expectEqual(utils.HttpMethod.OPTIONS, utils.HttpMethod.fromString("options").?);
    try testing.expectEqual(utils.HttpMethod.HEAD, utils.HttpMethod.fromString("HEAD").?);
    try testing.expectEqual(utils.HttpMethod.TRACE, utils.HttpMethod.fromString("trace").?);
    try testing.expectEqual(utils.HttpMethod.CONNECT, utils.HttpMethod.fromString("Connect").?);

    // Test invalid method
    try testing.expectEqual(@as(?utils.HttpMethod, null), utils.HttpMethod.fromString("INVALID"));
    try testing.expectEqual(@as(?utils.HttpMethod, null), utils.HttpMethod.fromString(""));

    // Test toString
    try testing.expectEqualStrings("GET", utils.HttpMethod.GET.toString());
    try testing.expectEqualStrings("POST", utils.HttpMethod.POST.toString());
    try testing.expectEqualStrings("PUT", utils.HttpMethod.PUT.toString());
    try testing.expectEqualStrings("DELETE", utils.HttpMethod.DELETE.toString());
    try testing.expectEqualStrings("PATCH", utils.HttpMethod.PATCH.toString());
    try testing.expectEqualStrings("OPTIONS", utils.HttpMethod.OPTIONS.toString());
    try testing.expectEqualStrings("HEAD", utils.HttpMethod.HEAD.toString());
    try testing.expectEqualStrings("TRACE", utils.HttpMethod.TRACE.toString());
    try testing.expectEqualStrings("CONNECT", utils.HttpMethod.CONNECT.toString());
}

test "Headers management" {
    const testing = std.testing;
    const allocator = testing.allocator;

    var headers = utils.Headers.init(allocator);
    defer headers.deinit();

    // Test setting headers
    try headers.set("Content-Type", "application/json");
    try headers.set("Authorization", "Bearer token123");
    try headers.set("X-Custom", "custom-value");

    // Test getting headers
    try testing.expectEqualStrings("application/json", headers.get("Content-Type").?);
    try testing.expectEqualStrings("Bearer token123", headers.get("Authorization").?);
    try testing.expectEqualStrings("custom-value", headers.get("X-Custom").?);

    // Test getting non-existent header
    try testing.expectEqual(@as(?[]const u8, null), headers.get("Non-Existent"));

    // Test removing headers
    try testing.expect(headers.remove("Content-Type"));
    try testing.expectEqual(@as(?[]const u8, null), headers.get("Content-Type"));

    // Test removing non-existent header
    try testing.expect(!headers.remove("Non-Existent"));
}

test "HttpRequest lifecycle" {
    const testing = std.testing;
    const allocator = testing.allocator;

    var request = utils.HttpRequest.init(allocator, .GET, "/api/test");
    defer request.deinit();

    // Test initial state
    try testing.expectEqual(utils.HttpMethod.GET, request.method);
    try testing.expectEqualStrings("/api/test", request.path);
    try testing.expectEqual(@as(?[]const u8, null), request.body);

    // Test headers
    try request.headers.set("Accept", "application/json");
    try testing.expectEqualStrings("application/json", request.headers.get("Accept").?);

    // Test query params
    try request.query_params.put("limit", "10");
    try request.query_params.put("offset", "0");
    try testing.expectEqualStrings("10", request.query_params.get("limit").?);
    try testing.expectEqualStrings("0", request.query_params.get("offset").?);

    // Test with body
    const body = "test body";
    var request_with_body = utils.HttpRequest.init(allocator, .POST, "/api/data");
    defer request_with_body.deinit();
    request_with_body.body = body;
    try testing.expectEqualStrings(body, request_with_body.body.?);
}

test "HttpResponse lifecycle" {
    const testing = std.testing;
    const allocator = testing.allocator;

    var response = utils.HttpResponse.init(allocator, .ok);
    defer response.deinit();

    // Test initial state
    try testing.expectEqual(utils.HttpStatus.ok, response.status);
    try testing.expectEqual(@as(?[]const u8, null), response.body);

    // Test setting content types
    try response.setJson();
    try testing.expectEqualStrings("application/json", response.headers.get("Content-Type").?);

    try response.setText();
    try testing.expectEqualStrings("text/plain", response.headers.get("Content-Type").?);

    try response.setHtml();
    try testing.expectEqualStrings("text/html", response.headers.get("Content-Type").?);

    // Test custom content type
    try response.setContentType("application/xml");
    try testing.expectEqualStrings("application/xml", response.headers.get("Content-Type").?);

    // Test with body
    const body = "{\"message\": \"success\"}";
    var response_with_body = utils.HttpResponse.init(allocator, .created);
    defer response_with_body.deinit();
    response_with_body.body = body;
    try testing.expectEqual(utils.HttpStatus.created, response_with_body.status);
    try testing.expectEqualStrings(body, response_with_body.body.?);
}

test "StringUtils comprehensive tests" {
    const testing = std.testing;
    const allocator = testing.allocator;

    // Test isEmptyOrWhitespace
    try testing.expect(utils.StringUtils.isEmptyOrWhitespace(""));
    try testing.expect(utils.StringUtils.isEmptyOrWhitespace("   "));
    try testing.expect(utils.StringUtils.isEmptyOrWhitespace("\t\r\n"));
    try testing.expect(utils.StringUtils.isEmptyOrWhitespace(" \t  \r\n "));
    try testing.expect(!utils.StringUtils.isEmptyOrWhitespace("hello"));
    try testing.expect(!utils.StringUtils.isEmptyOrWhitespace("  hello  "));
    try testing.expect(!utils.StringUtils.isEmptyOrWhitespace("hello world"));

    // Test toLower
    const lower = try utils.StringUtils.toLower(allocator, "HELLO WORLD");
    defer allocator.free(lower);
    try testing.expectEqualStrings("hello world", lower);

    const lower_mixed = try utils.StringUtils.toLower(allocator, "HeLLo WoRLd");
    defer allocator.free(lower_mixed);
    try testing.expectEqualStrings("hello world", lower_mixed);

    const lower_empty = try utils.StringUtils.toLower(allocator, "");
    defer allocator.free(lower_empty);
    try testing.expectEqualStrings("", lower_empty);

    // Test toUpper
    const upper = try utils.StringUtils.toUpper(allocator, "hello world");
    defer allocator.free(upper);
    try testing.expectEqualStrings("HELLO WORLD", upper);

    const upper_mixed = try utils.StringUtils.toUpper(allocator, "HeLLo WoRLd");
    defer allocator.free(upper_mixed);
    try testing.expectEqualStrings("HELLO WORLD", upper_mixed);

    const upper_empty = try utils.StringUtils.toUpper(allocator, "");
    defer allocator.free(upper_empty);
    try testing.expectEqualStrings("", upper_empty);
}

test "ArrayUtils comprehensive tests" {
    const testing = std.testing;

    // Test with integers
    const int_arr = [_]i32{ 1, 2, 3, 4, 5 };
    try testing.expect(utils.ArrayUtils.contains(i32, &int_arr, 3));
    try testing.expect(utils.ArrayUtils.contains(i32, &int_arr, 1));
    try testing.expect(utils.ArrayUtils.contains(i32, &int_arr, 5));
    try testing.expect(!utils.ArrayUtils.contains(i32, &int_arr, 0));
    try testing.expect(!utils.ArrayUtils.contains(i32, &int_arr, 6));

    try testing.expectEqual(@as(?usize, 0), utils.ArrayUtils.indexOf(i32, &int_arr, 1));
    try testing.expectEqual(@as(?usize, 2), utils.ArrayUtils.indexOf(i32, &int_arr, 3));
    try testing.expectEqual(@as(?usize, 4), utils.ArrayUtils.indexOf(i32, &int_arr, 5));
    try testing.expectEqual(@as(?usize, null), utils.ArrayUtils.indexOf(i32, &int_arr, 0));

    // Test with strings
    const str_arr = [_][]const u8{ "apple", "banana", "cherry" };
    try testing.expect(utils.ArrayUtils.contains([]const u8, &str_arr, "banana"));
    try testing.expect(!utils.ArrayUtils.contains([]const u8, &str_arr, "grape"));

    try testing.expectEqual(@as(?usize, 1), utils.ArrayUtils.indexOf([]const u8, &str_arr, "banana"));
    try testing.expectEqual(@as(?usize, null), utils.ArrayUtils.indexOf([]const u8, &str_arr, "grape"));

    // Test with empty array
    const empty_arr = [_]i32{};
    try testing.expect(!utils.ArrayUtils.contains(i32, &empty_arr, 1));
    try testing.expectEqual(@as(?usize, null), utils.ArrayUtils.indexOf(i32, &empty_arr, 1));
}

test "TimeUtils comprehensive tests" {
    const testing = std.testing;
    const allocator = testing.allocator;

    // Test timestamp functions (values should be reasonable)
    const ms = utils.TimeUtils.nowMs();
    const us = utils.TimeUtils.nowUs();
    const ns = utils.TimeUtils.nowNs();

    // All timestamps should be positive and reasonable
    try testing.expect(ms > 0);
    try testing.expect(us > 0);
    try testing.expect(ns > 0);

    // Microseconds should be larger than milliseconds (roughly)
    try testing.expect(us > ms * 1000);
    try testing.expect(ns > us * 1000);

    // Test formatDuration
    const duration_1ms = std.time.ns_per_ms;
    const formatted_1ms = try utils.TimeUtils.formatDuration(allocator, duration_1ms);
    defer allocator.free(formatted_1ms);
    try testing.expect(std.mem.indexOf(u8, formatted_1ms, "ms") != null);

    const duration_100us = 100 * std.time.ns_per_us;
    const formatted_100us = try utils.TimeUtils.formatDuration(allocator, duration_100us);
    defer allocator.free(formatted_100us);
    try testing.expect(std.mem.indexOf(u8, formatted_100us, "μs") != null);

    const duration_500ns = 500;
    const formatted_500ns = try utils.TimeUtils.formatDuration(allocator, duration_500ns);
    defer allocator.free(formatted_500ns);
    try testing.expect(std.mem.indexOf(u8, formatted_500ns, "ns") != null);

    // Test edge case: zero duration
    const formatted_zero = try utils.TimeUtils.formatDuration(allocator, 0);
    defer allocator.free(formatted_zero);
    try testing.expect(std.mem.indexOf(u8, formatted_zero, "ns") != null);
}

test "Version constants" {
    const testing = std.testing;

    // Test version structure
    try testing.expectEqual(@as(u32, 1), utils.VERSION.major);
    try testing.expectEqual(@as(u32, 0), utils.VERSION.minor);
    try testing.expectEqual(@as(u32, 0), utils.VERSION.patch);
    try testing.expectEqualStrings("alpha", utils.VERSION.pre_release);
}

test "HTTP structures integration" {
    const testing = std.testing;
    const allocator = testing.allocator;

    // Create a complete HTTP request
    var request = utils.HttpRequest.init(allocator, .POST, "/api/users");
    defer request.deinit();

    try request.headers.set("Content-Type", "application/json");
    try request.headers.set("Authorization", "Bearer token123");
    try request.query_params.put("limit", "10");
    try request.query_params.put("sort", "name");

    const request_body = "{\"name\": \"John\", \"age\": 30}";
    request.body = request_body;

    // Verify request
    try testing.expectEqual(utils.HttpMethod.POST, request.method);
    try testing.expectEqualStrings("/api/users", request.path);
    try testing.expectEqualStrings("application/json", request.headers.get("Content-Type").?);
    try testing.expectEqualStrings("Bearer token123", request.headers.get("Authorization").?);
    try testing.expectEqualStrings("10", request.query_params.get("limit").?);
    try testing.expectEqualStrings("name", request.query_params.get("sort").?);
    try testing.expectEqualStrings(request_body, request.body.?);

    // Create a complete HTTP response
    var response = utils.HttpResponse.init(allocator, .created);
    defer response.deinit();

    try response.setJson();
    try response.headers.set("Location", "/api/users/123");

    const response_body = "{\"id\": 123, \"name\": \"John\", \"age\": 30}";
    response.body = response_body;

    // Verify response
    try testing.expectEqual(utils.HttpStatus.created, response.status);
    try testing.expectEqualStrings("application/json", response.headers.get("Content-Type").?);
    try testing.expectEqualStrings("/api/users/123", response.headers.get("Location").?);
    try testing.expectEqualStrings(response_body, response.body.?);
}

test "Edge cases and error handling" {
    const testing = std.testing;
    const allocator = testing.allocator;

    // Test empty strings in HTTP methods
    try testing.expectEqual(@as(?utils.HttpMethod, null), utils.HttpMethod.fromString(""));
    try testing.expectEqual(@as(?utils.HttpMethod, null), utils.HttpMethod.fromString("   "));

    // Test case sensitivity in HTTP methods (should be case insensitive)
    try testing.expectEqual(utils.HttpMethod.GET, utils.HttpMethod.fromString("get").?);
    try testing.expectEqual(utils.HttpMethod.POST, utils.HttpMethod.fromString("Post").?);

    // Test headers with empty values
    var headers = utils.Headers.init(allocator);
    defer headers.deinit();

    try headers.set("empty", "");
    try testing.expectEqualStrings("", headers.get("empty").?);

    // Test removing non-existent header
    try testing.expect(!headers.remove("nonexistent"));

    // Test array operations with single element
    const single_arr = [_]i32{42};
    try testing.expect(utils.ArrayUtils.contains(i32, &single_arr, 42));
    try testing.expect(!utils.ArrayUtils.contains(i32, &single_arr, 0));
    try testing.expectEqual(@as(?usize, 0), utils.ArrayUtils.indexOf(i32, &single_arr, 42));
}

test "Memory management and cleanup" {
    const testing = std.testing;
    const allocator = testing.allocator;

    // Test Headers cleanup
    var headers = utils.Headers.init(allocator);
    try headers.set("test1", "value1");
    try headers.set("test2", "value2");
    headers.deinit(); // Should clean up all allocations

    // Test HttpRequest cleanup
    var request = utils.HttpRequest.init(allocator, .GET, "/test");
    try request.headers.set("test", "value");
    try request.query_params.put("param", "value");
    request.deinit(); // Should clean up headers and query params

    // Test HttpResponse cleanup
    var response = utils.HttpResponse.init(allocator, .ok);
    try response.headers.set("test", "value");
    response.deinit(); // Should clean up headers
}
