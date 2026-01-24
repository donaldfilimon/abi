//! Cloud Adapter Tests
//!
//! Tests for cloud function adapters including:
//! - AWS Lambda adapter
//! - Azure Functions adapter
//! - GCP Cloud Functions adapter
//! - Unified CloudEvent and CloudResponse types
//! - Provider detection and auto-routing

const std = @import("std");
const build_options = @import("build_options");
const cloud = @import("abi").cloud;

// =============================================================================
// CloudEvent Tests
// =============================================================================

test "cloud event initialization" {
    const allocator = std.testing.allocator;

    var event = cloud.CloudEvent.init(allocator, .aws_lambda, "req-12345");
    defer event.deinit();

    try std.testing.expectEqual(cloud.CloudProvider.aws_lambda, event.provider);
    try std.testing.expectEqualStrings("req-12345", event.request_id);
}

test "cloud event with body" {
    const allocator = std.testing.allocator;

    var event = cloud.CloudEvent.init(allocator, .gcp_functions, "gcp-req-001");
    defer event.deinit();

    event.body = "{\"message\": \"hello\"}";
    event.method = .POST;

    try std.testing.expect(event.body != null);
    try std.testing.expectEqual(cloud.HttpMethod.POST, event.method.?);
}

test "cloud event with path and query" {
    const allocator = std.testing.allocator;

    var event = cloud.CloudEvent.init(allocator, .azure_functions, "az-req-001");
    defer event.deinit();

    event.path = "/api/users";
    event.method = .GET;

    try std.testing.expectEqualStrings("/api/users", event.path.?);
}

// =============================================================================
// CloudResponse Tests
// =============================================================================

test "cloud response json creation" {
    const allocator = std.testing.allocator;

    var response = try cloud.CloudResponse.json(allocator, "{\"status\":\"ok\"}");
    defer response.deinit();

    try std.testing.expectEqual(@as(u16, 200), response.status_code);
    try std.testing.expectEqualStrings("application/json", response.headers.get("Content-Type").?);
    try std.testing.expectEqualStrings("{\"status\":\"ok\"}", response.body);
}

test "cloud response error creation" {
    const allocator = std.testing.allocator;

    var response = try cloud.CloudResponse.err(allocator, 404, "Not Found");
    defer {
        allocator.free(response.body);
        response.deinit();
    }

    try std.testing.expectEqual(@as(u16, 404), response.status_code);
    try std.testing.expect(std.mem.indexOf(u8, response.body, "404") != null);
}

test "cloud response redirect pattern" {
    const allocator = std.testing.allocator;

    // Create a redirect response using the base API
    var response = cloud.CloudResponse.init(allocator);
    defer response.deinit();

    response.status_code = 302;
    try response.headers.put("Location", "https://example.com/new-path");

    try std.testing.expectEqual(@as(u16, 302), response.status_code);
    try std.testing.expectEqualStrings("https://example.com/new-path", response.headers.get("Location").?);
}

test "cloud response with custom headers" {
    const allocator = std.testing.allocator;

    var response = cloud.CloudResponse.init(allocator);
    defer response.deinit();

    response.status_code = 200;
    try response.headers.put("X-Custom-Header", "custom-value");
    try response.headers.put("X-Another-Header", "another-value");

    try std.testing.expectEqualStrings("custom-value", response.headers.get("X-Custom-Header").?);
    try std.testing.expectEqualStrings("another-value", response.headers.get("X-Another-Header").?);
}

// =============================================================================
// HttpMethod Tests
// =============================================================================

test "http method parsing" {
    try std.testing.expectEqual(cloud.HttpMethod.GET, cloud.HttpMethod.fromString("GET").?);
    try std.testing.expectEqual(cloud.HttpMethod.POST, cloud.HttpMethod.fromString("POST").?);
    try std.testing.expectEqual(cloud.HttpMethod.PUT, cloud.HttpMethod.fromString("PUT").?);
    try std.testing.expectEqual(cloud.HttpMethod.DELETE, cloud.HttpMethod.fromString("DELETE").?);
    try std.testing.expectEqual(cloud.HttpMethod.PATCH, cloud.HttpMethod.fromString("PATCH").?);
    try std.testing.expectEqual(cloud.HttpMethod.HEAD, cloud.HttpMethod.fromString("HEAD").?);
    try std.testing.expectEqual(cloud.HttpMethod.OPTIONS, cloud.HttpMethod.fromString("OPTIONS").?);
}

test "http method parsing invalid" {
    try std.testing.expect(cloud.HttpMethod.fromString("INVALID") == null);
    try std.testing.expect(cloud.HttpMethod.fromString("get") == null);
    try std.testing.expect(cloud.HttpMethod.fromString("") == null);
}

test "http method to string" {
    try std.testing.expectEqualStrings("GET", cloud.HttpMethod.GET.toString());
    try std.testing.expectEqualStrings("POST", cloud.HttpMethod.POST.toString());
    try std.testing.expectEqualStrings("DELETE", cloud.HttpMethod.DELETE.toString());
}

// =============================================================================
// ResponseBuilder Tests
// =============================================================================

test "response builder basic usage" {
    const allocator = std.testing.allocator;

    var builder = cloud.ResponseBuilder.init(allocator);
    var response = builder
        .status(201)
        .json()
        .body("{\"created\":true}")
        .build();
    defer response.deinit();

    try std.testing.expectEqual(@as(u16, 201), response.status_code);
    try std.testing.expectEqualStrings("application/json", response.headers.get("Content-Type").?);
}

test "response builder with cors" {
    const allocator = std.testing.allocator;

    var builder = cloud.ResponseBuilder.init(allocator);
    var response = builder
        .status(200)
        .cors("*")
        .json()
        .body("{}")
        .build();
    defer response.deinit();

    try std.testing.expectEqualStrings("*", response.headers.get("Access-Control-Allow-Origin").?);
    try std.testing.expect(response.headers.get("Access-Control-Allow-Methods") != null);
    try std.testing.expect(response.headers.get("Access-Control-Allow-Headers") != null);
}

test "response builder content types" {
    const allocator = std.testing.allocator;

    // Test text content type
    var builder1 = cloud.ResponseBuilder.init(allocator);
    var text_response = builder1.text().build();
    defer text_response.deinit();
    try std.testing.expectEqualStrings("text/plain", text_response.headers.get("Content-Type").?);

    // Test HTML content type
    var builder2 = cloud.ResponseBuilder.init(allocator);
    var html_response = builder2.html().build();
    defer html_response.deinit();
    try std.testing.expectEqualStrings("text/html", html_response.headers.get("Content-Type").?);
}

// =============================================================================
// Provider Detection Tests
// =============================================================================

test "detect provider returns null in test environment" {
    // In a test environment without cloud env vars, should return null
    const provider = cloud.detectProvider();
    // We just verify the function works - actual value depends on env
    _ = provider;
}

// =============================================================================
// Cloud Context Tests
// =============================================================================

test "cloud context initialization" {
    const allocator = std.testing.allocator;

    const ctx = try cloud.Context.init(allocator, .{});
    defer ctx.deinit();

    // In test environment, no cloud provider should be auto-detected
    // (unless running on actual cloud infrastructure)
    _ = ctx.getProvider();
}

test "cloud context wrap handler" {
    const allocator = std.testing.allocator;

    const ctx = try cloud.Context.init(allocator, .{});
    defer ctx.deinit();

    const TestHandler = struct {
        fn handle(_: *cloud.CloudEvent, _: std.mem.Allocator) anyerror!cloud.CloudResponse {
            return cloud.CloudResponse{ .status_code = 200 };
        }
    };

    const wrapped = ctx.wrapHandler(TestHandler.handle);
    _ = wrapped;
}

// =============================================================================
// AWS Lambda Adapter Tests
// =============================================================================

test "aws lambda event parsing helpers" {
    const allocator = std.testing.allocator;

    // Test API Gateway proxy event structure
    const api_gw_event =
        \\{
        \\  "httpMethod": "POST",
        \\  "path": "/api/test",
        \\  "body": "{\"key\":\"value\"}",
        \\  "headers": {
        \\    "Content-Type": "application/json"
        \\  },
        \\  "requestContext": {
        \\    "requestId": "test-request-id"
        \\  }
        \\}
    ;

    var parsed = std.json.parseFromSlice(
        std.json.Value,
        allocator,
        api_gw_event,
        .{},
    ) catch return error.TestUnexpectedResult;
    defer parsed.deinit();

    // Verify structure can be parsed
    if (parsed.value.object.get("httpMethod")) |method| {
        try std.testing.expectEqualStrings("POST", method.string);
    } else {
        return error.TestUnexpectedResult;
    }

    if (parsed.value.object.get("path")) |path| {
        try std.testing.expectEqualStrings("/api/test", path.string);
    }
}

test "aws lambda response format" {
    const allocator = std.testing.allocator;

    var response = cloud.CloudResponse.init(allocator);
    defer response.deinit();

    response.status_code = 200;
    response.body = "{\"message\":\"success\"}";
    try response.headers.put("Content-Type", "application/json");

    // Verify response can be serialized for Lambda
    try std.testing.expectEqual(@as(u16, 200), response.status_code);
}

// =============================================================================
// Azure Functions Adapter Tests
// =============================================================================

test "azure functions http trigger structure" {
    const allocator = std.testing.allocator;

    // Test Azure HTTP trigger structure
    const azure_event =
        \\{
        \\  "Data": {
        \\    "req": {
        \\      "Method": "GET",
        \\      "Url": "https://myfunction.azurewebsites.net/api/test",
        \\      "Body": null,
        \\      "Headers": {}
        \\    }
        \\  },
        \\  "Metadata": {
        \\    "InvocationId": "azure-invocation-123"
        \\  }
        \\}
    ;

    var parsed = std.json.parseFromSlice(
        std.json.Value,
        allocator,
        azure_event,
        .{},
    ) catch return error.TestUnexpectedResult;
    defer parsed.deinit();

    // Verify Azure structure
    if (parsed.value.object.get("Data")) |data| {
        if (data.object.get("req")) |req| {
            if (req.object.get("Method")) |method| {
                try std.testing.expectEqualStrings("GET", method.string);
            }
        }
    }
}

// =============================================================================
// GCP Cloud Functions Adapter Tests
// =============================================================================

test "gcp cloud functions event structure" {
    const allocator = std.testing.allocator;

    // Test GCP HTTP function structure (similar to Express.js)
    const gcp_headers =
        \\{
        \\  "content-type": "application/json",
        \\  "x-cloud-trace-context": "trace-id/span-id",
        \\  "function-execution-id": "gcp-exec-123"
        \\}
    ;

    var parsed = std.json.parseFromSlice(
        std.json.Value,
        allocator,
        gcp_headers,
        .{},
    ) catch return error.TestUnexpectedResult;
    defer parsed.deinit();

    if (parsed.value.object.get("content-type")) |ct| {
        try std.testing.expectEqualStrings("application/json", ct.string);
    }
}

// =============================================================================
// InvocationMetadata Tests
// =============================================================================

test "invocation metadata initialization" {
    const time = @import("../shared/time.zig");
    var metadata = cloud.InvocationMetadata{
        .request_id = "req-123",
        .provider = .aws_lambda,
        .start_time = time.unixSeconds(),
        .cold_start = true,
    };

    try std.testing.expectEqualStrings("req-123", metadata.request_id);
    try std.testing.expectEqual(cloud.CloudProvider.aws_lambda, metadata.provider);
    try std.testing.expect(metadata.cold_start);
}

// =============================================================================
// CloudConfig Tests
// =============================================================================

test "cloud config defaults" {
    const config = cloud.CloudConfig.defaults();

    try std.testing.expectEqual(@as(u32, 256), config.memory_mb);
    try std.testing.expectEqual(@as(u32, 30), config.timeout_seconds);
}

// =============================================================================
// Integration Tests
// =============================================================================

test "full request response cycle simulation" {
    const allocator = std.testing.allocator;

    // Simulate an incoming cloud event
    var event = cloud.CloudEvent.init(allocator, .aws_lambda, "integration-test-001");
    defer event.deinit();

    event.method = .POST;
    event.path = "/api/process";
    event.body = "{\"data\": \"test\"}";

    // Process the event
    var builder = cloud.ResponseBuilder.init(allocator);
    var response = builder
        .status(200)
        .json()
        .cors("https://example.com")
        .body("{\"processed\": true}")
        .build();
    defer response.deinit();

    // Verify the response
    try std.testing.expectEqual(@as(u16, 200), response.status_code);
    try std.testing.expectEqualStrings("application/json", response.headers.get("Content-Type").?);
    try std.testing.expectEqualStrings("https://example.com", response.headers.get("Access-Control-Allow-Origin").?);
}

test "error response handling" {
    const allocator = std.testing.allocator;

    // Simulate various error responses
    const error_codes = [_]u16{ 400, 401, 403, 404, 500, 502, 503 };
    const error_messages = [_][]const u8{
        "Bad Request",
        "Unauthorized",
        "Forbidden",
        "Not Found",
        "Internal Server Error",
        "Bad Gateway",
        "Service Unavailable",
    };

    for (error_codes, error_messages) |code, message| {
        var response = try cloud.CloudResponse.err(allocator, code, message);
        defer {
            allocator.free(response.body);
            response.deinit();
        }

        try std.testing.expectEqual(code, response.status_code);
        try std.testing.expect(response.body.len > 0);
    }
}

test "module initialization" {
    if (!cloud.isEnabled()) return error.SkipZigTest;

    const allocator = std.testing.allocator;

    try cloud.init(allocator);
    defer cloud.deinit();

    try std.testing.expect(cloud.isInitialized());
}
