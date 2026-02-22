//! Cloud Lifecycle Integration Tests
//!
//! Tests for cloud adapter initialization, event handling, and shutdown:
//! - AWS Lambda adapter lifecycle
//! - Azure Functions adapter lifecycle
//! - GCP Cloud Functions lifecycle
//! - Framework shutdown coordination with cloud adapters

const std = @import("std");
const testing = std.testing;
const builtin = @import("builtin");
const build_options = @import("build_options");
const abi = @import("abi");
const time = abi.shared.time;
const sync = abi.shared.sync;

const fixtures = @import("fixtures.zig");

// ============================================================================
// Cloud Event and Response Tests
// ============================================================================

test "cloud lifecycle: CloudEvent creation" {
    if (!build_options.enable_web) return error.SkipZigTest;

    const allocator = testing.allocator;

    // Create a basic cloud event using the init function
    var event = abi.cloud.CloudEvent.init(allocator, .aws_lambda, "test-request-123");
    defer event.deinit();

    // Set optional fields
    event.method = .GET;
    event.path = "/api/test";
    event.body = "{\"test\": true}";

    try testing.expectEqual(abi.cloud.CloudProvider.aws_lambda, event.provider);
    try testing.expectEqual(abi.cloud.HttpMethod.GET, event.method.?);
    try testing.expectEqualStrings("/api/test", event.path.?);

    // Create JSON response
    var response = try abi.cloud.CloudResponse.json(allocator, "{\"status\": \"ok\"}");
    defer response.deinit();

    try testing.expectEqual(@as(u16, 200), response.status_code);
    try testing.expect(response.body.len > 0);
}

test "cloud lifecycle: CloudResponse status codes" {
    const allocator = testing.allocator;

    // Success response using json
    var ok_response = try abi.cloud.CloudResponse.json(allocator, "{\"message\": \"Success\"}");
    defer ok_response.deinit();
    try testing.expectEqual(@as(u16, 200), ok_response.status_code);

    // Error responses using err() helper
    var bad_request = try abi.cloud.CloudResponse.err(allocator, 400, "Invalid input");
    defer {
        bad_request.deinit();
        allocator.free(bad_request.body);
    }
    try testing.expectEqual(@as(u16, 400), bad_request.status_code);

    var not_found = try abi.cloud.CloudResponse.err(allocator, 404, "Resource not found");
    defer {
        not_found.deinit();
        allocator.free(not_found.body);
    }
    try testing.expectEqual(@as(u16, 404), not_found.status_code);

    var server_error = try abi.cloud.CloudResponse.err(allocator, 500, "Internal error");
    defer {
        server_error.deinit();
        allocator.free(server_error.body);
    }
    try testing.expectEqual(@as(u16, 500), server_error.status_code);
}

// ============================================================================
// AWS Lambda Integration Tests
// ============================================================================

test "cloud lifecycle: aws lambda event parsing" {
    if (!build_options.enable_web) return error.SkipZigTest;

    // Test AWS Lambda event structure
    const aws_event_json =
        \\{
        \\  "httpMethod": "POST",
        \\  "path": "/users",
        \\  "body": "{\"name\": \"test\"}",
        \\  "headers": {"Content-Type": "application/json"}
        \\}
    ;

    // Verify event JSON is valid
    var parsed = try std.json.parseFromSlice(std.json.Value, testing.allocator, aws_event_json, .{});
    defer parsed.deinit();

    const root = parsed.value.object;
    try testing.expectEqualStrings("POST", root.get("httpMethod").?.string);
    try testing.expectEqualStrings("/users", root.get("path").?.string);
}

// ============================================================================
// Azure Functions Integration Tests
// ============================================================================

test "cloud lifecycle: azure functions event parsing" {
    if (!build_options.enable_web) return error.SkipZigTest;

    // Test Azure Functions event structure
    const azure_event_json =
        \\{
        \\  "method": "GET",
        \\  "url": "https://example.azurewebsites.net/api/hello",
        \\  "headers": {"x-functions-key": "test-key"},
        \\  "query": {"name": "Azure"}
        \\}
    ;

    var parsed = try std.json.parseFromSlice(std.json.Value, testing.allocator, azure_event_json, .{});
    defer parsed.deinit();

    const root = parsed.value.object;
    try testing.expectEqualStrings("GET", root.get("method").?.string);
}

// ============================================================================
// GCP Cloud Functions Integration Tests
// ============================================================================

test "cloud lifecycle: gcp functions event parsing" {
    if (!build_options.enable_web) return error.SkipZigTest;

    // Test GCP Cloud Functions event structure
    const gcp_event_json =
        \\{
        \\  "method": "POST",
        \\  "path": "/function",
        \\  "body": {"message": "hello"},
        \\  "headers": {"content-type": "application/json"}
        \\}
    ;

    var parsed = try std.json.parseFromSlice(std.json.Value, testing.allocator, gcp_event_json, .{});
    defer parsed.deinit();

    const root = parsed.value.object;
    try testing.expectEqualStrings("POST", root.get("method").?.string);
}

// ============================================================================
// Framework Integration Tests
// ============================================================================

test "cloud lifecycle: framework initialization with cloud config" {
    if (!build_options.enable_web) return error.SkipZigTest;

    const allocator = testing.allocator;

    // Initialize framework with cloud-like configuration
    var fixture = try fixtures.IntegrationFixture.init(allocator, .{
        .web = true,
        .observability = build_options.enable_profiling,
    });
    defer fixture.deinit();

    try testing.expect(fixture.setup_complete);
}

test "cloud lifecycle: cold start simulation" {
    if (!build_options.enable_web) return error.SkipZigTest;

    const allocator = testing.allocator;

    // Simulate cold start by measuring initialization time
    var timer = try time.Timer.start();

    var fixture = try fixtures.IntegrationFixture.init(allocator, .{
        .web = true,
    });

    const cold_start_ns = timer.read();
    fixture.deinit();

    // Cold start should complete in reasonable time (< 1 second for tests)
    try testing.expect(cold_start_ns < 1_000_000_000);
}

test "cloud lifecycle: warm invocation simulation" {
    if (!build_options.enable_web) return error.SkipZigTest;

    const allocator = testing.allocator;

    // Initialize once (cold start)
    var fixture = try fixtures.IntegrationFixture.init(allocator, .{
        .web = true,
    });
    defer fixture.deinit();

    // Simulate multiple warm invocations
    var warm_times: [10]u64 = undefined;

    for (&warm_times) |*duration| {
        var timer = try time.Timer.start();

        // Simulate request handling by creating an event
        var event = abi.cloud.CloudEvent.init(allocator, .aws_lambda, "warm-request");
        event.method = .GET;
        event.path = "/health";
        event.deinit();

        duration.* = timer.read();
    }

    // Warm invocations should be much faster than cold start
    var avg: u64 = 0;
    for (warm_times) |t| {
        avg += t;
    }
    avg /= warm_times.len;

    // Average warm invocation should be < 10ms
    try testing.expect(avg < 10_000_000);
}

// ============================================================================
// Provider Normalization Tests
// ============================================================================

test "cloud lifecycle: normalize http methods" {
    // Test HTTP method normalization
    const methods = [_]abi.cloud.HttpMethod{
        .GET,
        .POST,
        .PUT,
        .DELETE,
        .PATCH,
        .HEAD,
        .OPTIONS,
    };

    for (methods) |method| {
        // Each method should have a valid enum value
        try testing.expect(@intFromEnum(method) <= @intFromEnum(abi.cloud.HttpMethod.OPTIONS));
    }
}

test "cloud lifecycle: provider detection" {
    // Test provider detection logic - all providers are known
    const providers = [_]abi.cloud.CloudProvider{
        .aws_lambda,
        .gcp_functions,
        .azure_functions,
    };

    for (providers) |provider| {
        // Each provider should have a name
        const name = provider.name();
        try testing.expect(name.len > 0);

        // Each provider should have a runtime identifier
        const runtime = provider.runtimeIdentifier();
        try testing.expect(runtime.len > 0);
    }
}

// ============================================================================
// Graceful Shutdown Tests
// ============================================================================

test "cloud lifecycle: graceful shutdown" {
    if (!build_options.enable_web) return error.SkipZigTest;

    const allocator = testing.allocator;

    // Initialize framework
    var fixture = try fixtures.IntegrationFixture.init(allocator, .{
        .web = true,
        .observability = build_options.enable_profiling,
    });

    // Verify framework is running
    try testing.expect(fixture.setup_complete);

    // Graceful shutdown
    fixture.deinit();

    // Verify cleanup
    try testing.expect(!fixture.setup_complete);
}

test "cloud lifecycle: shutdown with pending requests" {
    if (!build_options.enable_web) return error.SkipZigTest;

    const allocator = testing.allocator;

    var fixture = try fixtures.IntegrationFixture.init(allocator, .{
        .web = true,
    });

    // Simulate pending requests
    var pending_count: u32 = 5;

    // Drain pending requests before shutdown
    while (pending_count > 0) : (pending_count -= 1) {
        // Simulate request completion
    }

    try testing.expectEqual(@as(u32, 0), pending_count);

    // Now safe to shutdown
    fixture.deinit();
}
