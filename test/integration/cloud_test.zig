//! Integration Tests: Cloud Functions
//!
//! Verifies the cloud module's public types, provider detection,
//! response builders, event lifecycle, and configuration through
//! the `abi.cloud` facade.

const std = @import("std");
const abi = @import("abi");

const cloud = abi.cloud;

// ── Type availability ──────────────────────────────────────────────────

test "cloud: core types are accessible" {
    _ = cloud.CloudEvent;
    _ = cloud.CloudResponse;
    _ = cloud.CloudProvider;
    _ = cloud.CloudHandler;
    _ = cloud.CloudConfig;
    _ = cloud.CloudError;
    _ = cloud.HttpMethod;
    _ = cloud.InvocationMetadata;
    _ = cloud.ResponseBuilder;
}

// ── CloudProvider ──────────────────────────────────────────────────────

test "cloud: CloudProvider names" {
    try std.testing.expectEqualStrings("AWS Lambda", cloud.CloudProvider.aws_lambda.name());
    try std.testing.expectEqualStrings("Google Cloud Functions", cloud.CloudProvider.gcp_functions.name());
    try std.testing.expectEqualStrings("Azure Functions", cloud.CloudProvider.azure_functions.name());
}

test "cloud: CloudProvider runtime identifiers" {
    try std.testing.expectEqualStrings("provided.al2023", cloud.CloudProvider.aws_lambda.runtimeIdentifier());
    try std.testing.expectEqualStrings("zig-runtime", cloud.CloudProvider.gcp_functions.runtimeIdentifier());
    try std.testing.expectEqualStrings("custom", cloud.CloudProvider.azure_functions.runtimeIdentifier());
}

// ── HttpMethod ─────────────────────────────────────────────────────────

test "cloud: HttpMethod fromString round-trip" {
    const methods = [_]struct { str: []const u8, val: cloud.HttpMethod }{
        .{ .str = "GET", .val = .GET },
        .{ .str = "POST", .val = .POST },
        .{ .str = "PUT", .val = .PUT },
        .{ .str = "DELETE", .val = .DELETE },
        .{ .str = "PATCH", .val = .PATCH },
        .{ .str = "HEAD", .val = .HEAD },
        .{ .str = "OPTIONS", .val = .OPTIONS },
    };

    for (methods) |m| {
        const parsed = cloud.HttpMethod.fromString(m.str);
        try std.testing.expect(parsed != null);
        try std.testing.expectEqual(m.val, parsed.?);
        try std.testing.expectEqualStrings(m.str, parsed.?.toString());
    }

    try std.testing.expect(cloud.HttpMethod.fromString("INVALID") == null);
}

// ── CloudEvent ─────────────────────────────────────────────────────────

test "cloud: CloudEvent init and deinit" {
    const allocator = std.testing.allocator;
    var event = cloud.CloudEvent.init(allocator, .aws_lambda, "req-001");
    defer event.deinit();

    try std.testing.expectEqual(cloud.CloudProvider.aws_lambda, event.provider);
    try std.testing.expectEqualStrings("req-001", event.request_id);
    try std.testing.expect(event.timestamp != 0);
}

test "cloud: CloudEvent is not HTTP by default" {
    const allocator = std.testing.allocator;
    var event = cloud.CloudEvent.init(allocator, .gcp_functions, "req-002");
    defer event.deinit();

    try std.testing.expect(!event.isHttpRequest());
}

test "cloud: CloudEvent becomes HTTP with method" {
    const allocator = std.testing.allocator;
    var event = cloud.CloudEvent.init(allocator, .azure_functions, "req-003");
    defer event.deinit();

    event.method = .POST;
    try std.testing.expect(event.isHttpRequest());
}

test "cloud: CloudEvent provider context defaults" {
    const ctx = cloud.CloudEvent.ProviderContext{};
    try std.testing.expect(ctx.function_arn == null);
    try std.testing.expect(ctx.project_id == null);
    try std.testing.expect(ctx.invocation_id == null);
    try std.testing.expect(ctx.remaining_time_ms == null);
}

// ── CloudResponse ──────────────────────────────────────────────────────

test "cloud: CloudResponse init defaults" {
    const allocator = std.testing.allocator;
    var resp = cloud.CloudResponse.init(allocator);
    defer resp.deinit();

    try std.testing.expectEqual(@as(u16, 200), resp.status_code);
    try std.testing.expectEqualStrings("", resp.body);
    try std.testing.expect(!resp.is_base64_encoded);
}

test "cloud: CloudResponse json factory" {
    const allocator = std.testing.allocator;
    var resp = try cloud.CloudResponse.json(allocator, "{\"ok\":true}");
    defer resp.deinit();

    try std.testing.expectEqual(@as(u16, 200), resp.status_code);
    try std.testing.expectEqualStrings("application/json", resp.headers.get("Content-Type").?);
    try std.testing.expectEqualStrings("{\"ok\":true}", resp.body);
}

test "cloud: CloudResponse text factory" {
    const allocator = std.testing.allocator;
    var resp = try cloud.CloudResponse.text(allocator, "hello");
    defer resp.deinit();

    try std.testing.expectEqualStrings("text/plain", resp.headers.get("Content-Type").?);
    try std.testing.expectEqualStrings("hello", resp.body);
}

test "cloud: CloudResponse error factory" {
    const allocator = std.testing.allocator;
    var resp = try cloud.CloudResponse.err(allocator, 500, "Internal Server Error");
    defer {
        allocator.free(resp.body);
        resp.deinit();
    }

    try std.testing.expectEqual(@as(u16, 500), resp.status_code);
    try std.testing.expect(resp.body.len > 0);
}

test "cloud: CloudResponse setHeader and setStatus" {
    const allocator = std.testing.allocator;
    var resp = cloud.CloudResponse.init(allocator);
    defer resp.deinit();

    try resp.setHeader("X-Custom", "value");
    resp.setStatus(201);
    resp.setBody("created");

    try std.testing.expectEqual(@as(u16, 201), resp.status_code);
    try std.testing.expectEqualStrings("value", resp.headers.get("X-Custom").?);
    try std.testing.expectEqualStrings("created", resp.body);
}

// ── ResponseBuilder ────────────────────────────────────────────────────

test "cloud: ResponseBuilder fluent API" {
    const allocator = std.testing.allocator;
    var builder = cloud.ResponseBuilder.init(allocator);
    var resp = builder
        .status(201)
        .json()
        .header("X-Request-Id", "abc-123")
        .body("{\"created\":true}")
        .build();
    defer resp.deinit();

    try std.testing.expectEqual(@as(u16, 201), resp.status_code);
    try std.testing.expectEqualStrings("application/json", resp.headers.get("Content-Type").?);
    try std.testing.expectEqualStrings("abc-123", resp.headers.get("X-Request-Id").?);
    try std.testing.expectEqualStrings("{\"created\":true}", resp.body);
}

test "cloud: ResponseBuilder cors helper" {
    const allocator = std.testing.allocator;
    var builder = cloud.ResponseBuilder.init(allocator);
    var resp = builder.cors("*").build();
    defer resp.deinit();

    try std.testing.expectEqualStrings("*", resp.headers.get("Access-Control-Allow-Origin").?);
    try std.testing.expect(resp.headers.get("Access-Control-Allow-Methods") != null);
    try std.testing.expect(resp.headers.get("Access-Control-Allow-Headers") != null);
}

// ── CloudConfig ────────────────────────────────────────────────────────

test "cloud: CloudConfig defaults" {
    const cfg = cloud.CloudConfig.defaults();
    try std.testing.expectEqual(@as(u32, 256), cfg.memory_mb);
    try std.testing.expectEqual(@as(u32, 30), cfg.timeout_seconds);
    try std.testing.expect(cfg.logging_enabled);
    try std.testing.expect(!cfg.tracing_enabled);
    try std.testing.expect(cfg.cors == null);
}

test "cloud: CloudConfig LogLevel toString" {
    try std.testing.expectEqualStrings("info", cloud.CloudConfig.LogLevel.info.toString());
    try std.testing.expectEqualStrings("debug", cloud.CloudConfig.LogLevel.debug.toString());
}

// ── InvocationMetadata ─────────────────────────────────────────────────

test "cloud: InvocationMetadata duration_ms" {
    const meta = cloud.InvocationMetadata{
        .request_id = "inv-001",
        .provider = .aws_lambda,
        .start_time = 1000,
        .end_time = 1250,
    };

    try std.testing.expectEqual(@as(i64, 250), meta.duration_ms().?);
}

test "cloud: InvocationMetadata duration_ms null when incomplete" {
    const meta = cloud.InvocationMetadata{
        .request_id = "inv-002",
        .provider = .gcp_functions,
        .start_time = 1000,
    };

    try std.testing.expect(meta.duration_ms() == null);
}

// ── Context lifecycle ──────────────────────────────────────────────────

test "cloud: Context init and deinit" {
    const allocator = std.testing.allocator;
    const cfg = cloud.CloudConfig.defaults();
    const ctx = try cloud.Context.init(allocator, cfg);
    defer ctx.deinit();

    // In a test environment, provider detection is expected to return null.
    _ = ctx.getProvider();
    _ = ctx.isCloudEnvironment();
}

// ── Module lifecycle ───────────────────────────────────────────────────

test "cloud: isEnabled returns true" {
    try std.testing.expect(cloud.isEnabled());
}

test "cloud: init and deinit lifecycle" {
    const allocator = std.testing.allocator;
    try cloud.init(allocator);
    try std.testing.expect(cloud.isInitialized());

    cloud.deinit();
    try std.testing.expect(!cloud.isInitialized());
}

// ── Provider detection ─────────────────────────────────────────────────

test "cloud: detectProvider runs without crash" {
    // In test environments no cloud provider should be detected, but the
    // function must not panic or leak.
    _ = cloud.detectProvider();
}

test {
    std.testing.refAllDecls(@This());
}
