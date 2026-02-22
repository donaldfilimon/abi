//! AWS Lambda Adapter
//!
//! Provides event parsing, response formatting, and context extraction
//! for AWS Lambda functions. Supports both HTTP API Gateway events and
//! direct Lambda invocations.
//!
//! ## Usage
//!
//! ```zig
//! const cloud = @import("abi").cloud;
//!
//! pub fn handler(event: *cloud.CloudEvent, allocator: std.mem.Allocator) !cloud.CloudResponse {
//!     return try cloud.CloudResponse.json(allocator, "{\"message\":\"Hello from Lambda!\"}");
//! }
//!
//! pub fn main() !void {
//!     var gpa = std.heap.GeneralPurposeAllocator(.{}){};
//!     defer _ = gpa.deinit();
//!     const allocator = gpa.allocator();
//!
//!     try cloud.aws_lambda.runHandler(allocator, handler);
//! }
//! ```

const std = @import("std");
const types = @import("types.zig");
const utils = @import("../../services/shared/utils.zig");
const this_module = @This();

pub const CloudEvent = types.CloudEvent;
pub const CloudResponse = types.CloudResponse;
pub const CloudHandler = types.CloudHandler;
pub const CloudError = types.CloudError;
pub const HttpMethod = types.HttpMethod;
pub const InvocationMetadata = types.InvocationMetadata;

// Shared helpers from types.zig
const jsonStringOrNull = types.jsonStringOrNull;
const parseJsonStringMap = types.parseJsonStringMap;
const parseJsonHeaderMap = types.parseJsonHeaderMap;

/// AWS Lambda runtime API client.
pub const LambdaRuntime = struct {
    allocator: std.mem.Allocator,
    runtime_api: []const u8,
    handler: CloudHandler,
    cold_start: bool = true,

    /// Initialize the Lambda runtime.
    pub fn init(allocator: std.mem.Allocator, handler: CloudHandler) !LambdaRuntime {
        const runtime_api_ptr = std.c.getenv("AWS_LAMBDA_RUNTIME_API") orelse {
            return CloudError.InvalidConfig;
        };
        const runtime_api = std.mem.span(runtime_api_ptr);

        return .{
            .allocator = allocator,
            .runtime_api = runtime_api,
            .handler = handler,
        };
    }

    /// Run the Lambda runtime loop.
    /// This function does not return under normal operation.
    pub fn run(self: *LambdaRuntime) !void {
        while (true) {
            try self.processNextInvocation();
            self.cold_start = false;
        }
    }

    /// Process a single invocation.
    fn processNextInvocation(self: *LambdaRuntime) !void {
        // Get next invocation from runtime API
        const invocation = try self.getNextInvocation();
        defer invocation.deinit();

        const request_id = invocation.request_id;
        const start_time = utils.unixMs();

        // Parse the event
        var event = try parseEvent(self.allocator, invocation.body, request_id);
        defer event.deinit();

        // Set AWS-specific context
        event.context.function_arn = invocation.function_arn;
        event.context.log_group = invocation.log_group;
        event.context.log_stream = invocation.log_stream;
        event.context.remaining_time_ms = invocation.deadline_ms;

        // Execute handler
        var response: CloudResponse = undefined;
        var error_message: ?[]const u8 = null;

        if (self.handler(&event, self.allocator)) |resp| {
            response = resp;
        } else |err| {
            var err_name_buf: [128]u8 = undefined;
            error_message = std.fmt.bufPrint(&err_name_buf, "{t}", .{err}) catch "HandlerError";
            response = try CloudResponse.err(self.allocator, 500, error_message.?);
        }
        defer response.deinit();

        // Send response back to runtime API
        try self.sendResponse(request_id, &response);

        // Log invocation metadata
        const metadata = InvocationMetadata{
            .request_id = request_id,
            .provider = .aws_lambda,
            .start_time = start_time,
            .end_time = utils.unixMs(),
            .status_code = response.status_code,
            .error_message = error_message,
            .cold_start = self.cold_start,
        };
        _ = metadata;
    }

    /// Invocation data from the runtime API.
    const Invocation = struct {
        request_id: []const u8,
        body: []const u8,
        function_arn: ?[]const u8 = null,
        log_group: ?[]const u8 = null,
        log_stream: ?[]const u8 = null,
        deadline_ms: ?u64 = null,
        allocator: std.mem.Allocator,

        pub fn deinit(self: *const Invocation) void {
            self.allocator.free(self.body);
        }
    };

    /// Get the next invocation from the runtime API.
    /// Makes an HTTP GET request to the Lambda runtime API endpoint:
    /// http://{runtime_api}/2018-06-01/runtime/invocation/next
    /// This call blocks until the next invocation is available.
    fn getNextInvocation(self: *LambdaRuntime) !Invocation {
        const path = "/2018-06-01/runtime/invocation/next";

        // Build and send the HTTP GET request
        var req_buf: [512]u8 = undefined;
        const request = std.fmt.bufPrint(&req_buf, "GET {s} HTTP/1.1\r\nHost: {s}\r\nAccept: */*\r\n\r\n", .{
            path, self.runtime_api,
        }) catch return CloudError.ProviderError;

        const response_data = try this_module.httpRequest(self.allocator, self.runtime_api, request);
        errdefer self.allocator.free(response_data.body);

        return .{
            .request_id = response_data.request_id orelse "unknown",
            .body = response_data.body,
            .function_arn = response_data.function_arn,
            .log_group = response_data.log_group,
            .log_stream = response_data.log_stream,
            .deadline_ms = response_data.deadline_ms,
            .allocator = self.allocator,
        };
    }

    /// Send the response back to the runtime API.
    /// Makes an HTTP POST to:
    /// http://{runtime_api}/2018-06-01/runtime/invocation/{request_id}/response
    fn sendResponse(self: *LambdaRuntime, request_id: []const u8, response: *const CloudResponse) !void {
        // Format the response body
        const body = try this_module.formatResponse(self.allocator, response);
        defer self.allocator.free(body);

        // Build path
        var path_buf: [512]u8 = undefined;
        const path = std.fmt.bufPrint(&path_buf, "/2018-06-01/runtime/invocation/{s}/response", .{
            request_id,
        }) catch return CloudError.ProviderError;

        // Build full HTTP POST request
        var req_buf = std.ArrayListUnmanaged(u8).empty;
        defer req_buf.deinit(self.allocator);
        const writer = req_buf.writer(self.allocator);

        try std.fmt.format(writer, "POST {s} HTTP/1.1\r\n", .{path});
        try std.fmt.format(writer, "Host: {s}\r\n", .{self.runtime_api});
        try writer.writeAll("Content-Type: application/json\r\n");
        try std.fmt.format(writer, "Content-Length: {d}\r\n", .{body.len});
        try writer.writeAll("\r\n");
        try writer.writeAll(body);

        const resp = try this_module.httpRequest(self.allocator, self.runtime_api, req_buf.items);
        self.allocator.free(resp.body);
    }

    /// Report an error to the runtime API.
    /// Makes an HTTP POST to:
    /// http://{runtime_api}/2018-06-01/runtime/invocation/{request_id}/error
    fn sendError(self: *LambdaRuntime, request_id: []const u8, error_type: []const u8, message: []const u8) !void {
        // Build error body JSON
        var body_buf = std.ArrayListUnmanaged(u8).empty;
        defer body_buf.deinit(self.allocator);
        const body_writer = body_buf.writer(self.allocator);
        try body_writer.writeAll("{\"errorType\":\"");
        try body_writer.writeAll(error_type);
        try body_writer.writeAll("\",\"errorMessage\":\"");
        try body_writer.writeAll(message);
        try body_writer.writeAll("\"}");

        // Build path
        var path_buf: [512]u8 = undefined;
        const path = std.fmt.bufPrint(&path_buf, "/2018-06-01/runtime/invocation/{s}/error", .{
            request_id,
        }) catch return CloudError.ProviderError;

        // Build full HTTP POST request
        var req_buf = std.ArrayListUnmanaged(u8).empty;
        defer req_buf.deinit(self.allocator);
        const writer = req_buf.writer(self.allocator);

        try std.fmt.format(writer, "POST {s} HTTP/1.1\r\n", .{path});
        try std.fmt.format(writer, "Host: {s}\r\n", .{self.runtime_api});
        try writer.writeAll("Content-Type: application/json\r\n");
        try writer.writeAll("Lambda-Runtime-Function-Error-Type: Runtime.HandlerError\r\n");
        try std.fmt.format(writer, "Content-Length: {d}\r\n", .{body_buf.items.len});
        try writer.writeAll("\r\n");
        try writer.writeAll(body_buf.items);

        const resp = try this_module.httpRequest(self.allocator, self.runtime_api, req_buf.items);
        self.allocator.free(resp.body);
    }
};

// ============================================================================
// HTTP Client using POSIX sockets
// ============================================================================

/// Parsed response from an HTTP request to the Lambda runtime API.
const HttpResponse = struct {
    body: []u8,
    request_id: ?[]const u8 = null,
    function_arn: ?[]const u8 = null,
    log_group: ?[]const u8 = null,
    log_stream: ?[]const u8 = null,
    deadline_ms: ?u64 = null,
};

/// Parse a dotted-decimal IPv4 address string into 4 bytes.
/// Returns null if the string is not a valid IPv4 address.
fn parseIp4Bytes(text: []const u8) ?[4]u8 {
    var bytes: [4]u8 = undefined;
    var index: u8 = 0;
    var current: u16 = 0;
    var saw_digit = false;

    for (text) |ch| {
        switch (ch) {
            '.' => {
                if (!saw_digit or index >= 3) return null;
                bytes[index] = @intCast(current);
                index += 1;
                current = 0;
                saw_digit = false;
            },
            '0'...'9' => {
                saw_digit = true;
                current = current * 10 + @as(u16, ch - '0');
                if (current > 255) return null;
            },
            else => return null,
        }
    }

    if (!saw_digit or index != 3) return null;
    bytes[index] = @intCast(current);
    return bytes;
}

/// Parse a "host:port" string. If no port, defaults to 80.
fn parseHostPort(host_port: []const u8) struct { host: []const u8, port: u16 } {
    if (std.mem.lastIndexOfScalar(u8, host_port, ':')) |colon| {
        const port = std.fmt.parseInt(u16, host_port[colon + 1 ..], 10) catch 80;
        return .{ .host = host_port[0..colon], .port = port };
    }
    return .{ .host = host_port, .port = 80 };
}

/// Perform an HTTP request to the Lambda runtime API via POSIX sockets.
/// Connects to the host specified in `host_port`, sends `request_bytes`,
/// reads the full response, and extracts Lambda-specific headers.
fn httpRequest(
    allocator: std.mem.Allocator,
    host_port: []const u8,
    request_bytes: []const u8,
) !HttpResponse {
    const hp = parseHostPort(host_port);

    // Parse the host as an IPv4 dotted-decimal address.
    // The Lambda runtime API is typically a local address (e.g. 127.0.0.1:9001).
    const ip_bytes = parseIp4Bytes(hp.host) orelse [4]u8{ 127, 0, 0, 1 };

    const sock = std.c.socket(std.c.AF.INET, std.c.SOCK.STREAM, 0);
    if (sock < 0) return CloudError.ProviderError;
    defer _ = std.c.close(sock);

    var addr: std.c.sockaddr.in = .{
        .port = std.mem.nativeToBig(u16, hp.port),
        .addr = @bitCast(ip_bytes),
    };
    if (std.c.connect(sock, @ptrCast(&addr), @sizeOf(std.c.sockaddr.in)) < 0) {
        return CloudError.ProviderError;
    }

    // Send the request
    var total_sent: usize = 0;
    while (total_sent < request_bytes.len) {
        const n = std.c.send(sock, @ptrCast(request_bytes[total_sent..].ptr), request_bytes.len - total_sent, 0);
        if (n <= 0) return CloudError.ProviderError;
        total_sent += @intCast(n);
    }

    // Read response (header + body) into a dynamic buffer
    var resp_buf = std.ArrayListUnmanaged(u8).empty;
    defer resp_buf.deinit(allocator);

    var read_buf: [8192]u8 = undefined;
    var headers_complete = false;
    var content_length: ?usize = null;
    var header_end_pos: usize = 0;

    while (true) {
        const n = std.c.recv(sock, @ptrCast(&read_buf), read_buf.len, 0);
        if (n <= 0) break;
        try resp_buf.appendSlice(allocator, read_buf[0..@intCast(n)]);

        if (!headers_complete) {
            if (std.mem.indexOf(u8, resp_buf.items, "\r\n\r\n")) |end| {
                headers_complete = true;
                header_end_pos = end + 4;

                // Parse Content-Length from headers
                const headers_section = resp_buf.items[0..end];
                var lines = std.mem.splitSequence(u8, headers_section, "\r\n");
                while (lines.next()) |line| {
                    if (std.mem.indexOf(u8, line, ": ")) |sep| {
                        const key = line[0..sep];
                        const value = line[sep + 2 ..];
                        if (std.ascii.eqlIgnoreCase(key, "content-length")) {
                            content_length = std.fmt.parseInt(usize, value, 10) catch null;
                        }
                    }
                }
            }
        }

        // Check if we have the complete body
        if (headers_complete) {
            if (content_length) |cl| {
                const body_received = resp_buf.items.len - header_end_pos;
                if (body_received >= cl) break;
            }
        }
    }

    if (!headers_complete) return CloudError.ProviderError;

    // Extract Lambda-specific headers
    var result: HttpResponse = .{
        .body = undefined,
    };

    const headers_section = resp_buf.items[0 .. header_end_pos - 4];
    var lines = std.mem.splitSequence(u8, headers_section, "\r\n");
    while (lines.next()) |line| {
        if (std.mem.indexOf(u8, line, ": ")) |sep| {
            const key = line[0..sep];
            const value = line[sep + 2 ..];
            if (std.ascii.eqlIgnoreCase(key, "lambda-runtime-aws-request-id")) {
                result.request_id = value;
            } else if (std.ascii.eqlIgnoreCase(key, "lambda-runtime-invoked-function-arn")) {
                result.function_arn = value;
            } else if (std.ascii.eqlIgnoreCase(key, "lambda-runtime-log-group-name")) {
                result.log_group = value;
            } else if (std.ascii.eqlIgnoreCase(key, "lambda-runtime-log-stream-name")) {
                result.log_stream = value;
            } else if (std.ascii.eqlIgnoreCase(key, "lambda-runtime-deadline-ms")) {
                result.deadline_ms = std.fmt.parseInt(u64, value, 10) catch null;
            }
        }
    }

    // Copy body into an owned allocation
    const body_start = header_end_pos;
    const body_len = if (content_length) |cl| cl else resp_buf.items.len - body_start;
    const body = try allocator.alloc(u8, body_len);
    @memcpy(body, resp_buf.items[body_start..][0..body_len]);
    result.body = body;

    return result;
}

/// Parse an AWS Lambda event into a CloudEvent.
pub fn parseEvent(allocator: std.mem.Allocator, raw_event: []const u8, request_id: []const u8) !CloudEvent {
    var event = CloudEvent.init(allocator, .aws_lambda, request_id);
    errdefer event.deinit();

    // Parse the JSON event
    const parsed = std.json.parseFromSlice(std.json.Value, allocator, raw_event, .{}) catch {
        return CloudError.EventParseFailed;
    };
    defer parsed.deinit();

    const root = parsed.value;

    // Detect event type and parse accordingly
    if (root.object.get("httpMethod")) |method_val| {
        // API Gateway REST API event
        try parseApiGatewayEvent(&event, root, method_val.string);
    } else if (root.object.get("requestContext")) |ctx| {
        if (ctx.object.get("http")) |_| {
            // API Gateway HTTP API v2 event
            try parseHttpApiEvent(&event, root);
        }
    } else if (root.object.get("Records")) |_| {
        // Event source (S3, SQS, SNS, etc.)
        try parseEventSourceEvent(&event, root);
    } else {
        // Direct invocation - store raw body
        event.body = raw_event;
    }

    return event;
}

/// Parse API Gateway REST API event.
fn parseApiGatewayEvent(event: *CloudEvent, root: std.json.Value, method: []const u8) !void {
    event.method = HttpMethod.fromString(method);

    if (root.object.get("path")) |path| {
        event.path = path.string;
    }

    if (root.object.get("body")) |body| event.body = jsonStringOrNull(body);

    // Parse query parameters
    if (root.object.get("queryStringParameters")) |params| {
        if (params != .null) {
            event.query_params = try parseJsonStringMap(event.allocator, params);
        }
    }

    // Parse headers (lowercase keys for case-insensitive lookup)
    if (root.object.get("headers")) |headers| {
        if (headers != .null) {
            event.headers = try parseJsonHeaderMap(event.allocator, headers);
        }
    }
}

/// Parse API Gateway HTTP API v2 event.
fn parseHttpApiEvent(event: *CloudEvent, root: std.json.Value) !void {
    if (root.object.get("requestContext")) |ctx| {
        if (ctx.object.get("http")) |http| {
            if (http.object.get("method")) |method| {
                event.method = HttpMethod.fromString(method.string);
            }
            if (http.object.get("path")) |path| {
                event.path = path.string;
            }
        }
    }

    if (root.object.get("body")) |body| event.body = jsonStringOrNull(body);

    // Parse query parameters (already decoded in v2)
    if (root.object.get("queryStringParameters")) |params| {
        if (params != .null) {
            event.query_params = try parseJsonStringMap(event.allocator, params);
        }
    }

    // Parse headers (already lowercase in v2, but normalize for consistency)
    if (root.object.get("headers")) |headers| {
        if (headers != .null) {
            event.headers = try parseJsonHeaderMap(event.allocator, headers);
        }
    }
}

/// Parse event source records (S3, SQS, SNS, etc.).
fn parseEventSourceEvent(event: *CloudEvent, root: std.json.Value) !void {
    if (root.object.get("Records")) |records| {
        if (records.array.items.len > 0) {
            const first_record = records.array.items[0];

            if (first_record.object.get("eventSource")) |source| {
                event.source = source.string;
            }

            if (first_record.object.get("eventName")) |name| {
                event.event_type = name.string;
            }

            // For SQS, get the message body
            if (first_record.object.get("body")) |body| {
                event.body = body.string;
            }

            // For SNS, get the message
            if (first_record.object.get("Sns")) |sns| {
                if (sns.object.get("Message")) |msg| {
                    event.body = msg.string;
                }
            }
        }
    }
}

/// Format a CloudResponse for API Gateway.
pub fn formatResponse(allocator: std.mem.Allocator, response: *const CloudResponse) ![]const u8 {
    var buffer = std.ArrayListUnmanaged(u8).empty;
    errdefer buffer.deinit(allocator);

    const writer = buffer.writer(allocator);

    try writer.writeAll("{");

    // Status code
    try std.fmt.format(writer, "\"statusCode\":{d}", .{response.status_code});

    // Headers
    try writer.writeAll(",\"headers\":{");
    var first = true;
    var iter = response.headers.iterator();
    while (iter.next()) |entry| {
        if (!first) try writer.writeAll(",");
        first = false;
        try std.json.stringify(entry.key_ptr.*, .{}, writer);
        try writer.writeAll(":");
        try std.json.stringify(entry.value_ptr.*, .{}, writer);
    }
    try writer.writeAll("}");

    // Body
    try writer.writeAll(",\"body\":");
    try std.json.stringify(response.body, .{}, writer);

    // Base64 encoding flag
    if (response.is_base64_encoded) {
        try writer.writeAll(",\"isBase64Encoded\":true");
    }

    try writer.writeAll("}");

    return buffer.toOwnedSlice(allocator);
}

/// Run a handler in the Lambda runtime environment.
/// This is the main entry point for Lambda functions.
pub fn runHandler(allocator: std.mem.Allocator, handler: CloudHandler) !void {
    var runtime = try LambdaRuntime.init(allocator, handler);
    try runtime.run();
}

/// Create a simple handler wrapper for testing.
pub fn createTestHandler(comptime handler_fn: anytype) CloudHandler {
    return struct {
        pub fn handle(event: *CloudEvent, allocator: std.mem.Allocator) anyerror!CloudResponse {
            return handler_fn(event, allocator);
        }
    }.handle;
}

// ============================================================================
// Tests
// ============================================================================

test "parseEvent API Gateway REST" {
    const allocator = std.testing.allocator;

    const raw_event =
        \\{
        \\  "httpMethod": "POST",
        \\  "path": "/api/test",
        \\  "headers": {"Content-Type": "application/json"},
        \\  "queryStringParameters": {"id": "123"},
        \\  "body": "{\"message\":\"hello\"}"
        \\}
    ;

    var event = try parseEvent(allocator, raw_event, "test-id");
    defer event.deinit();

    try std.testing.expectEqual(HttpMethod.POST, event.method.?);
    try std.testing.expectEqualStrings("/api/test", event.path.?);
    try std.testing.expectEqualStrings("{\"message\":\"hello\"}", event.body.?);
    try std.testing.expectEqualStrings("123", event.query_params.?.get("id").?);
}

test "parseEvent HTTP API v2" {
    const allocator = std.testing.allocator;

    const raw_event =
        \\{
        \\  "requestContext": {
        \\    "http": {
        \\      "method": "GET",
        \\      "path": "/items"
        \\    }
        \\  },
        \\  "headers": {"content-type": "text/plain"},
        \\  "body": null
        \\}
    ;

    var event = try parseEvent(allocator, raw_event, "test-id");
    defer event.deinit();

    try std.testing.expectEqual(HttpMethod.GET, event.method.?);
    try std.testing.expectEqualStrings("/items", event.path.?);
}

test "formatResponse" {
    const allocator = std.testing.allocator;

    var response = try CloudResponse.json(allocator, "{\"ok\":true}");
    defer response.deinit();

    const formatted = try formatResponse(allocator, &response);
    defer allocator.free(formatted);

    // Verify it's valid JSON
    const parsed = try std.json.parseFromSlice(std.json.Value, allocator, formatted, .{});
    defer parsed.deinit();

    try std.testing.expectEqual(@as(i64, 200), parsed.value.object.get("statusCode").?.integer);
}
